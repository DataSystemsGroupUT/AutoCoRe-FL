import logging
import os
import itertools
import pickle
import random
import numpy as np
import pandas as pd
import torch
import yaml 
from AutoCore_FL.federated.client import FederatedClient
from AutoCore_FL.federated.server import FederatedServer
from AutoCore_FL.federated.utils import setup_logging,  save_config, load_config, SAM2Filter, evaluate_global_AutoCore_model, load_labels_from_manifest
from AutoCore_FL.concepts.vectorizer import build_image_concept_vectors


# --- Main HPO Orchestration Function ---
def main_autocore_fl_hpo():
    # --- Load Base Configuration ---
    base_config_path = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/AutoCore_FL/configs/config_autocore_fl_auto_ade20k.yaml" #  main config file
    if not os.path.exists(base_config_path):
        print(f"ERROR: Base config file {base_config_path} not found. Please create it.")
        return
    
    base_config = load_config(base_config_path)
    
    # --- Setup Main Logging for HPO Process ---
    hpo_run_id_base = base_config.get('run_id_base', 'autocore_hpo') + "_HPO_main"
    hpo_main_log_dir = os.path.join(base_config.get("output_base_dir_fl_run", "./autocore_fl_run_results"), hpo_run_id_base, "hpo_logs")
    os.makedirs(hpo_main_log_dir, exist_ok=True)
    hpo_logger = setup_logging(log_dir=hpo_main_log_dir, run_id=hpo_run_id_base, log_level_str=base_config.get("log_level", "INFO"))
    root_logger_instance_hpo = logging.getLogger() # Get root to add filter
    root_logger_instance_hpo.addFilter(SAM2Filter()) # Add filter once for the HPO process
    hpo_logger.info(f"======== Starting AutoCoRe-FL Hyperparameter Optimization: {hpo_run_id_base} ========")
        
    # --- Define Hyperparameter Grid for AutoCore AND Server Aggregation ---
    hparam_grid = {
        'figs_max_rules': [20, 30, 40, 60 , 100],
        'figs_max_trees': [1,3, 5, None], # None means no limit, use all trees
        'figs_max_features': [None, 'sqrt'],
        'use_accuracy_weighting_server': [False,True],
        'server_rule_validation_min_precision': [0.1, 0.2, 0.3],
        'server_rule_min_coverage_count': [3, 5, 10]
    }
    # Create all combinations
    hparam_keys, hparam_values = zip(*hparam_grid.items())
    hyperparameter_combinations = [dict(zip(hparam_keys, v)) for v in itertools.product(*hparam_values)]
    hpo_logger.info(f"Generated {len(hyperparameter_combinations)} total hyperparameter combinations for tuning.")
    save_config(base_config, os.path.join(hpo_main_log_dir, "base_config.yaml"))
    # --- Seed & Device ---
    np.random.seed(base_config['seed'])
    torch.manual_seed(base_config['seed'])
    random.seed(base_config['seed']) 
    if base_config['device'] == 'cuda' and torch.cuda.is_available():
        torch_device_main = torch.device("cuda"); torch.cuda.manual_seed_all(base_config['seed'])
    else:
        torch_device_main = torch.device("cpu")
    hpo_logger.info(f"Global HPO using device: {torch_device_main}")

    # === STAGE 0: ONE-TIME DATA LOADING AND PREPROCESSING (K-Means, Detectors, Vectorization) ===
    hpo_logger.info("======== STAGE 0: Performing One-Time Data Loading, K-Means, Detector Sync, and Vectorization ========")
    
    # Create a config copy for this one-time setup, ensuring it doesn't get overwritten by HPO loop's figs_params
    setup_config = base_config.copy()
    setup_config['current_run_id'] = base_config.get('run_id_base', 'autocore_setup') + "_data_prep" # Unique ID for this phase
    # Use a sub-logger for this setup phase if desired, or use hpo_logger
    
    # Load Global Class Mapping
    cached_data_base_path = setup_config['cached_data_base_dir']
    path_to_scene_map_json = os.path.join(cached_data_base_path, setup_config['partition_manifest_dir_input'], setup_config['scene_to_idx_map_filename_input'])
    try:
        with open(path_to_scene_map_json, 'r') as f: scene_to_global_idx_map = yaml.safe_load(f)
        setup_config['num_classes'] = len(scene_to_global_idx_map)
        hpo_logger.info(f"Loaded scene_to_idx_map ({setup_config['num_classes']} classes).")
    except Exception as e:
        hpo_logger.error(f"Failed to load scene_to_idx_map.json for setup: {e}. HPO cannot continue."); return

    # Initialize Clients with Pre-cached Data
    clients_setup = []
    num_configured_clients_setup = setup_config.get('num_clients', 10)
    for i in range(num_configured_clients_setup):
        client_id_str = f"client_{i}"
        client_cfg_setup = setup_config.copy(); client_cfg_setup['client_id'] = i
        client_obj = FederatedClient(client_id=i, config=client_cfg_setup, data_partition=None, scene_to_idx=scene_to_global_idx_map, skip_model_initialization=False)
        try:
            manifest_path = os.path.join(cached_data_base_path, setup_config['partition_manifest_dir_input'], client_id_str, 'image_manifest.json')
            seg_infos_path = os.path.join(cached_data_base_path, setup_config['partition_segment_infos_cache_dir_input'], f"{client_id_str}_segment_infos_with_crops.pkl")
            embeddings_path = os.path.join(cached_data_base_path, setup_config['embedding_cache_dir_input'], f"embeddings_{setup_config['embedding_type']}_{client_id_str}.pkl")
            client_base_ids, client_labels_int = load_labels_from_manifest(manifest_path, hpo_logger)
            if not client_base_ids: raise FileNotFoundError(f"Manifest empty/not loaded for {client_id_str}")
            with open(seg_infos_path, 'rb') as f: client_obj.state['filtered_segment_infos'] = pickle.load(f)
            with open(embeddings_path, 'rb') as f: client_obj.state['final_embeddings'] = pickle.load(f)
            client_obj.state['loaded_base_ids'] = client_base_ids
            client_obj.state['loaded_labels_int'] = client_labels_int
            if client_obj.state['filtered_segment_infos'] is None or client_obj.state['final_embeddings'] is None or \
               len(client_obj.state['filtered_segment_infos']) != client_obj.state['final_embeddings'].shape[0]:
                hpo_logger.error(f"SETUP: Client {i} Data Mismatch. Marking inactive."); client_obj.state['final_embeddings'] = None
        except Exception as e_load_setup:
            hpo_logger.error(f"SETUP: Failed to load data for client {i}: {e_load_setup}. Marking inactive."); client_obj.state['final_embeddings'] = None
        clients_setup.append(client_obj)

    server_setup = FederatedServer(setup_config.copy())

    # STAGE 1 (K-Means) for Setup
    hpo_logger.info(f"--- Starting Phase 1: Federated K-Means ---")
    current_centroids = server_setup.initialize_centroids(setup_config["embedding_dim"], setup_config["num_clusters"])
    client_kmeans_stats_final_round_setup = []
    for r_km_setup in range(setup_config["kmeans_rounds"]):
        hpo_logger.info(f"KMeans Round {r_km_setup + 1}/{setup_config['kmeans_rounds']}")
        client_kmeans_stats_current_round_setup = []
        active_clients_this_round_km_setup = 0
        for client_s in clients_setup:
            if client_s.state.get('final_embeddings') is None or client_s.state['final_embeddings'].shape[0] == 0: continue
            sums_s, counts_s = client_s.run_local_pipeline(global_centroids=current_centroids)
            if sums_s is not None and counts_s is not None: client_kmeans_stats_current_round_setup.append((sums_s, counts_s)); active_clients_this_round_km_setup +=1
        if active_clients_this_round_km_setup == 0: 
            if not client_kmeans_stats_final_round_setup: hpo_logger.error("SETUP: K-Means failed."); return
            break
        current_centroids, live_mask = server_setup.aggregate_kmeans(client_kmeans_stats_current_round_setup)
        client_kmeans_stats_final_round_setup = client_kmeans_stats_current_round_setup
        hpo_logger.info(f"KMeans R{r_km_setup+1}: Aggregated Centroids shape: {current_centroids.shape if current_centroids is not None else 'None'}, Live concepts (mask sum): {live_mask.sum() if live_mask is not None else 'N/A'}")

        if current_centroids is None or live_mask is None or live_mask.sum()==0:
            if not client_kmeans_stats_final_round_setup: hpo_logger.error("SETUP: K-Means agg failed."); return
            break
    
    if not client_kmeans_stats_final_round_setup: hpo_logger.error("SETUP: K-Means no valid stats."); return
    final_counts_agg_list_setup = [c for s,c in client_kmeans_stats_final_round_setup if c is not None and c.ndim > 0 and c.size > 0 and c.shape[0] == setup_config["num_clusters"]]
    if not final_counts_agg_list_setup: hpo_logger.error("SETUP: No K-Means counts."); return
    final_counts_agg_setup = np.sum(np.stack(final_counts_agg_list_setup), axis=0)
    keep_mask_concepts_setup = final_counts_agg_setup >= setup_config.get('min_samples_per_concept_cluster', 30)
    initial_shared_concept_indices_setup = np.where(keep_mask_concepts_setup)[0].tolist()
    if not initial_shared_concept_indices_setup: hpo_logger.error("SETUP: No K-Means concepts survived."); return
    hpo_logger.info(f"SETUP: Stage 1 K-Means Done. {len(initial_shared_concept_indices_setup)} initial concepts.")

    # --- PHASE 2 Preparation: Detector Sync ---
    hpo_logger.info(f"--- Starting Phase 2 Prep: Detector Sync ---")
    for client_s in clients_setup:
        if client_s.state.get('final_embeddings') is not None and client_s.state.get('cluster_labels') is not None: client_s.train_concept_detectors(setup_config)
    all_detector_updates_setup = [client_s.get_detector_update(initial_shared_concept_indices_setup) for client_s in clients_setup]
    final_phase2_concept_indices_ordered_setup, canonical_detector_params_for_broadcast_setup = server_setup.aggregate_detectors(all_detector_updates_setup, initial_shared_concept_indices_setup)
    num_AutoCore_features_setup = len(final_phase2_concept_indices_ordered_setup)
    if num_AutoCore_features_setup == 0: hpo_logger.error("SETUP: No detectors survived."); return
    num_AutoCore_features_setup = [f"concept_{i}" for i in range(num_AutoCore_features_setup)]; server_setup.feature_names_for_figs = num_AutoCore_features_setup
    final_original_to_dense_map_setup = {orig_idx: dense_idx for dense_idx, orig_idx in enumerate(final_phase2_concept_indices_ordered_setup)}
    server_setup.phase2_prep_broadcast(clients_setup, final_phase2_concept_indices_ordered_setup, final_original_to_dense_map_setup, canonical_detector_params_for_broadcast_setup)
    hpo_logger.info(f"SETUP: Stage 2 Detector Sync Done. {num_AutoCore_features_setup} final AutoCoRe-FL features.")

    # Client-Side Vectorization (ONCE)
    hpo_logger.info("SETUP: Client-Side Vectorization (occurs once)...")
    client_X_concepts_fixed = {} # Store {client_id: X_concepts_np_array}
    client_Y_labels_fixed = {}   # Store {client_id: Y_labels_np_array}
    for client_s in clients_setup:
        if client_s.state.get('final_embeddings') is None: continue # Skip inactive
        client_concept_vectors_s, client_kept_image_ids_s = client_s.build_concept_vectors(setup_config)
        if client_concept_vectors_s is not None and client_concept_vectors_s.shape[0] > 0:
            map_base_id_to_label_client_s = {bid: lbl for bid, lbl in zip(client_s.state['loaded_base_ids'], client_s.state['loaded_labels_int'])}
            client_true_labels_for_figs_s = np.array([map_base_id_to_label_client_s[bid] for bid in client_kept_image_ids_s if bid in map_base_id_to_label_client_s])
            if client_true_labels_for_figs_s.shape[0] == client_concept_vectors_s.shape[0]:
                client_X_concepts_fixed[client_s.client_id] = client_concept_vectors_s
                client_Y_labels_fixed[client_s.client_id] = client_true_labels_for_figs_s
                client_s.state['concept_vecs'] = client_concept_vectors_s # Update client state with these fixed vectors
            else: hpo_logger.warning(f"SETUP: Label/Vector mismatch for client {client_s.client_id} during fixed vectorization.")
        else: hpo_logger.warning(f"SETUP: No concept vectors generated for client {client_s.client_id}.")
    
    server_X_val_concepts_fixed, server_y_val_labels_fixed = None, None
    server_X_test_concepts_fixed, server_y_test_labels_fixed = None, None

    for holdout_set_name_setup in ["server_validation_set", "server_test_set"]:
        try:
            manifest_path_s_setup = os.path.join(cached_data_base_path, setup_config['partition_manifest_dir_input'], holdout_set_name_setup, 'image_manifest.json')
            server_holdout_base_ids_s, server_holdout_labels_int_s = load_labels_from_manifest(manifest_path_s_setup, hpo_logger)
            if server_holdout_base_ids_s:
                seg_infos_path_s = os.path.join(cached_data_base_path, setup_config['partition_segment_infos_cache_dir_input'], f"{holdout_set_name_setup}_segment_infos_with_crops.pkl")
                embeddings_path_s = os.path.join(cached_data_base_path, setup_config['embedding_cache_dir_input'], f"embeddings_{setup_config['embedding_type']}_{holdout_set_name_setup}.pkl")
                with open(seg_infos_path_s, 'rb') as f: server_seg_infos_holdout_s = pickle.load(f)
                with open(embeddings_path_s, 'rb') as f: server_embeddings_holdout_s = pickle.load(f)
                
                if server_seg_infos_holdout_s is not None and server_embeddings_holdout_s is not None and len(server_seg_infos_holdout_s) == server_embeddings_holdout_s.shape[0]:
                    server_dense_detectors_s = {
                        final_original_to_dense_map_setup[ok]: mo
                        for ok, mo in server_setup.canonical_detector_model_objects.items()
                        if ok in final_original_to_dense_map_setup
                    }
                    server_dense_thresholds_s = {
                        final_original_to_dense_map_setup[ok]: th
                        for ok, th in server_setup.canonical_thresholds.items()
                        if ok in final_original_to_dense_map_setup
                    }
                    for dense_idx, thresh_val in list(server_dense_thresholds_s.items()):
                        if isinstance(thresh_val, (list, tuple, np.ndarray)):
                            server_dense_thresholds_s[dense_idx] = float(np.mean(thresh_val))
                        else:
                            server_dense_thresholds_s[dense_idx] = float(thresh_val)
                    if server_dense_detectors_s:
                        current_X_s_concepts, kept_ids_s = build_image_concept_vectors(server_seg_infos_holdout_s,
                                                                                        server_dense_detectors_s, 
                                                                                        server_dense_thresholds_s, 
                                                                                        server_embeddings_holdout_s, 
                                                                                        num_AutoCore_features_setup, 
                                                                                        setup_config)
                        if current_X_s_concepts is not None and current_X_s_concepts.shape[0] > 0:
                            map_bid_lbl_s = {bid:lbl for bid,lbl in zip(server_holdout_base_ids_s, server_holdout_labels_int_s)}
                            aligned_lbls_s = np.array([map_bid_lbl_s[bid] for bid in kept_ids_s if bid in map_bid_lbl_s])
                            if current_X_s_concepts.shape[0] == aligned_lbls_s.shape[0]:
                                if holdout_set_name_setup == "server_validation_set": server_X_val_concepts_fixed, server_y_val_labels_fixed = current_X_s_concepts, aligned_lbls_s
                                elif holdout_set_name_setup == "server_test_set": server_X_test_concepts_fixed, server_y_test_labels_fixed = current_X_s_concepts, aligned_lbls_s
        except Exception as e_s_setup: hpo_logger.error(f"SETUP: Error processing server set {holdout_set_name_setup}: {e_s_setup}")
    
    if server_X_val_concepts_fixed is None or server_y_val_labels_fixed is None:
        hpo_logger.error("SETUP: Server validation concept vectors could not be generated. HPO cannot proceed.")
        return
    if server_X_test_concepts_fixed is None or server_y_test_labels_fixed is None:
        hpo_logger.warning("SETUP: Server test concept vectors could not be generated. Final testing will be skipped.")
        # Allow HPO to proceed on validation set, but final test won't happen.

    hpo_logger.info(f"SETUP: Val concepts: {server_X_val_concepts_fixed.shape}, Test concepts: {server_X_test_concepts_fixed.shape if server_X_test_concepts_fixed is not None else 'None'}")
    hpo_logger.info("======== STAGE 0: One-Time Data Processing Complete ========")
    # --- START: Noise Preparation (Done ONCE before HPO loop if noisy clients are fixed for all HPO trials) ---
    client_Y_labels_for_hpo_training = {} # This will hold the labels (clean or noisy) for each client
    noisy_client_ids_for_hpo = []

    if base_config.get("noise_experiment_enabled", False):
        hpo_logger.info(f"--- Preparing Noisy Labels for HPO Phase ---")
        num_total_clients_for_hpo = base_config.get("num_clients", 10)
        noise_percentage_hpo = base_config.get("noise_client_percentage", 0.0)
        num_noisy_clients_hpo = int(num_total_clients_for_hpo * noise_percentage_hpo)
        noise_degree_hpo = base_config.get("noise_label_shuffle_degree", 0.0)
        
        hpo_logger.info(f"HPO Noise: Designating {num_noisy_clients_hpo}/{num_total_clients_for_hpo} clients as noisy.")
        hpo_logger.info(f"HPO Noise: Noisy clients will have {noise_degree_hpo*100:.1f}% of their labels shuffled.")

        all_client_ids_for_hpo_selection = list(client_Y_labels_fixed.keys()) # Use clients that have data
        random.shuffle(all_client_ids_for_hpo_selection)
        noisy_client_ids_for_hpo = all_client_ids_for_hpo_selection[:num_noisy_clients_hpo]
        hpo_logger.info(f"HPO Noise: Selected noisy client IDs: {sorted(noisy_client_ids_for_hpo)}")

        for client_id_hpo_noise, original_clean_labels_hpo in client_Y_labels_fixed.items():
            if client_id_hpo_noise in noisy_client_ids_for_hpo:
                hpo_logger.info(f"HPO Noise: Applying label noise to client {client_id_hpo_noise}")
                
                current_labels_to_noise = original_clean_labels_hpo.copy()
                num_samples_client_hpo = len(current_labels_to_noise)
                num_to_shuffle_hpo = int(num_samples_client_hpo * noise_degree_hpo)

                if num_to_shuffle_hpo > 0 and num_samples_client_hpo > 1:
                    indices_to_shuffle_hpo = np.random.choice(num_samples_client_hpo, num_to_shuffle_hpo, replace=False)
                    
                    selected_labels_to_shuffle = current_labels_to_noise[indices_to_shuffle_hpo].copy()
                    np.random.shuffle(selected_labels_to_shuffle) # Shuffle this subset
                    
                    noisy_version_for_client = current_labels_to_noise.copy()
                    noisy_version_for_client[indices_to_shuffle_hpo] = selected_labels_to_shuffle
                    
                    client_Y_labels_for_hpo_training[client_id_hpo_noise] = noisy_version_for_client
                    changed_count_hpo = np.sum(current_labels_to_noise != noisy_version_for_client)
                    hpo_logger.debug(f"  Client {client_id_hpo_noise}: {changed_count_hpo}/{num_samples_client_hpo} labels effectively changed by noise.")
                else:
                    client_Y_labels_for_hpo_training[client_id_hpo_noise] = original_clean_labels_hpo # No actual shuffling
                    hpo_logger.debug(f"  Client {client_id_hpo_noise}: No noise applied (num_to_shuffle={num_to_shuffle_hpo}).")
            else:
                # For non-noisy clients, use their original fixed clean labels
                client_Y_labels_for_hpo_training[client_id_hpo_noise] = original_clean_labels_hpo
    else:
        # If noise is not enabled, all clients use their original fixed clean labels
        hpo_logger.info("--- Noise experiment NOT enabled. Using clean labels for all clients in HPO. ---")
        for client_id_hpo_clean, original_clean_labels_hpo_clean in client_Y_labels_fixed.items():
            client_Y_labels_for_hpo_training[client_id_hpo_clean] = original_clean_labels_hpo_clean
    # --- END: Noise Preparation ---
    # === HPO LOOP for STAGE 3 (AutoCore Rule Learning & Server Aggregation) ===
    hpo_results = []
    best_val_accuracy = -1.0
    best_overall_hparams = None # Will store the entire best hparam dict
    best_F_M_terms_val = None 

    for idx_combo, current_hparam_combo in enumerate(hyperparameter_combinations):
        # Construct a unique ID string for this combination
        hparam_combo_id_parts = []
        for k_hp, v_hp in current_hparam_combo.items():
            key_short = k_hp.replace("figs_", "").replace("server_rule_validation_", "serv_val_").replace("server_","serv_")
            val_short = str(v_hp).replace("None","N")
            hparam_combo_id_parts.append(f"{key_short}{val_short}")
        hparam_combo_id_str = "_".join(hparam_combo_id_parts)
        
        hpo_logger.info(f"\n======== HPO Iteration {idx_combo + 1}/{len(hyperparameter_combinations)}: Combo ID: {hparam_combo_id_str} ========")
        hpo_logger.info(f"Current Full HParams: {current_hparam_combo}")

        current_run_config = base_config.copy() # Use a fresh copy of base_config
        
        # Populate figs_params for the client
        current_run_config['figs_params'] = {
            'max_rules': current_hparam_combo['figs_max_rules'],
            'max_trees': current_hparam_combo['figs_max_trees'],
            'max_features': current_hparam_combo['figs_max_features']
            # Add other figs_params from base_config if they exist and are not being tuned
            # Example: 'min_impurity_decrease': base_config.get('figs_params',{}).get('min_impurity_decrease', 0.0)
        }
        # Ensure all expected figs_params are there, falling back to base_config if not in hparam_combo
        base_figs_params = base_config.get('figs_params', {})
        for fp_key, fp_val in base_figs_params.items():
            if fp_key not in current_run_config['figs_params']: # Only add if not tuned
                current_run_config['figs_params'][fp_key] = fp_val


        # Update server aggregation parameters directly in current_run_config
        current_run_config['use_accuracy_weighting_server'] = current_hparam_combo['use_accuracy_weighting_server']
        current_run_config['server_rule_validation_min_precision'] = current_hparam_combo['server_rule_validation_min_precision']
        current_run_config['server_rule_min_coverage_count'] = current_hparam_combo['server_rule_min_coverage_count']
        
        current_run_config['current_run_id'] = f"{base_config.get('run_id_base', 'autocore')}_hpo_{hparam_combo_id_str}"
        
        # --- Initialize Server for THIS HPO iteration's FL Rule Learning ---
        # The server object needs to be fresh for each HPO iteration to reset its state
        # and use the current_run_config which contains the server aggregation HPs.
        server_hpo_iter = FederatedServer(current_run_config.copy()) # Pass the config with updated server HPs
        server_hpo_iter.feature_names_for_figs = num_AutoCore_features_setup # From one-time setup
        server_hpo_iter.server_X_test_concepts_for_validation = server_X_val_concepts_fixed
        server_hpo_iter.server_y_test_labels_for_validation = server_y_val_labels_fixed
        server_hpo_iter.server_feature_names_for_validation = num_AutoCore_features_setup
        
        # Initialize FL rule learning state
        for client_s in clients_setup: client_s.state['accumulated_global_model_Fm_terms'] = []
        server_hpo_iter.accumulated_Fm_global_terms = []
        rounds_no_model_change_hpo = 0
        
        # --- Federated Rule Learning Loop (Stage 3) ---
        for r_fl_hpo in range(current_run_config["phase2_rounds"]): # Phase2_rounds from base_config
            # Key part: client_round_cfg_hpo['figs_params'] = current_run_config['figs_params']
            # The server_hpo_iter will use its own config (which has the server HPs)
            # when its aggregate_figs_models method is called (inside phase2_aggregate).

            hpo_logger.debug(f"  HPO Iter {hparam_combo_id_str} - FL Round {r_fl_hpo + 1}")
            client_figs_updates_hpo = []
            active_clients_hpo = 0
            for client_s in clients_setup:
                if client_s.client_id not in client_X_concepts_fixed: continue

                client_s.state['concept_vecs'] = client_X_concepts_fixed[client_s.client_id]
                
                # Client uses its (potentially noisy) FIXED labels for this HPO run
                current_training_labels_for_client = client_Y_labels_for_hpo_training.get(client_s.client_id)
                if current_training_labels_for_client is None:
                    hpo_logger.warning(f"Client {client_s.client_id} missing labels for HPO training. Skipping.")
                    continue
                client_round_cfg_hpo = client_s.config.copy() 
                client_round_cfg_hpo['current_round'] = r_fl_hpo
                # THIS IS WHERE THE TUNED FIGS PARAMS ARE PASSED TO THE CLIENT FOR PatchedFIGSClassifier
                client_round_cfg_hpo['figs_params'] = current_run_config['figs_params'] 
                client_round_cfg_hpo['num_classes'] = current_run_config['num_classes']

                _, terms_hk_client = client_s.train_figs(current_training_labels_for_client, client_round_cfg_hpo)
                if terms_hk_client:
                    update_payload_hpo = client_s.get_model_update()
                    update_payload_hpo['client_id'] = client_s.client_id
                    client_figs_updates_hpo.append(update_payload_hpo)
                active_clients_hpo +=1
            
            if active_clients_hpo == 0: hpo_logger.warning(f"  HPO Iter {hparam_combo_id_str} - FL Round {r_fl_hpo+1}: No active clients."); break

            # Server uses its own current_run_config (passed during its init) for aggregation params
            aggregated_hm_server_hpo, converged_hpo = server_hpo_iter.phase2_aggregate(client_figs_updates_hpo, r_fl_hpo + 1)
            server_hpo_iter.broadcast_model(clients_setup, is_residual_model=True)
            
            if converged_hpo:
                rounds_no_model_change_hpo += 1
                if rounds_no_model_change_hpo >= current_run_config.get("rule_structure_convergence_patience", 3):
                    hpo_logger.info(f"  HPO Iter {hparam_combo_id_str}: Converged after {r_fl_hpo + 1} FL rounds."); break
            else: rounds_no_model_change_hpo = 0
        # --- End of FL Rule Learning for this HPO Iteration ---

        current_F_M_hpo = server_hpo_iter.accumulated_Fm_global_terms
        val_acc, val_rp, val_rc, val_rl, val_rf = 0.0,0.0,0.0,0.0,0.0
        if current_F_M_hpo and server_X_val_concepts_fixed is not None:
            val_acc, val_rp, val_rc, val_rl, val_rf = evaluate_global_AutoCore_model(
                current_F_M_hpo, server_X_val_concepts_fixed, server_y_val_labels_fixed,
                num_AutoCore_features_setup, current_run_config['num_classes'], hpo_logger, eval_set_name="HPO_Validation"
            )
        hpo_logger.info(f"HPO Iter {hparam_combo_id_str} - Validation Set: Acc={val_acc:.4f}, RuleP={val_rp:.3f}, RuleC={val_rc:.3f}, RuleL={val_rl:.2f}, RuleF={val_rf:.3f}")
        
        hpo_results.append({
            "hparam_combo_id": hparam_combo_id_str,
            "full_hparams": current_hparam_combo, # Store the full combo
            "val_accuracy": val_acc, "val_rule_precision": val_rp, "val_rule_coverage": val_rc,
            "val_rule_complexity": val_rl, "val_rule_fidelity": val_rf,
            "num_F_M_terms": len(current_F_M_hpo)
        })

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_overall_hparams = current_hparam_combo # Store the full dictionary of best hparams
            best_F_M_terms_val = current_F_M_hpo 
            hpo_logger.info(f"NEW BEST Validation Accuracy: {best_val_accuracy:.4f} with HParams: {best_overall_hparams}")
    hpo_results_df = pd.DataFrame(hpo_results)
    hpo_results_path = os.path.join(hpo_main_log_dir, f"hpo_full_tuning_results_{hpo_run_id_base}.csv")
    hpo_results_df.to_csv(hpo_results_path, index=False)
    hpo_logger.info(f"HPO results saved to: {hpo_results_path}")
    hpo_logger.info(f"Best Val Acc: {best_val_accuracy:.4f} with HParams: {best_overall_hparams}")

    # === FINAL TEST RUN with Best Overall Hyperparameters ===
    if best_overall_hparams is None:
        hpo_logger.error("No best HPs found. Skipping final test."); return

    hpo_logger.info("\n======== FINAL TEST RUN with Best Overall Hyperparameters ========")
    final_run_config = base_config.copy()
    final_run_config['figs_params'] = {
        'max_rules': best_overall_hparams['figs_max_rules'], 'max_trees': best_overall_hparams['figs_max_trees'],
        'max_features': best_overall_hparams['figs_max_features']
    } # Add other fixed figs params from base_config
    base_figs_params_final_run = base_config.get('figs_params', {})
    for fp_k, fp_v in base_figs_params_final_run.items():
        if fp_k not in final_run_config['figs_params']: final_run_config['figs_params'][fp_k] = fp_v
    final_run_config['use_accuracy_weighting_server'] = best_overall_hparams['use_accuracy_weighting_server']
    final_run_config['server_rule_validation_min_precision'] = best_overall_hparams['server_rule_validation_min_precision']
    final_run_config['server_rule_min_coverage_count'] = best_overall_hparams['server_rule_min_coverage_count']
    final_run_config['current_run_id'] = f"{base_config.get('run_id_base', 'autocore')}_final_test_best_hps"
    
    server_final_run = FederatedServer(final_run_config.copy())
    server_final_run.feature_names_for_figs = num_AutoCore_features_setup
    server_final_run.server_X_test_concepts_for_validation = server_X_val_concepts_fixed 
    server_final_run.server_y_test_labels_for_validation = server_y_val_labels_fixed
    server_final_run.server_feature_names_for_validation = num_AutoCore_features_setup
    
    for client_s_final in clients_setup: client_s_final.state['accumulated_global_model_Fm_terms'] = []
    server_final_run.accumulated_Fm_global_terms = []
    
    # --- Final FL Run with Best HParams (and potentially different noise application if needed) ---
    # If noise for the final run should be freshly determined:
    final_run_noisy_client_ids = []
    if final_run_config.get("noise_experiment_enabled", False): # Check if noise is enabled FOR THE FINAL RUN
        num_total_clients_final = final_run_config.get("num_clients", 10)
        noise_percentage_final = final_run_config.get("noise_client_percentage", 0.0)
        num_noisy_clients_final = int(num_total_clients_final * noise_percentage_final)
        all_client_ids_final_sel = list(client_Y_labels_fixed.keys())
        random.shuffle(all_client_ids_final_sel)
        final_run_noisy_client_ids = all_client_ids_final_sel[:num_noisy_clients_final]
        hpo_logger.info(f"FINAL RUN: Noisy clients: {sorted(final_run_noisy_client_ids)}")
    
    client_Y_labels_for_final_training = {}
    for cid, clabels in client_Y_labels_fixed.items():
        if final_run_config.get("noise_experiment_enabled", False) and cid in final_run_noisy_client_ids:
            noise_deg_final = final_run_config.get("noise_label_shuffle_degree", 0.0)
            labels_to_noise_final = clabels.copy()
            n_samples_final_cl = len(labels_to_noise_final)
            n_shuffle_final = int(n_samples_final_cl * noise_deg_final)
            if n_shuffle_final > 0 and n_samples_final_cl > 1:
                idx_shuffle_final = np.random.choice(n_samples_final_cl, n_shuffle_final, replace=False)
                vals_shuffle_final = labels_to_noise_final[idx_shuffle_final].copy(); np.random.shuffle(vals_shuffle_final)
                labels_to_noise_final[idx_shuffle_final] = vals_shuffle_final
                client_Y_labels_for_final_training[cid] = labels_to_noise_final
            else: client_Y_labels_for_final_training[cid] = clabels
        else: client_Y_labels_for_final_training[cid] = clabels


    hpo_logger.info(f"Starting final FL rule learning with best HPs: {best_overall_hparams}")
    # ... (FL loop for final run - using client_Y_labels_for_final_training) ...
    # ... (Final evaluation on TEST SET: server_X_test_concepts_fixed, server_y_test_labels_fixed) ...
    # ... (Logging and saving final summary and model) ...
    final_fl_rounds_completed_final_run = 0; rounds_no_model_change_final = 0
    for r_fl_final_loop in range(final_run_config["phase2_rounds"]):
        final_fl_rounds_completed_final_run = r_fl_final_loop + 1
        client_figs_updates_final_loop = []
        active_clients_final_loop = 0
        for client_s_final_loop in clients_setup:
            if client_s_final_loop.client_id not in client_X_concepts_fixed: continue
            client_s_final_loop.state['concept_vecs'] = client_X_concepts_fixed[client_s_final_loop.client_id]
            current_training_labels_final = client_Y_labels_for_final_training.get(client_s_final_loop.client_id)
            if current_training_labels_final is None: continue

            client_round_cfg_final_loop = client_s_final_loop.config.copy()
            client_round_cfg_final_loop['current_round'] = r_fl_final_loop
            client_round_cfg_final_loop['figs_params'] = final_run_config['figs_params']
            client_round_cfg_final_loop['num_classes'] = final_run_config['num_classes']
            _, terms_hk_final_loop = client_s_final_loop.train_figs(current_training_labels_final, client_round_cfg_final_loop)
            if terms_hk_final_loop:
                update_payload_final_loop = client_s_final_loop.get_model_update(); update_payload_final_loop['client_id'] = client_s_final_loop.client_id
                client_figs_updates_final_loop.append(update_payload_final_loop)
            active_clients_final_loop +=1
        if active_clients_final_loop == 0: hpo_logger.warning(f"FINAL RUN - FL Rnd {r_fl_final_loop+1}: No active clients."); break
        aggregated_hm_final_loop, converged_final_loop = server_final_run.phase2_aggregate(client_figs_updates_final_loop, r_fl_final_loop + 1)
        server_final_run.broadcast_model(clients_setup, is_residual_model=True)
        if converged_final_loop:
            rounds_no_model_change_final += 1
            if rounds_no_model_change_final >= final_run_config.get("rule_structure_convergence_patience", 3):
                hpo_logger.info(f"Final Run: Converged after {r_fl_final_loop + 1} FL rounds."); break
        else: rounds_no_model_change_final = 0
    final_F_M_terms_for_test = server_final_run.accumulated_Fm_global_terms
    hpo_logger.info(f"Final Run: Evaluating model with {len(final_F_M_terms_for_test)} terms on TEST SET.")
    test_acc, test_rp, test_rc, test_rl, test_rf = 0.0,0.0,0.0,0.0,0.0
    if final_F_M_terms_for_test and server_X_test_concepts_fixed is not None and server_y_test_labels_fixed is not None:
        test_acc, test_rp, test_rc, test_rl, test_rf = evaluate_global_AutoCore_model(
            final_F_M_terms_for_test, server_X_test_concepts_fixed, server_y_test_labels_fixed,
            num_AutoCore_features_setup, final_run_config['num_classes'], hpo_logger, eval_set_name="FINAL_TEST_BEST_HPARAMS"
        )
    else: hpo_logger.warning("Final Test: Model empty or server test data missing.")
    hpo_logger.info(f"======== FINAL TEST RESULTS (Best HParams: {best_overall_hparams}) for HPO Run ID: {hpo_run_id_base} ========")
    final_summary_data = { "hpo_run_id": hpo_run_id_base, "best_overall_hparams": str(best_overall_hparams),
        "best_val_accuracy_during_hpo": best_val_accuracy, "final_test_accuracy": test_acc,
        "final_test_rule_precision": test_rp, "final_test_rule_coverage": test_rc,
        "final_test_rule_complexity": test_rl, "final_test_rule_fidelity": test_rf,
        "fl_rounds_in_final_run": final_fl_rounds_completed_final_run,
        "num_noisy_clients_final_run": len(final_run_noisy_client_ids) if final_run_config.get("noise_experiment_enabled",False) else 0,
        "noise_degree_final_run": final_run_config.get("noise_label_shuffle_degree",0.0) if final_run_config.get("noise_experiment_enabled",False) else 0.0
    }
    final_summary_df = pd.DataFrame([final_summary_data])
    final_summary_path = os.path.join(hpo_main_log_dir, f"final_test_summary_best_overall_hparams_{hpo_run_id_base}.csv")
    final_summary_df.to_csv(final_summary_path, index=False)
    hpo_logger.info(f"Final test summary saved to: {final_summary_path}")
    if best_F_M_terms_val:
        best_val_model_path = os.path.join(hpo_main_log_dir, f"best_validated_model_terms_overall_{hpo_run_id_base}.pkl")
        with open(best_val_model_path, "wb") as f: pickle.dump(best_F_M_terms_val, f)
        hpo_logger.info(f"Best model terms from validation phase saved to: {best_val_model_path}")
    hpo_logger.info(f"======== AutoCoRe-FL HPO Run {hpo_run_id_base} Complete. Results in {hpo_main_log_dir} ========")
if __name__ == '__main__':
    main_autocore_fl_hpo()