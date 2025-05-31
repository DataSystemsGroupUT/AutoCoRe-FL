import logging
import os
import random
import sys
import yaml
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root_path not in sys.path: sys.path.insert(0, project_root_path)
grandparent_dir = os.path.abspath(os.path.join(project_root_path, '..'))
if grandparent_dir not in sys.path: sys.path.insert(0, grandparent_dir)

from AutoCore_FL.federated.client import FederatedClient
from AutoCore_FL.federated.server import FederatedServer
from AutoCore_FL.federated.utils import setup_logging, generate_run_id, save_config, load_config,SAM2Filter, evaluate_global_AutoCore_model, load_labels_from_manifest
from AutoCore_FL.concepts.vectorizer import build_image_concept_vectors


def main_autocore_fl():
    config_path = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/AutoCore_FL/configs/config_autocore_fl_auto_sun.yaml" # Create this YAML config file
    if not os.path.exists(config_path):
        print(f"ERROR: Config file {config_path} not found. Please create it.")
        return
    
    config = load_config(config_path)
    run_id = generate_run_id(config.get('method_name', 'AutoCoReFL_auto_sun'))
    if config.get('run_id_base'): # Allows for more structured run_id
        run_id = f"{config['run_id_base']}_{run_id}"
    config['current_run_id'] = run_id

    # Setup output directory for this specific FL run
    fl_run_output_dir = os.path.join(config.get("output_base_dir_fl_run", "./autocore_fl_run_results"), run_id)
    log_dir_fl = os.path.join(fl_run_output_dir, "logs")
    os.makedirs(log_dir_fl, exist_ok=True)
    
    config['log_dir_run'] = log_dir_fl # For outputs specific to this FL run (like metrics.csv)
    
    main_logger = setup_logging(log_dir=log_dir_fl, run_id=run_id, log_level_str=config.get("log_level", "INFO"))
    root_logger_instance = logging.getLogger()
    root_logger_instance.addFilter(SAM2Filter())
    main_logger.info(f"======== Starting AutoCoRe-FL from V2 Cached Data: {run_id} ========")
    save_config(config, os.path.join(fl_run_output_dir, "config_this_fl_run.json")) # Save FL run config
    # --- Seed & Device ---
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    if config['device'] == 'cuda' and torch.cuda.is_available():
        torch_device_main = torch.device("cuda"); torch.cuda.manual_seed_all(config['seed'])
    else:
        torch_device_main = torch.device("cpu")
    main_logger.info(f"Using device: {torch_device_main}")

    # --- Load Global Class Mapping ---
    cached_data_base_path = config['cached_data_base_dir']
    path_to_scene_map_json = os.path.join(cached_data_base_path, config['partition_manifest_dir_input'], config['scene_to_idx_map_filename_input'])
    try:
        with open(path_to_scene_map_json, 'r') as f:
            scene_to_global_idx_map = yaml.safe_load(f)
        config['num_classes'] = len(scene_to_global_idx_map) # Set num_classes based on loaded map
        main_logger.info(f"Loaded scene_to_idx_map ({config['num_classes']} classes) from: {path_to_scene_map_json}")
    except Exception as e:
        main_logger.error(f"Failed to load scene_to_idx_map.json: {e}. Exiting.")
        return

    # --- Initialize Clients with Pre-cached Data ---
    main_logger.info("--- Initializing Clients with Pre-cached Data ---")
    clients = []
    num_configured_clients = config.get('num_clients', 10) # Default to 10 if not in config

    for i in range(num_configured_clients):
        client_id_str = f"client_{i}"
        client_config_copy = config.copy() # Give each client its own copy of the main config
        client_config_copy['client_id'] = i # Ensure client_id is in its config for logging/caching

        client = FederatedClient(
            client_id=i,
            config=client_config_copy,
            data_partition=None, # Not used for data loading in this version
            scene_to_idx=scene_to_global_idx_map, # Pass the global map
            skip_model_initialization=False # SAM/DINO models are still needed by client for Stage 2
        )
        # Trigger data loading within the client
        try:
            manifest_path = os.path.join(cached_data_base_path, config['partition_manifest_dir_input'], client_id_str, 'image_manifest.json')
            seg_infos_path = os.path.join(cached_data_base_path, config['partition_segment_infos_cache_dir_input'], f"{client_id_str}_segment_infos_with_crops.pkl")
            embeddings_path = os.path.join(cached_data_base_path, config['embedding_cache_dir_input'], f"embeddings_{config['embedding_type']}_{client_id_str}.pkl")

            client_base_ids, client_labels_int = load_labels_from_manifest(manifest_path, main_logger)
            if not client_base_ids: raise FileNotFoundError(f"Manifest empty/not loaded for {client_id_str}")

            with open(seg_infos_path, 'rb') as f: client.state['filtered_segment_infos'] = pickle.load(f)
            with open(embeddings_path, 'rb') as f: client.state['final_embeddings'] = pickle.load(f)

            client.state['loaded_base_ids'] = client_base_ids
            client.state['loaded_labels_int'] = client_labels_int
            if client.state['filtered_segment_infos'] is None or client.state['final_embeddings'] is None:
                raise ValueError("Segment infos or embeddings are None after loading.")
            if len(client.state['filtered_segment_infos']) != client.state['final_embeddings'].shape[0]:
                mismatched_error_msg = (f"Client {i} Data Mismatch: "
                                 f"{len(client.state['filtered_segment_infos'])} segment infos vs "
                                 f"{client.state['final_embeddings'].shape[0]} embeddings.")
                main_logger.error(mismatched_error_msg)
                client.state['filtered_segment_infos'] = None 
                client.state['final_embeddings'] = None
            
            main_logger.info(f"Client {i}: Loaded {len(client.state['loaded_base_ids'])} images, "
                             f"{len(client.state['filtered_segment_infos']) if client.state['filtered_segment_infos'] is not None else 0} segments, "
                             f"{client.state['final_embeddings'].shape if client.state['final_embeddings'] is not None else 'No'} embeddings.")
        except Exception as e_load:
            main_logger.error(f"Failed to load cached data for client {i}: {e_load}. Client may be inactive.")
            client.state['filtered_segment_infos'] = None 
            client.state['final_embeddings'] = None
        
        clients.append(client)

    # --- Initialize Server ---
    server = FederatedServer(config) # Server also uses the main config

    # --- PHASE 1: Federated K-Means (using pre-loaded embeddings) ---
    main_logger.info(f"--- Starting Phase 1: Federated K-Means ---")
    # Server initializes global centroids
    # Ensure embedding_dim is correctly set in config, matching the loaded embeddings
    current_centroids = server.initialize_centroids(config["embedding_dim"], config["num_clusters"])
    client_kmeans_stats_final_round = []

    for r_km in range(config["kmeans_rounds"]):
        main_logger.info(f"KMeans Round {r_km + 1}/{config['kmeans_rounds']}")
        client_kmeans_stats_current_round = []
        active_clients_this_round_km = 0
        for client in tqdm(clients, desc=f"KMeans R{r_km+1}-Clients", file=sys.stdout, leave=False):
            if client.state.get('final_embeddings') is None or client.state['final_embeddings'].shape[0] == 0:
                main_logger.debug(f"Client {client.client_id} has no embeddings. Skipping KMeans contribution.")
                continue

            sums, counts = client.run_local_pipeline(global_centroids=current_centroids)
            if sums is not None and counts is not None and sums.size > 0 and counts.size > 0:
                client_kmeans_stats_current_round.append((sums, counts))
                active_clients_this_round_km += 1
        
        if active_clients_this_round_km == 0:
            main_logger.error(f"No active clients in KMeans Round {r_km+1}. Using previous round's stats if available.")
            if not client_kmeans_stats_final_round: # No previous good round
                main_logger.error("KMeans failed: No client activity in any round resulting in stats.")
                return # Critical failure
            break # Exit K-Means loop, use centroids from client_kmeans_stats_final_round

        current_centroids, live_mask = server.aggregate_kmeans(client_kmeans_stats_current_round)
        client_kmeans_stats_final_round = client_kmeans_stats_current_round # Update with latest successful stats
        main_logger.info(f"KMeans R{r_km+1}: Aggregated Centroids shape: {current_centroids.shape if current_centroids is not None else 'None'}, Live concepts (mask sum): {live_mask.sum() if live_mask is not None else 'N/A'}")
        
        if current_centroids is None or live_mask is None or live_mask.sum() == 0:
            main_logger.error(f"KMeans aggregation failed in Round {r_km+1} or no live centroids. Using previous stats.")
            if not client_kmeans_stats_final_round:
                 main_logger.error("KMeans failed: No client activity led to valid centroids at any point.")
                 return
            break 
    
    if not client_kmeans_stats_final_round:
        main_logger.error("KMeans phase completed with no valid stats from any client round. Cannot proceed.")
        return

    final_counts_agg_list = [c for s,c in client_kmeans_stats_final_round if c is not None and c.ndim > 0 and c.size > 0 and c.shape[0] == config["num_clusters"]]
    if not final_counts_agg_list:
        main_logger.error("No valid final counts from KMeans to determine initial shared concepts.")
        return
    
    final_counts_agg = np.sum(np.stack(final_counts_agg_list), axis=0)
    keep_mask_concepts = final_counts_agg >= config.get('min_samples_per_concept_cluster', 30)
    initial_shared_concept_indices = np.where(keep_mask_concepts)[0].tolist() # These are original K-Means cluster indices
    if not initial_shared_concept_indices:
        main_logger.error(f"No concepts met min_samples_per_concept_cluster. Cannot proceed.")
        return
    main_logger.info(f"Phase 1 Done. Found {len(initial_shared_concept_indices)} potential K-Means concepts after filtering: {sorted(initial_shared_concept_indices)[:20]}...")


    # --- PHASE 2 Preparation: Detector Sync ---
    main_logger.info(f"--- Starting Phase 2 Prep: Detector Sync ---")
    for client in tqdm(clients, desc="Local Detectors Training", file=sys.stdout, leave=False):
        if client.state.get('final_embeddings') is not None and client.state.get('cluster_labels') is not None:
             # Pass the main FL run config to client.train_concept_detectors
            client.train_concept_detectors(config)
        else:
            main_logger.warning(f"Client {client.client_id} cannot train detectors: missing embeddings or cluster_labels.")
    
    all_detector_updates = [client.get_detector_update(initial_shared_concept_indices) for client in clients]
    
    # Server aggregates detectors (selects best, extracts params)
    final_phase2_concept_indices_ordered, canonical_detector_params_for_broadcast = server.aggregate_detectors(
        all_detector_updates, initial_shared_concept_indices
    )
    num_figs_features = len(final_phase2_concept_indices_ordered)
    if num_figs_features == 0:
        main_logger.error("No concepts survived detector aggregation for AutoCore. Cannot proceed.")
        return
    
    figs_feature_names = [f"concept_{i}" for i in range(num_figs_features)]
    server.feature_names_for_figs = figs_feature_names # For server's internal eval use
    final_original_to_dense_map = {orig_idx: dense_idx for dense_idx, orig_idx in enumerate(final_phase2_concept_indices_ordered)}
    main_logger.info(f"Phase 2 Prep Done. Using {num_figs_features} final concepts for AutoCore: {figs_feature_names[:5]}...")

    # Server broadcasts canonical detector definitions (params, map, ordered indices)
    server.phase2_prep_broadcast(
        clients,
        final_phase2_concept_indices_ordered,
        final_original_to_dense_map,
        canonical_detector_params_for_broadcast
    )

    # --- Server Holdout Data Vectorization (Using ached data for server) ---
    main_logger.info("--- Preparing Server Holdout Concept Data (Val & Test) ---")
    # Server needs to load its own cached segment_infos_with_crops and embeddings
    server_X_val_concepts, server_y_val_labels = None, None
    server_X_test_concepts, server_y_test_labels = None, None

    for holdout_set_name in ["server_validation_set", "server_test_set"]:
        main_logger.info(f"Processing server holdout: {holdout_set_name}")
        try:
            manifest_path_server = os.path.join(cached_data_base_path, config['partition_manifest_dir_input'], holdout_set_name, 'image_manifest.json')
            seg_infos_path_server = os.path.join(cached_data_base_path, config['partition_segment_infos_cache_dir_input'], f"{holdout_set_name}_segment_infos_with_crops.pkl")
            embeddings_path_server = os.path.join(cached_data_base_path, config['embedding_cache_dir_input'], f"embeddings_{config['embedding_type']}_{holdout_set_name}.pkl")

            server_holdout_base_ids, server_holdout_labels_int = load_labels_from_manifest(manifest_path_server, main_logger)
            if not server_holdout_base_ids:
                main_logger.warning(f"Manifest empty for {holdout_set_name}. Skipping."); continue

            with open(seg_infos_path_server, 'rb') as f: server_seg_infos_holdout = pickle.load(f)
            with open(embeddings_path_server, 'rb') as f: server_embeddings_holdout = pickle.load(f)

            if server_seg_infos_holdout is None or server_embeddings_holdout is None or \
               len(server_seg_infos_holdout) != server_embeddings_holdout.shape[0]:
                main_logger.error(f"Data mismatch for {holdout_set_name}. Segs: {len(server_seg_infos_holdout) if server_seg_infos_holdout is not None else 'None'}, Embs: {server_embeddings_holdout.shape if server_embeddings_holdout is not None else 'None'}. Skipping.")
                continue
            
            # Server vectorizes using its canonical_detector_model_objects and canonical_thresholds
            # build_image_concept_vectors needs model objects and thresholds keyed by DENSE index.
            server_dense_detectors_for_vec = {
                final_original_to_dense_map[orig_k]: model_obj
                for orig_k, model_obj in server.canonical_detector_model_objects.items()
                if orig_k in final_original_to_dense_map # Ensure only valid concepts are used
            }
            server_dense_thresholds_for_vec = {
                final_original_to_dense_map[orig_k]: thresh
                for orig_k, thresh in server.canonical_thresholds.items()
                if orig_k in final_original_to_dense_map
            }

            if not server_dense_detectors_for_vec:
                main_logger.warning(f"No canonical detectors available on server for vectorizing {holdout_set_name}. Skipping.")
                continue

            current_X_holdout_concepts, holdout_kept_image_ids = build_image_concept_vectors(
                filtered_segment_infos=server_seg_infos_holdout, # List of dicts with local img_idx, seg_crop_bgr
                linear_models=server_dense_detectors_for_vec,    # Dict {dense_idx: model_obj}
                optimal_thresholds=server_dense_thresholds_for_vec, # Dict {dense_idx: threshold}
                final_embeddings=server_embeddings_holdout,          # All segment embeddings for this holdout set
                target_num_features=num_figs_features,
                config=config # Main config for vectorizer_min_activating_segments
            )

            if current_X_holdout_concepts is not None and current_X_holdout_concepts.shape[0] > 0:
                # Align labels with the output of build_image_concept_vectors
                # Map holdout_kept_image_ids (base_ids) back to their labels
                map_base_id_to_label = {bid: lbl for bid, lbl in zip(server_holdout_base_ids, server_holdout_labels_int)}
                aligned_holdout_labels = np.array([map_base_id_to_label[bid] for bid in holdout_kept_image_ids if bid in map_base_id_to_label])

                if current_X_holdout_concepts.shape[0] == aligned_holdout_labels.shape[0]:
                    if holdout_set_name == "server_validation_set":
                        server_X_val_concepts, server_y_val_labels = current_X_holdout_concepts, aligned_holdout_labels
                    elif holdout_set_name == "server_test_set":
                        server_X_test_concepts, server_y_test_labels = current_X_holdout_concepts, aligned_holdout_labels
                    main_logger.info(f"Vectorized {holdout_set_name}: X_concepts {current_X_holdout_concepts.shape}, Y_labels {aligned_holdout_labels.shape}")
                else:
                     main_logger.error(f"Label alignment error for {holdout_set_name}. X: {current_X_holdout_concepts.shape[0]}, Y: {aligned_holdout_labels.shape[0]}.")
            else:
                main_logger.warning(f"No concept vectors generated for {holdout_set_name}.")

        except FileNotFoundError:
            main_logger.warning(f"Cached data not found for server holdout set: {holdout_set_name}. It will not be available.")
        except Exception as e_server_vec:
            main_logger.error(f"Error vectorizing server data for {holdout_set_name}: {e_server_vec}", exc_info=True)
            
    # Configure server for validation
    if server_X_val_concepts is not None and server_y_val_labels is not None:
        server.server_X_test_concepts_for_validation = server_X_val_concepts # Misnomer in server class, it's for validation
        server.server_y_test_labels_for_validation = server_y_val_labels
        server.server_feature_names_for_validation = figs_feature_names
        main_logger.info(f"Server validation data (for rule filtering) configured: X_shape={server_X_val_concepts.shape}")
    else:
        main_logger.warning("Server validation data not available. Server-side rule validation will be heuristic or skipped.")


    # --- PHASE 3: AutoCore FL Rule Learning ---
    main_logger.info(f"--- Starting Phase 3: AutoCore FL Rule Learning ---")

    # Initialize client's and server's accumulated global model F_m (list of terms)
    for client in clients: client.state['accumulated_global_model_Fm_terms'] = []
    server.accumulated_Fm_global_terms = []
    rounds_no_model_change = 0
    all_round_metrics = [] # To store metrics per round
    final_fl_round_completed = 0

    for r_fl in range(config["phase2_rounds"]): # Phase2_rounds is for FL rule learning
        final_fl_round_completed = r_fl + 1
        main_logger.info(f"--- AutoCore FL Round {r_fl + 1}/{config['phase2_rounds']} ---")
        client_figs_updates_this_round = []
        active_clients_this_fl_round = 0

        for client_idx, client in enumerate(tqdm(clients, desc=f"FL Round {r_fl+1} Clients", file=sys.stdout, leave=False)):
            if client.state.get('final_embeddings') is None: # Check if client has data
                main_logger.debug(f"Client {client.client_id} inactive (no embeddings). Skipping FL round participation.")
                continue
            client_concept_vectors, client_kept_image_ids = client.build_concept_vectors(config) # Pass main config
            
            if client_concept_vectors is None or client_concept_vectors.shape[0] == 0:
                main_logger.warning(f"Client {client.client_id}: No concept vectors for round {r_fl + 1}. Skipping.")
                continue

            # Get labels corresponding to client_kept_image_ids
            # This relies on client.state['loaded_base_ids'] and client.state['loaded_labels_int'] being correctly populated at init
            map_base_id_to_label_client = {bid: lbl for bid, lbl in zip(client.state['loaded_base_ids'], client.state['loaded_labels_int'])}
            client_true_labels_for_figs = np.array([map_base_id_to_label_client[bid] for bid in client_kept_image_ids if bid in map_base_id_to_label_client])

            if client_true_labels_for_figs.shape[0] != client_concept_vectors.shape[0]:
                main_logger.error(f"Client {client.client_id}: Label/ConceptVector mismatch in FL round. Labels: {client_true_labels_for_figs.shape[0]}, Vecs: {client_concept_vectors.shape[0]}. Skipping.")
                continue
            
            active_clients_this_fl_round +=1
            
            client_round_config_for_train = client.config.copy() # Client has its own base config
            client_round_config_for_train['current_round'] = r_fl # Pass 0-indexed round
            client_round_config_for_train['figs_params'] = config['figs_params'] # Global FIGS params
            client_round_config_for_train['num_classes'] = config['num_classes'] # Global num_classes
            
            # Client trains its local residual model h_k^(m)
            _, terms_hk_from_client = client.train_figs(client_true_labels_for_figs, client_round_config_for_train)
            
            if terms_hk_from_client:
                update_payload = client.get_model_update() # Sends terms of h_k^(m)
                update_payload['client_id'] = client.client_id # Ensure server knows who sent it
                client_figs_updates_this_round.append(update_payload)
            else:
                main_logger.info(f"Client {client.client_id}: No residual terms (h_k) generated in FL round {r_fl+1}.")
        
        if active_clients_this_fl_round == 0:
            main_logger.warning(f"FL Round {r_fl+1}: No clients active. Server model h^(m) will be empty.")
            # Server.phase2_aggregate should handle empty client_figs_updates_this_round
        
        # Server aggregates h_k^(m) to h^(m), updates its F_M, and checks convergence of h^(m)
        # server.global_figs_model_terms will store terms of h^(m)
        # server.accumulated_Fm_global_terms will store F_M^(m)
        aggregated_hm_terms_server, converged_this_round = server.phase2_aggregate(client_figs_updates_this_round, r_fl + 1)
        
        # Server broadcasts h^(m) terms to clients. Clients update their F_k^(m).
        server.broadcast_model(clients, is_residual_model=True) # Ensure clients receive h^(m)
        main_logger.info(f"FL R{r_fl+1} done. Aggregated h^(m) has {len(server.global_figs_model_terms)} terms. Converged: {converged_this_round}")
        
        # Intermediate evaluation of server's F_M^(m) on its test set
        if server_X_test_concepts is not None and server_y_test_labels is not None and server_X_test_concepts.shape[0] > 0:
            intermediate_F_M_on_server = server.accumulated_Fm_global_terms
            if intermediate_F_M_on_server:
                acc_im, rp_im, rc_im, rl_im, rf_im = evaluate_global_AutoCore_model(
                    intermediate_F_M_on_server, server_X_test_concepts, server_y_test_labels, 
                    figs_feature_names, config['num_classes'], main_logger)
                main_logger.info(f"  FL R{r_fl+1} Server F_M Test Metrics: Acc={acc_im:.3f}, RuleP={rp_im:.3f}, RuleC={rc_im:.3f}, RuleL={rl_im:.2f}, RuleF={rf_im:.3f}")
                all_round_metrics.append({
                    'round': r_fl + 1, 'F_M_acc': acc_im, 'F_M_rule_p': rp_im, 
                    'F_M_rule_c': rc_im, 'F_M_rule_l': rl_im, 'F_M_rule_f': rf_im, 
                    'h_m_terms_count': len(aggregated_hm_terms_server), # Terms in current h_m
                    'F_M_terms_count': len(intermediate_F_M_on_server)  # Terms in current F_M
                })

        if converged_this_round:
            rounds_no_model_change +=1
            if rounds_no_model_change >= config.get("rule_structure_convergence_patience", 3):
                main_logger.info(f"FL converged: h^(m) stable for {rounds_no_model_change} rounds. Stopping rule learning."); break
        else:
            rounds_no_model_change = 0

    # --- Final Evaluation ---
    main_logger.info(f"--- Final Evaluation after {final_fl_round_completed} FL rounds ---")
    final_F_M_terms_on_server = server.accumulated_Fm_global_terms
    main_logger.info(f"Evaluating server's final F_M model with {len(final_F_M_terms_on_server)} total effective terms.")
    
    model_acc, rule_p, rule_c, rule_l, rule_f = 0.0,0.0,0.0,0.0,0.0
    if final_F_M_terms_on_server and server_X_test_concepts is not None and server_y_test_labels is not None and server_X_test_concepts.shape[0] > 0:
        model_acc, rule_p, rule_c, rule_l, rule_f = evaluate_global_AutoCore_model(
            final_F_M_terms_on_server, server_X_test_concepts, server_y_test_labels, 
            figs_feature_names, config['num_classes'], main_logger
        )
    else:
        main_logger.warning("Final F_M empty or no server test data. Cannot evaluate final model.")

    main_logger.info(f"======== FINAL RESULTS for Run ID: {run_id} (AutoCoRe-FL) ========")
    main_logger.info(f"  Completed FL Rounds: {final_fl_round_completed}")
    main_logger.info(f"  Num Clients: {num_configured_clients}, Num AutoCore Features: {num_figs_features}")
    main_logger.info(f"  Global AutoCore Model Accuracy: {model_acc:.4f}")
    main_logger.info(f"  Global AutoCore Rule Metrics: P={rule_p:.3f}, C={rule_c:.3f}, L={rule_l:.2f}, F={rule_f:.3f}")
    main_logger.info(f"  Number of Terms in final F_M: {len(final_F_M_terms_on_server)}")
    
    # Save round-wise metrics
    if all_round_metrics:
        df_round_metrics = pd.DataFrame(all_round_metrics)
        round_metrics_path = os.path.join(fl_run_output_dir, "fl_round_metrics.csv")
        df_round_metrics.to_csv(round_metrics_path, index=False)
        main_logger.info(f"Round-wise metrics saved to {round_metrics_path}")

    # Save final summary metrics to a CSV
    summary_metrics = {
        "run_id": run_id, "method": config.get('method_name'), "completed_fl_rounds": final_fl_round_completed,
        "num_clients": num_configured_clients, "num_figs_features": num_figs_features,
        "final_model_accuracy": model_acc, "final_rule_precision": rule_p, "final_rule_coverage": rule_c,
        "final_rule_complexity": rule_l, "final_rule_fidelity": rule_f,
        "final_F_M_terms_count": len(final_F_M_terms_on_server),
        **config # Add all config params to the summary
    }
    # Remove complex objects from summary if any were in config
    for k,v in summary_metrics.items():
        if isinstance(v, (list,dict)) and k != 'figs_params': # Keep figs_params if simple enough
             summary_metrics[k] = str(v)
    
    summary_df = pd.DataFrame([summary_metrics])
    summary_csv_path = os.path.join(fl_run_output_dir, "final_summary_metrics.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    main_logger.info(f"Final summary metrics saved to {summary_csv_path}")


    # Save the final global model terms (F_M^*)
    final_model_path = os.path.join(fl_run_output_dir, f"final_global_model_F_M_terms_{run_id}.pkl")
    try:
        with open(final_model_path, "wb") as f:
            pickle.dump(final_F_M_terms_on_server, f)
        main_logger.info(f"Saved final global model terms (F_M^*) to {final_model_path}")
    except Exception as e_save_model:
        main_logger.error(f"Failed to save final global model terms: {e_save_model}")


    main_logger.info(f"======== AutoCoRe-FL Run {run_id} Complete. Results in {fl_run_output_dir} ========")


if __name__ == '__main__':
    main_autocore_fl()