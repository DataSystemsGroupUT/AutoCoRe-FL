# run_autocore_then_lrxfl_v5_exact_mirror.py

import logging
import os
import random
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
import time
import copy 
from collections import Counter 
from sklearn.metrics import accuracy_score # Direct use as in LR-XFL

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger as PL_CSVLogger
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
import torch 
from torch import stack, squeeze # Explicitly import stack and squeeze as used in LR-XFL

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
autocore_fl_package_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if autocore_fl_package_root not in sys.path: sys.path.insert(0, autocore_fl_package_root)
logic_explained_networks_base_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))
lrxfl_project_root = os.path.join(logic_explained_networks_base_dir, 'LR-XFL')
lrxfl_experiments_submodule_path = os.path.join(lrxfl_project_root, 'experiments')
original_sys_path = list(sys.path) 
if os.path.isdir(lrxfl_experiments_submodule_path):
    if lrxfl_experiments_submodule_path not in sys.path: sys.path.insert(0, lrxfl_experiments_submodule_path)
else: print(f"FATAL: LR-XFL experiments path not found: {lrxfl_experiments_submodule_path}."); sys.exit(1)
lrxfl_entropy_submodule_path = os.path.join(lrxfl_project_root, 'entropy_lens')
if os.path.isdir(lrxfl_entropy_submodule_path):
    if lrxfl_entropy_submodule_path not in sys.path: sys.path.insert(0, lrxfl_entropy_submodule_path)
if os.path.isdir(lrxfl_project_root):
    if lrxfl_project_root not in sys.path: sys.path.insert(0, lrxfl_project_root)
else: print(f"FATAL: LR-XFL project root not found: {lrxfl_project_root}."); sys.exit(1)

# --- AutoCoRe-FL Component Imports ---
from federated.client import FederatedClient as AutoCoReClient
from federated.server import FederatedServer as AutoCoReServer
from federated.utils import setup_logging, generate_run_id, save_config, load_config, SAM2Filter, load_labels_from_manifest
from concepts.vectorizer import build_image_concept_vectors

# --- LR-XFL Component Imports ---
try:
    from entropy_lens.models.explainer import Explainer 
    from entropy_lens.logic.metrics import formula_consistency, test_explanation 
    from entropy_lens.logic.utils import replace_names 
    import utils as lrxfl_internal_experiment_utils
    from local_training import local_train 
    from global_logic_aggregate import _global_aggregate_explanations, client_selection_class 
    from data.data_sampling_ade20k import ade20k_iid, ade20k_noniid_by_concept # For LR-XFL client data splitting
except ImportError as e:
    print(f"ERROR: Import LR-XFL components failed: {e}\nSys.path: {sys.path}"); sys.exit(1)

average_weights = lrxfl_internal_experiment_utils.average_weights
weighted_weights = lrxfl_internal_experiment_utils.weighted_weights

def get_xy(d):
    """
    Given a TensorDataset **or an arbitrary nesting of torch.utils.data.Subset
    objects that eventually wrap a TensorDataset**, return (x, y) tensors that
    correspond to *exactly* the samples in d.
    """
    # Base case – we’ve reached the real data
    if isinstance(d, torch.utils.data.TensorDataset):
        return d.tensors

    # Recursive case – unwrap one layer of Subset and propagate the indices
    if isinstance(d, torch.utils.data.Subset):
        # Recurse until we hit the TensorDataset
        x_parent, y_parent = get_xy(d.dataset)
        return x_parent[d.indices], y_parent[d.indices]

    raise TypeError(f"Unexpected dataset type: {type(d)}")


def main():
    base_config_path = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/AutoCore_FL/configs/config_lrxfl_fl_autocore_sun.yaml"
    if not os.path.exists(base_config_path): print(f"ERROR: Config {base_config_path} not found."); return
    config = load_config(base_config_path)
    run_id = generate_run_id(config.get('method_name','AC_LRXFL'))
    if config.get('run_id_base'): run_id = f"{config['run_id_base']}_{run_id}"
    config['current_run_id'] = run_id
    combined_run_output_dir = os.path.join(config.get("output_base_dir_combined_run", "./results_ac_lrxfl"), run_id)
    log_dir_combined = os.path.join(combined_run_output_dir, "logs")
    os.makedirs(log_dir_combined, exist_ok=True); config['log_dir_run'] = log_dir_combined
    main_logger = setup_logging(log_dir=log_dir_combined, run_id=run_id, log_level_str=config.get("log_level", "INFO"))
    root_logger_instance = logging.getLogger(); root_logger_instance.addFilter(SAM2Filter())
    main_logger.info(f"======== Starting {config.get('method_name','AC_LRXFL')} : {run_id} ========")
    save_config(config, os.path.join(combined_run_output_dir, "config_this_run.json"))
    
    SEED = config['seed']
    seed_everything(SEED, workers=True) 
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    
    torch_device_main = torch.device(config['device']) # Use DEVICE as in original LR-XFL script for this var
    if config['device'] == 'cuda' and torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    main_logger.info(f"Using device: {torch_device_main}") # Changed var name to match

    main_logger.info("\n======== PART 1: AutoCoRe-FL Concept Discovery & Vectorization ========")
    cached_data_base_path = config['cached_data_base_dir']
    path_to_scene_map_json = os.path.join(cached_data_base_path, config['partition_manifest_dir_input'], config['scene_to_idx_map_filename_input'])
    try:
        with open(path_to_scene_map_json, 'r') as f: scene_to_global_idx_map = yaml.safe_load(f)
        config['num_classes'] = len(scene_to_global_idx_map)
        config['loaded_scene_class_names'] = sorted(list(scene_to_global_idx_map.keys()))
    except Exception as e: main_logger.error(f"Failed to load scene_to_idx_map.json: {e}. Exiting."); return
    autocore_clients = []
    for i in range(config["num_clients"]):
        client_id_str = f"client_{i}"; client_cfg = config.copy(); client_cfg['client_id'] = i
        client_obj = AutoCoReClient(client_id=i, config=client_cfg, data_partition=None, scene_to_idx=scene_to_global_idx_map, skip_model_initialization=False)
        try:
            manifest_p = os.path.join(cached_data_base_path, config['partition_manifest_dir_input'], client_id_str, 'image_manifest.json')
            seg_infos_p = os.path.join(cached_data_base_path, config['partition_segment_infos_cache_dir_input'], f"{client_id_str}_segment_infos_with_crops.pkl")
            embed_p = os.path.join(cached_data_base_path, config['embedding_cache_dir_input'], f"embeddings_{config['embedding_type']}_{client_id_str}.pkl")
            c_bids, c_lbls = load_labels_from_manifest(manifest_p, main_logger)
            if not c_bids: raise FileNotFoundError(f"Manifest empty for client {i}")
            with open(seg_infos_p,'rb') as f: client_obj.state['filtered_segment_infos']=pickle.load(f)
            with open(embed_p,'rb') as f: client_obj.state['final_embeddings']=pickle.load(f)
            client_obj.state['loaded_base_ids'] = c_bids; client_obj.state['loaded_labels_int'] = c_lbls
            if not (client_obj.state['filtered_segment_infos'] is not None and \
                    client_obj.state['final_embeddings'] is not None and \
                    len(client_obj.state['filtered_segment_infos']) == client_obj.state['final_embeddings'].shape[0]):
                main_logger.error(f"Client {i} Data Mismatch. Segs: {len(client_obj.state.get('filtered_segment_infos',[]))}, Embs: {client_obj.state.get('final_embeddings', np.empty(0)).shape}. Inactive."); client_obj.state['final_embeddings'] = None
        except Exception as e_load_ac: main_logger.error(f"Failed to load data for AC Client {i}: {e_load_ac}. Inactive."); client_obj.state['final_embeddings'] = None
        autocore_clients.append(client_obj)
    autocore_server = AutoCoReServer(config.copy())
    main_logger.info("--- AutoCoRe Stage 1: Federated K-Means ---")
    ac_current_centroids = autocore_server.initialize_centroids(config["embedding_dim"], config["num_clusters"])
    ac_client_kmeans_stats_final_round = []
    for r_km_ac in range(config["kmeans_rounds"]):
        main_logger.debug(f"AC K-Means Rnd {r_km_ac + 1}")
        ac_client_kmeans_stats_current_round = []; ac_active_clients_km = 0
        for ac_client_km in autocore_clients:
            if ac_client_km.state.get('final_embeddings') is None: continue
            ac_sums_km, ac_counts_km = ac_client_km.run_local_pipeline(global_centroids=ac_current_centroids)
            if ac_sums_km is not None and ac_counts_km is not None: ac_client_kmeans_stats_current_round.append((ac_sums_km, ac_counts_km)); ac_active_clients_km +=1
        if ac_active_clients_km == 0:
            if not ac_client_kmeans_stats_final_round: main_logger.error("AC K-Means failed."); return
            break
        ac_current_centroids, ac_live_mask_km = autocore_server.aggregate_kmeans(ac_client_kmeans_stats_current_round)
        ac_client_kmeans_stats_final_round = ac_client_kmeans_stats_current_round
        if ac_current_centroids is None or ac_live_mask_km is None or ac_live_mask_km.sum()==0:
            if not ac_client_kmeans_stats_final_round: main_logger.error("AC K-Means agg failed."); return
            break
    if not ac_client_kmeans_stats_final_round: main_logger.error("AC K-Means no valid stats."); return
    ac_final_counts_list_km = [c for s,c in ac_client_kmeans_stats_final_round if c is not None and c.ndim > 0 and c.size > 0 and c.shape[0] == config["num_clusters"]]
    if not ac_final_counts_list_km: main_logger.error("AC No K-Means counts."); return
    ac_final_counts_agg_km = np.sum(np.stack(ac_final_counts_list_km), axis=0)
    ac_keep_mask_km = ac_final_counts_agg_km >= config.get('min_samples_per_concept_cluster',30)
    ac_initial_shared_concept_indices = np.where(ac_keep_mask_km)[0].tolist()
    if not ac_initial_shared_concept_indices: main_logger.error("AC No K-Means concepts survived."); return
    main_logger.info(f"AC K-Means Done. {len(ac_initial_shared_concept_indices)} initial concepts.")
    main_logger.info("--- AutoCoRe Stage 2: Detector Sync & Vectorization ---")
    for ac_client_det in autocore_clients:
        if ac_client_det.state.get('final_embeddings') is not None and ac_client_det.state.get('cluster_labels') is not None:
            ac_client_det.train_concept_detectors(config)
    ac_all_detector_updates = [ac_client_det.get_detector_update(ac_initial_shared_concept_indices) for ac_client_det in autocore_clients]
    ac_final_concept_indices_ordered, ac_canonical_detector_params = autocore_server.aggregate_detectors(ac_all_detector_updates, ac_initial_shared_concept_indices)
    num_autocore_concepts = len(ac_final_concept_indices_ordered)
    if num_autocore_concepts == 0: main_logger.error("AC No concepts survived detector agg."); return
    autocore_concept_names = [f"autocore_concept_{i}" for i in range(num_autocore_concepts)] 
    ac_original_to_dense_map = {orig_idx: dense_idx for dense_idx, orig_idx in enumerate(ac_final_concept_indices_ordered)}
    autocore_server.phase2_prep_broadcast(autocore_clients, ac_final_concept_indices_ordered, ac_original_to_dense_map, ac_canonical_detector_params)
    client_X_autocore_concepts_map = {} 
    client_Y_labels_map = {}            
    for ac_client_vec in autocore_clients:
        if ac_client_vec.state.get('final_embeddings') is None: continue
        ac_client_vecs, ac_client_kept_ids = ac_client_vec.build_concept_vectors(config)
        if ac_client_vecs is not None and ac_client_vecs.shape[0] > 0:
            map_bid_lbl_ac_vec = {bid:lbl for bid,lbl in zip(ac_client_vec.state['loaded_base_ids'], ac_client_vec.state['loaded_labels_int'])}
            ac_client_lbls_vec = np.array([map_bid_lbl_ac_vec[bid] for bid in ac_client_kept_ids if bid in map_bid_lbl_ac_vec])
            if ac_client_lbls_vec.shape[0] == ac_client_vecs.shape[0]:
                client_X_autocore_concepts_map[ac_client_vec.client_id] = ac_client_vecs
                client_Y_labels_map[ac_client_vec.client_id] = ac_client_lbls_vec
    server_X_val_autocore_concepts, server_y_val_labels = None, None 
    server_X_test_autocore_concepts, server_y_test_labels = None, None 
    for holdout_set_name_ac in ["server_validation_set", "server_test_set"]:
        try:
            manifest_p_s_ac = os.path.join(cached_data_base_path, config['partition_manifest_dir_input'], holdout_set_name_ac, 'image_manifest.json')
            s_bids_ac, s_lbls_int_ac = load_labels_from_manifest(manifest_p_s_ac, main_logger)
            if not s_bids_ac: continue
            seg_infos_p_s_ac = os.path.join(cached_data_base_path, config['partition_segment_infos_cache_dir_input'], f"{holdout_set_name_ac}_segment_infos_with_crops.pkl")
            embed_p_s_ac = os.path.join(cached_data_base_path, config['embedding_cache_dir_input'], f"embeddings_{config['embedding_type']}_{holdout_set_name_ac}.pkl")
            with open(seg_infos_p_s_ac,'rb') as f: s_seg_infos_ac = pickle.load(f)
            with open(embed_p_s_ac,'rb') as f: s_embeds_ac = pickle.load(f)
            if s_seg_infos_ac is not None and s_embeds_ac is not None and len(s_seg_infos_ac) == s_embeds_ac.shape[0]:
                s_dense_detectors_ac = {ac_original_to_dense_map[ok]:mo for ok,mo in autocore_server.canonical_detector_model_objects.items() if ok in ac_original_to_dense_map}
                s_dense_thresholds_ac = {ac_original_to_dense_map[ok]:th for ok,th in autocore_server.canonical_thresholds.items() if ok in ac_original_to_dense_map}
                if s_dense_detectors_ac:
                    curr_X_s_concepts_ac, kept_ids_s_ac = build_image_concept_vectors(s_seg_infos_ac,s_dense_detectors_ac,s_dense_thresholds_ac,s_embeds_ac,num_autocore_concepts,config)
                    if curr_X_s_concepts_ac is not None and curr_X_s_concepts_ac.shape[0] > 0:
                        map_s_bid_lbl_ac = {bid:lbl for bid,lbl in zip(s_bids_ac, s_lbls_int_ac)}
                        aligned_s_lbls_ac = np.array([map_s_bid_lbl_ac[bid] for bid in kept_ids_s_ac if bid in map_s_bid_lbl_ac])
                        if curr_X_s_concepts_ac.shape[0] == aligned_s_lbls_ac.shape[0]:
                            if holdout_set_name_ac == "server_validation_set": server_X_val_autocore_concepts, server_y_val_labels = curr_X_s_concepts_ac, aligned_s_lbls_ac
                            elif holdout_set_name_ac == "server_test_set": server_X_test_autocore_concepts, server_y_test_labels = curr_X_s_concepts_ac, aligned_s_lbls_ac
        except Exception as e_s_vec_ac: main_logger.error(f"Error vectorizing AC server data for {holdout_set_name_ac}: {e_s_vec_ac}")
    if server_X_val_autocore_concepts is None or server_X_test_autocore_concepts is None: main_logger.error("Server val/test AC concepts failed. Exiting LR-XFL part."); return
    main_logger.info("AutoCoRe-FL Stages 1 & 2 Complete.")

    main_logger.info("\n======== PART 2: LR-XFL Federated Rule Learning (using AutoCoRe Concepts) ========")
    
    # --- LR-XFL Variable Naming and Setup (to match 3_run_xfl_with_predicted_concepts.py) ---
    n_concepts = num_autocore_concepts       
    n_classes = config['num_classes']        
    concept_names = autocore_concept_names   
    num_users = config['num_clients']        
    max_epoch = config['lrxfl_max_epoch']
    LOCAL_EPOCHS = config['lrxfl_local_epochs'] 
    loaded_scene_class_names = config.get('loaded_scene_class_names', []) # From AutoCoRe part
    SEED = config['seed']
    SAMPLE = config.get('lrxfl_sample_type','iid') # 'iid' or 'non-iid' for client data distribution FROM global train_data
    TOPK_EXPLANATIONS_LOCAL = config['lrxfl_topk_explanations_local']
    TOPK_EXPLANATIONS_GLOBAL = config['lrxfl_topk_explanations_global']
    LOGIC_GENERATION_THRESHOLD = config['lrxfl_logic_generation_threshold']
    LEARNING_RATE = config['lrxfl_learning_rate']
    L1_REG = config['lrxfl_l1_reg']
    TEMPERATURE = config['lrxfl_temperature']
    EXPLAINER_HIDDEN = config['lrxfl_explainer_hidden']
    BATCH_SIZE = config['lrxfl_batch_size'] 
    DEVICE = torch_device_main 

    # --- Data Preparation for LR-XFL (Faithful to 3_run_xfl_with_predicted_concepts.py structure) ---
    # 1. Create the single global dataset for LR-XFL by pooling all available AutoCoRe concept data.
    all_X_for_lrxfl_dataset = []
    all_Y_int_for_lrxfl_dataset = []

    # Add data from all clients that have it
    for client_id_pool in range(num_users):
        if client_id_pool in client_X_autocore_concepts_map:
            all_X_for_lrxfl_dataset.append(client_X_autocore_concepts_map[client_id_pool])
            all_Y_int_for_lrxfl_dataset.append(client_Y_labels_map[client_id_pool])
    
    # Add AutoCoRe server's validation and test data to this initial pool
    # This makes LR-XFL's subsequent train/val/test split operate on the true "all available data"
    if server_X_val_autocore_concepts is not None and server_y_val_labels is not None:
        all_X_for_lrxfl_dataset.append(server_X_val_autocore_concepts)
        all_Y_int_for_lrxfl_dataset.append(server_y_val_labels)
    if server_X_test_autocore_concepts is not None and server_y_test_labels is not None:
        all_X_for_lrxfl_dataset.append(server_X_test_autocore_concepts)
        all_Y_int_for_lrxfl_dataset.append(server_y_test_labels)

    if not all_X_for_lrxfl_dataset:
        main_logger.error("LR-XFL: No AutoCoRe concepts available to form the global dataset pool. Exiting.")
        return

    x_data_np = np.vstack(all_X_for_lrxfl_dataset)
    y_scenes_scalar_np = np.concatenate(all_Y_int_for_lrxfl_dataset)

    x_data_tensor = torch.FloatTensor(x_data_np)
    y_scenes_one_hot = torch.nn.functional.one_hot(
        torch.from_numpy(y_scenes_scalar_np).long(), num_classes=n_classes
    ).float()

    dataset = TensorDataset(x_data_tensor, y_scenes_one_hot) # This is the global dataset
    main_logger.info(f"LR-XFL: Global dataset created from all AutoCoRe concepts: X_shape={x_data_tensor.shape}, Y_shape={y_scenes_one_hot.shape}")

    # 2. Split this dataset into LR-XFL's global train_data, val_data, test_data (Subsets)
    #    Using naming from original LR-XFL script for these global splits.
    train_size = int(len(dataset) * config.get('lrxfl_global_train_split_ratio', 0.9))
    val_test_remaining = len(dataset) - train_size
    val_size = val_test_remaining // 2
    test_size = val_test_remaining - val_size
    
    train_data, val_data, test_data = random_split( # These are Subset objects
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    main_logger.info(f"LR-XFL Global Data Split: train_data={len(train_data)}, val_data={len(val_data)}, test_data={len(test_data)}")

    # DataLoaders for global model evaluation (using these Subset objects)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=config.get('dataloader_num_workers',0))
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=config.get('dataloader_num_workers',0))
    
    # For global_trainer.predict (X from test_data Subset, Y_dummy)
    lrxfl_test_X_for_predict = test_data.dataset.tensors[0][test_data.indices] 
    lrxfl_dummy_Y_for_predict = torch.zeros(lrxfl_test_X_for_predict.size(0), n_classes).float()
    lrxfl_dataloader_for_global_predict = DataLoader(
        TensorDataset(lrxfl_test_X_for_predict, lrxfl_dummy_Y_for_predict),
        batch_size=BATCH_SIZE, num_workers=config.get('dataloader_num_workers',0)
    )

    # 3. Distribute LR-XFL's global train_data (Subset) among clients
    if SAMPLE == 'iid': # Using SAMPLE from LR-XFL config section
        user_groups_train = ade20k_iid(train_data, num_users, seed=SEED)
    else: # non-iid
        user_groups_train = ade20k_noniid_by_concept(train_data, num_users, num_concepts=n_concepts, seed=SEED)
    
    train_loader_users = []; val_loader_users = []; test_loader_users = []

    # --- Determine which clients will be noisy for the LR-XFL phase ---
    noisy_client_ids_for_lrxfl = []
    if config.get("noise_experiment_enabled", False):
        main_logger.info(f"--- Preparing for Client Label Noise in LR-XFL Phase ---")
        num_total_lrxfl_clients = config.get("num_clients", 10) # Should be same as AutoCoRe num_clients
        num_noisy_clients_to_select = int(num_total_lrxfl_clients * config.get("noise_client_percentage", 0.0))
        
        all_lrxfl_client_indices = list(range(num_total_lrxfl_clients))
        random.shuffle(all_lrxfl_client_indices) # Shuffle to pick random clients
        noisy_client_ids_for_lrxfl = all_lrxfl_client_indices[:num_noisy_clients_to_select]
        main_logger.info(f"LR-XFL: Designating {len(noisy_client_ids_for_lrxfl)} clients as noisy: {sorted(noisy_client_ids_for_lrxfl)}")
        main_logger.info(f"LR-XFL: Noisy clients will have {config.get('noise_label_shuffle_degree', 0.0)*100:.1f}% of their labels shuffled.")


    train_loader_users = []; val_loader_users = []; test_loader_users = []
    for user in range(num_users): # user is the iterator, matching original LR-XFL
        client_indices_in_global_train = list(user_groups_train[user]) 
        if not client_indices_in_global_train:
            empty_ds_cl_user = TensorDataset(torch.empty(0, n_concepts), torch.empty(0, n_classes))
            train_loader_users.append(DataLoader(empty_ds_cl_user)); val_loader_users.append(DataLoader(empty_ds_cl_user)); test_loader_users.append(DataLoader(empty_ds_cl_user))
            continue
        
        # This client_data_subset is a Subset of train_data (which itself is a Subset of original dataset)
        client_data_subset_for_user = Subset(train_data, client_indices_in_global_train)
        
        current_X_for_client, current_Y_onehot_for_client = get_xy(client_data_subset_for_user) # Use helper
        current_X_for_client = current_X_for_client.to(DEVICE) # Ensure on device
        current_Y_onehot_for_client = current_Y_onehot_for_client.to(DEVICE)

        # --- Apply label noise if this client is selected as noisy ---
        if config.get("noise_experiment_enabled", False) and user in noisy_client_ids_for_lrxfl:
            main_logger.info(f"LR-XFL Client {user}: Applying label noise.")
            noise_degree = config.get("noise_label_shuffle_degree", 0.0)
            num_samples_this_client = current_Y_onehot_for_client.size(0)
            num_to_shuffle = int(num_samples_this_client * noise_degree)

            if num_to_shuffle > 0 and num_samples_this_client > 1:
                # Convert one-hot Y to integer labels for shuffling
                y_int_for_shuffling_np = torch.argmax(current_Y_onehot_for_client, dim=1).cpu().numpy()
                
                indices_to_shuffle_np = np.random.choice(num_samples_this_client, num_to_shuffle, replace=False)
                
                # Get the labels at these specific indices
                labels_at_indices_to_shuffle = y_int_for_shuffling_np[indices_to_shuffle_np].copy()
                np.random.shuffle(labels_at_indices_to_shuffle) # Shuffle these selected labels
                
                # Place them back into a copy of the original integer labels
                y_int_noisy_np = y_int_for_shuffling_np.copy()
                y_int_noisy_np[indices_to_shuffle_np] = labels_at_indices_to_shuffle
                
                # Convert noisy integer labels back to one-hot and to the correct device
                current_Y_onehot_for_client = torch.nn.functional.one_hot(
                    torch.from_numpy(y_int_noisy_np).long(), num_classes=n_classes
                ).float().to(DEVICE)
                
                changed_count = np.sum(y_int_for_shuffling_np[indices_to_shuffle_np] != y_int_noisy_np[indices_to_shuffle_np])
                main_logger.debug(f"  Client {user}: {changed_count}/{num_to_shuffle} (intended) labels shuffled.")
            else:
                main_logger.debug(f"  Client {user}: No noise applied (num_to_shuffle={num_to_shuffle}).")
        # --- End of noise application for this client ---

        # Create the TensorDataset for this client using (possibly noised) Y
        client_dataset_for_local_split = TensorDataset(current_X_for_client, current_Y_onehot_for_client)
        
        # Local train/val/test split for this client's PTL trainer
        train_size_user = int(len(client_dataset_for_local_split) * 0.9) 
        val_size_user = (len(client_dataset_for_local_split) - train_size_user) // 2
        test_size_user = len(client_dataset_for_local_split) - train_size_user - val_size_user
        
        if len(client_dataset_for_local_split) < 3 : 
             train_data_user_dl_obj, val_data_user_dl_obj, test_data_user_dl_obj = client_dataset_for_local_split, Subset(client_dataset_for_local_split,[]), Subset(client_dataset_for_local_split,[])
        else:
            train_data_user_dl_obj, val_data_user_dl_obj, test_data_user_dl_obj = random_split(client_dataset_for_local_split, 
                [train_size_user, val_size_user, test_size_user], 
                generator=torch.Generator().manual_seed(SEED))
        
        train_loader_users.append(DataLoader(train_data_user_dl_obj, batch_size=max(1, len(train_data_user_dl_obj)), num_workers=config.get('dataloader_num_workers',0)))
        val_loader_users.append(DataLoader(val_data_user_dl_obj, batch_size=max(1, len(val_data_user_dl_obj)), num_workers=config.get('dataloader_num_workers',0)))
        test_loader_users.append(DataLoader(test_data_user_dl_obj, batch_size=max(1, len(test_data_user_dl_obj)), num_workers=config.get('dataloader_num_workers',0)))
    # --- Initialize LR-XFL Global Model and Trainer (variables match original LR-XFL script) ---
    global_model = Explainer(
        n_concepts=n_concepts, n_classes=n_classes, 
        l1=L1_REG, temperature=TEMPERATURE, 
        lr=LEARNING_RATE, explainer_hidden=EXPLAINER_HIDDEN
    )
    global_model.to(DEVICE)

    lrxfl_run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name_suffix_lrxfl = (f"{SAMPLE}_users{num_users}_flEp{max_epoch}_locEp{LOCAL_EPOCHS}"
                           f"_scenes21_geneTh{LOGIC_GENERATION_THRESHOLD}_AC_{lrxfl_run_timestamp}")
    base_dir_results = os.path.join(combined_run_output_dir, "lrxfl_phase_results", f"Explainer_{exp_name_suffix_lrxfl}")
    os.makedirs(base_dir_results, exist_ok=True)
    output_log_filename = os.path.join(base_dir_results, f"Log_{exp_name_suffix_lrxfl}.txt")
    
    eval_logger = PL_CSVLogger(save_dir=os.path.join(base_dir_results, "_pl_eval_logs"), name="global_server_eval", version=f"run_{lrxfl_run_timestamp}")
    global_trainer = Trainer(
        accelerator=config['device'], devices=1, deterministic=True, logger=eval_logger,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False, max_epochs=1 
    )
    
    global_weights = global_model.state_dict()
    local_weights = {}
    local_explanation_f = {} 
    explanations = {i: [] for i in range(n_classes)}
    global_model_results_list = []

    with open(output_log_filename, 'w') as file:
        file.write(f"Experiment Run: {exp_name_suffix_lrxfl}\n") # Use the LR-XFL specific suffix
        file.write(f"Input Features: AutoCoRe Concepts ({n_concepts} concepts)\n")
        file.write(f"Target Scenes: {loaded_scene_class_names}\n")
        file.write(f"Seed: {SEED}, Num Users: {num_users}\n")
        file.write(f"FL Epochs: {max_epoch}, Local Epochs: {LOCAL_EPOCHS}\n")
        file.write(f"Sampling for LR-XFL client distribution from global train_data: {SAMPLE}\n")
        file.write(f"Explainer Hidden Layers: {EXPLAINER_HIDDEN}\n")
        file.write("---------------------------\n\n")

    main_logger.info(f"Starting LR-XFL federated training loop (Epochs: {max_epoch})...")

    # --- LR-XFL Federated Loop (Exact Replication from here) ---
    for epoch in range(max_epoch):
        main_logger.info(f"--- LR-XFL Global Epoch {epoch + 1}/{max_epoch} ---")
        global_connector_class = [[] for _ in range(n_classes)] 
        users_for_train = [i for i in range(num_users)]
        
        for user_id in users_for_train:
            if not train_loader_users[user_id].dataset or len(train_loader_users[user_id].dataset) == 0:
                local_weights[user_id] = None; local_explanation_f[user_id] = None; continue

            model_for_local_train = copy.deepcopy(global_model) # Matches 'model' in local_train call
            model_for_local_train.to(DEVICE)
            client_lrxfl_ckpt_dir_epoch = os.path.join(base_dir_results, f"client_{user_id}_checkpoints_epoch{epoch+1}")
            os.makedirs(client_lrxfl_ckpt_dir_epoch, exist_ok=True)

            # Call to LR-XFL's local_train
            # The local_train from LR-XFL returns: weights, concept_mask, results, explanation_f
            local_weights[user_id], _, _, local_explanation_f[user_id] = local_train(
                user_id=user_id, epochs=LOCAL_EPOCHS, 
                train_loader=train_loader_users[user_id],
                val_loader=val_loader_users[user_id],
                test_loader=test_loader_users[user_id],
                n_classes=n_classes, n_concepts=n_concepts, 
                concept_names=concept_names, # Pass AutoCoRe concept names
                base_dir=client_lrxfl_ckpt_dir_epoch, 
                results_list=[], explanations=[], model=model_for_local_train,
                topk_explanations=TOPK_EXPLANATIONS_LOCAL, 
                verbose=False,
                logic_generation_threshold=LOGIC_GENERATION_THRESHOLD
            )
            
            if local_explanation_f[user_id] is not None:
                for f_item_loop in local_explanation_f[user_id]: # Renamed 'f'
                    if f_item_loop.get('explanation_connector') is not None:
                        target_class_loop_idx = f_item_loop.get('target_class')
                        if target_class_loop_idx is not None and 0 <= target_class_loop_idx < n_classes:
                             global_connector_class[target_class_loop_idx].append(f_item_loop['explanation_connector'])
        
        # Filter for active clients for weight aggregation
        active_local_weights = {uid: w for uid,w in local_weights.items() if w is not None}
        if not active_local_weights: main_logger.warning(f"LR-XFL Ep {epoch+1}: No active clients with weights."); continue

        # global_y_test_out: predictions of current global_model on global test_data
        global_model.eval()
        # Original: global_y_test_out = global_trainer.predict(global_model, dataloaders=test_data.dataset.tensors[0][test_data.indices])
        # Use the lrxfl_dataloader_for_global_predict which has (X_test, Y_dummy)
        predict_batches_out_loop = global_trainer.predict(global_model, dataloaders=lrxfl_dataloader_for_global_predict)
        
        if predict_batches_out_loop and isinstance(predict_batches_out_loop, list):
            try: # Attempt stack and squeeze as in original
                global_y_test_out = squeeze(stack(predict_batches_out_loop, dim=0), dim=1)
                # Additional checks for squeeze behavior if PTL predict returns differently
                if global_y_test_out.ndim > 2 and global_y_test_out.shape[0] == 1 and global_y_test_out.shape[1] != n_classes:
                    global_y_test_out = global_y_test_out.squeeze(0)
                elif global_y_test_out.ndim > 2 and global_y_test_out.shape[1] == 1 and global_y_test_out.shape[0] != n_classes: # Error in original logic, should be shape[0] != num_samples
                    global_y_test_out = global_y_test_out.squeeze(1)
                # If only one batch was predicted, PTL predict might return the tensor directly or a list with one tensor
                if len(predict_batches_out_loop) == 1 and isinstance(predict_batches_out_loop[0], torch.Tensor) and global_y_test_out.shape != predict_batches_out_loop[0].shape:
                     global_y_test_out = predict_batches_out_loop[0]
            except RuntimeError: global_y_test_out = torch.cat(predict_batches_out_loop, dim=0)
        elif isinstance(predict_batches_out_loop, torch.Tensor): global_y_test_out = predict_batches_out_loop
        else: global_y_test_out = torch.empty(0, n_classes).to(DEVICE); main_logger.warning("Predict global_y_test_out empty.")

        user_to_engage_class, local_explanations_accuracy_class, local_explanations_support_class = \
            client_selection_class(n_classes, num_users, local_explanation_f)
        
        users_to_aggregate = set([])
        users_aggregation_weight = {u: 0 for u in range(num_users)} # Corrected from range(10)
        global_explanation_accuracy = 0 
        global_explanation_fidelity = 0 

        for target_class in range(n_classes):
            if len(global_connector_class[target_class]) != 0:
                counts = Counter(global_connector_class[target_class])
                global_connector = counts.most_common(1)[0][0]
            else: global_connector = 'AND'
            
            # Use val_data (Subset) correctly for _global_aggregate_explanations
            x_val_for_agg = val_data.dataset.tensors[0][val_data.indices].to(DEVICE)
            y_val_for_agg = val_data.dataset.tensors[1][val_data.indices].to(DEVICE)
            
            global_explanation_class, _, user_to_engage_class[target_class] = _global_aggregate_explanations( # global_accuracy_class from here not used
                local_explanations_accuracy_class.get(target_class, {}), local_explanations_support_class.get(target_class, {}),
                TOPK_EXPLANATIONS_GLOBAL, target_class, x_val_for_agg, y_val_for_agg,
                concept_names, 'large', global_connector, 1 
            )
            
            # Use test_data (Subset) correctly for test_explanation
            x_test_for_eval_rule = test_data.dataset.tensors[0][test_data.indices].to(DEVICE)
            y_test_for_eval_rule = test_data.dataset.tensors[1][test_data.indices].to(DEVICE)
            
            global_explanation_accuracy_class, global_y_formula_class = test_explanation(
                global_explanation_class, x_test_for_eval_rule, y_test_for_eval_rule, target_class
            )
            global_explanation_accuracy += global_explanation_accuracy_class
            
            global_explanation_fidelity_class = 0.0 
            if global_y_formula_class is not None and global_y_test_out.numel() > 0 :
                model_preds = global_y_test_out.argmax(dim=1)            
                model_bool = (model_preds == target_class)               
                y_true_np = model_bool.detach().cpu().numpy() 
                y_pred_np = global_y_formula_class.cpu().numpy() if isinstance(global_y_formula_class, torch.Tensor) else np.array(global_y_formula_class)
                if y_true_np.shape == y_pred_np.shape:
                    global_explanation_fidelity_class = accuracy_score(y_true_np, y_pred_np)
            global_explanation_fidelity += global_explanation_fidelity_class
            
            global_explanation_class_named = replace_names(global_explanation_class, concept_names) if global_explanation_class and concept_names else (global_explanation_class or "False")
            explanations[target_class].append(global_explanation_class_named)
            
            if global_explanation_class: 
                with open(output_log_filename, 'a') as flog_loop: 
                    flog_loop.write('------------------\n')
                    flog_loop.write(f'Epoch {epoch+1}, Class: {target_class} ({loaded_scene_class_names[target_class]})\n')
                    flog_loop.write(f'Global explanation: {global_explanation_class_named}\n')
                    flog_loop.write(f'Explanation accuracy: {global_explanation_accuracy_class:.4f}\n')
                    flog_loop.write(f'Explanation fidelity: {global_explanation_fidelity_class:.4f}\n')
            users_to_aggregate = users_to_aggregate.union(user_to_engage_class.get(target_class, set()))
        
        for class_idx_uw in range(n_classes):
            if user_to_engage_class.get(class_idx_uw):
                for user_engaged_id in user_to_engage_class[class_idx_uw]:
                    if user_engaged_id in users_aggregation_weight: users_aggregation_weight[user_engaged_id] +=1
        
        # Weight aggregation logic from original LR-XFL script
        if users_to_aggregate is not None and len(users_to_aggregate) != 0:
            global_weights = weighted_weights(active_local_weights, users_aggregation_weight) # Use active_local_weights
        else:
            global_weights = average_weights(active_local_weights) # Use active_local_weights
        global_model.load_state_dict(global_weights)
        # global_model_parameters_list.append(global_weights)

        global_model_validation_results = global_trainer.test(copy.deepcopy(global_model), dataloaders=val_loader, verbose=False)
        global_model_results_list.append(global_model_validation_results)
        current_gm_val_acc_report = global_model_validation_results[0]['test_acc_epoch'] if global_model_validation_results else 0.0
        avg_ep_rule_acc_report = global_explanation_accuracy / n_classes if n_classes > 0 else 0
        avg_ep_rule_fid_report = global_explanation_fidelity / n_classes if n_classes > 0 else 0
        main_logger.info(f"LR-XFL Ep{epoch+1}: GlobalModelValAcc={current_gm_val_acc_report:.4f}, AvgRuleAcc(Test)={avg_ep_rule_acc_report:.4f}, AvgRuleFid(Test)={avg_ep_rule_fid_report:.4f}")
        with open(output_log_filename, 'a') as flog_sum: 
            flog_sum.write('---------------------------\n')
            flog_sum.write(f'Epoch {epoch+1} Summary: Global Model Val Acc: {current_gm_val_acc_report:.4f}\n')
            flog_sum.write(f'Avg Global Rule Accuracy (Test): {avg_ep_rule_acc_report:.4f}\n')
            flog_sum.write(f'Avg Global Rule Fidelity (Test): {avg_ep_rule_fid_report:.4f}\n')
            flog_sum.write('---------------------------\n')

        if len(global_model_results_list) > 1 and config.get("lrxfl_early_stopping", True):
            if global_model_results_list[-1][0]['test_acc_epoch'] < global_model_results_list[-2][0]['test_acc_epoch']:
                main_logger.info('LR-XFL Global model validation accuracy decreased. Breaking federated loop.'); break
    
    # --- Final LR-XFL Evaluation (Matches original LR-XFL script) ---
    main_logger.info("\n--- Final LR-XFL Evaluation on Global Test Set (test_loader) ---")
    global_model_results = global_trainer.test(copy.deepcopy(global_model), dataloaders=test_loader, verbose=True)
    final_lrxfl_model_test_acc = global_model_results[0]['test_acc_epoch'] if global_model_results else 0.0
    
    final_global_explanation_accuracy = 0; final_global_explanation_fidelity = 0
    final_global_rules_report = []
    
    final_predict_batches_report = global_trainer.predict(global_model, dataloaders=lrxfl_dataloader_for_global_predict)
    if final_predict_batches_report and isinstance(final_predict_batches_report, list):
        final_global_y_test_out = torch.cat(final_predict_batches_report, dim=0)
    elif isinstance(final_predict_batches_report, torch.Tensor): final_global_y_test_out = final_predict_batches_report
    else: final_global_y_test_out = torch.empty(0, n_classes).to(DEVICE)

    scene_to_global_idx_map_inv_report = {v: k for k, v in scene_to_global_idx_map.items()}
    for target_class_rep_final in range(n_classes):
        last_epoch_rule_final_report = explanations[target_class_rep_final][-1] if explanations[target_class_rep_final] else "False"
        final_global_rules_report.append(f"Class {target_class_rep_final} ({scene_to_global_idx_map_inv_report.get(target_class_rep_final, 'Unk')}): {last_epoch_rule_final_report}")
        
        test_X_final_eval_report = test_data.dataset.tensors[0][test_data.indices].to(DEVICE)
        test_Y_final_eval_report = test_data.dataset.tensors[1][test_data.indices].to(DEVICE)
        final_global_model_test_results = global_trainer.test(global_model.to(next(global_trainer.model.parameters()).device if global_trainer.model else 'cpu'), dataloaders=test_loader, verbose=False)
        final_test_acc = final_global_model_test_results[0]['test_acc_epoch']

            
    avg_final_global_rule_acc = final_global_explanation_accuracy / n_classes if n_classes > 0 else 0
    avg_final_global_rule_fid = final_global_explanation_fidelity / n_classes if n_classes > 0 else 0
    
    consistencies = [formula_consistency(explanations[j_c_final]) for j_c_final in range(n_classes) if explanations[j_c_final]]
    explanation_consistency = np.mean(consistencies) if consistencies else 0.0
    
    main_logger.info(f"======== FINAL AutoCoRe-then-LRXFL RESULTS for Run ID: {run_id} ========")
    main_logger.info(f"  LR-XFL Global Model Test Accuracy: {final_lrxfl_model_test_acc:.4f}")
    main_logger.info(f"  LR-XFL Avg Global Rule Accuracy (Test): {avg_final_global_rule_acc:.4f}")
    main_logger.info(f"  LR-XFL Avg Global Rule Fidelity (Test): {avg_final_global_rule_fid:.4f}")
    main_logger.info(f"  LR-XFL Avg Rule Consistency: {explanation_consistency:.4f}")
    main_logger.info(f"  Number of AutoCoRe Concepts: {n_concepts}")
    main_logger.info(f"  Final Global Rules (Last Epoch):")
    for rule_str_final_print in final_global_rules_report: main_logger.info(f"    {rule_str_final_print}")

    # Save summary metrics
    # results_list in original LR-XFL script was for multi-seed runs. Here we make a direct summary.
    final_summary_data_dict = { "run_id": run_id, "method": config.get('method_name'), "num_autocore_concepts": n_concepts,
        "lrxfl_model_test_accuracy": final_lrxfl_model_test_acc, "lrxfl_avg_global_rule_accuracy": avg_final_global_rule_acc,
        "lrxfl_avg_global_rule_fidelity": avg_final_global_rule_fid, "lrxfl_avg_rule_consistency": explanation_consistency,
        **{k: (str(v) if isinstance(v,(list,dict)) else v) for k,v in config.items()}}
    summary_df_final_output = pd.DataFrame([final_summary_data_dict]) # Renamed
    summary_csv_path_final_output = os.path.join(combined_run_output_dir, "final_summary_metrics_autocore_lrxfl.csv")
    summary_df_final_output.to_csv(summary_csv_path_final_output, index=False)
    main_logger.info(f"Final summary metrics saved to {summary_csv_path_final_output}")
    
    # Append to LR-XFL specific text log
    with open(output_log_filename, 'a') as flog_final_summary: 
        flog_final_summary.write('---------------------------\nFINAL EVALUATION ON GLOBAL TEST SET\n')
        flog_final_summary.write(f'Global LR-XFL Model Accuracy: {final_lrxfl_model_test_acc:.4f}\n')
        flog_final_summary.write(f'Avg Global Rule Accuracy (Test): {avg_final_global_rule_acc:.4f}\n')
        flog_final_summary.write(f'Avg Global Rule Fidelity (Test): {avg_final_global_rule_fid:.4f}\n')
        flog_final_summary.write(f'Avg Rule Consistency: {explanation_consistency:.4f}\n')
        flog_final_summary.write('---------------------------\n')

    main_logger.info(f"======== Run {run_id} Complete. Results in {combined_run_output_dir} ========")

if __name__ == '__main__':
    main()