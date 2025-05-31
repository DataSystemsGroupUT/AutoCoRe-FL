import os
import sys
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import TensorDataset 
import logging
from sklearn.model_selection import train_test_split 
from tqdm import tqdm


project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from AutoCore_FL.scripts.utils import setup_logging, generate_run_id, save_config, load_config, SAM2Filter, evaluate_global_AutoCore_model, ade20k_iid, ade20k_noniid_by_concept
from AutoCore_FL.federated.client import FederatedClient 
from AutoCore_FL.federated.server import FederatedServer

def main():
    config = load_config("/gpfs/helios/home/soliman/logic_explained_networks/experiments/AutoCore_FL/configs/config_autocore_fl_resnet_ade20k.yaml")
    run_id = generate_run_id(config['method_name'])
    config['current_run_id'] = run_id 

    base_results_dir = os.path.join("/gpfs/helios/home/soliman/logic_explained_networks/experiments/AutoCore_FL/results", run_id)
    log_dir = os.path.join(base_results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    config['log_dir_run'] = log_dir # For outputs specific to this run

    logger = setup_logging(log_dir=log_dir, run_id=run_id, log_level_str=config.get("log_level", "INFO"))
    root_logger_instance = logging.getLogger()
    root_logger_instance.addFilter(SAM2Filter()) 
    logger.info(f"***** Starting AutoCore FL with ResNet Concepts: {run_id} *****")
    save_config(config, os.path.join(base_results_dir, "config_this_run.json"))

    # --- Seed ---
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed']) # For python's random, used by some samplers
    if config['device'] == 'cuda' and torch.cuda.is_available():
        torch_device = torch.device("cuda"); torch.cuda.manual_seed_all(config['seed'])
    else:
        torch_device = torch.device("cpu")
    logger.info(f"Using device: {torch_device}")

    # --- 1. Load Pre-computed ResNet18 Concept Features and Labels ---
    logger.info("--- Stage 1: Loading Pre-computed ResNet18 Concept Data ---")
    scenes_for_fn_list = config.get('scenes_for_filename_part', config.get('chosen_classes'))
    scene_fn_part_cfg = '_'.join(sorted([s.replace("_", "") for s in scenes_for_fn_list]))[:50]
    model_arch_fn_part_cfg = config['model_arch_fn_part']
    
    pred_x_name = f"{model_arch_fn_part_cfg}_binary_X_scenes_{scene_fn_part_cfg}.npy"
    pred_y_name = f"{model_arch_fn_part_cfg}_binary_Y_scenes_{scene_fn_part_cfg}.npy"
    scene_names_npy_name = f"{model_arch_fn_part_cfg}_binary_scene_names_{scene_fn_part_cfg}.npy"
    obj_names_npy_name = f"{model_arch_fn_part_cfg}_binary_object_names.npy"

    features_dir = config['features_load_dir']
    try:
        X_all_resnet_concepts = np.load(os.path.join(features_dir, pred_x_name))
        y_all_scene_labels_int = np.load(os.path.join(features_dir, pred_y_name)) # 1D int labels
        scene_class_names_ordered = np.load(os.path.join(features_dir, scene_names_npy_name), allow_pickle=True).tolist()
        concept_feature_names = np.load(os.path.join(features_dir, obj_names_npy_name), allow_pickle=True).tolist()
    except FileNotFoundError as e:
        logger.error(f"CRITICAL NPY FILE NOT FOUND: {e}. Ensure paths and filenames in config are correct and "
                       f"2_generate_predicted_concept_features.py was run for these scenes/model.")
        return

    config['num_classes'] = len(scene_class_names_ordered)
    num_concept_features = X_all_resnet_concepts.shape[1]
    if num_concept_features != len(concept_feature_names):
        logger.error(f"Feature name/data mismatch. X feats:{num_concept_features}, Names:{len(concept_feature_names)}. Exiting."); return
    
    logger.info(f"Loaded ResNet concepts: X {X_all_resnet_concepts.shape}, Y {y_all_scene_labels_int.shape}")
    logger.info(f"{config['num_classes']} scenes: {scene_class_names_ordered[:3]}...")
    logger.info(f"{num_concept_features} ResNet concepts: {concept_feature_names[:3]}...")

    # --- 2. Split Data: Server Holdout & Client Pool ---
    logger.info("--- Stage 2: Splitting Data for Federation ---")
    try:
        server_holdout_X, client_pool_X_np, server_holdout_y, client_pool_y_np = train_test_split(
            X_all_resnet_concepts, y_all_scene_labels_int,
            test_size=(1.0 - config['server_split_ratio']), random_state=config['seed'], stratify=y_all_scene_labels_int
        )
        server_val_X, server_test_X, server_val_y, server_test_y = train_test_split(
            server_holdout_X, server_holdout_y,
            test_size=config['server_val_test_split_ratio'], random_state=config['seed'], stratify=server_holdout_y
        )
    except ValueError as e_split: # Fallback for stratification issues
        logger.warning(f"Stratification failed ({e_split}). Using non-stratified splits.")
        server_holdout_X, client_pool_X_np, server_holdout_y, client_pool_y_np = train_test_split(
            X_all_resnet_concepts, y_all_scene_labels_int, test_size=(1.0 - config['server_split_ratio']), random_state=config['seed']
        )
        server_val_X, server_test_X, server_val_y, server_test_y = train_test_split(
            server_holdout_X, server_holdout_y, test_size=config['server_val_test_split_ratio'], random_state=config['seed']
        )
    logger.info(f"Server Val: X{server_val_X.shape}, Y{server_val_y.shape}. Server Test: X{server_test_X.shape}, Y{server_test_y.shape}")
    logger.info(f"Client Pool: X{client_pool_X_np.shape}, Y{client_pool_y_np.shape}")

    # Create PyTorch Dataset for client_pool to pass to your samplers
    client_pool_tensor_X = torch.FloatTensor(client_pool_X_np)
    # For ade20k_noniid_by_concept, y is not directly used for sharding, but dataset expects it.
    # For ade20k_iid, y is not used.
    # If your samplers were to use Y for sharding, ensure it's in the format they expect (e.g., one-hot if needed).
    # Here, passing 1D integer labels.
    client_pool_tensor_y_int = torch.LongTensor(client_pool_y_np) 
    client_pool_pytorch_dataset = TensorDataset(client_pool_tensor_X, client_pool_tensor_y_int)

    if config['federated_sampling'] == 'iid':
        user_groups_indices_map = ade20k_iid(client_pool_pytorch_dataset, config['num_users'], seed=config['seed'])
    elif config['federated_sampling'] == 'non-iid':
        logger.info(f"Calling ade20k_noniid_by_concept with num_concepts_for_sharding = {num_concept_features}")
        user_groups_indices_map = ade20k_noniid_by_concept(
            client_pool_pytorch_dataset, config['num_users'],
            num_concepts_for_sharding=num_concept_features, # Shard by the 150 ResNet concept features
            seed=config['seed']
        )
    else:
        raise ValueError(f"Unsupported federated_sampling: {config['federated_sampling']}")
    logger.info("Client data partitioned.")

    # --- 3. Initialize Clients and Server ---
    logger.info("--- Stage 3: Initializing Clients and Server ---")
    clients = []
    client_true_y_labels_for_fl_loop_list = [] 

    for i in range(config['num_users']):
        client_config_copy = config.copy(); client_config_copy['client_id'] = i
        client = FederatedClient(client_id=i, config=client_config_copy, data_partition=[], scene_to_idx={}, skip_model_initialization=True)
        
        indices_for_this_client_raw = user_groups_indices_map.get(i, [])
        client_indices_list = sorted(list(indices_for_this_client_raw)) if isinstance(indices_for_this_client_raw, set) else [int(idx) for idx in indices_for_this_client_raw]

        if not client_indices_list:
            client.state['concept_vecs'] = np.empty((0, num_concept_features), dtype=np.float32)
            client_true_y_labels_for_fl_loop_list.append(np.array([], dtype=np.int64))
        else:
            client.state['concept_vecs'] = client_pool_X_np[client_indices_list]
            client_true_y_labels_for_fl_loop_list.append(client_pool_y_np[client_indices_list])
        
        client.state['feature_names_for_figs'] = concept_feature_names
        client.state['accumulated_global_model_Fm_terms'] = []
        client.state['learning_rate_gbm'] = config['learning_rate_gbm']
        client.state['num_classes'] = config['num_classes'] # Ensure PatchedFIGSClassifier knows this
        clients.append(client)
        logger.info(f"Client {i}: X_concepts {client.state['concept_vecs'].shape}, Y_labels {client_true_y_labels_for_fl_loop_list[-1].shape}")

    server = FederatedServer(config)
    server.feature_names_for_figs = concept_feature_names
    server.server_X_test_concepts_for_validation = server_val_X
    server.server_y_test_labels_for_validation = server_val_y
    server.server_feature_names_for_validation = concept_feature_names
    server.accumulated_Fm_global_terms = []
    logger.info(f"Server validation data (ResNet concepts) set: X_shape={server_val_X.shape}")

    logger.info(f"--- Stage 4: AutoCore FL on ResNet Concepts ---")
    rounds_no_model_change = 0; final_fl_round_completed = 0
    all_round_metrics = []

    for r_fl in range(config["fl_rounds"]):
        final_fl_round_completed = r_fl + 1
        logger.info(f"--- AutoCore FL Round {r_fl + 1}/{config['fl_rounds']} ---")
        client_figs_updates_this_round = []
        active_clients_this_fl_round = 0

        for client_idx, client in enumerate(tqdm(clients, desc=f"FL Round {r_fl+1} Clients", file=sys.stdout, leave=False)):
            client_resnet_concept_vectors = client.state['concept_vecs']
            client_true_labels_for_figs = client_true_y_labels_for_fl_loop_list[client_idx]

            if client_resnet_concept_vectors is None or client_resnet_concept_vectors.shape[0] == 0:
                logger.debug(f"Client {client_idx}: No data. Skip."); continue
            if client_true_labels_for_figs.shape[0] != client_resnet_concept_vectors.shape[0]:
                logger.error(f"Client {client_idx}: Mismatch X({client_resnet_concept_vectors.shape}) Y({client_true_labels_for_figs.shape}). Skipping."); continue
            
            active_clients_this_fl_round +=1
            
            client_round_config_for_train = client.config.copy()
            client_round_config_for_train['current_round'] = r_fl
            client_round_config_for_train['figs_params'] = config['figs_params']
            client_round_config_for_train['num_classes'] = config['num_classes'] # Ensure client's train_figs knows this

            _, terms_hk_from_client = client.train_figs(client_true_labels_for_figs, client_round_config_for_train)
            
            if terms_hk_from_client:
                update_payload = client.get_model_update()
                client_figs_updates_this_round.append(update_payload)
            else:
                logger.info(f"Client {client_idx}: No terms generated in FL round {r_fl+1}.")
        
        if active_clients_this_fl_round == 0: logger.warning(f"FL Round {r_fl+1}: No clients active. Skip server agg."); continue
        
        aggregated_hm_terms_server = [] # Default to empty if no updates
        converged_this_round = False    # Default
        if not client_figs_updates_this_round:
            logger.warning(f"FL Round {r_fl+1}: No client updates. Global h^(m) will be empty.")
            converged_this_round = server.has_converged(str(aggregated_hm_terms_server))
        else:
            aggregated_hm_terms_server, converged_this_round = server.phase2_aggregate(client_figs_updates_this_round, r_fl + 1)
        
        server.broadcast_model(clients, is_residual_model=True)
        logger.info(f"FL R{r_fl+1} done. Aggregated h^(m) has {len(server.global_figs_model_terms)} terms. Converged: {converged_this_round}")
        
        if server_test_X.size > 0 and server_test_y.size > 0:
            intermediate_F_M = server.accumulated_Fm_global_terms
            if intermediate_F_M: # Only evaluate if there are terms
                df_server_test_X_for_eval = pd.DataFrame(server_test_X, columns=concept_feature_names)
                acc_im, rp_im, rc_im, rl_im, rf_im = evaluate_global_AutoCore_model(
                    intermediate_F_M, server_test_X, server_test_y, concept_feature_names, config['num_classes'], logger)
                logger.info(f"  FL R{r_fl+1} Server F_M Test Metrics: Acc={acc_im:.3f}, RuleP={rp_im:.3f}, RuleC={rc_im:.3f}, RuleL={rl_im:.2f}, RuleF={rf_im:.3f}")
                all_round_metrics.append({'round': r_fl + 1, 'F_M_acc': acc_im, 'F_M_rule_p': rp_im, 
                                          'rc_im': rc_im, 'F_M_rule_l': rl_im, 'F_M_rule_f': rf_im, 
                                          'h_m_terms': len(aggregated_hm_terms_server), 'F_M_terms': len(intermediate_F_M)})

        if converged_this_round:
            rounds_no_model_change +=1
            if rounds_no_model_change >= config["rule_structure_convergence_patience"]:
                logger.info(f"FL converged: h^(m) stable for {rounds_no_model_change} rounds. Stop."); break
        else: rounds_no_model_change = 0

    # --- 5. Final Evaluation ---
    logger.info(f"--- Final Evaluation after {final_fl_round_completed} FL rounds ---")
    final_F_M_terms = server.accumulated_Fm_global_terms
    logger.info(f"Evaluating server's F_M model with {len(final_F_M_terms)} total effective terms.")
    
    model_acc, rule_p, rule_c, rule_l, rule_f = 0.0,0.0,0.0,0.0,0.0
    if final_F_M_terms and server_test_X.size > 0 and server_test_y.size > 0:
        model_acc, rule_p, rule_c, rule_l, rule_f = evaluate_global_AutoCore_model(
            final_F_M_terms, server_test_X, server_test_y, concept_feature_names, config['num_classes'], logger
        )
    else: logger.warning("Final F_M empty or no server test data. Cannot evaluate.")

    logger.info(f"======== FINAL RESULTS for Run ID: {run_id} (ResNet Concepts) ========")
    logger.info(f"  Completed FL Rounds: {final_fl_round_completed}")
    logger.info(f"  Num Clients: {config['num_users']}, Sampling: {config['federated_sampling']}")
    logger.info(f"  Num Concept Features (ResNet18): {num_concept_features}")
    logger.info(f"  Global AutoCore FL Model Accuracy: {model_acc:.4f}")
    logger.info(f"  Global AutoCore FL Rule Metrics: P={rule_p:.3f}, C={rule_c:.3f}, L={rule_l:.2f}, F={rule_f:.3f}")
    logger.info(f"  Number of Terms in final F_M: {len(final_F_M_terms)}")
    
    if all_round_metrics:
        df_round_metrics = pd.DataFrame(all_round_metrics)
        round_metrics_path = os.path.join(base_results_dir, "round_metrics.csv")
        df_round_metrics.to_csv(round_metrics_path, index=False)
        logger.info(f"Round-wise metrics saved to {round_metrics_path}")

    logger.info(f"======== Run {run_id} Complete. Results in {base_results_dir} ========")

if __name__ == '__main__':
    main()