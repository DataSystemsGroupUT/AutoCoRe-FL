
import logging
import os
import re
import sys
import yaml
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import time
import operator
import torch # For torch.device
from scipy.stats import pointbiserialr # For correlation diagnostic

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Adjust if script is elsewhere
if project_root not in sys.path: sys.path.insert(0, project_root)

try:
    from AutoCore_FL.data.ade20k_parition import get_filtered_image_paths, stratified_partition, load_scene_categories
    from AutoCore_FL.segmentation.sam_loader import load_sam_model
    from AutoCore_FL.segmentation.segment_crops import generate_segments_and_masks, filter_zero_segment_images, load_cached_segments
    from AutoCore_FL.embedding.dino_loader import init_dino, init_target_model
    from AutoCore_FL.embedding.compute_embeddings import compute_final_embeddings
    # from federated_logic_xai_figs_svm.concepts.detector import train_concept_detector # Now called from client
    from AutoCore_FL.concepts.vectorizer import build_image_concept_vectors
    # from federated_logic_xai_figs_svm.clustering.federated_kmeans import FederatedKMeans # Now called from client
    from AutoCore_FL.federated.client import FederatedClient
    from AutoCore_FL.federated.server import FederatedServer
    from AutoCore_FL.federated.utils import setup_logging
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Critical Import Error in run_federated_train_boosting.py: {e}. Execution cannot continue.")
    sys.exit(1)

from sklearn.metrics import accuracy_score
from scipy.special import softmax

# --- Helper Functions ---
def load_labels_for_images(image_ids: list, scene_map: dict, scene_to_idx_map: dict) -> np.ndarray:
    labels = []
    processed_ids = set()
    for img_id in image_ids:
        if img_id in processed_ids:
            # This can happen if build_image_concept_vectors returns duplicate image_ids
            # (e.g. if an image_id was associated with multiple img_idx due to some upstream issue,
            #  though typically it should be one base_id per img_idx).
            # For safety, just skip if already processed.
            continue
        scene = scene_map.get(img_id)
        if scene is None:
            # logging.warning(f"Image ID {img_id} not found in scene_map during label loading.")
            continue
        idx = scene_to_idx_map.get(scene)
        if idx is not None:
            labels.append(idx)
            processed_ids.add(img_id)
        # else:
            # logging.warning(f"Scene {scene} for image ID {img_id} not in scene_to_idx_map.")
    return np.array(labels, dtype=np.int64)

class SAM2Filter(logging.Filter):
    def filter(self, record):
        sam2_patterns = ["For numpy array image, we assume", "Computing image embeddings", "Image embeddings computed"]
        # Check if any pattern is a substring of the record's message
        return not any(p in record.getMessage() for p in sam2_patterns)


def generate_config_and_get_path(run_specific_id_base):
    METHOD_NAME = "FIGS_Boosting_V3"
    num_clusters = 75  # Default, can be overridden by a loaded config
    num_clients = 4
    chosen_classes = ['street', 'bedroom', 'living_room', 'bathroom', 'kitchen', 'skyscraper', 'highway']

    seed = 42
    num_global_classes = len(chosen_classes)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ade20k_root = "/gpfs/helios/home/soliman/logic_explained_networks/data/ade20k/ADEChallengeData2016/"
    sam_cfg_path = "configs/sam2.1/sam2.1_hiera_t.yaml" # Relative to project root or ensure absolute
    sam_ckpt_path = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/sam2.1_hiera_tiny.pt"

    run_specific_id_suffix = "dino_only_svm_server_val_final_fix" 
    effective_run_id = f"{run_specific_id_base}_{run_specific_id_suffix}"

    experiment_base_path = os.path.join(script_dir, f"experiment_results/{METHOD_NAME.lower()}_run_{effective_run_id}")
    base_dir = os.path.join(experiment_base_path) 
    os.makedirs(base_dir, exist_ok=True)

    # Cache directory naming convention
    cache_base_name = f"ade20k_{METHOD_NAME}_{num_clusters}c_{num_clients}cl_{effective_run_id}"
    segment_cache_dir = os.path.join(script_dir, "cache", f"segments_{cache_base_name}") # Store cache in a subfolder
    embedding_cache_dir = os.path.join(script_dir, "cache", f"embeddings_{cache_base_name}")
    os.makedirs(segment_cache_dir, exist_ok=True)
    os.makedirs(embedding_cache_dir, exist_ok=True)

    log_dir_path = os.path.join(base_dir, "logs")
    os.makedirs(log_dir_path, exist_ok=True)

    config = {
        "ade20k_root": ade20k_root,
        "processed_dataset_root": "/gpfs/helios/home/soliman/logic_explained_networks/data/ade20k/ADEChallengeData2016/images/", # Might be same as ade20k_root for some structures
        "scene_cat_file": os.path.join(ade20k_root, "sceneCategories.txt"),
        "chosen_classes": chosen_classes, "num_clients": num_clients, "seed": seed, "server_split": 0.1,
        "sam_cfg": sam_cfg_path, "sam_ckpt": sam_ckpt_path,
        "segment_cache_dir": segment_cache_dir,
        "embedding_cache_dir": embedding_cache_dir,
        "use_segment_cache": True, "use_embedding_cache": True,

        "min_mask_pixels": 500,
        "dino_model": "facebook/dinov2-base",
        "embedding_type": "dino_only",
        "embedding_dim": 768, # DINOv2-Base [CLS] token is 768 dimensional for dino_only
        
        "num_clusters": num_clusters,
        "kmeans_rounds": 5,
        "min_samples_per_concept_cluster": 30,
        "detector_type": "lr", # 'lr' or 'svm'
        "detector_min_samples_per_class": 5, # For concept detector training
        "detector_cv_splits": 3, # For concept detector training
        "pca_n_components": 128, # For concept detector training
        "svm_C": 1.0, # For SVM concept detector
        "lr_max_iter": 10000, # For LR concept detector
        "min_detector_score": 0.68, 
        "vectorizer_min_activating_segments": 2, 
        "method_name": METHOD_NAME,
        "figs_params": {
            "max_rules": 40, 
            "max_trees": 3,
            "min_impurity_decrease": 0.0,
            "max_features": None
        },
        "learning_rate_gbm": 0.1,
        "max_global_figs_terms": 100,
        "min_clients_for_figs_term": 1,
        "server_rule_validation_min_precision": 0.3, # Min precision on server test data for rule to be kept by server
        "server_rule_min_coverage_count": 5, # Min server test samples a rule must cover

        "phase2_rounds": 10,
        "rule_structure_convergence_patience": 3,
        "boosting_factor": 0.1,
        "use_accuracy_weighting_server": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        "global_rules_path": os.path.join(base_dir, f"ade20k_FIGS_{num_clusters}c_{num_clients}cl_global_rules.pkl"),
        "metrics_log_path": os.path.join(base_dir, f"final_metrics_{METHOD_NAME}.csv"),
        "log_dir": log_dir_path,
        "run_id": effective_run_id, # Use the full, unique run ID
        "num_classes": num_global_classes,
        "run_concept_label_correlation_diagnostic": True, # Optional diagnostic
    }
    cfg_path = os.path.join(base_dir, f"config_{METHOD_NAME}.yaml")
    with open(cfg_path, "w") as f: yaml.dump(config, f, default_flow_style=False)
    print(f"Config for {METHOD_NAME} (Run ID: {effective_run_id}) saved to: {cfg_path}")
    return cfg_path, config


# --- Helper: Evaluate Global FIGS Model (using terms) ---
def evaluate_global_figs_model(
    global_figs_terms: list, X_eval_data_concepts: np.ndarray, y_eval_labels: np.ndarray,
    feature_names: list, num_total_classes: int, main_logger: logging.Logger
):
    main_logger.info(f"Evaluating final global FIGS model with {len(global_figs_terms)} terms on data of shape {X_eval_data_concepts.shape}...")
    if X_eval_data_concepts is None or X_eval_data_concepts.shape[0] == 0 or not global_figs_terms:
        main_logger.warning("No data or no global terms to evaluate. Returning zeros.")
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    if not feature_names or len(feature_names) != X_eval_data_concepts.shape[1]:
        main_logger.error(f"Feature names count ({len(feature_names)}) mismatch with X_eval_data_concepts columns ({X_eval_data_concepts.shape[1]}). Cannot evaluate rules accurately.")
        try:
            summed_scores_for_pred_fallback = np.zeros((X_eval_data_concepts.shape[0], num_total_classes))
            # This part is tricky if feature_names are bad, as evaluate_figs_rule_str_on_sample_helper relies on them.
            # For now, just try to get model accuracy.
            X_eval_data_dicts_pred = [dict(zip(feature_names, row)) if feature_names else {} for row in X_eval_data_concepts]
            for i, x_dict in enumerate(X_eval_data_dicts_pred):
                sample_total_value = np.zeros(num_total_classes)
                for term in global_figs_terms:
                    if evaluate_figs_rule_str_on_sample_helper(term['rule_str'], x_dict, main_logger, feature_names): # Uses helper
                        term_val = term.get('aggregated_value_array', [])
                        if isinstance(term_val, list) and len(term_val) == num_total_classes:
                            sample_total_value += np.array(term_val)
                summed_scores_for_pred_fallback[i] = sample_total_value
            y_pred_probs_fallback = softmax(summed_scores_for_pred_fallback, axis=1)
            y_pred_model_labels_fallback = np.argmax(y_pred_probs_fallback, axis=1)
            model_accuracy_fallback = accuracy_score(y_eval_labels, y_pred_model_labels_fallback)
            main_logger.info(f"Fallback Model Accuracy (rules not eval'd due to feat name issue): {model_accuracy_fallback:.4f}")
            return model_accuracy_fallback, 0.0, 0.0, 0.0, 0.0
        except Exception as e_fallback_acc:
            main_logger.error(f"Fallback accuracy calculation failed: {e_fallback_acc}")
            return 0.0, 0.0, 0.0, 0.0, 0.0

    def evaluate_figs_rule_str_on_sample_helper(rule_str_eval, x_sample_dict_eval, logger_inst_eval, feat_names_list_eval):
        if rule_str_eval == "True": return True
        for cond_str_eval in rule_str_eval.split(' & '):
            match_eval = re.match(r"`(.+?)`\s*([><]=?)\s*([0-9eE.+-]+)", cond_str_eval.strip())
            if not match_eval: logger_inst_eval.debug(f"Helper: No match for cond '{cond_str_eval}' in '{rule_str_eval}'"); return False
            feat_eval, op_eval, val_s_eval = match_eval.groups()
            if feat_eval not in x_sample_dict_eval:
                if not hasattr(evaluate_figs_rule_str_on_sample_helper, 'logged_missing_helper'):
                    evaluate_figs_rule_str_on_sample_helper.logged_missing_helper = set()
                if feat_eval not in evaluate_figs_rule_str_on_sample_helper.logged_missing_helper: # Log only once
                    logger_inst_eval.warning(f"Helper: Feature '{feat_eval}' in rule not in sample. Avail keys (sample): {list(x_sample_dict_eval.keys())[:5]}")
                    evaluate_figs_rule_str_on_sample_helper.logged_missing_helper.add(feat_eval)
                return False
            s_val_eval, cond_v_eval = x_sample_dict_eval[feat_eval], float(val_s_eval)
            op_fn_eval = {'<=':operator.le,'>':operator.gt,'<':operator.lt,'>=':operator.ge,'==':operator.eq}.get(op_eval)
            if not (op_fn_eval and op_fn_eval(s_val_eval, cond_v_eval)): return False
        return True

    summed_scores = np.zeros((X_eval_data_concepts.shape[0], num_total_classes))
    X_eval_data_dicts = [dict(zip(feature_names, row)) for row in X_eval_data_concepts]

    for i, x_dict_loop in enumerate(tqdm(X_eval_data_dicts, desc="Eval Global Model Samples", file=sys.stdout, leave=False)):
        sample_total_value = np.zeros(num_total_classes)
        for term in global_figs_terms:
            if evaluate_figs_rule_str_on_sample_helper(term['rule_str'], x_dict_loop, main_logger, feature_names):
                term_val_arr = term.get('aggregated_value_array', [])
                if isinstance(term_val_arr, list) and len(term_val_arr) == num_total_classes:
                    sample_total_value += np.array(term_val_arr)
        summed_scores[i] = sample_total_value
    
    y_pred_probs = softmax(summed_scores, axis=1)
    y_pred_model_labels = np.argmax(y_pred_probs, axis=1)
    model_accuracy = accuracy_score(y_eval_labels, y_pred_model_labels)
    main_logger.info(f"Global Model Accuracy (from terms): {model_accuracy:.4f}")

    rule_precisions, rule_coverages, rule_complexities, rule_fidelities = [], [], [], []
    for term in tqdm(global_figs_terms, desc="Eval Global Rules", file=sys.stdout, leave=False):
        rule_str = term['rule_str']
        value_array = np.array(term.get('aggregated_value_array', []))
        rule_intended_class = np.argmax(value_array) if value_array.size > 0 else -1
        rule_complexities.append(rule_str.count('&') + 1 if rule_str != "True" else 0)
        
        fires_mask = np.array([evaluate_figs_rule_str_on_sample_helper(rule_str, x_d, main_logger, feature_names) for x_d in X_eval_data_dicts])
        
        coverage = np.mean(fires_mask); rule_coverages.append(coverage)
        num_covered = np.sum(fires_mask)

        if num_covered > 0:
            prec = np.mean(y_eval_labels[fires_mask] == rule_intended_class) if rule_intended_class != -1 else 0.0
            fid = np.mean(y_pred_model_labels[fires_mask] == rule_intended_class) if rule_intended_class != -1 else 0.0
            rule_precisions.append(prec); rule_fidelities.append(fid)
        else:
            rule_precisions.append(0.0); rule_fidelities.append(0.0)

    mean_prec = np.mean(rule_precisions) if rule_precisions else 0.0
    mean_cov = np.mean(rule_coverages) if rule_coverages else 0.0
    mean_comp = np.mean(rule_complexities) if rule_complexities else 0.0
    mean_fid = np.mean(rule_fidelities) if rule_fidelities else 0.0
    main_logger.info(f"  Mean Rule P={mean_prec:.3f}, C={mean_cov:.3f}, L={mean_comp:.2f}, F={mean_fid:.3f}")
    return model_accuracy, mean_prec, mean_cov, mean_comp, mean_fid



def main():
    # Use a base run_id, the suffix will be added in generate_config_and_get_path
    base_run_id = "20250511_225305" 
    cfg_path, config = generate_config_and_get_path(base_run_id)

    setup_logging(log_dir=config['log_dir'], filename=f"main_log_figs_boosting_{config['run_id']}.log")
    main_logger = logging.getLogger(f"MainFIGSBoost_{config['run_id']}") # Include run_id in logger name
    root_logger = logging.getLogger()
    root_logger.addFilter(SAM2Filter()) # Add filter to root to affect all child loggers
    main_logger.info(f"======== Starting FIGS Boosting Run ID: {config['run_id']} ========")
    main_logger.info(f"Full Config Path: {cfg_path}")
    main_logger.info(f"Config being used: {config}")


    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    main_logger.info(f"Using device: {config['device']}")

    # --- Phase 0: Data Preparation ---
    main_logger.info("--- Phase 0: Data Loading & Partitioning ---")
    try:
        scene_map = load_scene_categories(config["scene_cat_file"])
        all_paths_tuples = get_filtered_image_paths(config["ade20k_root"], scene_map, config["chosen_classes"], subset="training")
    except FileNotFoundError as e:
        main_logger.error(f"Data loading failed: {e}. Ensure paths in config are correct. Execution cannot continue.")
        return
    if not all_paths_tuples:
        main_logger.error("No images found for the chosen classes. Execution cannot continue.")
        return
    
    main_logger.info(f"Total images for chosen classes: {len(all_paths_tuples)}")
    server_holdout_frac = config['server_split']
    stratify_labels_full = [scene_map[base_id] for base_id, _ in all_paths_tuples]

    try:
        server_holdout_paths_tuples, client_data_paths_tuples = train_test_split(
            all_paths_tuples, test_size=(1.0 - server_holdout_frac), 
            random_state=config['seed'], stratify=stratify_labels_full
        )
        stratify_labels_server = [scene_map[base_id] for base_id, _ in server_holdout_paths_tuples]
        if len(server_holdout_paths_tuples) < 2 or len(np.unique(stratify_labels_server)) < min(2, config.get('num_classes', 2)):
            main_logger.warning("Server holdout too small for stratification, using random split for val/test.")
            server_val_paths_tuples, server_test_paths_tuples = train_test_split(
                server_holdout_paths_tuples, test_size=0.5, random_state=config['seed']
            )
        else:
            server_val_paths_tuples, server_test_paths_tuples = train_test_split(
                server_holdout_paths_tuples, test_size=0.5, 
                random_state=config['seed'], stratify=stratify_labels_server
            )
    except ValueError as e_split:
        main_logger.warning(f"Stratification error during server/client split: {e_split}. Falling back to random splits.")
        server_holdout_paths_tuples, client_data_paths_tuples = train_test_split(
            all_paths_tuples, test_size=(1.0 - server_holdout_frac), random_state=config['seed']
        )
        server_val_paths_tuples, server_test_paths_tuples = train_test_split(
            server_holdout_paths_tuples, test_size=0.5, random_state=config['seed']
        )

    main_logger.info(f"Server Val paths: {len(server_val_paths_tuples)}, Server Test paths: {len(server_test_paths_tuples)}, Total Client data paths: {len(client_data_paths_tuples)}")
    if not client_data_paths_tuples:
        main_logger.error("No client data after split. Execution cannot continue.")
        return
        
    client_partitions_paths = stratified_partition(client_data_paths_tuples, scene_map, config["chosen_classes"], config["num_clients"], seed=config["seed"])
    sorted_chosen_classes = sorted(config["chosen_classes"])
    scene_to_global_idx = {s: i for i, s in enumerate(sorted_chosen_classes)}
    main_logger.info(f"Scene to Global Index Map (Size {config['num_classes']}): {scene_to_global_idx}")

    # --- PHASE 1: Federated K-Means ---
    main_logger.info(f"--- Starting Phase 1: Federated K-Means (Embedding type: {config['embedding_type']}) ---")
    clients = [FederatedClient(i, config, client_partitions_paths[i], scene_to_global_idx) for i in range(config["num_clients"])]
    server = FederatedServer(config)
    
    current_centroids = server.initialize_centroids(config["embedding_dim"], config["num_clusters"])
    client_kmeans_stats_final_round = [] # To store stats from the last successful round for concept filtering

    for r_km in range(config["kmeans_rounds"]):
        main_logger.info(f"KMeans Round {r_km + 1}/{config['kmeans_rounds']}")
        client_kmeans_stats_current_round = []
        active_clients_this_round = 0
        for client in tqdm(clients, desc=f"KMeans R{r_km+1}-Clients"):
            sums, counts = client.run_local_pipeline(global_centroids=current_centroids, current_config=config) # Pass full config
            if sums is not None and counts is not None and sums.size > 0 and counts.size > 0:
                client_kmeans_stats_current_round.append((sums, counts))
                active_clients_this_round += 1
        
        if active_clients_this_round == 0:
            main_logger.error(f"No active clients in KMeans Round {r_km+1}. Stopping K-Means phase.")
            # Decide how to proceed: exit or use previous centroids if any
            if not client_kmeans_stats_final_round: # No previous good round
                main_logger.error("KMeans failed: No client activity in any round.")
                return
            break # Exit K-Means loop, use centroids from client_kmeans_stats_final_round

        current_centroids, live_mask = server.aggregate_kmeans(client_kmeans_stats_current_round)
        client_kmeans_stats_final_round = client_kmeans_stats_current_round # Update with latest stats
        main_logger.info(f"KMeans R{r_km+1}: Aggregated Centroids shape: {current_centroids.shape if current_centroids is not None else 'None'}, Live concepts (mask sum): {live_mask.sum() if live_mask is not None else 'N/A'}")
        if current_centroids is None or live_mask is None or live_mask.sum() == 0:
            main_logger.error(f"KMeans aggregation failed in Round {r_km+1} or resulted in no live centroids. Stopping K-Means phase.")
            if not client_kmeans_stats_final_round:
                 main_logger.error("KMeans failed: No client activity led to valid centroids.")
                 return
            break 

    if not client_kmeans_stats_final_round:
        main_logger.error("KMeans phase completed with no valid stats from any client round. Cannot proceed.")
        return

    # Use final_counts_agg from the last successful round for concept pruning
    final_counts_agg_list = [c for s,c in client_kmeans_stats_final_round if c is not None and c.ndim > 0 and c.size > 0 and c.shape[0] == config["num_clusters"]]
    if not final_counts_agg_list:
        main_logger.error("No valid final counts from KMeans to determine initial shared concepts.")
        return
    
    final_counts_agg = np.sum(np.stack(final_counts_agg_list), axis=0)
    keep_mask_concepts = final_counts_agg >= config['min_samples_per_concept_cluster']
    initial_shared_concept_indices = np.where(keep_mask_concepts)[0].tolist()
    if not initial_shared_concept_indices:
        main_logger.error(f"No concepts met min_samples_per_concept_cluster ({config['min_samples_per_concept_cluster']}). Cannot proceed.")
        return
    main_logger.info(f"Phase 1 Done. Found {len(initial_shared_concept_indices)} potential concepts post-filtering: {sorted(initial_shared_concept_indices)[:20]}...")


    # --- PHASE 2 Preparation: Detector Sync ---
    main_logger.info(f"--- Starting Phase 2 Prep: Detector Sync (Detector type: {config['detector_type']}) ---")
    for client in tqdm(clients, desc="Local Detectors Training"):
        client.train_concept_detectors(config) # Pass full config
    
    all_detector_updates = [client.get_detector_update(initial_shared_concept_indices) for client in clients]
    final_phase2_concept_indices_ordered, canonical_detector_params_for_broadcast = server.aggregate_detectors(
        all_detector_updates, initial_shared_concept_indices
    )
    # log how many detectors were aggregated
    main_logger.info(f"Phase 2 Detector Sync: {len(canonical_detector_params_for_broadcast)} detectors aggregated for {len(final_phase2_concept_indices_ordered)} concepts.")
    # log the keys and values of the first 5 detectors
    #main_logger.info(f"Phase 2 Detector Sync: First 5 detectors: {list(canonical_detector_params_for_broadcast.items())[:5]}")
    num_figs_features = len(final_phase2_concept_indices_ordered)
    if num_figs_features == 0:
        main_logger.error("No concepts survived detector aggregation for FIGS. Cannot proceed.")
        return
    
    figs_feature_names = [f"concept_{i}" for i in range(num_figs_features)] # Dense feature names
    server.feature_names_for_figs = figs_feature_names # Server needs this for eval
    final_original_to_dense_map = {orig_idx: dense_idx for dense_idx, orig_idx in enumerate(final_phase2_concept_indices_ordered)}
    
    main_logger.info(f"Phase 2 Prep Done. Using {num_figs_features} final concepts for FIGS: {figs_feature_names[:5]}...")

    # Initialize client's global model accumulator (F_m)
    for client in clients:
        client.state['accumulated_global_model_Fm_terms'] = [] # Start with F_0 = 0 (no terms)
        client.state['learning_rate_gbm'] = config.get('learning_rate_gbm', 0.05)

    # MODIFIED: Server broadcasts indices, map, AND detector parameters
    server.phase2_prep_broadcast(
        clients,
        final_phase2_concept_indices_ordered,
        final_original_to_dense_map,
        canonical_detector_params_for_broadcast 
    )
    
    # for client in clients:
    #     client.receive_final_concepts(final_phase2_concept_indices_ordered, final_original_to_dense_map)


    # --- Server Holdout Data Preparation (for final eval AND server-side rule validation) ---
    main_logger.info("--- Preparing Server Holdout Test Concept Data ---")
    torch_device = torch.device(config['device'])
    server_sam, server_mask_gen = load_sam_model(config['sam_cfg'], config['sam_ckpt'], torch_device)
    server_dino_proc, server_dino_model = init_dino(config['dino_model'], torch_device)
    server_target_model = init_target_model(torch_device) if config['embedding_type'] == 'combined' else None # Only if needed

    X_test_concepts, y_test_labels = np.empty((0, num_figs_features)), np.empty((0,))
    server_test_feature_names = figs_feature_names # Features for server test data are the final FIGS features

    if len(server_test_paths_tuples) > 0:
        try:
            # Use a unique client_id for server's data processing to avoid cache collisions
            server_data_client_id = f"server_test_data_{config['run_id']}"
            
            # Check for cached segments for server test data
            try:
                main_logger.info(f"Attempting to load cached segments for server test data ({server_data_client_id})...")
                test_segment_infos, test_images, test_masks, test_all_segments = load_cached_segments(config, client_id=server_data_client_id)
                main_logger.info(f"Loaded cached segments for server test data: {len(test_segment_infos)} segment infos.")
            except FileNotFoundError:
                main_logger.info(f"No cached segments for server test data ({server_data_client_id}). Generating...")
                test_segment_infos, test_images, test_masks, test_all_segments = generate_segments_and_masks(
                    server_test_paths_tuples, server_mask_gen, config, client_id=server_data_client_id
                )
            
            test_f_imgs, test_f_masks, _, test_f_seg_infos = filter_zero_segment_images(
                test_images, test_masks, test_all_segments, test_segment_infos
            )

            if len(test_f_seg_infos) > 0:
                test_final_embeddings = compute_final_embeddings(
                    test_f_seg_infos, test_f_imgs, test_f_masks, 
                    server_dino_proc, server_dino_model, server_target_model, 
                    torch_device, config, client_id=server_data_client_id # Use unique ID for server embedding cache
                )
                if test_final_embeddings is not None and test_final_embeddings.shape[0] > 0:
                    # Use canonical detectors (already aggregated by server, mapped to dense indices)
                    server_detectors_for_test_vecs_dense_keys = {
                        final_original_to_dense_map[orig_idx]: model_obj # Use model_obj
                        for orig_idx, model_obj in server.canonical_detector_model_objects.items() # Use model_objects here
                        if orig_idx in final_original_to_dense_map
                    }
                    server_thresholds_for_test_vecs_dense_keys = {
                        final_original_to_dense_map[orig_idx]: thresh
                        for orig_idx, thresh in server.canonical_thresholds.items() # Use thresholds here
                        if orig_idx in final_original_to_dense_map
                    }

                    if not server_thresholds_for_test_vecs_dense_keys:
                        main_logger.warning("No canonical detectors mapped for server test vector generation.")
                    else:
                        current_X_test_concepts, test_kept_image_ids = build_image_concept_vectors(
                            filtered_segment_infos=test_f_seg_infos, 
                            # Pass the model objects to build_image_concept_vectors for server's own use
                            # build_image_concept_vectors needs to be flexible or we need a separate function
                            # For simplicity, assuming build_image_concept_vectors can take model objects
                            linear_models=server_detectors_for_test_vecs_dense_keys, 
                            optimal_thresholds=server_thresholds_for_test_vecs_dense_keys, 
                            final_embeddings=test_final_embeddings, 
                            target_num_features=num_figs_features,
                            config=config
                        )
                        if current_X_test_concepts is not None and current_X_test_concepts.shape[0] > 0:
                            current_y_test_labels = load_labels_for_images(test_kept_image_ids, scene_map, scene_to_global_idx)
                            if current_X_test_concepts.shape[0] == current_y_test_labels.shape[0]:
                                X_test_concepts, y_test_labels = current_X_test_concepts, current_y_test_labels
                            else:
                                main_logger.error(f"Shape mismatch for server test data: Concepts {current_X_test_concepts.shape[0]}, Labels {current_y_test_labels.shape[0]}.")
        except Exception as e:
            main_logger.exception(f"Error during server holdout test data preparation: {e}")
    
    main_logger.info(f"Server Test concepts shape: {X_test_concepts.shape}, Labels shape: {y_test_labels.shape}")

    # Set server validation data for rule filtering during aggregation
    if X_test_concepts is not None and X_test_concepts.shape[0] > 0:
        server.server_X_test_concepts_for_validation = X_test_concepts
        server.server_y_test_labels_for_validation = y_test_labels
        # Feature names for server validation data are the final FIGS feature names
        server.server_feature_names_for_validation = server.feature_names_for_figs 
        main_logger.info(f"Server validation data configured for rule filtering: X_shape={X_test_concepts.shape}")
    else:
        main_logger.warning("No server test data available for server-side rule validation. Rule validation will be skipped by server.")


    # --- DIAGNOSTIC: Concept-Label Correlation ---
    if config.get("run_concept_label_correlation_diagnostic", False):
        main_logger.info("--- DIAGNOSTIC: Running Concept-Label Correlation (Client-wise, then Averaged) ---")
        client_avg_correlations = []
        for client_idx, client in enumerate(clients):
            client_vecs, client_ids = client.get_current_concept_vectors_and_ids() # Needs method in client
            if client_vecs is not None and client_vecs.shape[0] > 0:
                client_labels_for_vecs = load_labels_for_images(client_ids, scene_map, scene_to_global_idx)
                if client_labels_for_vecs.shape[0] == client_vecs.shape[0] and client_vecs.shape[0] > 1:
                    client_corrs = []
                    num_client_features = client_vecs.shape[1]
                    # Ensure client feature names align with server's view if possible, or use dense indices
                    client_feature_names_for_corr = client.state.get('feature_names_for_figs', [f"c_{i}" for i in range(num_client_features)])

                    for i in range(num_client_features):
                        concept_feature_col = client_vecs[:, i]
                        if np.std(concept_feature_col) > 1e-6 and len(np.unique(client_labels_for_vecs)) > 1:
                            try:
                                corr, _ = pointbiserialr(concept_feature_col, client_labels_for_vecs)
                                if not np.isnan(corr): client_corrs.append(corr)
                            except ValueError: # Can happen if one array is constant for pointbiserialr
                                pass # Skip if correlation cannot be computed
                    if client_corrs:
                        avg_abs_client_corr = np.mean(np.abs(client_corrs))
                        main_logger.info(f"Client {client_idx}: Avg Abs Concept-Label Corr: {avg_abs_client_corr:.4f} (over {len(client_corrs)} concepts with variance)")
                        client_avg_correlations.append(avg_abs_client_corr)
        if client_avg_correlations:
            main_logger.info(f"Overall Mean of Client Avg Abs Concept-Label Correlations: {np.mean(client_avg_correlations):.4f}")
    # --- PHASE 2: Federated FIGS Training Loop (GBM-Style) ---
    main_logger.info(f"--- Starting Phase 2: Federated FIGS Training (GBM-Style) ---")
    main_logger.info(f"GBM Learning Rate: {config.get('learning_rate_gbm', 0.05)}")
    main_logger.info(f"Rounds before residual fitting begins: {config.get('rounds_before_gbm_residual_fitting', 0)}")

    rounds_no_model_change = 0
    # Initialize client's accumulated global model F_m (list of terms)
    for client in clients:
        client.state['accumulated_global_model_Fm_terms'] = [] # F_0 is empty (predicts zeros)
    
    # Server also needs to accumulate its version of F_M for final evaluation
    server.accumulated_Fm_global_terms = [] # Initialize server's F_0

    final_round_completed = 0
    for r_ph2 in range(config["phase2_rounds"]):
        final_round_completed = r_ph2 + 1
        main_logger.info(f"--- FIGS GBM Round {r_ph2 + 1}/{config['phase2_rounds']} ---")
        client_figs_residual_updates = [] 

        for client_idx, client in enumerate(tqdm(clients, desc=f"FIGS R{r_ph2+1} Clients")):
            vecs, ids = client.build_concept_vectors(config) 
            if vecs is None or vecs.shape[0] == 0: main_logger.warning(f"Client {client_idx}: No vecs."); continue
            client_labels = load_labels_for_images(ids, scene_map, scene_to_global_idx)
            if client_labels.shape[0] != vecs.shape[0]: main_logger.warning(f"Client {client_idx}: Label/Vec mismatch."); continue

            train_config_client = config.copy()
            train_config_client['current_round'] = r_ph2 
            train_config_client['is_gbm_residual_round'] = True # This flag signals GBM logic

            # Client trains a model h_k^(m) to predict residuals
            # client.train_figs now uses client.state['accumulated_global_model_Fm_terms'] to get F_{m-1}
            # and targets residuals.
            _, residual_terms_hk = client.train_figs(client_labels, train_config_client) 
            
            if residual_terms_hk is not None: # residual_terms_hk are terms of h_k^(m)
                update = {'figs_terms': residual_terms_hk, 
                          'support': vecs.shape[0], 
                          'mse': client.state['local_model_mse'], # Accuracy of h_k^(m) on residuals (less interpretable)
                          'client_id': client.client_id}
                client_figs_residual_updates.append(update)
        
        if not client_figs_residual_updates:
            main_logger.warning(f"FIGS GBM Round {r_ph2+1}: No client updates for h^(m). Skipping server aggregation for h^(m).")
            # If h^(m) is effectively zero, F_m = F_{m-1}. Convergence check relies on h^(m) stability.
            aggregated_residual_model_hm_terms = [] # No update this round
            converged = server.has_converged(str(aggregated_residual_model_hm_terms)) # Check if h^(m) is empty like last time
        else:
            aggregated_residual_model_hm_terms, converged = server.phase2_aggregate(
                client_figs_residual_updates, r_ph2 + 1
            ) 
        
        # Server broadcasts h^(m) terms to clients
        # Server.global_figs_model_terms now holds terms of h^(m)
        server.broadcast_model(clients, is_residual_model=True) # Clients update their F_m
        
        main_logger.info(f"FIGS R{r_ph2+1} completed. Global residual model h^(m) has {len(server.global_figs_model_terms)} terms. Aggregated h^(m) Converged: {converged}")
        
        if converged: # Convergence of h^(m)
            rounds_no_model_change +=1
            main_logger.info(f"Global residual model h^(m) structure stable for {rounds_no_model_change} round(s).")
            if rounds_no_model_change >= config.get("rule_structure_convergence_patience", 3):
                main_logger.info("Federated FIGS-GBM converged: residual model h^(m) is stable. Stopping training.")
                break
        else:
            rounds_no_model_change = 0
    
        # Use the server's accumulated global model F_M for evaluation
        final_model_terms_to_evaluate = server.accumulated_Fm_global_terms
        model_accuracy, mean_rule_precision, mean_rule_coverage, mean_rule_complexity, mean_rule_fidelity = 0.0,0.0,0.0,0.0,0.0
        if final_model_terms_to_evaluate and X_test_concepts is not None and X_test_concepts.shape[0] > 0 : # Check if terms exist
            model_accuracy, mean_rule_precision, mean_rule_coverage, mean_rule_complexity, mean_rule_fidelity = evaluate_global_figs_model(
                global_figs_terms=final_model_terms_to_evaluate, 
                X_eval_data_concepts=X_test_concepts, 
                y_eval_labels=y_test_labels,
                feature_names=server.feature_names_for_figs, 
                num_total_classes=config['num_classes'], 
                main_logger=main_logger
            )
    # --- Final Evaluation ---
    main_logger.info(f"--- Final Evaluation Phase (FIGS-GBM) after {final_round_completed} rounds ---")
    
    # Use the server's accumulated global model F_M for evaluation
    final_model_terms_to_evaluate = server.accumulated_Fm_global_terms
    main_logger.info(f"Evaluating with Server's accumulated global model F_M ({len(final_model_terms_to_evaluate)} terms).")
    
    if not final_model_terms_to_evaluate and server.global_figs_model_terms: # Fallback if F_M is empty but last h_m exists
        main_logger.warning("Accumulated F_M is empty. Evaluating only with the last h_M (this is not the full model).")
        final_model_terms_to_evaluate = server.global_figs_model_terms


    model_accuracy, mean_rule_precision, mean_rule_coverage, mean_rule_complexity, mean_rule_fidelity = 0.0,0.0,0.0,0.0,0.0
    if final_model_terms_to_evaluate and X_test_concepts is not None and X_test_concepts.shape[0] > 0 : # Check if terms exist
        model_accuracy, mean_rule_precision, mean_rule_coverage, mean_rule_complexity, mean_rule_fidelity = evaluate_global_figs_model(
            global_figs_terms=final_model_terms_to_evaluate, 
            X_eval_data_concepts=X_test_concepts, 
            y_eval_labels=y_test_labels,
            feature_names=server.feature_names_for_figs, 
            num_total_classes=config['num_classes'], 
            main_logger=main_logger
        )

    else:
        main_logger.warning("Skipping final evaluation: Server test concept data (X_test_concepts) or labels (y_test_labels) are missing, empty, or mismatched.")

    num_final_global_terms = len(server.global_figs_model_terms)
    main_logger.info("--- Final FIGS Metrics ---")
    main_logger.info(f"Run ID: {config['run_id']}")
    main_logger.info(f"Completed Rounds: {final_round_completed}")
    main_logger.info(f"Number of Clients: {config['num_clients']}")
    main_logger.info(f"Number of Shared Concepts (FIGS features): {num_figs_features}")
    main_logger.info(f"Global FIGS Model Test Accuracy: {model_accuracy:.4f}")
    main_logger.info(f"Global Mean Rule Precision (heuristic): {mean_rule_precision:.4f}")
    main_logger.info(f"Global Mean Rule Coverage: {mean_rule_coverage:.4f}")
    main_logger.info(f"Global Mean Rule Complexity: {mean_rule_complexity:.4f}")
    main_logger.info(f"Global Mean Rule Fidelity (heuristic): {mean_rule_fidelity:.4f}")
    main_logger.info(f"Number of Final Global Terms: {num_final_global_terms}")
    
    print("\n--- Final FIGS Metrics ---")
    print(f"Run ID: {config['run_id']}")
    print(f"Global FIGS Model Test Acc: {model_accuracy:.4f}")
    print(f"Mean Rule Precision: {mean_rule_precision:.4f}, Coverage: {mean_rule_coverage:.4f}, Complexity: {mean_rule_complexity:.4f}, Fidelity: {mean_rule_fidelity:.4f}")
    print(f"Number of Final Global Terms: {num_final_global_terms}")

    results_summary = pd.DataFrame({
        "run_id": [config['run_id']],
        "Model Type": [f"{config['method_name']}_R{final_round_completed}"],
        "Embedding Type": [config['embedding_type']],
        "Detector Type": [config['detector_type']],
        "Min Activation Segments": [config['vectorizer_min_activating_segments']],
        "Max Rules Config": [config['figs_params']['max_rules']],
        "number of shared concepts": [num_figs_features],
        "Global Model Accuracy": [model_accuracy],
        "Global Rule Precision": [mean_rule_precision],
        "Global Rule Coverage": [mean_rule_coverage],
        "Global Rule Complexity": [mean_rule_complexity],
        "Global Rule Fidelity": [mean_rule_fidelity],
        "number final Global terms": [num_final_global_terms],
        "number of clusters": [config['num_clusters']],
        "number of Clients": [config['num_clients']],
        "number of kmeans rounds": [config['kmeans_rounds']],
        "number of boosting rounds": [final_round_completed],
        "number of classes": [config['num_classes']],
        "Boosting Factor": [config['boosting_factor']],
        "PCA Components": [config['pca_n_components']],
        "Use Accuracy for Boosting": [config['use_accuracy_weighting_server']],
        "Cluster Concept Minimum Samples": [config['min_samples_per_concept_cluster']],
        "Minimum Samples per Class": [config['detector_min_samples_per_class']],
        "Server Rule Minimum Percision": [config['server_rule_validation_min_precision']],
        "Server Rule Minimum Coverage": [config['server_rule_min_coverage_count']],
    })
    
    try:
        results_summary.to_csv(config['metrics_log_path'], index=False)
        main_logger.info(f"Saved metrics to {config['metrics_log_path']}")
    except Exception as e_csv:
        main_logger.error(f"Failed to save metrics CSV: {e_csv}")

    if server.global_figs_model_terms:
        try:
            with open(config["global_rules_path"], "wb") as f:
                pickle.dump(server.global_figs_model_terms, f)
            main_logger.info(f"Saved final global FIGS terms to {config['global_rules_path']}")
        except Exception as e_pkl:
            main_logger.error(f"Failed to save global rules pickle: {e_pkl}")
    
    main_logger.info(f"======== FIGS Run ID: {config['run_id']} Complete ========")

if __name__ == "__main__":
    # Optional: Add command line argument parsing for base_run_id or config file
    main()