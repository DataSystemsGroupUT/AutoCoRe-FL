
import logging
import os
import sys
import yaml
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from tqdm import tqdm
import pandas as pd
import torch
from collections import defaultdict
from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.ensemble import RandomForestClassifier as SklearnRF

from AutoCore_FL.data.ade20k_parition import load_scene_categories
from AutoCore_FL.embedding.dino_loader import init_dino, init_target_model
from AutoCore_FL.embedding.compute_embeddings import compute_final_embeddings
from AutoCore_FL.concepts.detector import train_concept_detector
from AutoCore_FL.federated.client import PatchedFIGSClassifier
from AutoCore_FL.federated.utils import setup_logging, calculate_metrics, build_centralized_concept_vectors_maxpool, add_seg_crop_bgr_to_split_infos
from AutoCore_FL.scripts.visualize_utils import *

dataset_name = "ade20k"
METHOD_NAME = f"AutoCore_Cent_AutoConcepts_{dataset_name}"
SEED = 42
CHOSEN_CLASSES = ['street', 'bedroom', 'living_room', 'bathroom', 'kitchen', 
        'skyscraper', 'highway', 'conference_room', 'mountain_snowy', 'office',
        'corridor', 'airport_terminal', 'attic', 'mountain', 'park', 'coast', 
        'alley','beach', 'childs_room', 'art_gallery','castle', 'dorm_room', 
        'nursery', 'lobby', 'reception', 'bar', 'house', 'bridge', 'classroom']

def evaluate_centralized_AutoCore_model(
    figs_model_instance,
    X_eval_data_concepts: np.ndarray,
    y_eval_data_labels: np.ndarray,
    feature_names_for_figs: List[str],
    num_total_classes: int,
    main_logger_passed: logging.Logger,
) -> Tuple[float, float, float, float, float]:
    """
    Evaluate a *centralised* AutoCore FIGS model on an **evaluation set only**
    and compute the rule–level metrics used in the AutoCoRe-FL paper.

    Parameters
    ----------
    figs_model_instance : imodels.FIGSClassifier
        A trained FIGS model (multi-class ready).
    X_eval_data_concepts : ndarray, shape (n_samples, n_features)
        Concept-vector inputs for evaluation.
    y_eval_data_labels : ndarray, shape (n_samples,)
        Ground-truth class labels (1-D, *not* one-hot).
    feature_names_for_figs : list of str
        Column names matching `X_eval_data_concepts`.
    num_total_classes : int
        Total number of distinct classes in the task.
    main_logger_passed : logging.Logger
        Logger for progress / debug messages.

    Returns
    -------
    accuracy           : float
    mean_rule_precision: float
    mean_rule_coverage : float
    mean_rule_complexity: float
    mean_rule_fidelity : float
    """
    log = main_logger_passed
    log.info("Evaluating centralised AutoCore model …")

    # -------- sanity guards -------------------------------------------------
    if X_eval_data_concepts is None or X_eval_data_concepts.shape[0] == 0:
        log.warning("No evaluation data supplied – returning zeros.")
        return (0.0,) * 5
    if not hasattr(figs_model_instance, "trees_") or len(figs_model_instance.trees_) == 0:
        log.warning("AutoCore model contains no trees – returning zeros.")
        return (0.0,) * 5

    # -------- 1) plain predictive accuracy ----------------------------------
    y_pred_labels = figs_model_instance.predict(X_eval_data_concepts)
    accuracy = accuracy_score(y_eval_data_labels, y_pred_labels)
    log.info(f"AutoCore accuracy on eval set: {accuracy:.4f}")

    # -------- 2) build rule coverage on *eval* set --------------------------
    X_df = pd.DataFrame(X_eval_data_concepts, columns=feature_names_for_figs)
    n_samples = X_df.shape[0]

    # Each unique rule string -> {indices, summed_contrib_vectors}
    rule_hits: Dict[str, List[int]] = defaultdict(list)
    rule_sum_contrib: Dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(num_total_classes, dtype=float)
    )
    # Helper to traverse one tree for a single sample
    def _path_conditions_and_leaf(node, x_row) -> Tuple[List[str], np.ndarray]:
        conds: List[str] = []
        while node.left is not None and node.right is not None:
            feat_idx = node.feature
            feat_name = feature_names_for_figs[feat_idx]
            thr = node.threshold
            if x_row[feat_idx] <= thr:
                conds.append(f"`{feat_name}` <= {thr:.6f}")
                node = node.left
            else:
                conds.append(f"`{feat_name}` > {thr:.6f}")
                node = node.right
        # node.value shape (1, C)
        return conds, node.value.flatten()

    for i_sample, x_row in enumerate(X_eval_data_concepts):
        path_conditions = set()
        contrib_vec = np.zeros(num_total_classes, dtype=float)

        for tree_root in figs_model_instance.trees_:
            conds, leaf_val = _path_conditions_and_leaf(tree_root, x_row)
            path_conditions.update(conds)
            contrib_vec += leaf_val

        rule_str = (
            " & ".join(sorted(path_conditions)) if path_conditions else "True"
        )
        rule_hits[rule_str].append(i_sample)
        rule_sum_contrib[rule_str] += contrib_vec

    # -------- 3) compute per-rule metrics -----------------------------------
    rule_precisions, rule_coverages, rule_complexities, rule_fidelities = [], [], [], []

    for rule_str, indices in rule_hits.items():
        coverage = len(indices) / n_samples
        rule_coverages.append(coverage)

        # Decide which class this rule *intends* to predict:
        intended_class = int(np.argmax(rule_sum_contrib[rule_str]))
        if intended_class < 0 or intended_class >= num_total_classes:
            continue  # skip un-interpretable rule

        # complexity: number of atomic conditions (≥1 even for root rule)
        complexity = max(1, rule_str.count("&") + 1)
        rule_complexities.append(complexity)

        # Precision: how often ground-truth matches intended class
        precision = np.mean(
            y_eval_data_labels[indices] == intended_class
        )
        rule_precisions.append(float(precision))

        # Fidelity: how often FIGS *prediction* matches intended class
        fidelity = np.mean(y_pred_labels[indices] == intended_class)
        rule_fidelities.append(float(fidelity))

    if not rule_precisions:  # should never happen, but guard anyway
        log.warning("No rules fired on evaluation set – rule metrics = 0.")
        return accuracy, 0.0, 0.0, 0.0, 0.0

    mean_prec = float(np.mean(rule_precisions))
    mean_cov = float(np.mean(rule_coverages))
    mean_comp = float(np.mean(rule_complexities))
    mean_fid = float(np.mean(rule_fidelities))

    log.info(
        f"Rule metrics – precision {mean_prec:.4f}, "
        f"coverage {mean_cov:.4f}, complexity {mean_comp:.2f}, "
        f"fidelity {mean_fid:.4f}"
    )
    return accuracy, mean_prec, mean_cov, mean_comp, mean_fid

# --- Config Generation  ---
def generate_config_AutoCore_auto(run_id_base=f"AutoCore_Cent_AutoConcepts_{dataset_name}"): 
    """
    Generates a configuration dictionary for the AutoCore centralized method with automatic concept detection.
    This config is tailored for the ADE20K dataset and includes paths, parameters, and settings
    for the centralized training and evaluation of the AutoCore model.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ade20k_root_path = "/gpfs/helios/home/soliman/logic_explained_networks/data/ade20k/ADEChallengeData2016/" 
    npy_base_path = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/" 

    effective_run_id = f"{run_id_base}"
    base_dir = os.path.join(script_dir, f"experiment_results_centralized/{METHOD_NAME.lower()}_run_{effective_run_id}")
    os.makedirs(base_dir, exist_ok=True)
    log_dir_path = os.path.join(base_dir, "logs")
    os.makedirs(log_dir_path, exist_ok=True)
    segment_cache_dir = os.path.join(script_dir, "cache_centralized_reused", f"segments_{effective_run_id}")
    embedding_cache_dir = os.path.join(script_dir, "cache_centralized_reused", f"embeddings_{effective_run_id}")
    os.makedirs(segment_cache_dir, exist_ok=True); os.makedirs(embedding_cache_dir, exist_ok=True)

    central_cache_dir_for_run = os.path.join(script_dir, "cache_centralized_reused", f"run_{effective_run_id}")
    os.makedirs(central_cache_dir_for_run, exist_ok=True)
    config = {
        "ade20k_root": ade20k_root_path, 
        "scene_cat_file": os.path.join(ade20k_root_path, "sceneCategories.txt"),
        "chosen_classes": CHOSEN_CLASSES,
        "num_classes": len(CHOSEN_CLASSES),
        "seed": SEED, "test_split_ratio": 0.2,
        "npy_base_path": npy_base_path, # Store path to npy files
        "dino_model": "facebook/dinov2-base", "embedding_type": "dino_only", "embedding_dim": 768,
        "num_clusters": 100, "min_samples_per_concept_cluster": 50,
        "detector_type": "lr", "detector_min_samples_per_class": 20, "detector_cv_splits": 3,
        "pca_n_components": 256, "lr_max_iter": 10000, "min_detector_score": 0.95,
        "vectorizer_strategy": "max_pool", # Key for new vectorization
        "figs_params": {"max_rules": 30, "min_impurity_decrease": 0.0, "max_trees": None, 'max_features':None}, 
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "metrics_log_path": os.path.join(base_dir, f"final_metrics_{METHOD_NAME}.csv"),
        "log_dir": log_dir_path, "run_id": effective_run_id,
        "final_model_path": os.path.join(base_dir, "final_centralized_figs_model.pkl"),
        "concept_definitions_path": os.path.join(base_dir, "concept_definitions.pkl"),
        "method_name": METHOD_NAME,
        "use_segment_cache": True, "segment_cache_dir": segment_cache_dir, 
        "use_embedding_cache": True, "embedding_cache_dir": embedding_cache_dir, 
        "central_run_cache_dir": central_cache_dir_for_run, 
        "use_seg_crops_cache": True,      # For add_seg_crop_bgr_to_split_infos
        "use_kmeans_cache": True,         # For K-Means clustering results
        "use_detectors_cache": True,      # For trained concept detectors
        "use_train_vectors_cache": True,  # For training concept vectors
        "use_test_vectors_cache": True,   # For test concept vectors
        "plot_dpi": 100, # Lower DPI for faster save during debug
        # Add min_mask_pixels here if compute_final_embeddings or add_seg_crop_bgr uses it for some fallback
        "min_mask_pixels": 100, # Fallback if needed, but ideally pre-gen segs are already filtered.
    }
    # Save config for reproducibility (optional but good)
    cfg_path = os.path.join(base_dir, f"config_{METHOD_NAME}.yaml")
    with open(cfg_path, "w") as f: yaml.dump(config, f, default_flow_style=False)
    print(f"Centralized Config for {METHOD_NAME} (Run ID: {effective_run_id}) saved to: {cfg_path}")
    return config



def main_AutoCore_centralized_auto():
    config = generate_config_AutoCore_auto()
    
    if not os.path.exists(config["npy_base_path"]):
        print(f"ERROR: NPY base path does not exist: {config['npy_base_path']}")
        return

    path_all_masks = os.path.join(config["npy_base_path"], "logic_ade20k_all_masksxx.npy")
    path_all_images = os.path.join(config["npy_base_path"], "logic_ade20k_all_imagesxx.npy")
    # path_all_segments_mapping = os.path.join(config["npy_base_path"], "logic_ade20k_all_segmentsxx.npy")
    path_segment_infos = os.path.join(config["npy_base_path"], "logic_ade20k_segment_infosxx.npy")


    setup_logging(log_dir=config['log_dir'],run_id = config['run_id'])
    main_logger = logging.getLogger(f"MainReusedSeg_{config['run_id']}")
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda' and torch.cuda.is_available(): torch.cuda.manual_seed_all(config['seed'])
    main_logger.info(f"Using device: {config['device']}")
    torch_device = torch.device(config['device'])

    main_logger.info(f"======== Starting Centralized FIGS with Reused Segments - Run ID: {config['run_id']} ========")
    main_logger.info(f"Full Config: {config}")

    # --- 1. Load Pre-generated Segmentation Data ---
    main_logger.info("--- Phase 1: Loading Pre-generated Segmentation Data ---")
    # These are loaded for the CHOSEN_CLASSES active during .npy generation.
    full_dataset_all_masks = np.load(path_all_masks, allow_pickle=True)
    full_dataset_all_images_rgb = np.load(path_all_images, allow_pickle=True)
    full_dataset_segment_infos = np.load(path_segment_infos, allow_pickle=True) # Flat list/array of dicts
    main_logger.info(f"Loaded pre-generated data: {len(full_dataset_all_images_rgb)} images, {len(full_dataset_segment_infos)} total segments.")

    # --- 2. Prepare Image Labels and Split Data ---
    main_logger.info("--- Phase 2: Preparing Labels and Splitting Data ---")
    scene_map = load_scene_categories(config["scene_cat_file"])
    
    sorted_chosen_classes_for_mapping = sorted(config["chosen_classes"]) # Used for consistent label mapping
    scene_to_global_idx_map = {s: i for i, s in enumerate(sorted_chosen_classes_for_mapping)}
    config['sorted_chosen_classes_for_mapping'] = sorted_chosen_classes_for_mapping # Store for later use

    # Get base_ids and create labels DIRECTLY based on the order in full_dataset_all_images_rgb
    # We need to find the base_id for each image in full_dataset_all_images_rgb.
    # The `segment_infos` links `img_idx` (index for full_dataset_all_images_rgb) to `base_id`.
    num_loaded_images = len(full_dataset_all_images_rgb)
    all_base_ids_ordered = [None] * num_loaded_images # Will store base_id for each loaded image
    
    temp_img_indices_found_in_segs = set()
    for seg_info in full_dataset_segment_infos:
        orig_img_idx = seg_info.get('img_idx')
        base_id = seg_info.get('base_id')
        if orig_img_idx is not None and base_id is not None and 0 <= orig_img_idx < num_loaded_images:
            if all_base_ids_ordered[orig_img_idx] is None:
                all_base_ids_ordered[orig_img_idx] = base_id
            elif all_base_ids_ordered[orig_img_idx] != base_id: # Should not happen with consistent data
                main_logger.warning(f"BaseID conflict for loaded image index {orig_img_idx}. Keeping first seen.")
            temp_img_indices_found_in_segs.add(orig_img_idx)

    valid_image_indices_for_run = [] # Indices into full_dataset_all_images_rgb that we can use
    base_ids_for_run = []
    labels_for_run_list = []

    for i in range(num_loaded_images):
        base_id = all_base_ids_ordered[i]
        if base_id is None:
            # main_logger.warning(f"Image at loaded index {i} has no base_id derived from segments. Skipping.")
            continue
        scene = scene_map.get(base_id)
        if scene in config['chosen_classes']: # Filter by CURRENT run's chosen_classes
            label_idx = scene_to_global_idx_map.get(scene)
            if label_idx is not None:
                valid_image_indices_for_run.append(i) # This is the original index
                base_ids_for_run.append(base_id)
                labels_for_run_list.append(label_idx)
            # else: main_logger.warning(f"Scene '{scene}' for base_id {base_id} not in current run's chosen_classes map.")
        # else: main_logger.debug(f"Base_id {base_id} (scene '{scene}') not in current run's chosen_classes.")


    if not valid_image_indices_for_run:
        main_logger.error(f"No images from loaded .npy files match the current config's chosen_classes: {config['chosen_classes']}. Exiting.")
        return
    main_logger.info(f"Proceeding with {len(valid_image_indices_for_run)} images that match current chosen_classes and have segment info.")
    
    labels_for_run_np = np.array(labels_for_run_list, dtype=np.int64)

    # Split the `valid_image_indices_for_run` into train and test indices
    # These indices still refer to positions in `full_dataset_all_images_rgb`
    indices_to_split_from = np.arange(len(valid_image_indices_for_run))
    try:
        train_relative_indices, test_relative_indices = train_test_split(
            indices_to_split_from,
            test_size=config['test_split_ratio'],
            random_state=config['seed'],
            stratify=labels_for_run_np # Stratify on the labels of the valid images
        )
    except ValueError:
        main_logger.warning("Stratification failed. Using random split on valid image indices.")
        train_relative_indices, test_relative_indices = train_test_split(
            indices_to_split_from, test_size=config['test_split_ratio'], random_state=config['seed']
        )

    # Get the original global indices for train and test images
    train_original_global_indices = [valid_image_indices_for_run[i] for i in train_relative_indices]
    test_original_global_indices = [valid_image_indices_for_run[i] for i in test_relative_indices]

    # Create train data structures (these lists will be ordered by the split)
    y_train_labels = labels_for_run_np[train_relative_indices]
    train_base_ids = [base_ids_for_run[i] for i in train_relative_indices]
    images_train_rgb_list = [full_dataset_all_images_rgb[original_idx] for original_idx in train_original_global_indices]
    masks_train_per_image_list = [full_dataset_all_masks[original_idx] for original_idx in train_original_global_indices]

    # Create test data structures
    y_test_labels = labels_for_run_np[test_relative_indices] # This is y_test_labels_final
    test_base_ids = [base_ids_for_run[i] for i in test_relative_indices]
    images_test_rgb_list = [full_dataset_all_images_rgb[original_idx] for original_idx in test_original_global_indices]
    masks_test_per_image_list = [full_dataset_all_masks[original_idx] for original_idx in test_original_global_indices]

    train_orig_to_local_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(train_original_global_indices)}
    test_orig_to_local_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(test_original_global_indices)}

    seg_infos_train_list = []
    seg_infos_test_list = []

    for seg_info_dict_orig in full_dataset_segment_infos: # Iterate all loaded segments
        original_img_idx_of_segment = seg_info_dict_orig.get('img_idx') # Global original index

        if original_img_idx_of_segment in train_orig_to_local_map:
            new_local_train_img_idx = train_orig_to_local_map[original_img_idx_of_segment]
            new_seg_info_dict = dict(seg_info_dict_orig)
            new_seg_info_dict['img_idx'] = new_local_train_img_idx # Re-index
            seg_infos_train_list.append(new_seg_info_dict)
        elif original_img_idx_of_segment in test_orig_to_local_map:
            new_local_test_img_idx = test_orig_to_local_map[original_img_idx_of_segment]
            new_seg_info_dict = dict(seg_info_dict_orig)
            new_seg_info_dict['img_idx'] = new_local_test_img_idx # Re-index
            seg_infos_test_list.append(new_seg_info_dict)
            
    seg_infos_train_np = np.array(seg_infos_train_list, dtype=object)
    seg_infos_test_np = np.array(seg_infos_test_list, dtype=object)

    main_logger.info(f"Train split: {len(images_train_rgb_list)} images, {len(seg_infos_train_np)} segments. Labels: {len(y_train_labels)}")
    main_logger.info(f"Test split: {len(images_test_rgb_list)} images, {len(seg_infos_test_np)} segments. Labels: {len(y_test_labels)}")

    main_logger.info("VERIFYING TEST SET ALIGNMENT:")
    for i_check in range(min(5, len(test_base_ids))):
        base_id_from_list = test_base_ids[i_check]
        label_from_list = y_test_labels[i_check]
        original_base_id_for_image_at_test_idx_i_check = None
        # The 'base_id' in seg_infos_test_np items IS the original base_id.
        # Its 'img_idx' is the new local test index.
        found_matching_seg_info = False
        for si_test_debug in seg_infos_test_np:
            if si_test_debug.get('img_idx') == i_check: # i_check is the local test image index
                original_base_id_for_image_at_test_idx_i_check = si_test_debug.get('base_id')
                found_matching_seg_info = True
                break # Found first segment for this local image index

        if not found_matching_seg_info:
             main_logger.error(f"  Test local_idx {i_check}: No segment found in seg_infos_test_np with local img_idx {i_check}!")
        elif base_id_from_list != original_base_id_for_image_at_test_idx_i_check:
            main_logger.error(f"  MISMATCH Test local_idx {i_check}: test_base_ids says '{base_id_from_list}', "
                              f"but seg_infos_test_np implies base_id '{original_base_id_for_image_at_test_idx_i_check}' for local img_idx {i_check}.")
        else:
            main_logger.info(f"  OK Test local_idx {i_check}: base_id '{base_id_from_list}' matches. Label: {label_from_list} ('{config['sorted_chosen_classes_for_mapping'][label_from_list]}')")
 
    main_logger.info(f"Train split: {len(images_train_rgb_list)} images, {len(seg_infos_train_np)} segments. Labels: {len(y_train_labels)}")
    main_logger.info(f"Test split: {len(images_test_rgb_list)} images, {len(seg_infos_test_np)} segments. Labels: {len(y_test_labels)}")
    y_test_labels_final = y_test_labels.copy() 
    # --- Prepare seg_crop_bgr for TRAINING data segments ---
    main_logger.info("Preparing 'seg_crop_bgr' for TRAINING segments if not present...")
    # Caching for add_seg_crop_bgr_to_split_infos (TRAIN)
    seg_crops_train_cache_file = os.path.join(
        config["central_run_cache_dir"], 
        f"seg_crops_train_{config['run_id']}.pkl"
    )
    if config.get("use_seg_crops_cache", True) and os.path.exists(seg_crops_train_cache_file):
        try:
            with open(seg_crops_train_cache_file, "rb") as f:
                seg_infos_train_np_with_crops = pickle.load(f)
            main_logger.info(f"Loaded cached train seg_crops from {seg_crops_train_cache_file}")
            if not isinstance(seg_infos_train_np_with_crops, np.ndarray) or \
               (seg_infos_train_np_with_crops.size > 0 and not isinstance(seg_infos_train_np_with_crops[0], dict)):
                raise ValueError("Cached train seg_crops not in expected format.")
        except Exception as e:
            main_logger.warning(f"Failed to load cached train seg_crops: {e}. Recomputing.")
            seg_infos_train_np_with_crops = add_seg_crop_bgr_to_split_infos(
                seg_infos_train_np, images_train_rgb_list, masks_train_per_image_list, main_logger
            )
            if config.get("use_seg_crops_cache", True):
                with open(seg_crops_train_cache_file, "wb") as f: pickle.dump(seg_infos_train_np_with_crops, f)
    else:
        seg_infos_train_np_with_crops = add_seg_crop_bgr_to_split_infos(
            seg_infos_train_np, images_train_rgb_list, masks_train_per_image_list, main_logger
        )
        if config.get("use_seg_crops_cache", True) and seg_infos_train_np_with_crops is not None:
            with open(seg_crops_train_cache_file, "wb") as f: pickle.dump(seg_infos_train_np_with_crops, f)
    
    main_logger.info("--- Phase 3: Embedding (Training Data Segments) ---")
    dino_processor, dino_model = init_dino(config['dino_model'], torch_device)
    target_model_resnet = init_target_model(torch_device) if config['embedding_type'] == 'combined' else None
    
    embeddings_train_segments = compute_final_embeddings(
        seg_infos_train_np_with_crops, images_train_rgb_list, None, # masks not directly used by compute_final_embeddings now
        dino_processor, dino_model, target_model_resnet,
        torch_device, config, client_id=f"central_train_reused_{config['run_id']}" # Unique cache ID for this split
    )
    if embeddings_train_segments is None or embeddings_train_segments.shape[0] == 0: main_logger.error("Embedding failed for train segments!"); return
    main_logger.info(f"Training segment embeddings computed. Shape: {embeddings_train_segments.shape}")

    visualize_embedding_tsne(embeddings_train_segments, None, "TrainSegEmbeds_PreKMeans", config, main_logger)


    # --- Phase 4: Concept Discovery (K-Means on training segment embeddings) ---
    main_logger.info(f"--- Phase 4: Concept Discovery (K-Means with k={config['num_clusters']}) ---")
    kmeans_cache_file = os.path.join(
        config["central_run_cache_dir"],
        f"kmeans_results_{config['run_id']}_k{config['num_clusters']}_minsamp{config['min_samples_per_concept_cluster']}.pkl"
    )
    if config.get("use_kmeans_cache", True) and os.path.exists(kmeans_cache_file):
        try:
            with open(kmeans_cache_file, "rb") as f:
                cluster_labels_train_segments, final_concept_original_indices = pickle.load(f)
            main_logger.info(f"Loaded cached K-Means results from {kmeans_cache_file}")
            if not isinstance(cluster_labels_train_segments, np.ndarray) or \
               not isinstance(final_concept_original_indices, list):
                raise ValueError("Cached K-Means results not in expected format.")
        except Exception as e:
            main_logger.warning(f"Failed to load cached K-Means results: {e}. Recomputing.")
            kmeans = KMeans(n_clusters=config['num_clusters'], random_state=config['seed'], n_init=10, verbose=0)
            cluster_labels_train_segments = kmeans.fit_predict(embeddings_train_segments)
            unique_labels_km, counts_km = np.unique(cluster_labels_train_segments, return_counts=True)
            keep_mask_km = counts_km >= config['min_samples_per_concept_cluster']
            final_concept_original_indices = unique_labels_km[keep_mask_km].tolist()
            if config.get("use_kmeans_cache", True):
                with open(kmeans_cache_file, "wb") as f: pickle.dump((cluster_labels_train_segments, final_concept_original_indices), f)
    else:
        # K-Means computation block
        kmeans = KMeans(n_clusters=config['num_clusters'], random_state=config['seed'], n_init=10, verbose=0)
        cluster_labels_train_segments = kmeans.fit_predict(embeddings_train_segments)
        unique_labels_km, counts_km = np.unique(cluster_labels_train_segments, return_counts=True)
        keep_mask_km = counts_km >= config['min_samples_per_concept_cluster']
        final_concept_original_indices = unique_labels_km[keep_mask_km].tolist()
        if config.get("use_kmeans_cache", True) and final_concept_original_indices is not None: # Ensure data is valid before saving
            with open(kmeans_cache_file, "wb") as f: pickle.dump((cluster_labels_train_segments, final_concept_original_indices), f)

    if not final_concept_original_indices: main_logger.error("No concept clusters survived filtering!"); return # Critical check

    main_logger.info(f"Found {len(final_concept_original_indices)} concepts after K-Means and filtering.")

    visualize_embedding_tsne(embeddings_train_segments, cluster_labels_train_segments, "TrainSegEmbeds_PostKMeans", config, main_logger)
    
    cluster_to_view = [idx for idx in [0, 5, 10] if idx in final_concept_original_indices and idx < config['num_clusters']]
    if cluster_to_view:
        for i_c in cluster_to_view[:min(3, len(cluster_to_view))]:
            visualize_cluster_segments_from_data(
                i_c, cluster_labels_train_segments,
                list(seg_infos_train_np_with_crops),
                list(masks_train_per_image_list), list(images_train_rgb_list),
                10, (2,5), (12,5), 0.7,
                f"cluster{i_c}_train_examples", config, main_logger
            )

    main_logger.info("--- Visualizing Concept Clusters vs. Random Segments ---")
    if 'final_concept_original_indices' in locals() and final_concept_original_indices and \
       'seg_infos_train_np_with_crops' in locals() and seg_infos_train_np_with_crops.size > 0 and \
       'cluster_labels_train_segments' in locals() and cluster_labels_train_segments is not None:
        
        num_concepts_to_visualize_detailed = min(20, len(final_concept_original_indices))
        main_logger.info(f"Will generate concept_vs_random plots for {num_concepts_to_visualize_detailed} clusters.")

        for i in range(num_concepts_to_visualize_detailed):
            concept_id_to_show = final_concept_original_indices[i] # Get K-Means cluster ID
            
            main_logger.info(f"Plotting Concept_vs_Random for K-Means Cluster ID: {concept_id_to_show}")

            visualize_concept_vs_random_six(
            concept_cluster_id=concept_id_to_show,
            all_segment_infos=list(seg_infos_train_np_with_crops), # Pass train segments with crops
            all_cluster_labels= cluster_labels_train_segments,
            num_each=6,
            config=config,
            logger=main_logger,
            plot_prefix="ttt"
        )
    else:
        main_logger.warning("Skipping Concept_vs_Random visualization: missing necessary data components.")


    # --- Phase 5: Concept Detector Training ---
    main_logger.info("--- Phase 5: Concept Detector Training ---")
    detectors_cache_file = os.path.join(
        config["central_run_cache_dir"],
        f"detectors_{config['run_id']}_type{config['detector_type']}_score{config['min_detector_score']}.pkl"
    )
    image_groups_train_segments = np.array([info["img_idx"] for info in seg_infos_train_np_with_crops]) # Define this once

    if config.get("use_detectors_cache", True) and os.path.exists(detectors_cache_file):
        try:
            with open(detectors_cache_file, "rb") as f:
                trained_detectors_loaded = pickle.load(f) # This should be a dict {orig_idx: (pipeline, threshold)}
            main_logger.info(f"Loaded cached detectors from {detectors_cache_file}")
            if not isinstance(trained_detectors_loaded, dict): raise ValueError("Cached detectors not a dict.")
            trained_detectors = trained_detectors_loaded 
        except Exception as e:
            main_logger.warning(f"Failed to load cached detectors: {e}. Recomputing.")
            trained_detectors = {}
            for original_idx in tqdm(final_concept_original_indices, desc="Training Detectors", file=sys.stdout):
                _, model_info, score = train_concept_detector(
                    original_idx, embeddings_train_segments, cluster_labels_train_segments,
                    image_groups_train_segments, config
                )
                if model_info and score >= config['min_detector_score']:
                    trained_detectors[original_idx] = model_info # model_info is (pipeline, optimal_threshold)
            if config.get("use_detectors_cache", True):
                with open(detectors_cache_file, "wb") as f: pickle.dump(trained_detectors, f)
    else:
        trained_detectors = {}
        for original_idx in tqdm(final_concept_original_indices, desc="Training Detectors", file=sys.stdout):
            _, model_info, score = train_concept_detector(
                original_idx, embeddings_train_segments, cluster_labels_train_segments,
                image_groups_train_segments, config
            )
            if model_info and score >= config['min_detector_score']:
                trained_detectors[original_idx] = model_info
        if config.get("use_detectors_cache", True) and trained_detectors: # Save if not empty
            with open(detectors_cache_file, "wb") as f: pickle.dump(trained_detectors, f)
    
    if not trained_detectors: main_logger.error("No concept detectors trained successfully!"); return # Critical
    ordered_final_concept_original_ids_for_features = sorted(list(trained_detectors.keys()))
    num_final_figs_features = len(ordered_final_concept_original_ids_for_features)
    main_logger.info(f"Trained and kept {num_final_figs_features} concept detectors.")


    # --- Phase 6: Symbolic Concept Vectorization (TRAIN DATA) ---
    main_logger.info(f"--- Phase 6: Concept Vectorization (Training Data) using {config['vectorizer_strategy']} ---")
    train_vectors_cache_file = os.path.join(
        config["central_run_cache_dir"],
        f"train_vectors_{config['run_id']}_strat{config['vectorizer_strategy']}.pkl"
    )
    if config.get("use_train_vectors_cache", True) and os.path.exists(train_vectors_cache_file):
        try:
            with open(train_vectors_cache_file, "rb") as f:
                X_train_concepts, train_ids_from_cache = pickle.load(f)
            main_logger.info(f"Loaded cached TRAIN concept vectors from {train_vectors_cache_file}")
            if not isinstance(X_train_concepts, np.ndarray) or not isinstance(train_ids_from_cache, list) or \
               X_train_concepts.shape[1] != num_final_figs_features: # Check feature dimension
                raise ValueError("Cached train vectors format/dimension error.")
            # Verify if train_base_ids match train_ids_from_cache if necessary, or just use cached IDs
        except Exception as e:
            main_logger.warning(f"Failed to load cached TRAIN concept vectors: {e}. Recomputing.")
            X_train_concepts, _ = build_centralized_concept_vectors_maxpool( # Original base_ids passed
                seg_infos_train_np, embeddings_train_segments,
                trained_detectors, ordered_final_concept_original_ids_for_features,
                len(images_train_rgb_list), train_base_ids,
                config, main_logger
            )
            if config.get("use_train_vectors_cache", True):
                with open(train_vectors_cache_file, "wb") as f: pickle.dump((X_train_concepts, train_base_ids), f)
    else:
        X_train_concepts, _ = build_centralized_concept_vectors_maxpool(
            seg_infos_train_np, embeddings_train_segments,
            trained_detectors, ordered_final_concept_original_ids_for_features,
            len(images_train_rgb_list), train_base_ids,
            config, main_logger
        )
        if config.get("use_train_vectors_cache", True) and X_train_concepts is not None: # Save if valid
            with open(train_vectors_cache_file, "wb") as f: pickle.dump((X_train_concepts, train_base_ids), f)
    

    #visualize_concept_vectors_pca(X_train_concepts, y_train_labels, "TrainImageConceptVecs_MaxPool", config, main_logger)

    figs_feature_names = [f"concept_{i}" for i in range(num_final_figs_features)]

    # --- Phase 8: Process Test Data (Embedding, Vectorization) ---
    # Caching for add_seg_crop_bgr_to_split_infos (TEST)
    seg_crops_test_cache_file = os.path.join(
        config["central_run_cache_dir"], 
        f"seg_crops_test_{config['run_id']}.pkl"
    )
    if len(seg_infos_test_np) > 0: # Only process if there's test data
        if config.get("use_seg_crops_cache", True) and os.path.exists(seg_crops_test_cache_file):
            try:
                with open(seg_crops_test_cache_file, "rb") as f: seg_infos_test_np_with_crops = pickle.load(f)
                main_logger.info(f"Loaded cached TEST seg_crops from {seg_crops_test_cache_file}")
            except Exception as e:
                main_logger.warning(f"Failed to load cached TEST seg_crops: {e}. Recomputing.")
                seg_infos_test_np_with_crops = add_seg_crop_bgr_to_split_infos(seg_infos_test_np, images_test_rgb_list, masks_test_per_image_list, main_logger)
                if config.get("use_seg_crops_cache", True): 
                    with open(seg_crops_test_cache_file, "wb") as f: pickle.dump(seg_infos_test_np_with_crops, f)
        else:
            seg_infos_test_np_with_crops = add_seg_crop_bgr_to_split_infos(seg_infos_test_np, images_test_rgb_list, masks_test_per_image_list, main_logger)
            if config.get("use_seg_crops_cache", True) and seg_infos_test_np_with_crops is not None:
                with open(seg_crops_test_cache_file, "wb") as f:
                    pickle.dump(seg_infos_test_np_with_crops, f)
        
        # Test embeddings (cached by compute_final_embeddings with unique client_id)
        embeddings_test_segments = compute_final_embeddings(
            seg_infos_test_np_with_crops, images_test_rgb_list, None,
            dino_processor, dino_model, target_model_resnet,
            torch_device, config, client_id=f"central_test_reused_{config['run_id']}"
        )
        # Caching for TEST concept vectors
        test_vectors_cache_file = os.path.join(
            config["central_run_cache_dir"],
            f"test_vectors_{config['run_id']}_strat{config['vectorizer_strategy']}.pkl"
        )
        if embeddings_test_segments is not None and embeddings_test_segments.shape[0] > 0:
            if config.get("use_test_vectors_cache", True) and os.path.exists(test_vectors_cache_file):
                try:
                    with open(test_vectors_cache_file, "rb") as f: X_test_concepts, test_ids_from_cache = pickle.load(f)
                    main_logger.info(f"Loaded cached TEST concept vectors from {test_vectors_cache_file}")
                    if not isinstance(X_test_concepts, np.ndarray) or X_test_concepts.shape[1] != num_final_figs_features: 
                        raise ValueError("Cached test vectors format/dim error.")
                except Exception as e:
                    main_logger.warning(f"Failed to load cached TEST concept vectors: {e}. Recomputing.")
                    X_test_concepts, _ = build_centralized_concept_vectors_maxpool(
                        seg_infos_test_np, embeddings_test_segments, trained_detectors, ordered_final_concept_original_ids_for_features,
                        len(images_test_rgb_list), test_base_ids, config, main_logger)
                    if config.get("use_test_vectors_cache", True): 
                        with open(test_vectors_cache_file, "wb") as f:
                            pickle.dump((X_test_concepts, test_base_ids), f)
            else:
                X_test_concepts, _ = build_centralized_concept_vectors_maxpool(
                    seg_infos_test_np, embeddings_test_segments, trained_detectors, ordered_final_concept_original_ids_for_features,
                    len(images_test_rgb_list), test_base_ids, config, main_logger)
                if config.get("use_test_vectors_cache", True) and X_test_concepts is not None:
                    with open(test_vectors_cache_file, "wb") as f:
                         pickle.dump((X_test_concepts, test_base_ids), f)
        else:
            main_logger.warning("No embeddings for test segments, X_test_concepts will be empty.")
            X_test_concepts = np.empty((0, num_final_figs_features)) # Ensure defined
    else:
        main_logger.warning("No segments for test data. X_test_concepts will be empty.")
        X_test_concepts = np.empty((0, num_final_figs_features)) # Ensure defined

    # --- 7. FIGS Model Training ---
    main_logger.info("--- Phase 7: AutoCore Model Training ---")
    
    main_logger.info("Sanity check: Training Sklearn Logistic Regression on concept vectors...")
    lr_sanity_model = SklearnLR(random_state=config['seed'], max_iter=1000)
    lr_sanity_model.fit(X_train_concepts, y_train_labels)
    if X_test_concepts.shape[0] > 0:
        lr_accuracy = lr_sanity_model.score(X_test_concepts, y_test_labels_final)
        main_logger.info(f"Sanity Sklearn LR Test Accuracy: {lr_accuracy:.4f}")

    main_logger.info("Sanity check: Training Sklearn Random Forest on concept vectors...")
    rf_sanity_model = SklearnRF(random_state=config['seed'], n_estimators=100)
    rf_sanity_model.fit(X_train_concepts, y_train_labels)
    if X_test_concepts.shape[0] > 0:
        rf_accuracy = rf_sanity_model.score(X_test_concepts, y_test_labels_final)
        main_logger.info(f"Sanity Sklearn RF Test Accuracy: {rf_accuracy:.4f}")
    max_rules_values = [20, 30, 40, 60, 100]
    max_trees_values = [ 1,3,5]
    max_features = ['sqrt', None]
    for max_rules in max_rules_values:
        for max_trees in max_trees_values:
            for current_max_features in max_features:
                config['figs_params']['max_rules'] = max_rules
                config['figs_params']['max_trees'] = max_trees
                config['figs_params']['max_features'] = current_max_features
                figs_model = PatchedFIGSClassifier(**config['figs_params'], random_state=config['seed'], n_outputs_global=config['num_classes'] )
                # parameters used
                main_logger.info(f"Patched imodels.FIGSClassifier Params: {config['figs_params']}")
                df_train_concepts = pd.DataFrame(X_train_concepts, columns=ordered_final_concept_original_ids_for_features)
                figs_model.fit(df_train_concepts, y_train_labels, feature_names=ordered_final_concept_original_ids_for_features)
                main_logger.info(f"AutoCore model trained. Complexity: {getattr(figs_model, 'complexity_', 'N/A')}")


                # --- 9. Evaluation ---
                main_logger.info("--- Phase 9: Final Evaluation ---")
                if X_test_concepts.shape[0] > 0 and y_test_labels_final.shape[0] == X_test_concepts.shape[0]:
                    accuracy, rule_prec, rule_fid = calculate_metrics(
                        figs_model, X_test_concepts, y_test_labels_final, figs_feature_names, main_logger)
                    main_logger.info(f"Run ID: {config['run_id']}")
                    main_logger.info(f"AutoCore Model Test Accuracy: {accuracy:.4f}, Mean Rule Precision: {rule_prec:.4f}, Fidelity: {rule_fid:.4f}")
                else:
                    main_logger.warning("Skipping final evaluation: No test concept data or label mismatch.")
    # --- 9. Evaluation ---
    main_logger.info("--- Phase 9: Final Evaluation ---")
    if X_test_concepts.shape[0] > 0 and y_test_labels_final.shape[0] == X_test_concepts.shape[0]:

        accuracy, rule_prec, rule_cov, rule_comp, rule_fid = evaluate_centralized_AutoCore_model(
            figs_model, X_test_concepts, y_test_labels_final, 
            figs_feature_names, config['num_classes'], main_logger
        ) 

        main_logger.info(f"Run ID: {config['run_id']}")
        main_logger.info(f"AutoCore Model Test Accuracy: {accuracy:.4f}, RuleP: {rule_prec:.4f}, RuleCov: {rule_cov:.4f}, RuleComp: {rule_comp:.2f}, RuleFid: {rule_fid:.4f}")
        
        if config.get("generate_paper_visualizations", True):
            y_pred_test_labels = figs_model.predict(X_test_concepts)
            correctly_classified_indices_in_test = np.where(y_pred_test_labels == y_test_labels_final)[0]
            
            num_viz_samples = min(config.get("num_paper_viz_samples", 50), len(correctly_classified_indices_in_test))
            
            if num_viz_samples > 0:
                chosen_indices_for_viz = np.random.choice(correctly_classified_indices_in_test, num_viz_samples, replace=False)
                
                for test_split_idx_to_viz in chosen_indices_for_viz:

                    original_rgb_image_to_viz = images_test_rgb_list[test_split_idx_to_viz] # From Phase 2 data split
                    concept_vector_to_viz = X_test_concepts[test_split_idx_to_viz]
                    predicted_label_idx_to_viz = y_pred_test_labels[test_split_idx_to_viz]
                    predicted_scene_name_to_viz = config['sorted_chosen_classes_for_mapping'][predicted_label_idx_to_viz]

                    main_logger.info(f"Generating paper viz for test sample (local_idx_in_test_split: {test_split_idx_to_viz}), "
                                     f"pred: {predicted_scene_name_to_viz}")

                    visualize_centralized_decision(
                        target_image_rgb_np=original_rgb_image_to_viz,
                        image_idx_in_test_split=test_split_idx_to_viz,
                        predicted_class_name_str=predicted_scene_name_to_viz,
                        image_concept_vector_np=concept_vector_to_viz,
                        figs_model_instance=figs_model,
                        ordered_final_concept_original_ids=ordered_final_concept_original_ids_for_features,
                        trained_concept_detectors_map_paper_viz=trained_detectors,
                        seg_infos_test_flat_paper_viz=seg_infos_test_np_with_crops, 
                        embeddings_test_flat_paper_viz=embeddings_test_segments,   
                        feature_names_for_figs_paper_viz=figs_feature_names,  
                        config_paper_viz=config,
                        main_logger_paper_viz=main_logger,
                        # These control the content of the 2x2 panel
                        target_num_top_concept_segments_from_image=config.get("paper_viz_target_concept_segments", 2),
                        # segments_per_concept is implicitly 1 if we pick distinct concepts for the panel
                        target_num_random_segments_from_dataset=config.get("paper_viz_target_random_segments", 2), # This will be adjusted by remaining 
                    )

    # --- Phase 10: Visualizing Decision Explanations ---
    main_logger.info("--- Phase 10: Visualizing Decision Explanations ---")
    if 'figs_model' in locals() and hasattr(figs_model, 'trees_') and \
       'X_test_concepts' in locals() and X_test_concepts is not None and X_test_concepts.shape[0] > 0 and \
       'y_test_labels' in locals() and y_test_labels is not None and y_test_labels.shape[0] > 0 and \
       'test_base_ids' in locals() and len(test_base_ids) > 0 and \
       'images_test_rgb_list' in locals() and len(images_test_rgb_list) > 0 and \
       'seg_infos_test_np' in locals() and seg_infos_test_np.size > 0 and \
       'masks_test_per_image_list' in locals() and len(masks_test_per_image_list) > 0 and \
       'embeddings_test_segments' in locals() and embeddings_test_segments is not None and embeddings_test_segments.shape[0] > 0 and \
       'trained_detectors' in locals() and \
       'ordered_final_concept_original_ids_for_features' in locals() and \
       'figs_feature_names' in locals():

        num_samples_to_explain = min(3, X_test_concepts.shape[0])
        main_logger.info(f"Attempting decision explanations for {num_samples_to_explain} test samples...")

        for i in range(num_samples_to_explain):
            # `i` is the local index for the test split data structures
            current_local_test_idx = i

            image_concept_vec_sample = X_test_concepts[current_local_test_idx]
            predicted_class_idx_sample = figs_model.predict(image_concept_vec_sample.reshape(1, -1))[0]
            actual_class_idx_sample = y_test_labels[current_local_test_idx]
            current_base_id_for_viz = test_base_ids[current_local_test_idx]
            # Use the sorted list for name lookup
            actual_class_name = config['sorted_chosen_classes_for_mapping'][actual_class_idx_sample]
            predicted_class_name = config['sorted_chosen_classes_for_mapping'][predicted_class_idx_sample]
            main_logger.info(f"VIZ PREP for TestSample local_idx {current_local_test_idx}:")
            main_logger.info(f"  BaseID from test_base_ids: {current_base_id_for_viz}")
            main_logger.info(f"  True Label: {actual_class_idx_sample} ({actual_class_name})")
            main_logger.info(f"  Predicted Label: {predicted_class_idx_sample} ({predicted_class_name})")
            # Sanity check: What image does the *scene_map* say this base_id is?
            scene_from_map_for_baseid = scene_map.get(current_base_id_for_viz, "UNKNOWN_SCENE_IN_MAP")
            label_idx_from_map_for_baseid = scene_to_global_idx_map.get(scene_from_map_for_baseid, -1) # This uses the correct map
            name_from_map = config['sorted_chosen_classes_for_mapping'][label_idx_from_map_for_baseid] if label_idx_from_map_for_baseid !=-1 else "N/A"
            main_logger.info(f"  For BaseID {current_base_id_for_viz}, scene_map gives: {scene_from_map_for_baseid} (label_idx: {label_idx_from_map_for_baseid}, name: {name_from_map})")
            if label_idx_from_map_for_baseid != actual_class_idx_sample:
                main_logger.error(f"CRITICAL MISMATCH: y_test_labels[{current_local_test_idx}] ({actual_class_idx_sample}) "
                                  f"!= label from scene_map for base_id {current_base_id_for_viz} ({label_idx_from_map_for_baseid}). "
                                  f"This indicates `y_test_labels` or `test_base_ids` is misaligned from the start!")
                # If this error occurs, the problem is in Phase 2 when test_df was created or when y_test_labels/test_base_ids were extracted.

            try:
                visualize_decision_explanation(
                    image_idx_in_split=current_local_test_idx, # This is the index for images_test_rgb_list etc.
                    images_split_rgb=images_test_rgb_list,
                    seg_infos_flat_split=seg_infos_test_np,
                    masks_split_per_image=masks_test_per_image_list,
                    embeddings_flat_split_segments=embeddings_test_segments,
                    image_concept_vector_for_explanation_np=image_concept_vec_sample,
                    predicted_class_name=predicted_class_name,
                    figs_model_instance=figs_model,
                    trained_detectors_map=trained_detectors,
                    ordered_final_concept_original_ids=ordered_final_concept_original_ids_for_features,
                    feature_names_for_figs=figs_feature_names,
                    config=config,
                    main_logger_passed=main_logger,
                    title_extra = f"_TestIdx{current_local_test_idx}_BaseID_{current_base_id_for_viz}_Pred{predicted_class_idx_sample}{predicted_class_name}_True{actual_class_idx_sample}{actual_class_name}"
                )
                
            except Exception as e_viz_main:
                main_logger.error(f"Error in visualize_decision_explanation call for sample {current_local_test_idx}: {e_viz_main}", exc_info=True)
    else:
        main_logger.warning("Skipping decision explanation: Missing necessary components for visualization.")

    main_logger.info(f"======== Centralized Reused Segments Run ID: {config['run_id']} Complete ========")

if __name__ == "__main__":
    main_AutoCore_centralized_auto()