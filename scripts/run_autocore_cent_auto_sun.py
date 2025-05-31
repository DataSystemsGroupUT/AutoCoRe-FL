# run_cent.py
import logging
import operator
import os
import re
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from tqdm import tqdm
import pandas as pd
import torch

import json 

from AutoCore_FL.embedding.dino_loader import init_dino, init_target_model
from AutoCore_FL.embedding.compute_embeddings import compute_final_embeddings
from AutoCore_FL.concepts.detector import train_concept_detector 
from AutoCore_FL.federated.client import PatchedFIGSClassifier
from AutoCore_FL.federated.utils import setup_logging, calculate_metrics, build_centralized_concept_vectors_maxpool, add_seg_crop_bgr_to_split_infos
from sklearn.metrics import accuracy_score 

SEED = 42
dataset_name = "sun"  # Change to "ade20k" or "sunrgbd" as needed

# --- Helper Function: Load Scene Categories (Generic) ---
def load_scene_categories_generic(scene_cat_file: str, main_logger_passed) -> dict:
    scene_map = {}
    try:
        with open(scene_cat_file, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    base, scene_str = parts
                    scene_map[base] = scene_str.lower().strip()
                # else: # Be less verbose about skipping lines unless it's critical
                    # main_logger_passed.debug(f"Skipping malformed line {line_num+1} in {scene_cat_file}: '{line.strip()}'")
    except FileNotFoundError:
        main_logger_passed.error(f"Scene category file not found: {scene_cat_file}")
        raise
    except Exception as e:
        main_logger_passed.error(f"Error reading scene category file {scene_cat_file}: {e}")
        raise
    return scene_map


# --- Config Generation ---
def generate_centralized_config( run_id_base=f"AutoCore_Cent_AutoConcepts_{dataset_name}"):
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]) if hasattr(sys, 'argv') else __file__)
    
    config = {
        "seed": SEED, "test_split_ratio": 0.2, "dino_model": "facebook/dinov2-base",
        "embedding_type": "dino_only", "embedding_dim": 768,
        "min_samples_per_concept_cluster": 50, "detector_type": "lr",
        "detector_min_samples_per_class": 20, "detector_cv_splits": 3,
        "pca_n_components": 256, "lr_max_iter": 10000, 
        "vectorizer_strategy": "max_pool",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "plot_dpi": 100, "min_mask_pixels": 100,
        "method_name": f"AutoCore_Cent_AutoConcepts_{dataset_name}", # Base name
        "dataset_name": dataset_name,
        "use_seg_crops_cache": True, "use_kmeans_cache": True, "use_detectors_cache": True,
        "use_train_vectors_cache": True, "use_test_vectors_cache": True,
        "use_embedding_cache": True,
        "sweep_num_clusters": [75, 100, 150],
        "sweep_min_detector_score": [0.60, 0.65, 0.70],
        "figs_params_base": {"min_impurity_decrease": 0.0, "random_state": SEED},
        "figs_max_rules_sweep": [20, 30, 40, 60,100], 
        "figs_max_trees_sweep": [1, 3, 5],
        "figs_max_features_sweep": ['sqrt', None],
    }

    if dataset_name == "ade20k":
        config["data_root"] = "/gpfs/helios/home/soliman/logic_explained_networks/data/ade20k/ADEChallengeData2016/"
        config["ade20k_segment_data_path"] = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/" 
        config["scene_cat_file"] = os.path.join(config["data_root"], "sceneCategories.txt")
        ade20k_classes_raw = ['street', 'bedroom', 'living_room', 'bathroom', 'kitchen', 'skyscraper',
                               'highway', 'conference_room', 'mountain_snowy', 'office', 'corridor', 
                               'airport_terminal', 'attic', 'mountain', 'park', 'coast', 'alley',
                               'beach', 'childs_room', 'art_gallery','castle', 'dorm_room',
                               'nursery', 'lobby', 'reception', 'bar', 'house', 'bridge', 'classroom']
        config["chosen_classes"] = sorted(list(set(c.lower().strip() for c in ade20k_classes_raw)))
    elif dataset_name == "sun":
        config["data_root"] = "/gpfs/helios/home/soliman/logic_explained_networks/data/sunrgbd/"
        config["sunrgbd_segment_data_path"] = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/"
        config["scene_cat_file"] = os.path.join(config["data_root"], "images", "sceneCategories.txt")
        
        sun_classes_raw = ['bathroom', 'bedroom', 'bookstore']
        config["chosen_classes"] = sorted(list(set(c.lower().strip() for c in sun_classes_raw)))
        
        temp_logger_sun_config = logging.getLogger("ConfigGenSUN_3Distinct")
        if not temp_logger_sun_config.handlers:
            _h = logging.StreamHandler(sys.stdout); _h.setFormatter(logging.Formatter('%(name)s-%(levelname)s: %(message)s')); temp_logger_sun_config.addHandler(_h); temp_logger_sun_config.setLevel(logging.INFO)
        temp_logger_sun_config.info(f"SUNRGBD: Using PREDEFINED list of {len(config['chosen_classes'])} distinct classes: {config['chosen_classes']}")
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    config["num_classes"] = len(config["chosen_classes"])
    
    distinct_marker = f"_distinct{config['num_classes']}"
    effective_run_id = f"{run_id_base}_{dataset_name}{distinct_marker}_{config['seed']}"
    config["run_id"] = effective_run_id
    config["method_name"] = f"AutoCoRe_Centralized_{dataset_name.upper()}{distinct_marker.upper()}"

    base_dir = os.path.join(script_dir, f"experiment_results_centralized/{config['method_name'].lower()}_run_{effective_run_id}")
    os.makedirs(base_dir, exist_ok=True)
    config["log_dir"] = os.path.join(base_dir, "logs")
    os.makedirs(config["log_dir"], exist_ok=True)
    config["central_run_cache_dir"] = os.path.join(script_dir, "cache_centralized_runs", f"run_{effective_run_id}")
    os.makedirs(config["central_run_cache_dir"], exist_ok=True)
    config["embedding_cache_dir"] = os.path.join(config["central_run_cache_dir"], "embeddings_compute_cache") 
    os.makedirs(config["embedding_cache_dir"], exist_ok=True)
    config["final_model_path"] = os.path.join(base_dir, f"final_figs_model.pkl")
    config["concept_definitions_path"] = os.path.join(base_dir, f"concept_defs.pkl")
    config["metrics_log_path"] = os.path.join(base_dir, f"final_metrics_summary.csv")
    
    return config

def run_single_configuration(config, main_logger_parent):
    run_details_for_log = f"k{config['num_clusters']}_ds{config['min_detector_score']:.2f}"
    current_run_logger = logging.getLogger(f"{main_logger_parent.name}.{run_details_for_log}")
    if not current_run_logger.handlers:
        for handler_parent in main_logger_parent.handlers: current_run_logger.addHandler(handler_parent)
    current_run_logger.setLevel(main_logger_parent.level); current_run_logger.propagate = False
    current_run_logger.info(f"--- Starting Sub-Run: num_clusters={config['num_clusters']}, min_detector_score={config['min_detector_score']:.2f} ---")
    embeddings_train_segments = config['_temp_embeddings_train_segments']
    seg_infos_train_np_with_crops = config['_temp_seg_infos_train_np_with_crops']
    images_train_rgb_list = config['_temp_images_train_rgb_list']
    y_train_labels = config['_temp_y_train_labels']
    train_base_ids = config['_temp_train_base_ids']
    embeddings_test_segments = config['_temp_embeddings_test_segments']
    seg_infos_test_np_with_crops = config['_temp_seg_infos_test_np_with_crops']
    images_test_rgb_list = config['_temp_images_test_rgb_list']
    y_test_labels_final = config['_temp_y_test_labels_final']
    test_base_ids = config['_temp_test_base_ids']
    dataset_name = config['dataset_name']
    kmeans_cache_file = os.path.join(config["central_run_cache_dir"], f"kmeans_k{config['num_clusters']}_results.pkl")
    if config.get("use_kmeans_cache", True) and os.path.exists(kmeans_cache_file):
        try:
            with open(kmeans_cache_file, "rb") as f: cluster_labels_train_segments, final_concept_original_indices = pickle.load(f)
        except Exception as e:
            current_run_logger.warning(f"K-Means cache load failed ({e}), recomputing for k={config['num_clusters']}")
            kmeans = KMeans(n_clusters=config['num_clusters'], random_state=config['seed'], n_init='auto', verbose=0).fit(embeddings_train_segments)
            cluster_labels_train_segments, unique_labels_km, counts_km = kmeans.labels_, np.unique(kmeans.labels_, return_counts=True)[0], np.unique(kmeans.labels_, return_counts=True)[1]
            final_concept_original_indices = unique_labels_km[counts_km >= config['min_samples_per_concept_cluster']].tolist()
            if config.get("use_kmeans_cache", True) and final_concept_original_indices: pickle.dump((cluster_labels_train_segments, final_concept_original_indices), open(kmeans_cache_file, "wb"))
    else:
        kmeans = KMeans(n_clusters=config['num_clusters'], random_state=config['seed'], n_init='auto', verbose=0).fit(embeddings_train_segments)
        cluster_labels_train_segments, unique_labels_km, counts_km = kmeans.labels_, np.unique(kmeans.labels_, return_counts=True)[0], np.unique(kmeans.labels_, return_counts=True)[1]
        final_concept_original_indices = unique_labels_km[counts_km >= config['min_samples_per_concept_cluster']].tolist()
        if config.get("use_kmeans_cache", True) and final_concept_original_indices: pickle.dump((cluster_labels_train_segments, final_concept_original_indices), open(kmeans_cache_file, "wb"))
    if not final_concept_original_indices: current_run_logger.error(f"No K-Means clusters k={config['num_clusters']}."); return None, None, -1.0, -1.0, 0
    detectors_cache_file = os.path.join(config["central_run_cache_dir"], f"detectors_k{config['num_clusters']}_ds{config['min_detector_score']:.2f}.pkl")
    image_groups_train_segments = np.array([info["img_idx"] for info in seg_infos_train_np_with_crops if "img_idx" in info])
    if len(image_groups_train_segments) != embeddings_train_segments.shape[0]: current_run_logger.error("Detector train: image_groups mismatch."); return None, None, -1.0, -1.0, 0
    trained_detectors_with_scores = {}
    if config.get("use_detectors_cache", True) and os.path.exists(detectors_cache_file):
        try: trained_detectors_with_scores = pickle.load(open(detectors_cache_file, "rb"))
        except Exception as e:
            current_run_logger.warning(f"Detector cache load fail ({e}), recomputing")
            trained_detectors_with_scores = {}
    if not trained_detectors_with_scores:
        for original_idx in tqdm(final_concept_original_indices, desc=f"Detectors k={config['num_clusters']}", file=sys.stdout, disable=True):
            _, model_info_tuple, score_val = train_concept_detector(original_idx, embeddings_train_segments, cluster_labels_train_segments, image_groups_train_segments, config)
            if model_info_tuple and score_val >= config['min_detector_score']: trained_detectors_with_scores[original_idx] = (model_info_tuple[0], model_info_tuple[1], score_val)
        if config.get("use_detectors_cache", True) and trained_detectors_with_scores: pickle.dump(trained_detectors_with_scores, open(detectors_cache_file, "wb"))
    if not trained_detectors_with_scores: current_run_logger.error(f"No detectors trained."); return None, None, -1.0, -1.0, 0
    trained_detectors = { orig_idx: (pipe, thresh) for orig_idx, (pipe, thresh, score) in trained_detectors_with_scores.items() }
    all_det_scores = [score for _,_,score in trained_detectors_with_scores.values()]; current_run_logger.info(f"Detectors Kept: {len(trained_detectors)}. Mean Score: {np.mean(all_det_scores) if all_det_scores else 'N/A'}")
    ordered_final_concept_original_ids_for_features = sorted(list(trained_detectors.keys()))
    num_final_figs_features = len(ordered_final_concept_original_ids_for_features)
    if num_final_figs_features == 0: current_run_logger.error("No concepts after detector training."); return None, None, -1.0, -1.0, 0
    train_vectors_cache_file = os.path.join(config["central_run_cache_dir"], f"train_vectors_k{config['num_clusters']}_ds{config['min_detector_score']:.2f}.pkl")
    if config.get("use_train_vectors_cache", True) and os.path.exists(train_vectors_cache_file):
        try: X_train_concepts, _ = pickle.load(open(train_vectors_cache_file, "rb"))
        except Exception as e:
            current_run_logger.warning(f"Train vector cache load fail ({e}), recomputing.")
            X_train_concepts, _ = build_centralized_concept_vectors_maxpool(seg_infos_train_np_with_crops, embeddings_train_segments, trained_detectors, ordered_final_concept_original_ids_for_features, len(images_train_rgb_list), train_base_ids, config, current_run_logger)
            if config.get("use_train_vectors_cache", True) and X_train_concepts.size > 0 : pickle.dump((X_train_concepts, train_base_ids), open(train_vectors_cache_file, "wb"))
    else:
        X_train_concepts, _ = build_centralized_concept_vectors_maxpool(seg_infos_train_np_with_crops, embeddings_train_segments, trained_detectors, ordered_final_concept_original_ids_for_features, len(images_train_rgb_list), train_base_ids, config, current_run_logger)
        if config.get("use_train_vectors_cache", True) and X_train_concepts.size > 0: pickle.dump((X_train_concepts, train_base_ids), open(train_vectors_cache_file, "wb"))
    if X_train_concepts is None or X_train_concepts.shape[0] != len(y_train_labels): current_run_logger.error(f"Train vector gen/shape error."); return None, None, -1.0, -1.0, num_final_figs_features
    X_test_concepts = np.empty((0, num_final_figs_features))
    if embeddings_test_segments is not None and embeddings_test_segments.shape[0] > 0 and len(seg_infos_test_np_with_crops) > 0:
        test_vectors_cache_file = os.path.join(config["central_run_cache_dir"], f"test_vectors_k{config['num_clusters']}_ds{config['min_detector_score']:.2f}.pkl")
        if config.get("use_test_vectors_cache", True) and os.path.exists(test_vectors_cache_file):
            try: X_test_concepts, _ = pickle.load(open(test_vectors_cache_file, "rb"))
            except Exception as e:
                current_run_logger.warning(f"Test vector cache load fail ({e}), recomputing.")
                X_test_concepts, _ = build_centralized_concept_vectors_maxpool(seg_infos_test_np_with_crops, embeddings_test_segments, trained_detectors, ordered_final_concept_original_ids_for_features, len(images_test_rgb_list), test_base_ids, config, current_run_logger)
                if config.get("use_test_vectors_cache", True) and X_test_concepts.size > 0: pickle.dump((X_test_concepts, test_base_ids), open(test_vectors_cache_file, "wb"))
        else:
            X_test_concepts, _ = build_centralized_concept_vectors_maxpool(seg_infos_test_np_with_crops, embeddings_test_segments, trained_detectors, ordered_final_concept_original_ids_for_features, len(images_test_rgb_list), test_base_ids, config, current_run_logger)
            if config.get("use_test_vectors_cache", True) and X_test_concepts.size > 0: pickle.dump((X_test_concepts, test_base_ids), open(test_vectors_cache_file, "wb"))
    if X_train_concepts.size > 0: current_run_logger.info(f"Sparsity X_train_concepts: {1.0 - (np.count_nonzero(X_train_concepts) / float(X_train_concepts.size)):.4f}, Mean active: {np.mean(np.sum(X_train_concepts, axis=1)):.2f}")
    if X_test_concepts.size > 0: current_run_logger.info(f"Sparsity X_test_concepts: {1.0 - (np.count_nonzero(X_test_concepts) / float(X_test_concepts.size)):.4f}, Mean active: {np.mean(np.sum(X_test_concepts, axis=1)):.2f}")
    lr_acc, rf_acc = -1.0, -1.0
    if X_train_concepts.shape[0] > 0 and X_test_concepts.shape[0] > 0 and y_test_labels_final.shape[0] == X_test_concepts.shape[0] and X_train_concepts.shape[1] == X_test_concepts.shape[1] and X_train_concepts.shape[1] > 0: # Ensure non-empty and matching features
        from sklearn.linear_model import LogisticRegression as SklearnLR
        from sklearn.ensemble import RandomForestClassifier as SklearnRF
        lr_sanity = SklearnLR(random_state=config['seed'], max_iter=1000, solver='liblinear').fit(X_train_concepts, y_train_labels)
        lr_acc = lr_sanity.score(X_test_concepts, y_test_labels_final); current_run_logger.info(f"  Sanity LR Acc: {lr_acc:.4f}")
        rf_sanity = SklearnRF(random_state=config['seed'], n_estimators=100).fit(X_train_concepts, y_train_labels)
        rf_acc = rf_sanity.score(X_test_concepts, y_test_labels_final); current_run_logger.info(f"  Sanity RF Acc: {rf_acc:.4f}")
    return X_train_concepts, X_test_concepts, lr_acc, rf_acc, num_final_figs_features

# --- Main Function (Orchestrator) ---
def main_centralized_autocore():
    initial_config = generate_centralized_config() 
    setup_logging(log_dir=initial_config['log_dir'],run_id = initial_config['run_id'])
    main_logger = logging.getLogger(f"MainAutoCoReSweep_{initial_config['run_id']}")
    np.random.seed(initial_config['seed']); torch.manual_seed(initial_config['seed'])
    if initial_config['device'] == 'cuda' and torch.cuda.is_available(): torch.cuda.manual_seed_all(initial_config['seed'])
    main_logger.info(f"Using device: {initial_config['device']}")
    main_logger.info(f"======== Starting Centralized AutoCoRe SWEEP ({dataset_name.upper()}) - Run ID: {initial_config['run_id']} ========")
    main_logger.info(f"Targeting {initial_config['num_classes']} classes: {initial_config['chosen_classes']}")
    main_logger.info(f"Full Initial Config for Sweep: {json.dumps(initial_config, indent=2, sort_keys=True)}")

    # --- Phase 1 & 2: Data Loading and Initial Splitting (Runs ONCE) ---
    # This populates: images_train_rgb_list, masks_train_per_image_list, seg_infos_train_np, y_train_labels, train_base_ids
    # and their _test_ counterparts.
    if dataset_name == "ade20k":
        seg_data_base_path = initial_config["ade20k_segment_data_path"]; npy_suffix = "xx" 
        path_all_masks_npy, path_all_images_npy, path_segment_infos_npy = os.path.join(seg_data_base_path, f"logic_ade20k_all_masks{npy_suffix}.npy"), os.path.join(seg_data_base_path, f"logic_ade20k_all_images{npy_suffix}.npy"), os.path.join(seg_data_base_path, f"logic_ade20k_segment_infos{npy_suffix}.npy")
    elif dataset_name == "sun":
        seg_data_base_path = initial_config["sunrgbd_segment_data_path"]
        path_all_masks_npy, path_all_images_npy, path_segment_infos_npy = os.path.join(seg_data_base_path, "logic_sunrgbd_all_masks.npy"), os.path.join(seg_data_base_path, "logic_sunrgbd_all_images.npy"), os.path.join(seg_data_base_path, "logic_sunrgbd_segment_infos.npy")
    else: main_logger.error(f"Dataset {dataset_name} not supported."); return
    try: full_dataset_all_masks_raw, full_dataset_all_images_rgb_raw, full_dataset_segment_infos_raw = np.load(path_all_masks_npy, allow_pickle=True), np.load(path_all_images_npy, allow_pickle=True), np.load(path_segment_infos_npy, allow_pickle=True)
    except FileNotFoundError: main_logger.error(f"Seg .npy files not found for {dataset_name} at {seg_data_base_path}."); return
    scene_map = load_scene_categories_generic(initial_config["scene_cat_file"], main_logger) 
    initial_config['sorted_chosen_classes_for_mapping'] = initial_config["chosen_classes"] 
    scene_to_global_idx_map = {s: i for i, s in enumerate(initial_config['sorted_chosen_classes_for_mapping'])}
    raw_idx_to_scenemap_key, raw_idx_to_original_seg_base_id = {}, {}
    for seg_info in full_dataset_segment_infos_raw:
        raw_img_idx, base_id_from_seg_npy = seg_info.get('img_idx'), seg_info.get('base_id')
        if raw_img_idx is not None and base_id_from_seg_npy is not None and raw_img_idx not in raw_idx_to_scenemap_key:
            key_for_scene_map = ""
            if dataset_name == "ade20k": key_for_scene_map = base_id_from_seg_npy
            elif dataset_name == "sun":
                if isinstance(base_id_from_seg_npy, str) and '_' in base_id_from_seg_npy: parts = base_id_from_seg_npy.split('_', 1); key_for_scene_map = parts[1] if len(parts) > 1 else base_id_from_seg_npy
                else: key_for_scene_map = str(base_id_from_seg_npy) if base_id_from_seg_npy is not None else ""
            if key_for_scene_map: raw_idx_to_scenemap_key[raw_img_idx], raw_idx_to_original_seg_base_id[raw_img_idx] = key_for_scene_map, base_id_from_seg_npy
    valid_raw_indices_for_run, base_ids_for_run, labels_for_run_list = [], [], []
    for raw_idx in range(len(full_dataset_all_images_rgb_raw)):
        scenemap_key_to_use = raw_idx_to_scenemap_key.get(raw_idx)
        if not scenemap_key_to_use: continue 
        scene_from_map = scene_map.get(scenemap_key_to_use) 
        if scene_from_map and scene_from_map in initial_config['sorted_chosen_classes_for_mapping']:
            label_idx = scene_to_global_idx_map.get(scene_from_map)
            if label_idx is not None: valid_raw_indices_for_run.append(raw_idx); base_ids_for_run.append(raw_idx_to_original_seg_base_id.get(raw_idx, f"ERROR_NO_ORIG_BASEID_FOR_{raw_idx}")); labels_for_run_list.append(label_idx)
    if not valid_raw_indices_for_run: main_logger.error(f"No images match chosen_classes after mapping for {dataset_name}. Exiting."); return
    labels_for_run_np = np.array(labels_for_run_list, dtype=np.int64)
    current_run_all_images_rgb = [full_dataset_all_images_rgb_raw[i] for i in valid_raw_indices_for_run]; current_run_all_masks = [full_dataset_all_masks_raw[i] for i in valid_raw_indices_for_run]
    raw_idx_to_current_run_local_idx = {raw_idx: new_idx for new_idx, raw_idx in enumerate(valid_raw_indices_for_run)}
    current_run_segment_infos_list = []
    for seg_info_raw in full_dataset_segment_infos_raw:
        original_raw_img_idx = seg_info_raw.get('img_idx')
        if original_raw_img_idx in raw_idx_to_current_run_local_idx: new_seg_info = dict(seg_info_raw); new_seg_info['img_idx'] = raw_idx_to_current_run_local_idx[original_raw_img_idx]; current_run_segment_infos_list.append(new_seg_info)
    current_run_segment_infos_np = np.array(current_run_segment_infos_list, dtype=object)
    indices_to_split_from_current_run = np.arange(len(current_run_all_images_rgb))
    try: train_local_indices_in_current_run, test_local_indices_in_current_run = train_test_split(indices_to_split_from_current_run, test_size=initial_config['test_split_ratio'], random_state=initial_config['seed'], stratify=labels_for_run_np)
    except ValueError: main_logger.warning("Stratification failed. Using non-stratified split."); train_local_indices_in_current_run, test_local_indices_in_current_run = train_test_split(indices_to_split_from_current_run, test_size=initial_config['test_split_ratio'], random_state=initial_config['seed'])
    y_train_labels = labels_for_run_np[train_local_indices_in_current_run]; train_base_ids = [base_ids_for_run[i] for i in train_local_indices_in_current_run]; images_train_rgb_list = [current_run_all_images_rgb[i] for i in train_local_indices_in_current_run]; masks_train_per_image_list = [current_run_all_masks[i] for i in train_local_indices_in_current_run]
    y_test_labels = labels_for_run_np[test_local_indices_in_current_run]; test_base_ids = [base_ids_for_run[i] for i in test_local_indices_in_current_run]; images_test_rgb_list = [current_run_all_images_rgb[i] for i in test_local_indices_in_current_run]; masks_test_per_image_list = [current_run_all_masks[i] for i in test_local_indices_in_current_run]
    current_run_idx_to_train_split_idx_map = {idx_curr: new_idx for new_idx, idx_curr in enumerate(train_local_indices_in_current_run)}; current_run_idx_to_test_split_idx_map = {idx_curr: new_idx for new_idx, idx_curr in enumerate(test_local_indices_in_current_run)}
    seg_infos_train_list, seg_infos_test_list = [], []
    for seg_info_from_current_run in current_run_segment_infos_np:
        idx_in_current_run = seg_info_from_current_run.get('img_idx'); final_seg_info_dict = dict(seg_info_from_current_run)
        if idx_in_current_run in current_run_idx_to_train_split_idx_map: final_seg_info_dict['img_idx'] = current_run_idx_to_train_split_idx_map[idx_in_current_run]; seg_infos_train_list.append(final_seg_info_dict)
        elif idx_in_current_run in current_run_idx_to_test_split_idx_map: final_seg_info_dict['img_idx'] = current_run_idx_to_test_split_idx_map[idx_in_current_run]; seg_infos_test_list.append(final_seg_info_dict)
    seg_infos_train_np = np.array(seg_infos_train_list, dtype=object) if seg_infos_train_list else np.array([], dtype=object)
    seg_infos_test_np = np.array(seg_infos_test_list, dtype=object) if seg_infos_test_list else np.array([], dtype=object)
    y_test_labels_final = y_test_labels.copy() # Crucial: this is the y_true for all test evaluations
    main_logger.info(f"Initial Data Split ({dataset_name}): Train Images {len(images_train_rgb_list)}, Test Images {len(images_test_rgb_list)}")

    # --- Phase 2b: Add seg_crop_bgr (ONCE for train, ONCE for test) ---
    train_split_key_params = f"len{len(seg_infos_train_np)}_seed{initial_config['seed']}_splitratio{initial_config['test_split_ratio']}"
    seg_crops_train_cache_file = os.path.join(initial_config["central_run_cache_dir"], f"seg_crops_train_{train_split_key_params}.pkl")
    if initial_config.get("use_seg_crops_cache", True) and os.path.exists(seg_crops_train_cache_file):
        try:
            with open(seg_crops_train_cache_file, "rb") as f: seg_infos_train_np_with_crops = pickle.load(f)
            if not (isinstance(seg_infos_train_np_with_crops, np.ndarray) and (seg_infos_train_np_with_crops.size == 0 or isinstance(seg_infos_train_np_with_crops[0], dict))): raise ValueError("Bad format")
            if len(seg_infos_train_np_with_crops) != len(seg_infos_train_np): raise ValueError("Cache length mismatch")
        except Exception as e:
            main_logger.warning(f"Train seg_crop cache fail: {e}. Recomputing.")
            seg_infos_train_np_with_crops = add_seg_crop_bgr_to_split_infos(seg_infos_train_np, images_train_rgb_list, masks_train_per_image_list, main_logger)
            if initial_config.get("use_seg_crops_cache", True) and seg_infos_train_np_with_crops.size > 0: pickle.dump(seg_infos_train_np_with_crops, open(seg_crops_train_cache_file, "wb"))
    else:
        seg_infos_train_np_with_crops = add_seg_crop_bgr_to_split_infos(seg_infos_train_np, images_train_rgb_list, masks_train_per_image_list, main_logger)
        if initial_config.get("use_seg_crops_cache", True) and seg_infos_train_np_with_crops.size > 0: pickle.dump(seg_infos_train_np_with_crops, open(seg_crops_train_cache_file, "wb"))
    if seg_infos_train_np_with_crops.size == 0 and seg_infos_train_np.size > 0 : main_logger.error(f"Train seg_crops empty but input was not."); return
    seg_infos_test_np_with_crops = np.array([], dtype=object)
    if len(seg_infos_test_np) > 0:
        test_split_key_params = f"len{len(seg_infos_test_np)}_seed{initial_config['seed']}_splitratio{initial_config['test_split_ratio']}"
        seg_crops_test_cache_file = os.path.join(initial_config["central_run_cache_dir"], f"seg_crops_test_{test_split_key_params}.pkl")
        if initial_config.get("use_seg_crops_cache", True) and os.path.exists(seg_crops_test_cache_file):
            try:
                with open(seg_crops_test_cache_file, "rb") as f: seg_infos_test_np_with_crops = pickle.load(f)
                if not (isinstance(seg_infos_test_np_with_crops, np.ndarray) and (seg_infos_test_np_with_crops.size == 0 or isinstance(seg_infos_test_np_with_crops[0], dict))): raise ValueError("Bad format")
                if len(seg_infos_test_np_with_crops) != len(seg_infos_test_np): raise ValueError("Cache length mismatch")
            except Exception as e:
                main_logger.warning(f"Test seg_crop cache fail: {e}. Recomputing.")
                seg_infos_test_np_with_crops = add_seg_crop_bgr_to_split_infos(seg_infos_test_np, images_test_rgb_list, masks_test_per_image_list, main_logger)
                if initial_config.get("use_seg_crops_cache", True) and seg_infos_test_np_with_crops.size > 0: pickle.dump(seg_infos_test_np_with_crops, open(seg_crops_test_cache_file, "wb"))
        else:
            seg_infos_test_np_with_crops = add_seg_crop_bgr_to_split_infos(seg_infos_test_np, images_test_rgb_list, masks_test_per_image_list, main_logger)
            if initial_config.get("use_seg_crops_cache", True) and seg_infos_test_np_with_crops.size > 0: pickle.dump(seg_infos_test_np_with_crops, open(seg_crops_test_cache_file, "wb"))
        if seg_infos_test_np_with_crops.size == 0 and seg_infos_test_np.size > 0 : main_logger.error(f"Test seg_crops empty but input was not.")
    
    # --- Phase 3: Embedding (ONCE for train, ONCE for test) ---
    main_logger.info(f"--- Phase 3: Embedding (Training & Test Segments) for {dataset_name.upper()} ---")
    torch_device_for_emb = torch.device(initial_config['device'])
    dino_processor, dino_model = init_dino(initial_config['dino_model'], torch_device_for_emb)
    target_model_resnet = init_target_model(torch_device_for_emb) if initial_config['embedding_type'] == 'combined' else None
    train_emb_client_id_cache = f"central_train_emb_{initial_config['run_id']}"
    embeddings_train_segments = compute_final_embeddings(seg_infos_train_np_with_crops, images_train_rgb_list, masks_train_per_image_list, dino_processor, dino_model, target_model_resnet, torch_device_for_emb, initial_config, client_id=train_emb_client_id_cache)
    if embeddings_train_segments is None or embeddings_train_segments.shape[0] == 0: main_logger.error(f"Embedding failed for train segments ({dataset_name})!"); return
    embeddings_test_segments = None
    if seg_infos_test_np_with_crops.size > 0:
        test_emb_client_id_cache = f"central_test_emb_{initial_config['run_id']}"
        embeddings_test_segments = compute_final_embeddings(seg_infos_test_np_with_crops, images_test_rgb_list, masks_test_per_image_list, dino_processor, dino_model, target_model_resnet, torch_device_for_emb, initial_config, client_id=test_emb_client_id_cache)
        if embeddings_test_segments is None or embeddings_test_segments.shape[0] == 0: main_logger.warning(f"Embedding failed for test segments ({dataset_name}), or no test segments with crops.")
    
    # Store these base pre-computed arrays in the config to pass to run_single_configuration
    initial_config['_temp_embeddings_train_segments'] = embeddings_train_segments
    initial_config['_temp_seg_infos_train_np_with_crops'] = seg_infos_train_np_with_crops
    initial_config['_temp_images_train_rgb_list'] = images_train_rgb_list
    initial_config['_temp_y_train_labels'] = y_train_labels
    initial_config['_temp_train_base_ids'] = train_base_ids
    initial_config['_temp_embeddings_test_segments'] = embeddings_test_segments
    initial_config['_temp_seg_infos_test_np_with_crops'] = seg_infos_test_np_with_crops
    initial_config['_temp_images_test_rgb_list'] = images_test_rgb_list
    initial_config['_temp_y_test_labels_final'] = y_test_labels_final # This is y_true for test
    initial_config['_temp_test_base_ids'] = test_base_ids

    # --- HYPERPARAMETER SWEEP LOOP for K-Means, Detectors, and FIGS ---
    num_clusters_sweep = initial_config.get("sweep_num_clusters")
    min_detector_score_sweep = initial_config.get("sweep_min_detector_score")
    figs_max_rules_values = initial_config.get("figs_max_rules_sweep")
    figs_max_trees_values = initial_config.get("figs_max_trees_sweep")
    figs_max_features_values = initial_config.get("figs_max_features_sweep")
    
    sweep_results_summary = []
    best_overall_figs_accuracy = -1.0
    best_overall_params = {} 

    for k_clusters in num_clusters_sweep:
        for det_score_thresh in min_detector_score_sweep:
            current_sweep_config = initial_config.copy() # Start with base config for this sub-run
            current_sweep_config['num_clusters'] = k_clusters
            current_sweep_config['min_detector_score'] = det_score_thresh
            
            # This call performs K-Means, Detector Training, Vectorization for the current k_clusters & det_score_thresh
            X_train_c, X_test_c, lr_sanity_acc, rf_sanity_acc, num_actual_concepts = run_single_configuration(current_sweep_config, main_logger)

            if X_train_c is None or num_actual_concepts == 0 or \
               (X_test_c is None and y_test_labels_final is not None and y_test_labels_final.size > 0) : 
                main_logger.warning(f"Concept vector generation yielded no/invalid concepts for k={k_clusters}, ds={det_score_thresh:.2f}. Skipping FIGS for this combo.")
                sweep_results_summary.append({
                    "num_clusters": k_clusters, "min_detector_score": f"{det_score_thresh:.2f}",
                    "lr_sanity_acc": f"{lr_sanity_acc:.4f}" if lr_sanity_acc != -1.0 else "N/A", 
                    "rf_sanity_acc": f"{rf_sanity_acc:.4f}" if rf_sanity_acc != -1.0 else "N/A",
                    "figs_max_rules": "N/A", "figs_max_trees": "N/A", "figs_max_features": "N/A",
                    "figs_accuracy": -1.0, "num_actual_concepts": num_actual_concepts if num_actual_concepts is not None else 0
                })
                continue
            
            # --- Nested FIGS Hyperparameter Sweep ---
            figs_feature_names_current_sweep = [f"concept_{i}" for i in range(num_actual_concepts)]
            # y_train_labels is already prepared and available from initial_config['_temp_y_train_labels']
            df_train_concepts_for_figs = pd.DataFrame(X_train_c, columns=figs_feature_names_current_sweep)

            for figs_rules in figs_max_rules_values:
                for figs_trees in figs_max_trees_values:
                    for figs_feats in figs_max_features_values:
                        current_figs_specific_params = initial_config['figs_params_base'].copy()
                        current_figs_specific_params.update({
                            "max_rules": figs_rules, "max_trees": figs_trees, "max_features": figs_feats
                        })
                        main_logger.info(f"Training AutoCore (k={k_clusters}, ds={det_score_thresh:.2f}) with params: {current_figs_specific_params}")
                        
                        figs_model_current = PatchedFIGSClassifier(
                            **current_figs_specific_params,
                            n_outputs_global=initial_config['num_classes'] 
                        )
                        try:
                            figs_model_current.fit(df_train_concepts_for_figs, y_train_labels, # y_train_labels is correct here
                                               feature_names=figs_feature_names_current_sweep, _y_fit_override=None)
                            
                            figs_accuracy_current_run = -1.0
                            if X_test_c is not None and X_test_c.shape[0] > 0 and y_test_labels_final.shape[0] == X_test_c.shape[0]:
                                y_pred_figs_current = figs_model_current.predict(X_test_c)
                                figs_accuracy_current_run = accuracy_score(y_test_labels_final, y_pred_figs_current)
                                main_logger.info(f"    FIGS Test Acc: {figs_accuracy_current_run:.4f} (k={k_clusters}, ds={det_score_thresh:.2f}, figs={current_figs_specific_params})")
                            
                            if X_test_c.shape[0] > 0 and y_test_labels_final.shape[0] == X_test_c.shape[0]:
                                accuracy, rule_prec, rule_fid = calculate_metrics(
                                    figs_model_current, X_test_c, y_test_labels_final, figs_feature_names_current_sweep, main_logger)
                                main_logger.info(f"FIGS Model Test Accuracy: {accuracy:.4f}, Mean Rule Precision: {rule_prec:.4f}, Fidelity: {rule_fid:.4f}")
                            else:
                                main_logger.warning("Skipping final evaluation: No test concept data or label mismatch.")
                            sweep_results_summary.append({
                                "num_clusters": k_clusters, "min_detector_score": f"{det_score_thresh:.2f}",
                                "lr_sanity_acc": f"{lr_sanity_acc:.4f}" if lr_sanity_acc != -1.0 else "N/A", 
                                "rf_sanity_acc": f"{rf_sanity_acc:.4f}" if rf_sanity_acc != -1.0 else "N/A",
                                "figs_max_rules": figs_rules, "figs_max_trees": figs_trees if figs_trees is not None else "None", 
                                "figs_max_features": figs_feats if figs_feats is not None else "None",
                                "figs_accuracy": f"{figs_accuracy_current_run:.4f}" if figs_accuracy_current_run != -1.0 else "N/A",
                                "num_actual_concepts": num_actual_concepts
                            })

                            if figs_accuracy_current_run > best_overall_figs_accuracy:
                                best_overall_figs_accuracy = figs_accuracy_current_run
                                best_overall_params = {
                                    "num_clusters": k_clusters, "min_detector_score": det_score_thresh,
                                    **current_figs_specific_params
                                }
                        except Exception as e_figs_fit_sweep:
                            main_logger.error(f"    Error during FIGS fit/eval for params {current_figs_specific_params}: {e_figs_fit_sweep}", exc_info=True)
                            sweep_results_summary.append({
                                "num_clusters": k_clusters, "min_detector_score": f"{det_score_thresh:.2f}",
                                "lr_sanity_acc": f"{lr_sanity_acc:.4f}" if lr_sanity_acc != -1.0 else "N/A", 
                                "rf_sanity_acc": f"{rf_sanity_acc:.4f}" if rf_sanity_acc != -1.0 else "N/A",
                                "figs_max_rules": figs_rules, "figs_max_trees": figs_trees if figs_trees is not None else "None", 
                                "figs_max_features": figs_feats if figs_feats is not None else "None",
                                "figs_accuracy": "ERROR", "error_msg": str(e_figs_fit_sweep)[:100],
                                "num_actual_concepts": num_actual_concepts
                            })
    if sweep_results_summary:
        sweep_summary_df = pd.DataFrame(sweep_results_summary)
        sweep_summary_path = os.path.join(initial_config['log_dir'], f"autocore_hyperparam_sweep_summary_{dataset_name}.csv")
        try: sweep_summary_df.to_csv(sweep_summary_path, index=False)
        except Exception as e_csv: main_logger.error(f"Failed to save sweep summary CSV: {e_csv}")
        main_logger.info(f"Full hyperparameter sweep summary saved to {sweep_summary_path}")
        main_logger.info(f"Best Overall FIGS Accuracy achieved: {best_overall_figs_accuracy:.4f} with params: {best_overall_params}")
        best_params_path = os.path.join(initial_config['log_dir'], f"best_hyperparams_{dataset_name}.json")
        with open(best_params_path, 'w') as f_params: json.dump({"best_accuracy": best_overall_figs_accuracy, "best_parameters": best_overall_params}, f_params, indent=4)
        main_logger.info(f"Best hyperparameters saved to {best_params_path}")
    else: main_logger.warning("No sweep results were generated.")
    main_logger.info(f"======== Centralized AutoCoRe SWEEP ({dataset_name.upper()}) Run ID: {initial_config['run_id']} Complete ========")

if __name__ == "__main__":
    main_centralized_autocore()