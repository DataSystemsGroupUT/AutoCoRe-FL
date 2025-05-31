# run_cent_multidataset.py (Revised for pre-prepared ADE20K-like datasets)

import logging
import os
import random
import sys
import yaml
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from tqdm import tqdm
import pandas as pd
import time
import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# import seaborn as sns # Not used
from collections import Counter, defaultdict
import cv2
from scipy.special import expit, softmax
import operator
import re
from typing import List, Tuple, Dict

# --- Project Imports ---
# Assuming these are in PYTHONPATH or accessible relative to where this script is run
# For ADE20K (original sceneCategories.txt format)
from AutoCore_FL.data.ade20k_parition import load_scene_categories as load_ade20k_original_scene_categories
from AutoCore_FL.segmentation.sam_loader import load_sam_model
from AutoCore_FL.segmentation.segment_crops import generate_segments_and_masks
from AutoCore_FL.embedding.dino_loader import init_dino, init_target_model
from AutoCore_FL.embedding.compute_embeddings import compute_final_embeddings
from AutoCore_FL.concepts.detector import train_concept_detector
from AutoCore_FL.federated.client import PatchedFIGSClassifier # LocalNode is defined within client.py
from AutoCore_FL.federated.utils import setup_logging, SAM2Filter
from AutoCore_FL.scripts.visualize_utils import (
    visualize_embedding_tsne, 
    visualize_cluster_segments_from_data,
    visualize_concept_vs_random_six,
    visualize_decision_explanation, 
    get_critical_active_concepts_from_figs,
    save_plot
)
# SAM and on-the-fly segment_crops are NOT needed if data is fully pre-prepared by another script
# from federated_logic_xai_figs_svm.segmentation.sam_loader import load_sam_model
# from federated_logic_xai_figs_svm.segmentation.segment_crops import generate_segments_and_masks as generate_sam_segments_and_masks_cached

# --- Visualization Helpers (from run_cent.py) ---
# (Keep find_segments_for_concept_in_image, get_random_segments_from_dataset, visualize_centralized_decision_for_paper as before)
def find_segments_for_concept_in_image(
    image_idx_in_split, target_concept_original_kmeans_idx, trained_concept_detectors_map,
    seg_infos_flat_for_split, embeddings_flat_for_split, max_segments_to_return=2
):
    activating_segment_crops = []
    if target_concept_original_kmeans_idx not in trained_concept_detectors_map: return []
    detector_pipeline, detector_threshold = trained_concept_detectors_map[target_concept_original_kmeans_idx]
    candidate_embeddings_list = []
    original_seg_info_for_candidates = []
    for global_seg_idx, seg_info in enumerate(seg_infos_flat_for_split):
        if seg_info.get('img_idx') == image_idx_in_split: # seg_info['img_idx'] should be local to the split
            # Check if embedding exists for this global_seg_idx AND seg_crop_bgr exists in seg_info
            if global_seg_idx < len(embeddings_flat_for_split) and embeddings_flat_for_split[global_seg_idx] is not None and \
               seg_info.get('seg_crop_bgr') is not None:
                candidate_embeddings_list.append(embeddings_flat_for_split[global_seg_idx])
                original_seg_info_for_candidates.append(seg_info)
            # else:
                # print(f"Debug: Skip seg {global_seg_idx} for img {image_idx_in_split}. Emb: {embeddings_flat_for_split[global_seg_idx] is not None}, Crop: {seg_info.get('seg_crop_bgr') is not None}")
    if not candidate_embeddings_list: return []
    candidate_embeddings_np = np.array(candidate_embeddings_list)
    try:
        if hasattr(detector_pipeline, "predict_proba"): probs = detector_pipeline.predict_proba(candidate_embeddings_np)[:, 1]
        elif hasattr(detector_pipeline, "decision_function"): probs = expit(detector_pipeline.decision_function(candidate_embeddings_np))
        else: return []
    except Exception as e_pred_viz:
        # print(f"Debug: Predict error in find_segments: {e_pred_viz}")
        return []
    activating_indices = np.where(probs >= detector_threshold)[0]
    for idx_in_candidates in activating_indices[:max_segments_to_return]:
        seg_info_for_crop = original_seg_info_for_candidates[idx_in_candidates]
        activating_segment_crops.append(seg_info_for_crop.get('seg_crop_bgr'))
    return [c for c in activating_segment_crops if c is not None]

def get_random_segments_from_dataset(
    all_segment_infos_in_split, num_random_segments=2, exclude_image_idx=None
):
    candidate_indices = [i for i, si in enumerate(all_segment_infos_in_split) if si.get('seg_crop_bgr') is not None and (exclude_image_idx is None or si.get('img_idx') != exclude_image_idx)]
    if not candidate_indices: return []
    num_to_sample = min(num_random_segments, len(candidate_indices))
    if num_to_sample == 0: return []
    return [all_segment_infos_in_split[i].get('seg_crop_bgr') for i in np.random.choice(candidate_indices, num_to_sample, replace=False)]

def visualize_centralized_decision_for_paper(
    target_image_rgb_np, image_idx_in_test_split, predicted_class_name_str, image_concept_vector_np,
    figs_model_instance, ordered_final_concept_original_ids, trained_concept_detectors_map_paper_viz,
    seg_infos_test_flat_paper_viz, embeddings_test_flat_paper_viz, feature_names_for_figs_paper_viz,
    config_paper_viz, main_logger_paper_viz, target_num_top_concept_segments_from_image=2,
    target_num_random_segments_from_dataset=2, output_filename_stem="paper_viz_decision"
):
    main_logger_paper_viz.info(f"PaperViz: img_idx {image_idx_in_test_split}, pred: {predicted_class_name_str}")
    fig_orig, ax_orig = plt.subplots(1, 1, figsize=(config_paper_viz.get("viz_figsize_orig",(6,6))))
    ax_orig.imshow(target_image_rgb_np); ax_orig.axis('off')
    save_plot(fig_orig, f"{output_filename_stem}_img{image_idx_in_test_split}_original", config_paper_viz, main_logger_paper_viz)
    
    panel_segment_crops = [None] * 4; current_panel_idx = 0
    critical_dense_indices = get_critical_active_concepts_from_figs(figs_model_instance, image_concept_vector_np, feature_names_for_figs_paper_viz)
    num_concept_segments_found = 0
    if critical_dense_indices:
        for dense_idx in critical_dense_indices:
            if num_concept_segments_found >= target_num_top_concept_segments_from_image: break
            if not (0 <= dense_idx < len(ordered_final_concept_original_ids)): continue
            original_kmeans_idx = ordered_final_concept_original_ids[dense_idx]
            # Ensure embeddings_test_flat_paper_viz is not None and has data
            if embeddings_test_flat_paper_viz is None or embeddings_test_flat_paper_viz.shape[0] == 0:
                main_logger_paper_viz.warning("PaperViz: embeddings_test_flat_paper_viz is empty, cannot find concept segments.")
                break
            crops = find_segments_for_concept_in_image(image_idx_in_test_split, original_kmeans_idx, trained_concept_detectors_map_paper_viz, seg_infos_test_flat_paper_viz, embeddings_test_flat_paper_viz, 1)
            if crops and crops[0] is not None and current_panel_idx < 4:
                panel_segment_crops[current_panel_idx] = crops[0]; current_panel_idx += 1; num_concept_segments_found += 1
    num_random_needed = 4 - current_panel_idx
    if num_random_needed > 0:
        random_crops = get_random_segments_from_dataset(seg_infos_test_flat_paper_viz, num_random_needed, image_idx_in_test_split)
        for rand_crop in random_crops:
            if current_panel_idx < 4 and rand_crop is not None: panel_segment_crops[current_panel_idx] = rand_crop; current_panel_idx += 1
    
    if not any(c is not None for c in panel_segment_crops): 
        main_logger_paper_viz.warning(f"PaperViz: No segments to plot for panel img {image_idx_in_test_split}.")
        return
    
    fig_segments_panel, axes_segments_panel = plt.subplots(2, 2, figsize=config_paper_viz.get("viz_figsize_panel",(5,5)), squeeze=False)
    axes_flat_panel = axes_segments_panel.flatten()
    for i_panel, crop_bgr_panel in enumerate(panel_segment_crops):
        ax_panel = axes_flat_panel[i_panel]
        if crop_bgr_panel is not None: ax_panel.imshow(cv2.cvtColor(crop_bgr_panel, cv2.COLOR_BGR2RGB))
        else: ax_panel.text(0.5,0.5, "N/A", ha='center',va='center', fontsize=8, color='grey')
        ax_panel.axis('off')
    plt.tight_layout(pad=0.1)
    save_plot(fig_segments_panel, f"{output_filename_stem}_img{image_idx_in_test_split}_segment_panel_2x2", config_paper_viz, main_logger_paper_viz)


# --- build_centralized_concept_vectors_maxpool (from run_cent.py) ---
def build_centralized_concept_vectors_maxpool(
    segment_infos_split_np, embeddings_for_segments_in_split_np, trained_concept_detectors_dict,
    ordered_final_concept_original_ids, num_total_images_in_split, image_base_ids_in_split,
    config_dict, main_logger_passed
):
    logger_main = main_logger_passed
    logger_main.info(f"Vectorizing {num_total_images_in_split} images using max-pool for {config_dict.get('dataset_name', 'UnknownSet')}...")
    num_final_figs_features = len(ordered_final_concept_original_ids)
    if num_final_figs_features == 0: return np.empty((num_total_images_in_split, 0)), [] # Return correct shape for empty features
    
    # Check if segment_infos_split_np has 'base_id', if not, cannot use base_id_to_split_img_idx mapping logic
    # This happens if segment_infos were directly from generate_sam_segments which might not populate 'base_id'
    # For reused NPY, 'base_id' should be there.
    has_base_id_in_seg_infos = False
    if segment_infos_split_np.size > 0 and isinstance(segment_infos_split_np[0], dict) and 'base_id' in segment_infos_split_np[0]:
        has_base_id_in_seg_infos = True
        base_id_to_split_img_idx = {bid: i for i, bid in enumerate(image_base_ids_in_split)}

    split_img_idx_to_global_seg_indices = defaultdict(list)
    for global_seg_idx, seg_info in enumerate(segment_infos_split_np):
        if has_base_id_in_seg_infos:
            bid = seg_info.get('base_id')
            if bid in base_id_to_split_img_idx:
                split_img_idx_to_global_seg_indices[base_id_to_split_img_idx[bid]].append(global_seg_idx)
        else: # If no base_id, assume 'img_idx' in seg_info is already the local split_img_idx
            local_img_idx_from_seg = seg_info.get('img_idx')
            if local_img_idx_from_seg is not None and 0 <= local_img_idx_from_seg < num_total_images_in_split:
                 split_img_idx_to_global_seg_indices[local_img_idx_from_seg].append(global_seg_idx)

    concept_vecs_for_split = np.zeros((num_total_images_in_split, num_final_figs_features), dtype=np.float32)
    for dense_feature_idx, original_kmeans_idx in tqdm(enumerate(ordered_final_concept_original_ids), desc=f"Vectorizing ({config_dict.get('dataset_name')})", total=num_final_figs_features, file=sys.stdout, disable=not config_dict.get("tqdm_enabled", True)):
        if original_kmeans_idx not in trained_concept_detectors_dict: continue
        model_pipeline, optimal_threshold = trained_concept_detectors_dict[original_kmeans_idx]
        if embeddings_for_segments_in_split_np is None or embeddings_for_segments_in_split_np.shape[0] == 0: continue
        try:
            if hasattr(model_pipeline, "predict_proba"): all_seg_probs = model_pipeline.predict_proba(embeddings_for_segments_in_split_np)[:, 1]
            elif hasattr(model_pipeline, "decision_function"): all_seg_probs = expit(model_pipeline.decision_function(embeddings_for_segments_in_split_np))
            else: continue
        except Exception: continue
        for split_img_idx in range(num_total_images_in_split):
            seg_indices_for_img = split_img_idx_to_global_seg_indices.get(split_img_idx, [])
            if not seg_indices_for_img: continue
            # Ensure indices are valid for all_seg_probs
            valid_seg_indices_for_probs = [idx for idx in seg_indices_for_img if idx < len(all_seg_probs)]
            if not valid_seg_indices_for_probs: continue
            probs_this_img = all_seg_probs[valid_seg_indices_for_probs]
            if probs_this_img.size > 0 and np.max(probs_this_img) >= optimal_threshold:
                concept_vecs_for_split[split_img_idx, dense_feature_idx] = 1.0
    return concept_vecs_for_split, list(image_base_ids_in_split)

# --- add_seg_crop_bgr_to_split_infos (from run_cent.py) ---
def add_seg_crop_bgr_to_split_infos(
    segment_infos_split, images_rgb_split_list, masks_per_image_split_list, main_logger_passed_crop,
    min_mask_pixels_for_crop=100
):
    processed_infos = []
    if masks_per_image_split_list is None: # Handle case where masks are not provided (e.g. from NPY)
        main_logger_passed_crop.warning("add_seg_crop_bgr: masks_per_image_split_list is None. "
                                        "Cannot generate crops if 'seg_crop_bgr' is missing from segment_infos.")
        # If 'seg_crop_bgr' is already in segment_infos_split (e.g. from NPY), this loop will just pass them through.
        # Otherwise, it will result in None for 'seg_crop_bgr'.
    
    for seg_info_item in tqdm(segment_infos_split, desc="Verifying/Preparing seg_crop_bgr", file=sys.stdout, disable=not main_logger_passed_crop.isEnabledFor(logging.DEBUG)):
        if 'seg_crop_bgr' in seg_info_item and seg_info_item['seg_crop_bgr'] is not None:
            processed_infos.append(seg_info_item); continue
        
        if masks_per_image_split_list is None: # Cannot generate if masks missing
            new_seg_info_item = dict(seg_info_item); new_seg_info_item['seg_crop_bgr'] = None
            processed_infos.append(new_seg_info_item); continue

        new_seg_info_item = dict(seg_info_item)
        try:
            local_img_idx = new_seg_info_item['img_idx']; seg_idx_in_img = new_seg_info_item['seg_idx']
            if not (0 <= local_img_idx < len(images_rgb_split_list) and \
                    0 <= local_img_idx < len(masks_per_image_split_list) and \
                    masks_per_image_split_list[local_img_idx] is not None and \
                    0 <= seg_idx_in_img < len(masks_per_image_split_list[local_img_idx])):
                new_seg_info_item['seg_crop_bgr'] = None; processed_infos.append(new_seg_info_item); continue
            
            original_img_rgb = images_rgb_split_list[local_img_idx]
            seg_mask_bool = masks_per_image_split_list[local_img_idx][seg_idx_in_img]

            if not isinstance(seg_mask_bool, np.ndarray) or seg_mask_bool.sum() < min_mask_pixels_for_crop:
                 new_seg_info_item['seg_crop_bgr'] = None; processed_infos.append(new_seg_info_item); continue
            
            ys, xs = np.where(seg_mask_bool)
            if len(ys) == 0: new_seg_info_item['seg_crop_bgr'] = None; processed_infos.append(new_seg_info_item); continue
            
            top, left, bottom, right = np.min(ys), np.min(xs), np.max(ys), np.max(xs)
            # Ensure crop dimensions are valid
            if top >= bottom or left >= right:
                 new_seg_info_item['seg_crop_bgr'] = None; processed_infos.append(new_seg_info_item); continue

            seg_crop_rgb_temp = original_img_rgb[top:bottom+1, left:right+1].copy()
            local_mask_temp = seg_mask_bool[top:bottom+1, left:right+1]
            seg_crop_rgb_temp[~local_mask_temp] = (0,0,0)
            new_seg_info_item['seg_crop_bgr'] = cv2.cvtColor(seg_crop_rgb_temp, cv2.COLOR_RGB2BGR)
        except Exception as e_crop_add: 
            main_logger_passed_crop.debug(f"Minor error creating seg_crop_bgr: {e_crop_add}")
            new_seg_info_item['seg_crop_bgr'] = None
        processed_infos.append(new_seg_info_item)
    return np.array(processed_infos, dtype=object)


# --- evaluate_centralized_figs_model & figs_lr_xfl_metrics (from run_cent.py, already provided) ---
# ... (These function definitions are assumed to be available from the previous context or imported) ...
from experiments.AutoCore_FL.scripts.run_autocore_cent_auto_ade20k import evaluate_centralized_AutoCore_model, figs_lr_xfl_metrics


# --- Config Generation for Multi-Dataset ---
def generate_centralized_multidataset_config(
    dataset_name: str = "ade20k", 
    run_id_base: str = "cent_multidataset_run"
):
    METHOD_NAME = f"FIGS_Centralized_{dataset_name.upper()}_ReusedNPY" # Indicate NPY usage
    seed = 42
    import os
    script_dir = os.getcwd()
    sam_cfg_path = "configs/sam2.1/sam2.1_hiera_t.yaml" # Relative to project root or ensure absolute
    sam_ckpt_path = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/sam2.1_hiera_tiny.pt"
    config = {
        "dataset_name": dataset_name,
        "seed": seed, "test_split_ratio": 0.2,
        "dino_model": "facebook/dinov2-base", "embedding_type": "dino_only", "embedding_dim": 768,
        "num_clusters": 100, "min_samples_per_concept_cluster": 30, # Adjusted for potentially smaller SUNRGBD subset
        "detector_type": "lr", "detector_min_samples_per_class": 10, # Adjusted
        "detector_cv_splits": 3, "pca_n_components": 128, 
        "lr_max_iter": 5000, "min_detector_score": 0.65, # Adjusted
        "vectorizer_strategy": "max_pool",
        "figs_params": {"max_rules": 45, "min_impurity_decrease": 0.0, "max_trees": 5, 'max_features':None, "random_state": seed}, # Example
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "method_name": METHOD_NAME,
        "plot_dpi": 100,
        "tqdm_enabled": True,
        "perform_on_the_fly_segmentation_if_npy_missing": True, # Set to False as we rely on pre-prepared NPYs
        "viz_figsize_orig": (4,4), "viz_figsize_panel": (3,3), # Smaller viz for speed,
        "sam_cfg": sam_cfg_path, "sam_ckpt": sam_ckpt_path,
    }

    if dataset_name == "ade20k":
        config["chosen_classes"] = ['street', 'bedroom', 'living_room', 'bathroom', 'kitchen', 
            'skyscraper', 'highway', 'conference_room', 'mountain_snowy', 'office',
            'corridor', 'airport_terminal', 'attic', 'mountain', 'park', 'coast', 
            'alley','beach', 'childs_room', 'art_gallery','castle', 'dorm_room', 
            'nursery', 'lobby', 'reception', 'bar', 'house', 'bridge', 'classroom']
        config["data_root_for_npy"] = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/" # Path to logic_ade20k_...xx.npy files
        config["scene_cat_file_for_labels"] = "/gpfs/helios/home/soliman/logic_explained_networks/data/ade20k/ADEChallengeData2016/sceneCategories.txt"
    elif dataset_name == "sunrgbd":
        config["chosen_classes"] = sorted(["bathroom", "bedroom", "bookstore"])
        # This is the root of your SUNRGBD_subset_3scenes_output (where attributes.npy etc. are)
        config["data_root_for_npy"] = "/gpfs/helios/home/soliman/logic_explained_networks/data/sun_final/" # !!! USER: VERIFY THIS PATH !!!
        config["scene_cat_file_for_labels"] = os.path.join(config["data_root_for_npy"], "sceneCategories.txt")
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    config["num_classes"] = len(config["chosen_classes"])
    
    effective_run_id = f"{run_id_base}_{dataset_name}"
    config["run_id"] = effective_run_id
    base_dir = os.path.join(script_dir, f"experiment_results_centralized/{METHOD_NAME.lower()}_run_{effective_run_id}")
    os.makedirs(base_dir, exist_ok=True)
    log_dir_path = os.path.join(base_dir, f"logs_{dataset_name}")
    os.makedirs(log_dir_path, exist_ok=True)
    config["log_dir"] = log_dir_path
    
    cache_root_dir = os.path.join(script_dir, "cache_centralized_runs_multidataset_reusednpy")
    config["central_run_cache_dir"] = os.path.join(cache_root_dir, f"run_{dataset_name}_{effective_run_id}")
    os.makedirs(config["central_run_cache_dir"], exist_ok=True)
    config["embedding_cache_dir"] = os.path.join(config["central_run_cache_dir"], "embedding_cache") # For compute_final_embeddings

    config["final_model_path"] = os.path.join(base_dir, f"final_centralized_figs_model_{dataset_name}.pkl")
    config["metrics_log_path"] = os.path.join(base_dir, f"final_metrics_{METHOD_NAME}.csv")
    
    config["use_seg_crops_cache"] = True; config["use_kmeans_cache"] = True
    config["use_detectors_cache"] = True; config["use_train_vectors_cache"] = True
    config["use_test_vectors_cache"] = True; config["use_embedding_cache"] = True  

    config["generate_paper_visualizations"] = False # Default to False
    config["num_paper_viz_samples"] = 3

    return config


# --- Main Centralized Logic ---
def main_centralized_multidataset(config):
    main_logger = logging.getLogger(f"MainCent_{config['dataset_name']}_{config['run_id']}") # Unique logger per run
    main_logger.propagate = False # Prevent duplicate logs if root logger is also configured
    if not main_logger.handlers: # Add handlers only if not already configured (e.g. by outer script)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        # File handler
        fh = logging.FileHandler(os.path.join(config['log_dir'], f"main_log_{config['run_id']}.log"), mode='a')
        fh.setFormatter(formatter)
        main_logger.addHandler(fh)
        # Stream handler (console)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        main_logger.addHandler(sh)
        main_logger.setLevel(logging.INFO) # Or from config
        #main_logger.addFilter(SAM2Filter())


    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda' and torch.cuda.is_available(): torch.cuda.manual_seed_all(config['seed'])
    main_logger.info(f"Using device: {config['device']}")
    torch_device = torch.device(config['device'])

    main_logger.info(f"======== Starting Centralized FIGS ({config['dataset_name']}) - Run ID: {config['run_id']} ========")
    main_logger.info(f"Full Config (first few items): { {k:v for i,(k,v) in enumerate(config.items()) if i < 10} } ...")
# Inside main_centralized_multidataset(config) in run_cent_multidataset.py

 # --- Phase 1: Data Loading (Try NPY, then On-the-Fly from PREPARED subset) ---
    main_logger.info(f"--- Phase 1: Loading/Preparing Data for {config['dataset_name']} ---")
    
    full_dataset_all_masks = None
    full_dataset_all_images_rgb = None
    full_dataset_segment_infos_with_crops = None 
    loaded_from_npy = False # This flag is for segment-level NPYs like logic_ade20k_...

    if config["dataset_name"] == "ade20k":
        # ... (ADE20K NPY loading logic remains the same) ...
        data_root_npy = config.get("data_root_for_npy") 
        if data_root_npy and os.path.exists(data_root_npy):
            try:
                path_masks = os.path.join(data_root_npy, "logic_ade20k_all_masksxx.npy")
                path_images = os.path.join(data_root_npy, "logic_ade20k_all_imagesxx.npy")
                path_seg_infos = os.path.join(data_root_npy, "logic_ade20k_segment_infosxx.npy")
                
                if all(os.path.exists(p) for p in [path_masks, path_images, path_seg_infos]):
                    full_dataset_all_masks = np.load(path_masks, allow_pickle=True)
                    full_dataset_all_images_rgb = np.load(path_images, allow_pickle=True)
                    full_dataset_segment_infos_with_crops = np.load(path_seg_infos, allow_pickle=True) 
                    main_logger.info(f"ADE20K: Loaded pre-generated data from NPY: {len(full_dataset_all_images_rgb) if full_dataset_all_images_rgb is not None else 0} images, {len(full_dataset_segment_infos_with_crops) if full_dataset_segment_infos_with_crops is not None else 0} total segments.")
                    loaded_from_npy = True
                else:
                    main_logger.info("ADE20K: Some NPY files missing for reused segments path.")
            except Exception as e:
                main_logger.warning(f"ADE20K: Error loading NPY files: {e}.")
        else:
            main_logger.info("ADE20K: data_root_for_npy not configured for reused segments path.")


    # If not loaded from segment-level NPYs, AND on-the-fly is enabled:
    if not loaded_from_npy and config.get("perform_on_the_fly_segmentation_if_npy_missing", False):
        main_logger.info(f"No segment-level NPYs found or specified. Proceeding with on-the-fly segmentation for {config['dataset_name']}...")
        
        raw_image_paths_for_segmentation = [] # List of (base_id, full_path_to_image_for_sam)

        if config["dataset_name"] == "ade20k":
            # ... (ADE20K raw image path gathering logic - same as before) ...
            temp_scene_map_ade = load_ade20k_original_scene_categories(config["scene_cat_file_for_labels"])
            from AutoCore_FL.data.ade20k_parition import get_filtered_image_paths
            raw_image_paths_for_segmentation.extend(get_filtered_image_paths(
                ade_root=config["ade20k_root"], scene_map=temp_scene_map_ade,
                chosen_classes=config["chosen_classes"], subset="training"
            ))
            raw_image_paths_for_segmentation.extend(get_filtered_image_paths(
                ade_root=config["ade20k_root"], scene_map=temp_scene_map_ade,
                chosen_classes=config["chosen_classes"], subset="validation"
            ))
            if not raw_image_paths_for_segmentation:
                main_logger.error("ADE20K (OTF): No raw image paths found. Exiting."); return

        elif config["dataset_name"] == "sunrgbd":
            # For SUNRGBD on-the-fly, we use images from the output of `prepare_sunrgbd_subset_scenes.py`
            # which is config["data_root_for_npy"] for SUNRGBD.
            sunrgbd_prepared_images_root = os.path.join(config["data_root_for_npy"], "images")
            if not os.path.isdir(sunrgbd_prepared_images_root):
                main_logger.error(f"SUNRGBD (OTF): Prepared image directory not found: {sunrgbd_prepared_images_root}. "
                                  "Run prepare_sunrgbd_subset_scenes.py first. Exiting.")
                return

            main_logger.info(f"SUNRGBD (OTF): Gathering image paths from prepared subset: {sunrgbd_prepared_images_root}")
            for scene_folder_name in config["chosen_classes"]: # e.g., "bathroom", "bedroom", "bookstore"
                scene_folder_path_actual = os.path.join(sunrgbd_prepared_images_root, scene_folder_name.replace(" ", "_").replace("/", "_"))
                if os.path.isdir(scene_folder_path_actual):
                    for img_file in os.listdir(scene_folder_path_actual):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_full_path = os.path.join(scene_folder_path_actual, img_file)
                            # The 'base_id' for these images is their filename without extension,
                            # as used in sceneCategories_subset.txt
                            img_base_id = os.path.splitext(img_file)[0]
                            raw_image_paths_for_segmentation.append((img_base_id, img_full_path))
            
            if not raw_image_paths_for_segmentation:
                main_logger.error(f"SUNRGBD (OTF): No images found in prepared subset directory {sunrgbd_prepared_images_root} for chosen classes. Exiting."); return
            main_logger.info(f"SUNRGBD (OTF): Gathered {len(raw_image_paths_for_segmentation)} image paths from prepared subset for SAM.")

        else:
            main_logger.error(f"On-the-fly segmentation not configured for dataset: {config['dataset_name']}. Exiting."); return

        # --- Common On-the-Fly Segmentation (SAM) ---
        sam_model_otf, mask_gen_otf = load_sam_model(
            config['sam_cfg'], config['sam_ckpt'], torch_device,

        )
        
        # generate_sam_segments_and_masks_cached is your segment_crops.generate_segments_and_masks
        # It needs: list of (base_id, path_to_image_for_sam)
        segment_infos_otf, images_otf, masks_otf, _ = generate_segments_and_masks(
            raw_image_paths_for_segmentation, 
            mask_gen_otf, config, 
            client_id=f"central_{config['dataset_name']}_otf_seg"
        )
        
        if not segment_infos_otf or not images_otf:
             main_logger.error(f"On-the-fly SAM segmentation for {config['dataset_name']} failed to produce segment_infos or images. Exiting."); return

        # The segment_infos_otf should already contain 'seg_crop_bgr'.
        # 'img_idx' in segment_infos_otf is local to the images_otf list.
        # 'base_id' in segment_infos_otf is what was passed in raw_image_paths_for_segmentation.
        full_dataset_segment_infos_with_crops = segment_infos_otf 
        full_dataset_all_images_rgb = images_otf
        full_dataset_all_masks = masks_otf 
        main_logger.info(f"{config['dataset_name']} (OTF): SAM segmentation complete. "
                         f"{len(full_dataset_segment_infos_with_crops)} segments from {len(full_dataset_all_images_rgb)} images.")
    
    elif not loaded_from_npy: # NPYs were expected (or on-the-fly disabled) but not found
        main_logger.error(f"No data loaded for {config['dataset_name']}. Pre-generated NPY files not found, and on-the-fly processing disabled or failed. Exiting.")
        return

    # --- Phase 2: Prepare Image Labels and Split Data (COMMON LOGIC) ---
    # This section should now work correctly for both ADE20K (from NPY) 
    # and SUNRGBD (from on-the-fly SAM on prepared subset images)
    main_logger.info(f"--- Phase 2: Preparing Labels and Splitting Data for {config['dataset_name']} ---")
    
    scene_map_full = {} # Maps base_id (from segment_infos) to scene_name
    if os.path.exists(config["scene_cat_file_for_labels"]):
        with open(config["scene_cat_file_for_labels"], 'r') as f_sc:
            for line in f_sc:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2: scene_map_full[parts[0]] = parts[1]
        main_logger.info(f"Loaded scene map for {config['dataset_name']} from: {config['scene_cat_file_for_labels']}")
    else:
        main_logger.error(f"Scene category file for labels not found: {config['scene_cat_file_for_labels']}. Cannot map base_ids to scenes. Exiting.")
        return

    sorted_chosen_classes_for_mapping = sorted(config["chosen_classes"])
    scene_to_global_idx_map = {s: i for i, s in enumerate(sorted_chosen_classes_for_mapping)}
    config['sorted_chosen_classes_for_mapping'] = sorted_chosen_classes_for_mapping # Store for later use

    num_loaded_images = len(full_dataset_all_images_rgb)
    # `all_base_ids_ordered_from_segs` maps global image index (0 to N-1) to its base_id.
    # This base_id comes from the 'base_id' field in `full_dataset_segment_infos_with_crops`.
    all_base_ids_ordered_from_segs = [None] * num_loaded_images 
    
    # Correctly build the map from global image index to its *first encountered* base_id from segments
    img_idx_to_first_base_id = {}
    for seg_info in full_dataset_segment_infos_with_crops:
        img_idx_global = seg_info.get('img_idx') # This is global index into full_dataset_all_images_rgb
        base_id_from_seg = seg_info.get('base_id')
        if img_idx_global is not None and base_id_from_seg is not None:
            if img_idx_global not in img_idx_to_first_base_id:
                 img_idx_to_first_base_id[img_idx_global] = base_id_from_seg
    
    for i in range(num_loaded_images):
        all_base_ids_ordered_from_segs[i] = img_idx_to_first_base_id.get(i)

    valid_image_indices_for_run = [] 
    base_ids_for_run = []            
    labels_for_run_list = []         

    for i_img_global in range(num_loaded_images):
        base_id_for_labeling = all_base_ids_ordered_from_segs[i_img_global]
        if base_id_for_labeling is None: 
            # This means image i_img_global had no segments in full_dataset_segment_infos_with_crops, or segments had no base_id
            # main_logger.debug(f"Image at global index {i_img_global} has no base_id from segments. Skipping for label assignment.")
            continue

        scene_name = scene_map_full.get(base_id_for_labeling) # Use the scene_map_full
        if scene_name and scene_name in config['chosen_classes']:
            label_idx = scene_to_global_idx_map.get(scene_name)
            if label_idx is not None:
                valid_image_indices_for_run.append(i_img_global) 
                base_ids_for_run.append(base_id_for_labeling) 
                labels_for_run_list.append(label_idx)
            # else: main_logger.debug(f"Scene '{scene_name}' for base_id '{base_id_for_labeling}' not in current chosen_classes map.")
        # else: main_logger.debug(f"Scene for base_id '{base_id_for_labeling}' ('{scene_name}') not in chosen_classes.")
    
    if not valid_image_indices_for_run:
        main_logger.error(f"No images from loaded data match current config's chosen_classes: {config['chosen_classes']}. "
                          f"Check scene_map and base_ids in segment_infos. Exiting for {config['dataset_name']}.")
        return
    
    labels_for_run_np = np.array(labels_for_run_list, dtype=np.int64)
    main_logger.info(f"Data Preparation ({config['dataset_name']}): {len(valid_image_indices_for_run)} images match chosen classes.")
    
    # ... (Rest of Phase 2: Splitting data into train/test using valid_image_indices_for_run)
    # ... (This includes creating images_train_rgb_list, masks_train_per_image_list, etc.)
    # ... (And re-indexing seg_infos_train_np_with_crops, seg_infos_test_np_with_crops)

    indices_to_split_from = np.arange(len(valid_image_indices_for_run)) 
    try:
        train_relative_indices, test_relative_indices = train_test_split(
            indices_to_split_from, test_size=config['test_split_ratio'],
            random_state=config['seed'], stratify=labels_for_run_np if len(np.unique(labels_for_run_np)) > 1 else None
        )
    except ValueError:
        main_logger.warning(f"Stratification failed for {config['dataset_name']}. Using random split.")
        train_relative_indices, test_relative_indices = train_test_split(
            indices_to_split_from, test_size=config['test_split_ratio'], random_state=config['seed']
        )

    train_original_global_indices = [valid_image_indices_for_run[i] for i in train_relative_indices]
    test_original_global_indices = [valid_image_indices_for_run[i] for i in test_relative_indices]

    y_train_labels = labels_for_run_np[train_relative_indices]
    train_base_ids = [base_ids_for_run[i] for i in train_relative_indices] 
    images_train_rgb_list = [full_dataset_all_images_rgb[original_idx] for original_idx in train_original_global_indices]
    masks_train_per_image_list = [full_dataset_all_masks[original_idx] if full_dataset_all_masks is not None and original_idx < len(full_dataset_all_masks) else [] for original_idx in train_original_global_indices]

    y_test_labels = labels_for_run_np[test_relative_indices]
    test_base_ids = [base_ids_for_run[i] for i in test_relative_indices]
    images_test_rgb_list = [full_dataset_all_images_rgb[original_idx] for original_idx in test_original_global_indices]
    masks_test_per_image_list = [full_dataset_all_masks[original_idx] if full_dataset_all_masks is not None and original_idx < len(full_dataset_all_masks) else [] for original_idx in test_original_global_indices]

    train_orig_to_local_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(train_original_global_indices)}
    test_orig_to_local_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(test_original_global_indices)}
    
    seg_infos_train_list_reindexed = []
    seg_infos_test_list_reindexed = []

    for seg_info_dict_orig in full_dataset_segment_infos_with_crops:
        original_img_idx_of_segment = seg_info_dict_orig.get('img_idx') # This is the global image index
        
        if original_img_idx_of_segment in train_orig_to_local_map:
            new_local_train_img_idx = train_orig_to_local_map[original_img_idx_of_segment]
            new_seg_info_dict = dict(seg_info_dict_orig); new_seg_info_dict['img_idx'] = new_local_train_img_idx # Re-index
            seg_infos_train_list_reindexed.append(new_seg_info_dict)
        elif original_img_idx_of_segment in test_orig_to_local_map:
            new_local_test_img_idx = test_orig_to_local_map[original_img_idx_of_segment]
            new_seg_info_dict = dict(seg_info_dict_orig); new_seg_info_dict['img_idx'] = new_local_test_img_idx # Re-index
            seg_infos_test_list_reindexed.append(new_seg_info_dict)
            
    seg_infos_train_np_with_crops = np.array(seg_infos_train_list_reindexed, dtype=object) 
    seg_infos_test_np_with_crops = np.array(seg_infos_test_list_reindexed, dtype=object)   
    main_logger.info(f"Train split ({config['dataset_name']}): {len(images_train_rgb_list)} images, {len(seg_infos_train_np_with_crops)} segments.")
    main_logger.info(f"Test split ({config['dataset_name']}): {len(images_test_rgb_list)} images, {len(seg_infos_test_np_with_crops)} segments.")
    y_test_labels_final = y_test_labels.copy() # For final evaluation

    # seg_crop_bgr should already be in seg_infos_train_np_with_crops from NPY or SAM output
    # We can call add_seg_crop_bgr_to_split_infos to verify or if masks were loaded separately for NPYs
    # but if generate_sam_segments_and_masks_cached correctly adds 'seg_crop_bgr', this might be redundant for OTF path
    min_pixels_crop_config = config.get("min_mask_pixels", 100)
    seg_infos_train_np_with_crops = add_seg_crop_bgr_to_split_infos(seg_infos_train_np_with_crops, images_train_rgb_list, masks_train_per_image_list, main_logger, min_pixels_crop_config)
    seg_infos_test_np_with_crops = add_seg_crop_bgr_to_split_infos(seg_infos_test_np_with_crops, images_test_rgb_list, masks_test_per_image_list, main_logger, min_pixels_crop_config)


    # --- Phase 3: Embedding (Training Data Segments) ---
    main_logger.info(f"--- Phase 3: Embedding (Training Data Segments) for {config['dataset_name']} ---")
    dino_processor, dino_model = init_dino(config['dino_model'], torch_device)
    target_model_resnet = init_target_model(torch_device) if config['embedding_type'] == 'combined' else None
    embeddings_train_segments = compute_final_embeddings(
        seg_infos_train_np_with_crops, 
        None, None, 
        dino_processor, dino_model, target_model_resnet,
        torch_device, config, client_id=f"central_train_{config['dataset_name']}_{config['run_id']}"
    )
    if embeddings_train_segments is None or embeddings_train_segments.shape[0] == 0: main_logger.error(f"({config['dataset_name']}) Embedding failed for train segments!"); return
    main_logger.info(f"Training segment embeddings computed ({config['dataset_name']}). Shape: {embeddings_train_segments.shape}")
    
    # --- Phases 4-10 ---
    # The rest of the pipeline (K-Means, Detectors, Vectorization, FIGS, Eval, Viz)
    # should now proceed correctly with the prepared data structures.
    # Remember to update cache file names and plot titles within these sections
    # to include `config['dataset_name']` to keep outputs for ADE20K and SUNRGBD separate.

    # ... (K-Means logic, adapted cache name) ...
    kmeans_cache_file = os.path.join(config["central_run_cache_dir"], f"kmeans_results_{config['dataset_name']}_{config['run_id']}.pkl")
    cluster_labels_train_segments = None; final_concept_original_indices = [] # Defaults
    if config.get("use_kmeans_cache", True) and os.path.exists(kmeans_cache_file):
        try:
            with open(kmeans_cache_file, "rb") as f: cluster_labels_train_segments, final_concept_original_indices = pickle.load(f)
            main_logger.info(f"Loaded cached K-Means results for {config['dataset_name']} from {kmeans_cache_file}")
        except Exception as e:
            main_logger.warning(f"Failed to load K-Means cache for {config['dataset_name']}: {e}. Recomputing.") # Fallthrough
    
    if cluster_labels_train_segments is None: # Recompute if cache failed or not used
        if embeddings_train_segments.shape[0] < config['num_clusters']: # Check if enough samples for K-Means
             main_logger.error(f"Not enough training segments ({embeddings_train_segments.shape[0]}) for K-Means with {config['num_clusters']} clusters. Reduce num_clusters or check data. Halting."); return
        kmeans = KMeans(n_clusters=config['num_clusters'], random_state=config['seed'], n_init=10, verbose=0 if not config.get("tqdm_enabled", True) else 1)
        cluster_labels_train_segments = kmeans.fit_predict(embeddings_train_segments)
        unique_labels_km, counts_km = np.unique(cluster_labels_train_segments, return_counts=True)
        final_concept_original_indices = unique_labels_km[counts_km >= config['min_samples_per_concept_cluster']].tolist()
        if config.get("use_kmeans_cache", True) and final_concept_original_indices: 
            with open(kmeans_cache_file, "wb") as f: pickle.dump((cluster_labels_train_segments, final_concept_original_indices),f)
            
    if not final_concept_original_indices: main_logger.error(f"({config['dataset_name']}) No K-Means concepts survived!"); return
    main_logger.info(f"({config['dataset_name']}) Found {len(final_concept_original_indices)} K-Means concepts after filtering.")

    # ... (Detector Training, adapted cache name) ...
    detectors_cache_file = os.path.join(config["central_run_cache_dir"], f"detectors_{config['dataset_name']}_{config['run_id']}.pkl")
    image_groups_train_segments = np.array([info["img_idx"] for info in seg_infos_train_np_with_crops if isinstance(info, dict) and "img_idx" in info])
    trained_detectors = {}
    if config.get("use_detectors_cache", True) and os.path.exists(detectors_cache_file):
        try:
            with open(detectors_cache_file, "rb") as f: trained_detectors = pickle.load(f)
            main_logger.info(f"Loaded cached detectors for {config['dataset_name']} from {detectors_cache_file}")
        except Exception as e: main_logger.warning(f"Failed to load detectors cache for {config['dataset_name']}: {e}. Recomputing.") # Fallthrough

    if not trained_detectors:
        trained_detectors = {}
        for original_idx in tqdm(final_concept_original_indices, desc=f"Training Detectors ({config['dataset_name']})", file=sys.stdout, disable=not config.get("tqdm_enabled",True)):
            _, model_info, score = train_concept_detector(original_idx, embeddings_train_segments, cluster_labels_train_segments, image_groups_train_segments, config)
            if model_info and score >= config['min_detector_score']: trained_detectors[original_idx] = model_info
        if config.get("use_detectors_cache", True) and trained_detectors: 
            with open(detectors_cache_file, "wb") as f: pickle.dump(trained_detectors, f)
    if not trained_detectors: main_logger.error(f"({config['dataset_name']}) No detectors trained!"); return
    ordered_final_concept_original_ids_for_features = sorted(list(trained_detectors.keys()))
    num_final_figs_features = len(ordered_final_concept_original_ids_for_features)
    main_logger.info(f"({config['dataset_name']}) Trained and kept {num_final_figs_features} concept detectors.")

    # ... (Train and Test Vectorization, adapted cache names) ...
    train_vectors_cache_file = os.path.join(config["central_run_cache_dir"], f"train_vectors_{config['dataset_name']}_{config['run_id']}.pkl")
    X_train_concepts = None
    if config.get("use_train_vectors_cache", True) and os.path.exists(train_vectors_cache_file):
        try:
            with open(train_vectors_cache_file, "rb") as f: X_train_concepts, _ = pickle.load(f)
            main_logger.info(f"Loaded cached TRAIN vectors for {config['dataset_name']} from {train_vectors_cache_file}")
        except Exception as e: main_logger.warning(f"Failed to load train vectors cache for {config['dataset_name']}: {e}. Recomputing.")

    if X_train_concepts is None:
        X_train_concepts, _ = build_centralized_concept_vectors_maxpool(seg_infos_train_np_with_crops, embeddings_train_segments, trained_detectors, ordered_final_concept_original_ids_for_features, len(images_train_rgb_list), train_base_ids, config, main_logger)
        if config.get("use_train_vectors_cache", True) and X_train_concepts is not None: 
            with open(train_vectors_cache_file, "wb") as f: pickle.dump((X_train_concepts, train_base_ids),f)
    if X_train_concepts is None or X_train_concepts.size == 0: main_logger.error(f"({config['dataset_name']}) Train concept vectors are empty!"); return
    main_logger.info(f"Train concept vectors created ({config['dataset_name']}). Shape: {X_train_concepts.shape}")
    figs_feature_names = [f"concept_{i}" for i in range(num_final_figs_features)] # Generic names for FIGS

    # Test Data Processing
    embeddings_test_segments = None; X_test_concepts = None
    if len(seg_infos_test_np_with_crops) > 0: # Check if there are test segments to process
        embeddings_test_segments = compute_final_embeddings(
            seg_infos_test_np_with_crops, None, None, dino_processor, dino_model, target_model_resnet,
            torch_device, config, client_id=f"central_test_{config['dataset_name']}_{config['run_id']}"
        )
        test_vectors_cache_file = os.path.join(config["central_run_cache_dir"], f"test_vectors_{config['dataset_name']}_{config['run_id']}.pkl")
        if embeddings_test_segments is not None and embeddings_test_segments.shape[0] > 0:
            if config.get("use_test_vectors_cache", True) and os.path.exists(test_vectors_cache_file):
                try:
                    with open(test_vectors_cache_file, "rb") as f: X_test_concepts, _ = pickle.load(f)
                    main_logger.info(f"Loaded cached TEST vectors for {config['dataset_name']} from {test_vectors_cache_file}")
                except Exception as e: main_logger.warning(f"Failed to load TEST vectors cache for {config['dataset_name']}: {e}. Recomputing.")
            
            if X_test_concepts is None:
                X_test_concepts, _ = build_centralized_concept_vectors_maxpool(seg_infos_test_np_with_crops, embeddings_test_segments, trained_detectors, ordered_final_concept_original_ids_for_features, len(images_test_rgb_list), test_base_ids, config, main_logger)
                if config.get("use_test_vectors_cache", True) and X_test_concepts is not None: 
                    with open(test_vectors_cache_file, "wb") as f: pickle.dump((X_test_concepts, test_base_ids),f)
        else: main_logger.warning(f"({config['dataset_name']}) No embeddings for test segments.")
    else: main_logger.warning(f"({config['dataset_name']}) No segments for test data, skipping test vectorization.")
    
    if X_test_concepts is None: X_test_concepts = np.empty((0, num_final_figs_features if num_final_figs_features > 0 else 1)) # Ensure defined, handle num_final_figs_features=0
    main_logger.info(f"Test concept vectors shape ({config['dataset_name']}): {X_test_concepts.shape}")


    # --- FIGS Model Training & Evaluation ---
    main_logger.info(f"--- Phase 7: FIGS Model Training ({config['dataset_name']}) ---")
    figs_model_instance = PatchedFIGSClassifier(**config['figs_params'], n_outputs_global=config['num_classes'])
    df_train_concepts_for_figs = pd.DataFrame(X_train_concepts, columns=figs_feature_names)
    figs_model_instance.fit(df_train_concepts_for_figs, y_train_labels, feature_names=figs_feature_names)
    main_logger.info(f"FIGS model trained ({config['dataset_name']}). Complexity: {getattr(figs_model_instance, 'complexity_', 'N/A')}")
    #with open(config["final_model_path"], "wb") as f: pickle.dump(figs_model_instance, f)
    #main_logger.info(f"Saved trained FIGS model ({config['dataset_name']}) to {config['final_model_path']}")
    
    main_logger.info(f"--- Phase 9: Final Evaluation ({config['dataset_name']}) ---")
    if X_test_concepts is not None and X_test_concepts.shape[0] > 0 and y_test_labels_final.shape[0] == X_test_concepts.shape[0]:
        accuracy_ac, rp_ac, rc_ac, rl_ac, rf_ac = evaluate_centralized_AutoCore_model(
            figs_model_instance, X_test_concepts, y_test_labels_final,
            figs_feature_names, config['num_classes'], main_logger
        )
        main_logger.info(f"({config['dataset_name']}) AutoCoRe-Style Metrics: Acc={accuracy_ac:.4f}, RuleP={rp_ac:.3f}, RuleCov={rc_ac:.3f}, RuleL={rl_ac:.2f}, RuleFid={rf_ac:.3f}")
        
        metrics_to_log = { "run_id": config['run_id'], "dataset": config['dataset_name'], "model_accuracy": accuracy_ac, "rule_precision": rp_ac, "rule_coverage": rc_ac, "rule_complexity": rl_ac, "rule_fidelity": rf_ac, **config['figs_params']}
        metrics_df = pd.DataFrame([metrics_to_log])
        metrics_log_path = config['metrics_log_path'] # Ensure this is correctly defined in config
        if os.path.exists(metrics_log_path): metrics_df.to_csv(metrics_log_path, mode='a', header=False, index=False)
        else: metrics_df.to_csv(metrics_log_path, mode='w', header=True, index=False)
        main_logger.info(f"Metrics for {config['dataset_name']} saved to {metrics_log_path}")
    else:
        main_logger.warning(f"Skipping final evaluation for {config['dataset_name']}: No test concept data or label mismatch.")

    # --- Paper Visualization (Phase 10) ---
    if config.get("generate_paper_visualizations", False) and \
       X_test_concepts is not None and X_test_concepts.shape[0] > 0 and \
       seg_infos_test_np_with_crops.size > 0 and \
       embeddings_test_segments is not None and embeddings_test_segments.shape[0] > 0: # Added check for embeddings
        main_logger.info(f"--- Phase 10: Visualizing Decision Explanations ({config['dataset_name']}) ---")
        # ... (Visualization logic as before, ensure all inputs are valid for the current dataset context) ...
        y_pred_test_labels_viz = figs_model_instance.predict(X_test_concepts)
        correctly_classified_indices_viz = np.where(y_pred_test_labels_viz == y_test_labels_final)[0]
        num_viz_samples = min(config.get("num_paper_viz_samples", 3), len(correctly_classified_indices_viz))
        if num_viz_samples > 0:
            chosen_indices_for_viz = np.random.choice(correctly_classified_indices_viz, num_viz_samples, replace=False)
            for test_split_local_idx_viz in chosen_indices_for_viz:
                original_rgb_image_to_viz = images_test_rgb_list[test_split_local_idx_viz]
                concept_vector_to_viz = X_test_concepts[test_split_local_idx_viz]
                predicted_label_idx_to_viz = y_pred_test_labels_viz[test_split_local_idx_viz]
                predicted_scene_name_to_viz = config['sorted_chosen_classes_for_mapping'][predicted_label_idx_to_viz]
                
                visualize_centralized_decision_for_paper(
                    target_image_rgb_np=original_rgb_image_to_viz,
                    image_idx_in_test_split=test_split_local_idx_viz,
                    predicted_class_name_str=predicted_scene_name_to_viz,
                    image_concept_vector_np=concept_vector_to_viz,
                    figs_model_instance=figs_model_instance,
                    ordered_final_concept_original_ids=ordered_final_concept_original_ids_for_features,
                    trained_concept_detectors_map_paper_viz=trained_detectors,
                    seg_infos_test_flat_paper_viz=seg_infos_test_np_with_crops, 
                    embeddings_test_flat_paper_viz=embeddings_test_segments,
                    feature_names_for_figs_paper_viz=figs_feature_names,
                    config_paper_viz=config, main_logger_paper_viz=main_logger,
                    output_filename_stem=f"paper_viz_{config['dataset_name']}_decision"
                )
    
    main_logger.info(f"======== Centralized FIGS ({config['dataset_name']}) Run ID: {config['run_id']} Complete ========")


if __name__ == "__main__":
    DATASET_TO_RUN = "sunrgbd" 
    # DATASET_TO_RUN = "ade20k" 
    
    current_run_config = generate_centralized_multidataset_config(dataset_name=DATASET_TO_RUN)
    
    # Ensure the logger passed to main_centralized_multidataset is correctly set up
    # If main_centralized_multidataset itself configures its logger using config['log_dir'] and config['run_id'],
    # then this outer logger setup might not be strictly necessary for that function, but good for the script overall.
    main_script_logger = setup_logging(log_dir=current_run_config['log_dir'], 
                                       filename_prefix=f"MAIN_SCRIPT_LOG_{current_run_config['run_id']}.log", # Distinct filename
                                       run_id=current_run_config['run_id'] + "_SCRIPT_RUNNER")
    #main_script_logger.addFilter(SAM2Filter())

    main_script_logger.info(f"Starting main script for dataset: {DATASET_TO_RUN} with Run ID: {current_run_config['run_id']}")
    try:
        main_centralized_multidataset(current_run_config)
    except Exception as e_main_run:
        main_script_logger.critical(f"CRITICAL ERROR in main_centralized_multidataset for {DATASET_TO_RUN}: {e_main_run}", exc_info=True)