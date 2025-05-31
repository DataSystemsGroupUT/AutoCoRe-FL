# run_cent_sunrgbd_autocore.py

import logging
import os
import sys
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score
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
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import cv2
from scipy.special import expit, softmax
import operator
import re
import itertools # For HPO loop

# --- Add project root to sys.path ---
# This assumes the script is run from a location where this relative path makes sense
# Or that the federated_logic_xai_figs_svm package is in PYTHONPATH
try:
    # Assuming standard project structure relative to where client.py is
    current_script_dir_for_proj = os.path.dirname(os.path.abspath(__file__))
    project_root_for_proj = os.path.abspath(os.path.join(current_script_dir_for_proj, '..'))
    if project_root_for_proj not in sys.path:
        sys.path.insert(0, project_root_for_proj)
except NameError: # __file__ not defined (e.g. in interactive environment)
    project_root_for_proj = os.getcwd() # Fallback
    if project_root_for_proj not in sys.path:
         sys.path.insert(0, project_root_for_proj)


# --- Project Module Imports ---
# These imports assume your project structure. Adjust if necessary.
from AutoCore_FL.data.sun_parition import load_scene_categories_sun # Reused for SUN RGB-D scene file
from AutoCore_FL.embedding.dino_loader import init_dino, init_target_model
from AutoCore_FL.embedding.compute_embeddings import compute_final_embeddings
from AutoCore_FL.concepts.detector import train_concept_detector
from AutoCore_FL.federated.client import PatchedFIGSClassifier, LocalNode
from AutoCore_FL.federated.utils import setup_logging, generate_run_id, save_config, load_config

# --- Helper function from sun_starter.py (to make script self-contained) ---
def filter_zero_segment_images(all_images_input, all_masks_input, all_segments_mapping_input, segment_infos_input):
    """
    Filters out images with zero segments based on all_masks_input.
    Rebuilds arrays so that img_idx is consecutive in the new lists.
    """
    valid_indices = []
    oldidx_to_newidx = {}
    new_idx = 0
    for old_idx, mask_list_for_img in enumerate(all_masks_input):
        if mask_list_for_img and len(mask_list_for_img) > 0:
            valid_indices.append(old_idx)
            oldidx_to_newidx[old_idx] = new_idx
            new_idx += 1

    # script_logger_v2.info(f"Filtering zero-segment images. Total: {len(all_images_input)}. Keeping {len(valid_indices)}.")

    filtered_images_out = []
    filtered_masks_out  = []
    for old_idx in valid_indices:
        filtered_images_out.append(all_images_input[old_idx])
        filtered_masks_out.append(all_masks_input[old_idx])

    new_all_segments_mapping_out = []
    if all_segments_mapping_input is not None and isinstance(all_segments_mapping_input, (list, np.ndarray)) and len(all_segments_mapping_input) > 0 : # check if it's array-like
        for (old_img_idx_map, seg_idx_map) in all_segments_mapping_input:
            if old_img_idx_map in oldidx_to_newidx:
                new_img_idx_map = oldidx_to_newidx[old_img_idx_map]
                new_all_segments_mapping_out.append((new_img_idx_map, seg_idx_map))
    
    new_segment_infos_out = []
    for info_item in segment_infos_input:
        old_img_idx_info = info_item.get("img_idx")
        if old_img_idx_info in oldidx_to_newidx:
            new_img_idx_info = oldidx_to_newidx[old_img_idx_info]
            new_info_item = dict(info_item); new_info_item["img_idx"] = new_img_idx_info
            new_segment_infos_out.append(new_info_item)

    return (np.array(filtered_images_out, dtype=object),
            np.array(filtered_masks_out,  dtype=object),
            np.array(new_all_segments_mapping_out,   dtype=object) if new_all_segments_mapping_out else np.array([]), # Handle empty case
            np.array(new_segment_infos_out,  dtype=object))


# --- Helper Function: Load Labels (from run_cent.py) ---
def load_labels_for_images(image_ids: list, scene_map: dict, scene_to_idx_map: dict, main_logger_passed) -> np.ndarray:
    labels = []
    processed_ids_for_labeling = set()
    for img_id in image_ids:
        if img_id in processed_ids_for_labeling: continue
        scene = scene_map.get(img_id)
        if scene is None:
            main_logger_passed.debug(f"Image base_id '{img_id}' not found in scene_map during label loading.")
            continue
        idx = scene_to_idx_map.get(scene)
        if idx is not None:
            labels.append(idx)
            processed_ids_for_labeling.add(img_id)
        else:
            main_logger_passed.debug(f"Scene '{scene}' for image base_id '{img_id}' not in scene_to_idx_map.")
    return np.array(labels, dtype=np.int64)

# --- Helper Function: Save Plot (from run_cent.py) ---
def save_plot(fig, plot_name_stem, config, main_logger_passed):
    plots_dir = os.path.join(config.get('log_dir', './logs'), "plots_sunrgbd_cent_autocore")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"{plot_name_stem}_{config.get('run_id', 'run')}.png")
    try:
        fig.savefig(path, bbox_inches='tight', dpi=config.get('plot_dpi', 150))
        plt.close(fig)
        main_logger_passed.info(f"Saved plot: {path}")
    except Exception as e:
        main_logger_passed.error(f"Failed to save plot {path}: {e}")

# --- Visualization Functions (Adapted from run_cent.py) ---
def visualize_cluster_segments_from_data(
    cluster_id_to_show, cluster_labels_for_all_segments,
    segment_infos_list_of_dicts, all_masks_per_image_in_split,
    all_images_rgb_in_split, n_samples, grid_size, figsize, mask_alpha,
    save_path_full_stem, config, main_logger_passed
):
    indices_in_cluster = np.where(cluster_labels_for_all_segments == cluster_id_to_show)[0]
    if len(indices_in_cluster) == 0:
        main_logger_passed.info(f"Cluster {cluster_id_to_show} has no segments for visualization.")
        return

    actual_n_samples = min(n_samples, len(indices_in_cluster))
    if actual_n_samples == 0: return
    sampled_global_seg_indices_in_split = np.random.choice(indices_in_cluster, actual_n_samples, replace=False)
    
    n_rows, n_cols = grid_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if actual_n_samples == 1: axes = np.array([axes]) # Ensure axes is iterable if only one plot
    axes = axes.flatten()

    for i, ax_idx in enumerate(range(actual_n_samples)):
        ax = axes[ax_idx]
        global_seg_idx_in_split = sampled_global_seg_indices_in_split[i] 
        seg_info_dict = segment_infos_list_of_dicts[global_seg_idx_in_split]
        split_local_img_idx = seg_info_dict.get('img_idx')
        seg_idx_in_image = seg_info_dict.get('seg_idx')

        valid_indices = True
        if not (split_local_img_idx is not None and 0 <= split_local_img_idx < len(all_images_rgb_in_split) and all_images_rgb_in_split[split_local_img_idx] is not None):
            main_logger_passed.warning(f"Invalid local img_idx ({split_local_img_idx}) for cluster viz. Max: {len(all_images_rgb_in_split)}")
            valid_indices = False
        if valid_indices and not (seg_idx_in_image is not None and 0 <= seg_idx_in_image < len(all_masks_per_image_in_split[split_local_img_idx])):
            main_logger_passed.warning(f"Invalid seg_idx_in_image ({seg_idx_in_image}) for local_img_idx {split_local_img_idx} for cluster viz.")
            valid_indices = False
            
        if not valid_indices:
            ax.text(0.5, 0.5, "Data Err", ha='center', va='center'); ax.axis('off'); continue

        image_rgb = all_images_rgb_in_split[split_local_img_idx]
        mask_bool = all_masks_per_image_in_split[split_local_img_idx][seg_idx_in_image]

        if mask_bool.sum() == 0: ax.text(0.5,0.5,"EmptyMask",ha='center',va='center'); ax.axis('off'); continue

        overlay_rgba = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
        color_for_mask = [0, 255, 0] 
        for c_idx_loop in range(3): overlay_rgba[mask_bool, c_idx_loop] = color_for_mask[c_idx_loop]
        overlay_rgba[mask_bool, 3] = int(mask_alpha * 255)
        
        ax.imshow(image_rgb); ax.imshow(overlay_rgba); ax.axis('off')

    for j_ax in range(actual_n_samples, len(axes)): axes[j_ax].axis('off')
    fig.suptitle(f"Cluster {cluster_id_to_show} Examples (Overlay)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot(fig, save_path_full_stem, config, main_logger_passed)

def visualize_embedding_tsne(embeddings, labels, title_prefix, config, main_logger_passed, perplexity=30, n_iter=300):
    if embeddings is None or embeddings.shape[0] < max(2, perplexity + 1):
        main_logger_passed.warning(f"TSNE: Not enough samples for {title_prefix} ({embeddings.shape[0] if embeddings is not None else 'None'} samples). Perplexity: {perplexity}")
        return
    from sklearn.manifold import TSNE 

    num_samples_for_tsne = min(config.get("tsne_plot_max_samples", 2000), embeddings.shape[0])
    indices = np.random.choice(embeddings.shape[0], num_samples_for_tsne, replace=False)
    
    actual_perplexity = min(perplexity, num_samples_for_tsne - 1)
    if actual_perplexity < 5:
        main_logger_passed.warning(f"TSNE: Perplexity {actual_perplexity} too low for {num_samples_for_tsne} samples. Skipping for {title_prefix}.")
        return

    tsne = TSNE(n_components=2, random_state=config['seed'], perplexity=actual_perplexity,
                n_iter=n_iter, init='pca', learning_rate='auto', n_jobs=-1)
    try:
        embeddings_2d = tsne.fit_transform(embeddings[indices])
    except Exception as e_tsne:
        main_logger_passed.error(f"t-SNE failed for {title_prefix}: {e_tsne}", exc_info=True)
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    labels_subset_for_plot = labels[indices] if labels is not None and len(labels) == embeddings.shape[0] else None
    cmap_to_use = 'viridis'
    if labels_subset_for_plot is not None:
        n_unique = len(np.unique(labels_subset_for_plot))
        if 1 < n_unique <= 20: cmap_to_use = plt.get_cmap('tab20', n_unique)
    
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_subset_for_plot, cmap=cmap_to_use, alpha=0.6, s=10)
    ax.set_title(f"{title_prefix} - t-SNE of Embeddings ({num_samples_for_tsne} samples)")
    
    if labels_subset_for_plot is not None and 1 < n_unique <= 20:
        try:
            legend_elements = scatter.legend_elements(num=min(n_unique, 20))[0]
            ax.legend(handles=legend_elements, labels=[str(int(ul)) for ul in np.unique(labels_subset_for_plot)[:len(legend_elements)]], title="Clusters/Classes", loc="best")
        except Exception as e_legend:
            main_logger_passed.warning(f"Could not create legend for t-SNE plot {title_prefix}: {e_legend}")
    save_plot(fig, f"{title_prefix.replace(' ', '_')}_embeddings_tsne", config, main_logger_passed)

def visualize_concept_vectors_pca(concept_vectors, image_labels, title_prefix, config, main_logger_passed):
    if concept_vectors is None or concept_vectors.shape[0] < 2:
        main_logger_passed.warning(f"PCA: Not enough concept vectors for {title_prefix} ({concept_vectors.shape[0] if concept_vectors is not None else 'None'} samples).")
        return
    from sklearn.decomposition import PCA

    num_components_for_pca = min(2, concept_vectors.shape[1])
    if num_components_for_pca == 0:
        main_logger_passed.warning(f"PCA: Concept vectors have 0 features for {title_prefix}.")
        return

    pca = PCA(n_components=num_components_for_pca, random_state=config['seed'])
    try:
        vectors_2d_or_1d = pca.fit_transform(concept_vectors)
    except Exception as e_pca:
        main_logger_passed.error(f"PCA transformation failed for {title_prefix}: {e_pca}", exc_info=True); return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    labels_for_plot = image_labels if image_labels is not None and len(image_labels) == vectors_2d_or_1d.shape[0] else None
    cmap_to_use = 'viridis'
    if labels_for_plot is not None:
        n_unique = len(np.unique(labels_for_plot))
        if 1 < n_unique <= 20: cmap_to_use = plt.get_cmap('tab20', n_unique)
    
    if num_components_for_pca == 1:
        y_jitter = np.random.rand(vectors_2d_or_1d.shape[0]) * 0.1 
        scatter = ax.scatter(vectors_2d_or_1d[:, 0], y_jitter, c=labels_for_plot, cmap=cmap_to_use, alpha=0.7, s=15)
    else:
        scatter = ax.scatter(vectors_2d_or_1d[:, 0], vectors_2d_or_1d[:, 1], c=labels_for_plot, cmap=cmap_to_use, alpha=0.7, s=15)
    ax.set_title(f"{title_prefix} - PCA of Image Concept Vectors")

    if labels_for_plot is not None and 1 < n_unique <= 20:
        try:
            legend_elements = scatter.legend_elements(num=min(n_unique, 20))[0]
            class_names_for_legend = config.get('sorted_chosen_classes_for_mapping', [str(int(ul)) for ul in np.unique(labels_for_plot)])
            display_labels = [class_names_for_legend[int(ul)] if 0 <= int(ul) < len(class_names_for_legend) else str(int(ul)) for ul in np.unique(labels_for_plot)[:len(legend_elements)]]
            ax.legend(handles=legend_elements, labels=display_labels, title="Image Classes", loc="best")
        except Exception as e_legend_pca:
            main_logger_passed.warning(f"Could not create legend for PCA plot {title_prefix}: {e_legend_pca}")
    save_plot(fig, f"{title_prefix.replace(' ', '_')}_concept_vectors_pca", config, main_logger_passed)


# --- Vectorization (Max-Pool strategy from run_cent.py) ---
def build_centralized_concept_vectors_maxpool(
    segment_infos_split_np, embeddings_for_segments_in_split_np,
    trained_concept_detectors_dict, ordered_final_concept_original_ids,
    num_total_images_in_split, image_base_ids_in_split,
    config_dict, main_logger_passed
    ):
    logger_main = main_logger_passed
    logger_main.info("Building centralized concept vectors using MAX-POOLING strategy...")
    num_final_figs_features = len(ordered_final_concept_original_ids)
    if num_final_figs_features == 0:
        logger_main.warning("No final concepts defined. Returning empty concept vectors.")
        return np.empty((0, 0)), []

    base_id_to_split_img_idx = {base_id: i for i, base_id in enumerate(image_base_ids_in_split)}
    split_img_idx_to_global_seg_indices_in_split = defaultdict(list)

    for global_seg_idx_in_split, seg_info_dict in enumerate(segment_infos_split_np):
        base_id = seg_info_dict.get('base_id')
        if base_id in base_id_to_split_img_idx:
            split_img_idx = base_id_to_split_img_idx[base_id]
            split_img_idx_to_global_seg_indices_in_split[split_img_idx].append(global_seg_idx_in_split)

    concept_vecs_for_split = np.zeros((num_total_images_in_split, num_final_figs_features), dtype=np.float32)

    for dense_feature_idx, original_kmeans_idx in tqdm(enumerate(ordered_final_concept_original_ids),
                                                       desc="Vectorizing (max-pool)", total=num_final_figs_features,
                                                       file=sys.stdout):
        if original_kmeans_idx not in trained_concept_detectors_dict:
            logger_main.debug(f"Detector not found for concept orig_idx {original_kmeans_idx} (dense_idx {dense_feature_idx}). Filling with zeros.")
            continue
        
        model_pipeline, optimal_threshold = trained_concept_detectors_dict[original_kmeans_idx]

        if embeddings_for_segments_in_split_np is None or embeddings_for_segments_in_split_np.shape[0] == 0:
            logger_main.debug(f"No embeddings to predict on for concept {original_kmeans_idx}.")
            continue

        try:
            if hasattr(model_pipeline, "predict_proba"):
                all_split_seg_probs_for_concept = model_pipeline.predict_proba(embeddings_for_segments_in_split_np)[:, 1]
            elif hasattr(model_pipeline, "decision_function"):
                all_split_seg_scores_for_concept = model_pipeline.decision_function(embeddings_for_segments_in_split_np)
                all_split_seg_probs_for_concept = expit(all_split_seg_scores_for_concept)
            else:
                logger_main.error(f"Detector for concept orig_idx {original_kmeans_idx} has no proba/decision_func.")
                continue
        except Exception as e:
            logger_main.error(f"Error predicting probs for concept orig_idx {original_kmeans_idx}: {e}")
            continue
            
        for split_img_idx in range(num_total_images_in_split):
            global_indices_for_this_image_segments_in_split = split_img_idx_to_global_seg_indices_in_split.get(split_img_idx, [])
            if not global_indices_for_this_image_segments_in_split: continue
                
            probs_for_this_image_segments = all_split_seg_probs_for_concept[global_indices_for_this_image_segments_in_split]
            max_prob_for_image = np.max(probs_for_this_image_segments) if probs_for_this_image_segments.size > 0 else 0.0
            
            if max_prob_for_image >= optimal_threshold:
                concept_vecs_for_split[split_img_idx, dense_feature_idx] = 1.0
    
    return concept_vecs_for_split, list(image_base_ids_in_split)

# --- Evaluation Metric Function (from run_cent.py) ---
def figs_lr_xfl_metrics(
    figs_model, X_eval: np.ndarray, y_eval: np.ndarray,
    feature_names: List[str], main_logger: logging.Logger,
) -> Tuple[float, float, float]:
    log = main_logger
    if not hasattr(figs_model, 'predict') or not hasattr(figs_model, 'classes_') or not hasattr(figs_model, 'trees_'):
        log.error("Provided figs_model is not a valid/fitted PatchedFIGSClassifier instance.")
        return 0.0, 0.0, 0.0

    if X_eval.shape[0] == 0:
        log.warning("X_eval is empty. Cannot compute metrics."); return 0.0, 0.0, 0.0
        
    df_X_eval = pd.DataFrame(X_eval, columns=feature_names)
    n_samples = df_X_eval.shape[0]
    try:
        y_hat_model_predictions = figs_model.predict(X_eval)
    except Exception as e_pred:
        log.error(f"Error during figs_model.predict(X_eval): {e_pred}"); return 0.0, 0.0, 0.0
        
    model_overall_accuracy = accuracy_score(y_eval, y_hat_model_predictions)
    log.info(f"FIGS Overall Model Predictive Accuracy: {model_overall_accuracy:.4f}")

    n_model_classes = len(figs_model.classes_)
    class_leaf_masks: Dict[int, List[np.ndarray]] = defaultdict(list)

    for tree_idx, tree_root_node in enumerate(figs_model.trees_):
        if tree_root_node is None: log.warning(f"Tree {tree_idx} is None. Skipping."); continue
        
        # --- get_all_leaf_paths_and_masks (defined inline for clarity) ---
        def get_all_leaf_paths_and_masks_inline(tree_root_inline, feature_names_inline, X_df_inline, n_samples_inline, logger_inline):
            leaf_details_inline = []
            def recurse_inline(node_inline, current_conditions_list_inline: List[str]):
                if node_inline is None: return
                if node_inline.left is None and node_inline.right is None: # Leaf
                    rule_str_inline = " & ".join(sorted(list(set(current_conditions_list_inline)))) if current_conditions_list_inline else "True"
                    current_mask_inline = np.ones(n_samples_inline, dtype=bool)
                    if rule_str_inline != "True":
                        try: current_mask_inline = X_df_inline.eval(rule_str_inline).to_numpy(dtype=bool)
                        except Exception: # Fallback manual parser
                            current_mask_inline.fill(True); parse_success_inline = True
                            for cond_str_part_inline in current_conditions_list_inline: 
                                m_inline = re.match(r"`(.+?)`\s*([><]=?)\s*([0-9eE.+-]+)", cond_str_part_inline.strip())
                                if not m_inline: current_mask_inline.fill(False); parse_success_inline = False; break 
                                feat_inline, op_inline, val_str_parsed_inline = m_inline.groups()
                                try: val_fl_inline = float(val_str_parsed_inline)
                                except ValueError: current_mask_inline.fill(False); parse_success_inline = False; break
                                if feat_inline not in X_df_inline.columns: current_mask_inline.fill(False); parse_success_inline = False; break
                                arr_feat_data_inline = X_df_inline[feat_inline].to_numpy()
                                op_func_inline = {'<=': operator.le, '>': operator.gt, '<': operator.lt, '>=': operator.ge, '==': operator.eq, '!=': operator.ne}.get(op_inline)
                                if not op_func_inline: current_mask_inline.fill(False); parse_success_inline = False; break
                                try: current_mask_inline &= op_func_inline(arr_feat_data_inline, val_fl_inline)
                                except Exception: current_mask_inline.fill(False); parse_success_inline = False; break
                            if not parse_success_inline: logger_inline.warning(f"Manual parse failed for: '{rule_str_inline[:100]}'.")
                    leaf_value_inline = node_inline.value.flatten() if isinstance(node_inline.value, np.ndarray) else np.zeros(len(feature_names_inline))
                    leaf_details_inline.append( (rule_str_inline, leaf_value_inline, current_mask_inline) ); return
                if not hasattr(node_inline, 'feature') or node_inline.feature is None or not hasattr(node_inline, 'threshold') or node_inline.threshold is None: return 
                f_idx_inline, thr_inline = node_inline.feature, node_inline.threshold
                if not (0 <= f_idx_inline < len(feature_names_inline)): return 
                f_name_inline = feature_names_inline[f_idx_inline]
                if node_inline.left: recurse_inline(node_inline.left, current_conditions_list_inline + [f"`{f_name_inline}` <= {thr_inline:.6f}"])
                if node_inline.right: recurse_inline(node_inline.right, current_conditions_list_inline + [f"`{f_name_inline}` > {thr_inline:.6f}"])
            if tree_root_inline is not None: recurse_inline(tree_root_inline, [])
            return leaf_details_inline
        # --- End of inline get_all_leaf_paths_and_masks ---

        all_paths_in_this_tree = get_all_leaf_paths_and_masks_inline(tree_root_node, feature_names, df_X_eval, n_samples, log)

        for _rule_str, leaf_value_vector, mask_for_rule in all_paths_in_this_tree:
            if mask_for_rule.sum() == 0: continue
            predicted_class_idx_for_leaf = int(np.argmax(leaf_value_vector)) if leaf_value_vector.size > 0 else -1
            if 0 <= predicted_class_idx_for_leaf < n_model_classes:
                class_leaf_masks[predicted_class_idx_for_leaf].append(mask_for_rule)

    dnf_rule_accuracies_per_class, dnf_rule_fidelities_per_class = [], []
    for class_idx_numeric in range(n_model_classes):
        actual_class_label_from_model = figs_model.classes_[class_idx_numeric]
        masks_for_this_class = class_leaf_masks.get(class_idx_numeric, [])
        dnf_mask_for_this_class = np.any(np.array(masks_for_this_class), axis=0) if masks_for_this_class else np.zeros(n_samples, dtype=bool)
        
        dnf_rule_accuracies_per_class.append(accuracy_score(y_eval == actual_class_label_from_model, dnf_mask_for_this_class))
        dnf_rule_fidelities_per_class.append(accuracy_score(y_hat_model_predictions == actual_class_label_from_model, dnf_mask_for_this_class))
        
    mean_dnf_rule_accuracy = float(np.mean(dnf_rule_accuracies_per_class)) if dnf_rule_accuracies_per_class else 0.0
    mean_dnf_rule_fidelity = float(np.mean(dnf_rule_fidelities_per_class)) if dnf_rule_fidelities_per_class else 0.0
    
    log.info(f"LR-XFL Style Metrics: ModelAcc={model_overall_accuracy:.4f} | DNF_RuleAcc={mean_dnf_rule_accuracy:.4f} | DNF_RuleFid={mean_dnf_rule_fidelity:.4f}")
    return model_overall_accuracy, mean_dnf_rule_accuracy, mean_dnf_rule_fidelity


# --- Configuration Generation ---
def generate_sunrgbd_cent_autocore_config(run_id_base="sunrgbd_cent_ac_run"):
    # --- USER ACTION REQUIRED: Verify these paths ---
    SUNRGBD_DATA_ROOT = "/gpfs/helios/home/soliman/logic_explained_networks/data/sunrgbd" # e.g., /gpfs/helios/home/soliman/logic_explained_networks/data/sunrgbd
    SUNRGBD_NPY_BASE_PATH = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/" # e.g., /gpfs/helios/home/soliman/logic_explained_networks/experiments/
    SUNRGBD_SCENE_CAT_PATH = os.path.join(SUNRGBD_DATA_ROOT, "images", "sceneCategories.txt") # From sun_starter.py structure

    # Check if placeholder paths are still there
    if "/path/to/your" in SUNRGBD_DATA_ROOT or "/path/to/your" in SUNRGBD_NPY_BASE_PATH:
        print("FATAL ERROR: Placeholder paths for SUNRGBD_DATA_ROOT or SUNRGBD_NPY_BASE_PATH detected in config. Please update them in the script.")
        sys.exit(1)
    if not os.path.exists(SUNRGBD_SCENE_CAT_PATH):
        print(f"FATAL ERROR: SUNRGBD_SCENE_CAT_PATH does not exist: {SUNRGBD_SCENE_CAT_PATH}")
        sys.exit(1)
    # Check for SUNRGBD NPY files
    expected_npy_files = [
        "logic_sunrgbd_all_masks.npy", "logic_sunrgbd_all_images.npy",
        "logic_sunrgbd_segment_infos.npy", "logic_sunrgbd_all_segments.npy"
    ]
    for fname in expected_npy_files:
        if not os.path.exists(os.path.join(SUNRGBD_NPY_BASE_PATH, fname)):
            print(f"FATAL ERROR: Expected SUN RGB-D NPY file not found: {os.path.join(SUNRGBD_NPY_BASE_PATH, fname)}")
            sys.exit(1)


    # Default SUN RGB-D classes (37 scenes from sun_starter.py log)
    SUNRGBD_CHOSEN_CLASSES = [ 'bathroom', 'bedroom','classroom', 'computer_room',  'conference_room', 'bookstore', 'cafeteria', 'basement'
                               'coffee_room',   'corridor', 'dancing_room', 'dinette', 'dining_area', 
                                 'discussion_area', 'furniture_store',  'home_office', 'kitchen','lab', 'library', 'office']

    
    METHOD_NAME = "FIGS_Centralized_SUNRGBD_AutoCoRe"
    seed = 42
    effective_run_id = f"{run_id_base}_{seed}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, f"experiment_results_centralized/{METHOD_NAME.lower()}_run_{effective_run_id}")
    os.makedirs(base_dir, exist_ok=True)
    log_dir_path = os.path.join(base_dir, "logs_sunrgbd_autocore")
    os.makedirs(log_dir_path, exist_ok=True)
    
    central_cache_dir_for_run = os.path.join(script_dir, "cache_centralized_sunrgbd_autocore", f"run_{effective_run_id}")
    os.makedirs(central_cache_dir_for_run, exist_ok=True)

    config = {
        "dataset_name": "sunrgbd", # Key for conditional logic
        "sunrgbd_data_root": SUNRGBD_DATA_ROOT,
        "sunrgbd_npy_base_path": SUNRGBD_NPY_BASE_PATH,
        "sunrgbd_scene_cat_path": SUNRGBD_SCENE_CAT_PATH,
        "sunrgbd_chosen_classes": SUNRGBD_CHOSEN_CLASSES, # Use the full list for SUN RGB-D
        "num_classes": len(SUNRGBD_CHOSEN_CLASSES), # Automatically set
        
        "seed": seed, "test_split_ratio": 0.2,
        "dino_model": "facebook/dinov2-base", "embedding_type": "dino_only", "embedding_dim": 768,
        "num_clusters": 100, "min_samples_per_concept_cluster": 50, # K-Means related
        
        "detector_type": "lr", "detector_min_samples_per_class": 20, "detector_cv_splits": 3,
        "pca_n_components": 256, "lr_max_iter": 10000, "min_detector_score": 0.70, # Adjusted score for potentially harder concepts
        
        "vectorizer_strategy": "max_pool", # As per original run_cent.py
        
        # FIGS params will be swept, these are just defaults if sweep is removed
        "figs_params": {"min_impurity_decrease": 0.0, "random_state": seed},
        
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "metrics_log_path": os.path.join(base_dir, f"final_metrics_{METHOD_NAME}.csv"),
        "log_dir": log_dir_path, "run_id": effective_run_id, "method_name": METHOD_NAME,
        "final_model_path": os.path.join(base_dir, f"final_model_{METHOD_NAME}.pkl"),
        "concept_definitions_path": os.path.join(base_dir, f"concept_definitions_{METHOD_NAME}.pkl"),
        
        "central_run_cache_dir": central_cache_dir_for_run,
        "use_seg_crops_cache": True, "use_kmeans_cache": True,
        "use_detectors_cache": True, "use_train_vectors_cache": True,
        "use_test_vectors_cache": True, "plot_dpi": 100,
        "min_mask_pixels_for_crop": 100, # From generate_deterministic_cached_data_v2
        "tsne_plot_max_samples": 2000, # For t-SNE visualization performance
        # --- compute_final_embeddings specific cache control (if its internal caching is used) ---
        "use_embedding_cache": True, # Let compute_final_embeddings manage its own cache
        "embedding_cache_dir": os.path.join(central_cache_dir_for_run, "embedding_cache_compute_final"), # Separate dir for compute_final_embeddings
    }
    os.makedirs(config["embedding_cache_dir"], exist_ok=True)
    
    return config

# --- Main Centralized AutoCoRe-FIGS Function for SUN RGB-D ---
def main_centralized_autocore_sunrgbd():
    config = generate_sunrgbd_cent_autocore_config() # Get SUN RGB-D specific config
    
    # Setup logging using project's utility
    main_logger = setup_logging(log_dir=config['log_dir'], run_id=config['run_id'], log_level_str="INFO")
    main_logger.info(f"======== Starting Centralized AutoCoRe-FIGS for SUN RGB-D - Run ID: {config['run_id']} ========")
    main_logger.info(f"Full Config: {config}")

    # Seed everything
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda' and torch.cuda.is_available(): torch.cuda.manual_seed_all(config['seed'])
    main_logger.info(f"Using device: {config['device']}")
    torch_device = torch.device(config['device'])

    # --- Phase 1: Loading SUN RGB-D Pre-generated Segmentation Data ---
    main_logger.info("--- Phase 1: Loading SUN RGB-D Pre-generated Segmentation Data ---")
    path_all_masks_sun = os.path.join(config["sunrgbd_npy_base_path"], "logic_sunrgbd_all_masks.npy")
    path_all_images_sun = os.path.join(config["sunrgbd_npy_base_path"], "logic_sunrgbd_all_images.npy") # Note the 'l'
    path_segment_infos_sun_raw = os.path.join(config["sunrgbd_npy_base_path"], "logic_sunrgbd_segment_infos.npy")
    path_all_segments_mapping_sun = os.path.join(config["sunrgbd_npy_base_path"], "logic_sunrgbd_all_segments.npy")

    try:
        full_dataset_all_masks_raw = np.load(path_all_masks_sun, allow_pickle=True)
        full_dataset_all_images_rgb_raw = np.load(path_all_images_sun, allow_pickle=True)
        full_dataset_segment_infos_raw = np.load(path_segment_infos_sun_raw, allow_pickle=True)
        full_dataset_all_segments_mapping_raw = np.load(path_all_segments_mapping_sun, allow_pickle=True)
    except Exception as e:
        main_logger.error(f"Failed to load SUN RGB-D NPY files: {e}", exc_info=True); return

    (full_dataset_all_images_rgb, full_dataset_all_masks, _, full_dataset_segment_infos
    ) = filter_zero_segment_images(full_dataset_all_images_rgb_raw, full_dataset_all_masks_raw, 
                                   full_dataset_all_segments_mapping_raw, full_dataset_segment_infos_raw)
    main_logger.info(f"SUN RGB-D: Loaded and filtered .npy data: {len(full_dataset_all_images_rgb)} images, {len(full_dataset_segment_infos)} segments.")

    # --- Phase 2: Preparing Labels and Splitting Data ---
    main_logger.info("--- Phase 2: Preparing Labels and Splitting Data ---")
    scene_map = load_scene_categories_sun(config["sunrgbd_scene_cat_path"])
    # log scene map
    main_logger.info(f"Scene map loaded: {len(scene_map)} unique scenes.")
    # log first 10 entries
    main_logger.info(f"First 10 entries in scene map: {list(scene_map.items())[:10]}")
    sorted_chosen_classes_for_mapping = sorted(config["sunrgbd_chosen_classes"])
    scene_to_global_idx_map = {s: i for i, s in enumerate(sorted_chosen_classes_for_mapping)}
    config['sorted_chosen_classes_for_mapping'] = sorted_chosen_classes_for_mapping
    main_logger.info(f"Scene to global index mapping: {scene_to_global_idx_map}")
    num_loaded_images = len(full_dataset_all_images_rgb)
    all_base_ids_ordered = [None] * num_loaded_images
    # log num of loaded images
    # ... after loading scene_map and full_dataset_segment_infos
    main_logger.info(f"Number of entries in scene_map: {len(scene_map)}")
    main_logger.info(f"Example scene_map keys (base_ids): {list(scene_map.keys())[:5]}")
    main_logger.info(f"Example scene_map values (scene names): {list(scene_map.values())[:5]}")

    base_ids_from_segments = set()
    for seg_info_debug in full_dataset_segment_infos:
        base_ids_from_segments.add(seg_info_debug.get('base_id'))
    main_logger.info(f"Number of unique base_ids in segment_infos: {len(base_ids_from_segments)}")
    main_logger.info(f"Example base_ids from segment_infos: {list(base_ids_from_segments)[:5]}")

    # Check for overlap
    overlap_base_ids = len(base_ids_from_segments.intersection(set(scene_map.keys())))
    main_logger.info(f"Number of base_ids from segments that are ALSO keys in scene_map: {overlap_base_ids}")

    # Check config['sunrgbd_chosen_classes']
    main_logger.info(f"config['sunrgbd_chosen_classes'] being used: {config['sunrgbd_chosen_classes']}")
    # ... then the loop to create scene_to_global_idx_map
    temp_img_indices_found_in_segs = set()
    for seg_info in full_dataset_segment_infos:
        orig_img_idx = seg_info.get('img_idx')
        base_id = seg_info.get('base_id')
        if orig_img_idx is not None and base_id is not None and 0 <= orig_img_idx < num_loaded_images:
            if all_base_ids_ordered[orig_img_idx] is None: all_base_ids_ordered[orig_img_idx] = base_id
            temp_img_indices_found_in_segs.add(orig_img_idx)

    valid_image_indices_for_run, labels_for_run_list, base_ids_for_run = [], [], []
    for i in range(num_loaded_images):
        base_id = all_base_ids_ordered[i]
        # log every 1000th image base_id
        if i % 1000 == 0: main_logger.debug(f"Processing image {i}/{num_loaded_images} with base_id {base_id}")
        if base_id is None: continue
        scene = scene_map.get(base_id)
        # log every 1000th image
        if i % 1000 == 0: main_logger.debug(f"Processing image {i}/{num_loaded_images} with base_id {base_id} and scene {scene}")
        if scene in scene_to_global_idx_map: # Filter by CURRENT run's chosen_classes via map
            label_idx = scene_to_global_idx_map[scene]
            valid_image_indices_for_run.append(i)
            base_ids_for_run.append(base_id)
            labels_for_run_list.append(label_idx)
    
    if not valid_image_indices_for_run: main_logger.error("No images match target classes. Exiting."); return
    main_logger.info(f"Proceeding with {len(valid_image_indices_for_run)} SUN RGB-D images matching chosen classes.")
    labels_for_run_np = np.array(labels_for_run_list, dtype=np.int64)

    indices_to_split_from = np.arange(len(valid_image_indices_for_run))
    try:
        train_relative_indices, test_relative_indices = train_test_split(
            indices_to_split_from, test_size=config['test_split_ratio'],
            random_state=config['seed'], stratify=labels_for_run_np
        )
    except ValueError: # Stratify fail
        main_logger.warning("Stratification failed. Using random split on valid image indices.")
        train_relative_indices, test_relative_indices = train_test_split(
            indices_to_split_from, test_size=config['test_split_ratio'], random_state=config['seed']
        )

    train_original_global_indices = [valid_image_indices_for_run[i] for i in train_relative_indices]
    test_original_global_indices = [valid_image_indices_for_run[i] for i in test_relative_indices]

    y_train_labels = labels_for_run_np[train_relative_indices]
    train_base_ids = [base_ids_for_run[i] for i in train_relative_indices]
    images_train_rgb_list = [full_dataset_all_images_rgb[original_idx] for original_idx in train_original_global_indices]
    masks_train_per_image_list = [full_dataset_all_masks[original_idx] for original_idx in train_original_global_indices]

    y_test_labels_final = labels_for_run_np[test_relative_indices]
    test_base_ids = [base_ids_for_run[i] for i in test_relative_indices]
    images_test_rgb_list = [full_dataset_all_images_rgb[original_idx] for original_idx in test_original_global_indices]
    masks_test_per_image_list = [full_dataset_all_masks[original_idx] for original_idx in test_original_global_indices]

    train_orig_to_local_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(train_original_global_indices)}
    test_orig_to_local_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(test_original_global_indices)}
    seg_infos_train_list, seg_infos_test_list = [], []
    for seg_info_dict_orig in full_dataset_segment_infos:
        original_img_idx_of_segment = seg_info_dict_orig.get('img_idx')
        if original_img_idx_of_segment in train_orig_to_local_map:
            new_local_train_img_idx = train_orig_to_local_map[original_img_idx_of_segment]
            new_seg_info_dict = dict(seg_info_dict_orig); new_seg_info_dict['img_idx'] = new_local_train_img_idx
            seg_infos_train_list.append(new_seg_info_dict)
        elif original_img_idx_of_segment in test_orig_to_local_map:
            new_local_test_img_idx = test_orig_to_local_map[original_img_idx_of_segment]
            new_seg_info_dict = dict(seg_info_dict_orig); new_seg_info_dict['img_idx'] = new_local_test_img_idx
            seg_infos_test_list.append(new_seg_info_dict)
    seg_infos_train_np = np.array(seg_infos_train_list, dtype=object)
    seg_infos_test_np = np.array(seg_infos_test_list, dtype=object)
    
    main_logger.info(f"Train split: {len(images_train_rgb_list)} images, {len(seg_infos_train_np)} segments.")
    main_logger.info(f"Test split: {len(images_test_rgb_list)} images, {len(seg_infos_test_np)} segments.")

    # --- Add seg_crop_bgr ---
    def add_seg_crop_bgr_to_split_infos_local(segment_infos_split, images_rgb_split_list, masks_per_image_split_list, main_logger_passed_crop, min_pixels_cfg):
        processed_infos = []
        for seg_info_item in tqdm(segment_infos_split, desc="Preparing seg_crop_bgr", file=sys.stdout, leave=False):
            if 'seg_crop_bgr' in seg_info_item and seg_info_item['seg_crop_bgr'] is not None:
                processed_infos.append(seg_info_item); continue
            new_seg_info_item = dict(seg_info_item); seg_crop_bgr_val = None
            try:
                local_img_idx, seg_idx_in_img = new_seg_info_item['img_idx'], new_seg_info_item['seg_idx']
                if not (0 <= local_img_idx < len(images_rgb_split_list) and \
                        0 <= local_img_idx < len(masks_per_image_split_list) and \
                        0 <= seg_idx_in_img < len(masks_per_image_split_list[local_img_idx])):
                    main_logger_passed_crop.debug(f"Idx OOB for seg_crop: img {local_img_idx}, seg {seg_idx_in_img}")
                else:
                    original_img_rgb, seg_mask_bool = images_rgb_split_list[local_img_idx], masks_per_image_split_list[local_img_idx][seg_idx_in_img]
                    if seg_mask_bool.sum() >= min_pixels_cfg:
                        ys, xs = np.where(seg_mask_bool)
                        if len(ys) > 0:
                            top, left, bottom, right = np.min(ys), np.min(xs), np.max(ys), np.max(xs)
                            seg_crop_rgb_temp = original_img_rgb[top:bottom+1, left:right+1].copy()
                            local_mask_temp = seg_mask_bool[top:bottom+1, left:right+1]
                            seg_crop_rgb_temp[~local_mask_temp] = (0,0,0)
                            seg_crop_bgr_val = cv2.cvtColor(seg_crop_rgb_temp, cv2.COLOR_RGB2BGR)
            except Exception as e_crop: main_logger_passed_crop.error(f"Error creating seg_crop_bgr: {e_crop}", exc_info=False)
            new_seg_info_item['seg_crop_bgr'] = seg_crop_bgr_val; processed_infos.append(new_seg_info_item)
        return np.array(processed_infos, dtype=object)

    # --- Process TRAIN data ---
    main_logger.info("Preparing 'seg_crop_bgr' for TRAINING segments...")
    seg_crops_train_cache_file = os.path.join(config["central_run_cache_dir"], f"seg_crops_train_{config['run_id']}.pkl")
    if config.get("use_seg_crops_cache", True) and os.path.exists(seg_crops_train_cache_file):
        try:
            with open(seg_crops_train_cache_file, "rb") as f: seg_infos_train_np_with_crops = pickle.load(f)
            main_logger.info(f"Loaded cached train seg_crops from {seg_crops_train_cache_file}")
        except Exception as e:
            main_logger.warning(f"Failed to load cached train seg_crops: {e}. Recomputing.")
            seg_infos_train_np_with_crops = add_seg_crop_bgr_to_split_infos_local(seg_infos_train_np, images_train_rgb_list, masks_train_per_image_list, main_logger, config["min_mask_pixels_for_crop"])
            if config.get("use_seg_crops_cache", True): 
                with open(seg_crops_train_cache_file, "wb") as f: pickle.dump(seg_infos_train_np_with_crops, f)
    else:
        seg_infos_train_np_with_crops = add_seg_crop_bgr_to_split_infos_local(seg_infos_train_np, images_train_rgb_list, masks_train_per_image_list, main_logger, config["min_mask_pixels_for_crop"])
        if config.get("use_seg_crops_cache", True) and seg_infos_train_np_with_crops is not None: 
            with open(seg_crops_train_cache_file, "wb") as f: pickle.dump(seg_infos_train_np_with_crops, f)
    
    main_logger.info("--- Phase 3: Embedding (Training Data Segments) ---")
    dino_processor, dino_model = init_dino(config['dino_model'], torch_device)
    target_model_resnet = init_target_model(torch_device) if config['embedding_type'] == 'combined' else None
    
    embeddings_train_segments = compute_final_embeddings(
        seg_infos_train_np_with_crops, images_train_rgb_list, None,
        dino_processor, dino_model, target_model_resnet,
        torch_device, config, client_id=f"central_train_sunrgbd_{config['run_id']}"
    )
    if embeddings_train_segments is None or embeddings_train_segments.shape[0] == 0: main_logger.error("Embedding failed for train segments!"); return
    main_logger.info(f"Training segment embeddings computed. Shape: {embeddings_train_segments.shape}")
    visualize_embedding_tsne(embeddings_train_segments, None, "TrainSegEmbeds_PreKMeans_SUNRGBD", config, main_logger)

    # --- Phase 4: Concept Discovery (K-Means) ---
    main_logger.info(f"--- Phase 4: Concept Discovery (K-Means with k={config['num_clusters']}) ---")
    kmeans_cache_file = os.path.join(config["central_run_cache_dir"], f"kmeans_results_sunrgbd_{config['run_id']}.pkl")
    if config.get("use_kmeans_cache", True) and os.path.exists(kmeans_cache_file):
        try:
            with open(kmeans_cache_file, "rb") as f: cluster_labels_train_segments, final_concept_original_indices = pickle.load(f)
            main_logger.info(f"Loaded cached K-Means results from {kmeans_cache_file}")
        except Exception as e:
            main_logger.warning(f"Failed to load K-Means cache: {e}. Recomputing.")
            kmeans = KMeans(n_clusters=config['num_clusters'], random_state=config['seed'], n_init=10, verbose=0)
            cluster_labels_train_segments = kmeans.fit_predict(embeddings_train_segments)
            unique_labels_km, counts_km = np.unique(cluster_labels_train_segments, return_counts=True)
            keep_mask_km = counts_km >= config['min_samples_per_concept_cluster']
            final_concept_original_indices = unique_labels_km[keep_mask_km].tolist()
            if config.get("use_kmeans_cache", True): 
                with open(kmeans_cache_file, "wb") as f: pickle.dump((cluster_labels_train_segments, final_concept_original_indices), f)
    else:
        kmeans = KMeans(n_clusters=config['num_clusters'], random_state=config['seed'], n_init=10, verbose=0)
        cluster_labels_train_segments = kmeans.fit_predict(embeddings_train_segments)
        unique_labels_km, counts_km = np.unique(cluster_labels_train_segments, return_counts=True)
        keep_mask_km = counts_km >= config['min_samples_per_concept_cluster']
        final_concept_original_indices = unique_labels_km[keep_mask_km].tolist()
        if config.get("use_kmeans_cache", True) and final_concept_original_indices is not None: 
            with open(kmeans_cache_file, "wb") as f: pickle.dump((cluster_labels_train_segments, final_concept_original_indices), f)
    if not final_concept_original_indices: main_logger.error("No concept clusters survived filtering!"); return
    main_logger.info(f"Found {len(final_concept_original_indices)} concepts after K-Means and filtering.")
    visualize_embedding_tsne(embeddings_train_segments, cluster_labels_train_segments, "TrainSegEmbeds_PostKMeans_SUNRGBD", config, main_logger)
    
    # --- Phase 5: Concept Detector Training ---
    main_logger.info("--- Phase 5: Concept Detector Training ---")
    detectors_cache_file = os.path.join(config["central_run_cache_dir"], f"detectors_sunrgbd_{config['run_id']}.pkl")
    image_groups_train_segments = np.array([info["img_idx"] for info in seg_infos_train_np_with_crops])
    if config.get("use_detectors_cache", True) and os.path.exists(detectors_cache_file):
        try:
            with open(detectors_cache_file, "rb") as f: trained_detectors = pickle.load(f)
            main_logger.info(f"Loaded cached detectors from {detectors_cache_file}")
        except Exception as e:
            main_logger.warning(f"Failed to load detectors cache: {e}. Recomputing.")
            trained_detectors = {}
            for original_idx in tqdm(final_concept_original_indices, desc="Training Detectors (SUN RGB-D)", file=sys.stdout):
                _, model_info, score = train_concept_detector(original_idx, embeddings_train_segments, cluster_labels_train_segments, image_groups_train_segments, config)
                if model_info and score >= config['min_detector_score']: trained_detectors[original_idx] = model_info
            if config.get("use_detectors_cache", True): 
                with open(detectors_cache_file, "wb") as f: pickle.dump(trained_detectors, f)
    else:
        trained_detectors = {}
        for original_idx in tqdm(final_concept_original_indices, desc="Training Detectors (SUN RGB-D)", file=sys.stdout):
            _, model_info, score = train_concept_detector(original_idx, embeddings_train_segments, cluster_labels_train_segments, image_groups_train_segments, config)
            if model_info and score >= config['min_detector_score']: trained_detectors[original_idx] = model_info
        if config.get("use_detectors_cache", True) and trained_detectors: 
            with open(detectors_cache_file, "wb") as f: pickle.dump(trained_detectors, f)
    if not trained_detectors: main_logger.error("No concept detectors trained successfully!"); return
    ordered_final_concept_original_ids_for_features = sorted(list(trained_detectors.keys()))
    num_final_figs_features = len(ordered_final_concept_original_ids_for_features)
    main_logger.info(f"Trained and kept {num_final_figs_features} concept detectors.")

    # --- Phase 6: Symbolic Concept Vectorization (TRAIN DATA) ---
    main_logger.info(f"--- Phase 6: Concept Vectorization (Training Data) using {config['vectorizer_strategy']} ---")
    train_vectors_cache_file = os.path.join(config["central_run_cache_dir"], f"train_vectors_sunrgbd_{config['run_id']}.pkl")
    if config.get("use_train_vectors_cache", True) and os.path.exists(train_vectors_cache_file):
        try:
            with open(train_vectors_cache_file, "rb") as f: X_train_concepts, _ = pickle.load(f) # Discard cached train_ids
            main_logger.info(f"Loaded cached TRAIN concept vectors. Shape: {X_train_concepts.shape}")
            if X_train_concepts.shape[1] != num_final_figs_features: raise ValueError("Dim mismatch.")
        except Exception as e:
            main_logger.warning(f"Failed to load TRAIN vectors cache: {e}. Recomputing.")
            X_train_concepts, _ = build_centralized_concept_vectors_maxpool(seg_infos_train_np, embeddings_train_segments, trained_detectors, ordered_final_concept_original_ids_for_features, len(images_train_rgb_list), train_base_ids, config, main_logger)
            if config.get("use_train_vectors_cache", True):
                with open(train_vectors_cache_file, "wb") as f: pickle.dump((X_train_concepts, train_base_ids), f)
    else:
        X_train_concepts, _ = build_centralized_concept_vectors_maxpool(seg_infos_train_np, embeddings_train_segments, trained_detectors, ordered_final_concept_original_ids_for_features, len(images_train_rgb_list), train_base_ids, config, main_logger)
        if config.get("use_train_vectors_cache", True) and X_train_concepts is not None:
            with open(train_vectors_cache_file, "wb") as f: pickle.dump((X_train_concepts, train_base_ids), f)
    if X_train_concepts is None: main_logger.error("Train concept vectorization failed!"); return
    visualize_concept_vectors_pca(X_train_concepts, y_train_labels, "TrainImageConceptVecs_SUNRGBD", config, main_logger)

    # --- Phase 7: Process Test Data ---
    main_logger.info("--- Phase 7: Processing Test Data (Embedding, Vectorization) ---")
    X_test_concepts = np.empty((0, num_final_figs_features)) # Default if no test data
    if len(seg_infos_test_np) > 0:
        main_logger.info("Preparing 'seg_crop_bgr' for TEST segments...")
        seg_crops_test_cache_file = os.path.join(config["central_run_cache_dir"], f"seg_crops_test_sunrgbd_{config['run_id']}.pkl")
        if config.get("use_seg_crops_cache", True) and os.path.exists(seg_crops_test_cache_file):
            try:
                with open(seg_crops_test_cache_file, "rb") as f: seg_infos_test_np_with_crops = pickle.load(f)
                main_logger.info(f"Loaded cached TEST seg_crops.")
            except Exception as e:
                main_logger.warning(f"Failed to load TEST seg_crops cache: {e}. Recomputing.")
                seg_infos_test_np_with_crops = add_seg_crop_bgr_to_split_infos_local(seg_infos_test_np, images_test_rgb_list, masks_test_per_image_list, main_logger, config["min_mask_pixels_for_crop"])
                if config.get("use_seg_crops_cache", True): 
                    with open(seg_crops_test_cache_file, "wb") as f: pickle.dump(seg_infos_test_np_with_crops, f)
        else:
            seg_infos_test_np_with_crops = add_seg_crop_bgr_to_split_infos_local(seg_infos_test_np, images_test_rgb_list, masks_test_per_image_list, main_logger, config["min_mask_pixels_for_crop"])
            if config.get("use_seg_crops_cache", True) and seg_infos_test_np_with_crops is not None: 
                with open(seg_crops_test_cache_file, "wb") as f: pickle.dump(seg_infos_test_np_with_crops, f)
        
        embeddings_test_segments = compute_final_embeddings(
            seg_infos_test_np_with_crops, images_test_rgb_list, None,
            dino_processor, dino_model, target_model_resnet,
            torch_device, config, client_id=f"central_test_sunrgbd_{config['run_id']}"
        )
        if embeddings_test_segments is not None and embeddings_test_segments.shape[0] > 0:
            main_logger.info(f"Test segment embeddings computed. Shape: {embeddings_test_segments.shape}")
            test_vectors_cache_file = os.path.join(config["central_run_cache_dir"], f"test_vectors_sunrgbd_{config['run_id']}.pkl")
            if config.get("use_test_vectors_cache", True) and os.path.exists(test_vectors_cache_file):
                try:
                    with open(test_vectors_cache_file, "rb") as f: X_test_concepts, _ = pickle.load(f)
                    main_logger.info(f"Loaded cached TEST concept vectors. Shape: {X_test_concepts.shape}")
                    if X_test_concepts.shape[1] != num_final_figs_features: raise ValueError("Dim mismatch.")
                except Exception as e:
                    main_logger.warning(f"Failed to load TEST vectors cache: {e}. Recomputing.")
                    X_test_concepts, _ = build_centralized_concept_vectors_maxpool(seg_infos_test_np, embeddings_test_segments, trained_detectors, ordered_final_concept_original_ids_for_features, len(images_test_rgb_list), test_base_ids, config, main_logger)
                    if config.get("use_test_vectors_cache", True): 
                        with open(test_vectors_cache_file, "wb") as f: pickle.dump((X_test_concepts, test_base_ids), f)
            else:
                X_test_concepts, _ = build_centralized_concept_vectors_maxpool(seg_infos_test_np, embeddings_test_segments, trained_detectors, ordered_final_concept_original_ids_for_features, len(images_test_rgb_list), test_base_ids, config, main_logger)
                if config.get("use_test_vectors_cache", True) and X_test_concepts is not None: 
                    with open(test_vectors_cache_file, "wb") as f: pickle.dump((X_test_concepts, test_base_ids), f)
            if X_test_concepts is None: X_test_concepts = np.empty((0, num_final_figs_features)) # Ensure defined
            visualize_concept_vectors_pca(X_test_concepts, y_test_labels_final, "TestImageConceptVecs_SUNRGBD", config, main_logger)
    else:
        main_logger.warning("No segments for test data. X_test_concepts will be empty.")
    
    # --- Phase 8: FIGS Model Training & Hyperparameter Sweep ---
    main_logger.info("--- Phase 8: FIGS Model Training & Hyperparameter Sweep ---")
    figs_feature_names = [f"concept_{i}" for i in range(num_final_figs_features)]
    
    max_rules_sweep = config.get("figs_max_rules_sweep", [20, 30, 50, 80, 100, 120, 150]) # Example sweep values
    max_trees_sweep = config.get("figs_max_trees_sweep", [1, 3, 5, None])
    max_features_sweep = config.get("figs_max_features_sweep", [None, 'sqrt']) # Example
    
    results_log = []
    best_figs_model_for_dataset = None
    best_accuracy_for_dataset = -1.0
    best_params_for_dataset = {}

    y_train_labels_1d_for_figs = y_train_labels.ravel() if y_train_labels.ndim > 1 else y_train_labels
    df_train_concepts_for_figs = pd.DataFrame(X_train_concepts, columns=figs_feature_names)

    for rules_val in max_rules_sweep:
        for trees_val in max_trees_sweep:
            for features_val in max_features_sweep:
                current_figs_params_combo = config['figs_params'].copy() # Start with base seed
                current_figs_params_combo['max_rules'] = rules_val
                current_figs_params_combo['max_trees'] = trees_val
                current_figs_params_combo['max_features'] = features_val
                
                main_logger.info(f"Testing FIGS with params: {current_figs_params_combo}")
                figs_model_iter = PatchedFIGSClassifier(**current_figs_params_combo, n_outputs_global=config['num_classes'])
                
                try:
                    figs_model_iter.fit(
                        df_train_concepts_for_figs, y_train_labels_1d_for_figs, 
                        feature_names=figs_feature_names, _y_fit_override=None
                    )
                    main_logger.info(f"  FIGS trained. Complexity: {getattr(figs_model_iter, 'complexity_', 'N/A')}, Trees: {len(getattr(figs_model_iter,'trees_',[]))}")

                    accuracy, rule_prec, rule_fid = 0.0, 0.0, 0.0
                    if X_test_concepts.shape[0] > 0 and y_test_labels_final.shape[0] == X_test_concepts.shape[0]:
                        y_test_1d_for_eval = y_test_labels_final.ravel() if y_test_labels_final.ndim > 1 else y_test_labels_final
                        accuracy, rule_prec, rule_fid = figs_lr_xfl_metrics(
                            figs_model_iter, X_test_concepts, y_test_1d_for_eval, figs_feature_names, main_logger
                        )
                        main_logger.info(f"  Results for {current_figs_params_combo}: ModelAcc={accuracy:.4f}, DNF_RuleAcc={rule_prec:.4f}, DNF_RuleFid={rule_fid:.4f}")
                        
                        results_log.append({**current_figs_params_combo, "ModelAcc": accuracy, "DNF_RuleAcc": rule_prec, "DNF_RuleFid": rule_fid})
                        if accuracy > best_accuracy_for_dataset:
                            best_accuracy_for_dataset = accuracy
                            best_params_for_dataset = current_figs_params_combo.copy()
                            best_figs_model_for_dataset = figs_model_iter # Store the best model instance
                    else:
                        main_logger.warning("  Skipping evaluation: No test data or label mismatch.")
                except Exception as e_figs_fit:
                    main_logger.error(f"  Error training/evaluating FIGS with {current_figs_params_combo}: {e_figs_fit}", exc_info=True)
                main_logger.info("-" * 30)

    main_logger.info(f"--- FIGS Hyperparameter Sweep Complete for SUN RGB-D ---")
    main_logger.info(f"Best Accuracy: {best_accuracy_for_dataset:.4f} with params: {best_params_for_dataset}")
    
    results_df = pd.DataFrame(results_log)
    results_filename = os.path.join(config['log_dir'], f"figs_sunrgbd_autocore_hparam_sweep_results_{config['run_id']}.csv")
    results_df.to_csv(results_filename, index=False)
    main_logger.info(f"Hyperparameter sweep results saved to {results_filename}")

    if best_figs_model_for_dataset:
        with open(config["final_model_path"], "wb") as f: pickle.dump(best_figs_model_for_dataset, f)
        main_logger.info(f"Best FIGS model saved to {config['final_model_path']}")
        # Save concept definitions (detectors and final chosen concepts)
        concept_defs_to_save = {
            'ordered_final_concept_original_ids': ordered_final_concept_original_ids_for_features,
            'trained_detectors_map': trained_detectors, # {orig_kmeans_idx: (pipeline, threshold)}
            'figs_feature_names': figs_feature_names
        }
        with open(config["concept_definitions_path"], "wb") as f: pickle.dump(concept_defs_to_save, f)
        main_logger.info(f"Concept definitions saved to {config['concept_definitions_path']}")
    
    main_logger.info(f"======== Centralized AutoCoRe-FIGS for SUN RGB-D - Run ID: {config['run_id']} Complete ========")

if __name__ == "__main__":
    main_centralized_autocore_sunrgbd()