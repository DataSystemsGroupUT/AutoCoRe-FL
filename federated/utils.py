import logging
import os
import sys
import time
import json
import cv2
import pandas as pd
import yaml 
import numpy as np
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Dict
from collections import defaultdict
import operator
import re
from scipy.special import expit 
from tqdm import tqdm
from scipy.special import softmax 
from torch.utils.data import TensorDataset, Subset 
import random


def ade20k_iid(dataset_subset, num_users, seed=42):
    """
    Sample I.I.D. client data from a given dataset_subset.
    dataset_subset can be a TensorDataset or a Subset.
    Returns dict: {user_id: set_of_indices_relative_to_dataset_subset}
    """
    np.random.seed(seed)
    num_items_in_subset = len(dataset_subset)
    all_indices_in_subset = list(range(num_items_in_subset))
    
    dict_users_subset_indices = {i: set() for i in range(num_users)}
    if num_users == 0 or num_items_in_subset == 0:
        return dict_users_subset_indices

    num_items_per_user = num_items_in_subset // num_users
    if num_items_per_user == 0: # Fewer items than users
        for i in range(num_items_in_subset):
            dict_users_subset_indices[i % num_users].add(all_indices_in_subset[i]) # Distribute one by one
        return dict_users_subset_indices

    temp_indices_pool = list(all_indices_in_subset) # Work with a copy for np.random.choice
    for i in range(num_users):
        current_items_for_this_user = num_items_per_user + (1 if i < (num_items_in_subset % num_users) else 0)
        items_to_pick = min(current_items_for_this_user, len(temp_indices_pool))
        if items_to_pick > 0:
            chosen_indices_arr = np.random.choice(temp_indices_pool, items_to_pick, replace=False)
            dict_users_subset_indices[i] = set(chosen_indices_arr)
            temp_indices_pool = [idx for idx in temp_indices_pool if idx not in chosen_indices_arr]
        else:
            dict_users_subset_indices[i] = set()
    return dict_users_subset_indices


def ade20k_noniid_by_concept(dataset_subset, num_users, num_concepts_for_sharding, seed=42):
    """
    Partitions dataset_subset non-IID based on a dominant concept per sample.
    Args:
        dataset_subset: A PyTorch Dataset (can be TensorDataset or Subset).
        num_users: Number of clients.
        num_concepts_for_sharding: Number of categories to shard concepts into.
                                   This should be the number of features in X if sharding by feature index.
        seed: Random seed.
    Returns:
        dict_users: {user_id: numpy_array_of_indices_relative_to_dataset_subset}
    """
    np.random.seed(seed)
    random.seed(seed)

    # Get the feature tensor (X) correctly from dataset_subset
    if isinstance(dataset_subset, TensorDataset):
        feature_tensor = dataset_subset.tensors[0] 
        num_samples_in_subset = feature_tensor.shape[0]
    elif isinstance(dataset_subset, Subset):
        # If it's a Subset, access the original dataset and use the subset's indices
        original_dataset_feature_tensor = dataset_subset.dataset.tensors[0]
        feature_tensor = original_dataset_feature_tensor[dataset_subset.indices]
        num_samples_in_subset = len(dataset_subset.indices)
    else:
        raise TypeError("dataset_subset must be a PyTorch TensorDataset or Subset.")

    if num_samples_in_subset == 0:
        return {i: np.array([], dtype=int) for i in range(num_users)}

    # Determine the "dominant concept class" for each sample in the subset
    dominant_concept_per_sample = []
    for i in range(num_samples_in_subset):
        row = feature_tensor[i] 
        max_val = row.max().item()
        indices_of_max_vals = (row == max_val).nonzero(as_tuple=True)[0].cpu().numpy()
        if indices_of_max_vals.size > 0:
            selected_dominant_concept_index = np.random.choice(indices_of_max_vals)
        else: 
            selected_dominant_concept_index = np.random.randint(0, feature_tensor.shape[1]) # Fallback: random concept if all are zero or negative uniformly
        dominant_concept_per_sample.append(selected_dominant_concept_index)
    
    input_class_for_sharding = np.array(dominant_concept_per_sample)

    # Create shards of these dominant concept indices
    # num_concepts_for_sharding IS the number of features (e.g., 150 for ResNet concepts)
    concept_indices_to_be_sharded = list(range(num_concepts_for_sharding)) 
    random.shuffle(concept_indices_to_be_sharded)

    user_concept_shards = {i: [] for i in range(num_users)}
    idx_counter = 0
    num_shards_per_user_base = num_concepts_for_sharding // num_users
    remainder_shards = num_concepts_for_sharding % num_users

    for user_id in range(num_users):
        take_count = num_shards_per_user_base + (1 if user_id < remainder_shards else 0)
        user_concept_shards[user_id] = concept_indices_to_be_sharded[idx_counter : idx_counter + take_count]
        idx_counter += take_count
        
    dict_users_sample_indices = {i: [] for i in range(num_users)}
    all_sample_indices_in_subset = np.arange(num_samples_in_subset) # These are 0 to N-1 for the current subset

    for user_id in range(num_users):
        concept_shard_for_this_user = user_concept_shards[user_id]
        if not concept_shard_for_this_user:
            dict_users_sample_indices[user_id] = np.array([], dtype=int)
            continue
        
        # Find samples whose dominant concept is in this user's shard
        # np.isin checks if elements of input_class_for_sharding are present in concept_shard_for_this_user
        mask_user_samples = np.isin(input_class_for_sharding, concept_shard_for_this_user)
        dict_users_sample_indices[user_id] = all_sample_indices_in_subset[mask_user_samples]
        
    return dict_users_sample_indices

# --- Helper: Load Labels for Base IDs (Ensure consistent with run_federated_train_boosting) ---
def load_labels_for_base_ids(
    base_ids_list: list,
    full_scene_map: dict, 
    scene_name_to_idx_map: dict,
    logger_instance: logging.Logger
) -> np.ndarray:
    labels = []
    missing_in_scene_map = 0
    missing_in_idx_map = 0
    for base_id in base_ids_list:
        scene_name = full_scene_map.get(base_id)
        if scene_name:
            label_idx = scene_name_to_idx_map.get(scene_name)
            if label_idx is not None:
                labels.append(label_idx)
            else:
                missing_in_idx_map +=1
                logger_instance.debug(f"Label Load: Scene '{scene_name}' for base_id '{base_id}' not in scene_name_to_idx_map.")
                labels.append(-1) # Placeholder, will be filtered
        else:
            missing_in_scene_map +=1
            logger_instance.debug(f"Label Load: Base ID '{base_id}' not found in full_scene_map.")
            labels.append(-1)
    if missing_in_scene_map > 0: logger_instance.warning(f"{missing_in_scene_map} base_ids not found in scene_map.")
    if missing_in_idx_map > 0: logger_instance.warning(f"{missing_in_idx_map} scene_names not found in scene_to_idx_map.")
    
    final_labels = np.array([l for l in labels if l != -1], dtype=np.int64)
    if len(final_labels) != len(base_ids_list):
        logger_instance.warning(f"Label loading resulted in {len(final_labels)} labels for {len(base_ids_list)} base_ids due to missing mappings.")
    return final_labels


def load_labels_from_manifest(manifest_path: str, main_logger_passed: logging.Logger) -> tuple[list, np.ndarray]:
    """
    Load base IDs and integer labels from a manifest file.
    """""
    base_ids = []
    labels_int = []
    try:
        with open(manifest_path, 'r') as f:
            manifest_data = yaml.safe_load(f) 
        for item in manifest_data:
            base_ids.append(item['base_id'])
            labels_int.append(item['label_int'])
        return base_ids, np.array(labels_int, dtype=np.int64)
    except FileNotFoundError:
        main_logger_passed.error(f"Manifest file not found: {manifest_path}")
        return [], np.array([], dtype=np.int64)
    except Exception as e:
        main_logger_passed.error(f"Error loading manifest {manifest_path}: {e}")
        return [], np.array([], dtype=np.int64)


def evaluate_global_AutoCore_model(
    global_AutoCore_terms: list, X_eval_data_concepts: np.ndarray, y_eval_labels: np.ndarray,
    feature_names: list, num_total_classes: int, main_logger: logging.Logger, eval_set_name: str = "Eval"
):
    main_logger.info(f"Evaluating global AutoCore model ({eval_set_name}) with {len(global_AutoCore_terms)} terms on data of shape {X_eval_data_concepts.shape}...")
    if X_eval_data_concepts is None or X_eval_data_concepts.shape[0] == 0 or not global_AutoCore_terms:
        main_logger.warning(f"No data or no global terms to evaluate for {eval_set_name}. Returning zeros.")
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    if not feature_names or len(feature_names) != X_eval_data_concepts.shape[1]:
        main_logger.error(f"Feature names count ({len(feature_names) if feature_names else 'None'}) mismatch with X_eval_data_concepts columns ({X_eval_data_concepts.shape[1]}) for {eval_set_name}. Cannot evaluate rules accurately.")
        if not global_AutoCore_terms or not isinstance(global_AutoCore_terms, list) or not global_AutoCore_terms[0].get('aggregated_value_array'):
             main_logger.error(f"Cannot perform fallback accuracy for {eval_set_name}: global_figs_terms malformed or empty.")
             return 0.0,0.0,0.0,0.0,0.0
        try:
            summed_scores_for_pred_fallback = np.zeros((X_eval_data_concepts.shape[0], num_total_classes))
            for i_sample_fallback in range(X_eval_data_concepts.shape[0]):
                sample_total_value_fallback = np.zeros(num_total_classes)
                for term_fallback in global_AutoCore_terms:
                    term_val_fallback = term_fallback.get('aggregated_value_array', [])
                    if isinstance(term_val_fallback, list) and len(term_val_fallback) == num_total_classes: sample_total_value_fallback += np.array(term_val_fallback)
                summed_scores_for_pred_fallback[i_sample_fallback] = sample_total_value_fallback
            y_pred_probs_fallback = softmax(summed_scores_for_pred_fallback, axis=1)
            y_pred_model_labels_fallback = np.argmax(y_pred_probs_fallback, axis=1)
            model_accuracy_fallback = accuracy_score(y_eval_labels, y_pred_model_labels_fallback)
            main_logger.info(f"Fallback Model Accuracy ({eval_set_name}, rules not eval'd due to feat name issue): {model_accuracy_fallback:.4f}")
            return model_accuracy_fallback, 0.0, 0.0, 0.0, 0.0
        except Exception as e_fallback_acc_eval:
            main_logger.error(f"Fallback accuracy calculation failed for {eval_set_name}: {e_fallback_acc_eval}")
            return 0.0, 0.0, 0.0, 0.0, 0.0

    def evaluate_AutoCore_rule_str_on_sample_helper(rule_str_eval, x_sample_dict_eval, logger_inst_eval, feat_names_list_eval):
        if rule_str_eval == "True": return True
        for cond_str_eval in rule_str_eval.split(' & '):
            match_eval = re.match(r"`(.+?)`\s*([><]=?)\s*([0-9eE.+-]+)", cond_str_eval.strip())
            if not match_eval: logger_inst_eval.debug(f"Helper: No match for cond '{cond_str_eval}' in '{rule_str_eval}'"); return False
            feat_eval, op_eval, val_s_eval = match_eval.groups()
            if feat_eval not in x_sample_dict_eval:
                if not hasattr(evaluate_AutoCore_rule_str_on_sample_helper, 'logged_missing_helper_eval'): evaluate_AutoCore_rule_str_on_sample_helper.logged_missing_helper_eval = set()
                if feat_eval not in evaluate_AutoCore_rule_str_on_sample_helper.logged_missing_helper_eval: 
                    logger_inst_eval.warning(f"Helper: Feature '{feat_eval}' in rule not in sample ({eval_set_name}). Avail keys: {list(x_sample_dict_eval.keys())[:5]}"); evaluate_AutoCore_rule_str_on_sample_helper.logged_missing_helper_eval.add(feat_eval)
                return False
            s_val_eval, cond_v_eval = x_sample_dict_eval[feat_eval], float(val_s_eval)
            op_fn_eval = {'<=':operator.le,'>':operator.gt,'<':operator.lt,'>=':operator.ge,'==':operator.eq}.get(op_eval)
            if not (op_fn_eval and op_fn_eval(s_val_eval, cond_v_eval)): return False
        return True

    summed_scores = np.zeros((X_eval_data_concepts.shape[0], num_total_classes))
    X_eval_data_dicts = [dict(zip(feature_names, row)) for row in X_eval_data_concepts]
    for i_sample, x_dict_loop in enumerate(tqdm(X_eval_data_dicts, desc=f"Eval Global Model Samples ({eval_set_name})", file=sys.stdout, leave=False)):
        sample_total_value = np.zeros(num_total_classes)
        for term in global_AutoCore_terms:
            if evaluate_AutoCore_rule_str_on_sample_helper(term['rule_str'], x_dict_loop, main_logger, feature_names):
                term_val_arr = term.get('aggregated_value_array', [])
                if isinstance(term_val_arr, list) and len(term_val_arr) == num_total_classes: sample_total_value += np.array(term_val_arr)
        summed_scores[i_sample] = sample_total_value
    y_pred_probs = softmax(summed_scores, axis=1); y_pred_model_labels = np.argmax(y_pred_probs, axis=1)
    model_accuracy = accuracy_score(y_eval_labels, y_pred_model_labels)
    main_logger.info(f"Global Model Accuracy ({eval_set_name}, from terms): {model_accuracy:.4f}")
    rule_precisions, rule_coverages, rule_complexities, rule_fidelities = [], [], [], []
    for term in tqdm(global_AutoCore_terms, desc=f"Eval Global Rules ({eval_set_name})", file=sys.stdout, leave=False):
        rule_str = term['rule_str']; value_array = np.array(term.get('aggregated_value_array', []))
        rule_intended_class = np.argmax(value_array) if value_array.size > 0 else -1
        rule_complexities.append(rule_str.count('&') + 1 if rule_str != "True" else 0)
        fires_mask = np.array([evaluate_AutoCore_rule_str_on_sample_helper(rule_str, x_d, main_logger, feature_names) for x_d in X_eval_data_dicts])
        coverage = np.mean(fires_mask); rule_coverages.append(coverage); num_covered = np.sum(fires_mask)
        if num_covered > 0:
            prec = np.mean(y_eval_labels[fires_mask] == rule_intended_class) if rule_intended_class != -1 else 0.0
            fid = np.mean(y_pred_model_labels[fires_mask] == rule_intended_class) if rule_intended_class != -1 else 0.0
            rule_precisions.append(prec); rule_fidelities.append(fid)
        else: rule_precisions.append(0.0); rule_fidelities.append(0.0)
    mean_prec = np.mean(rule_precisions) if rule_precisions else 0.0; mean_cov = np.mean(rule_coverages) if rule_coverages else 0.0
    mean_comp = np.mean(rule_complexities) if rule_complexities else 0.0; mean_fid = np.mean(rule_fidelities) if rule_fidelities else 0.0
    main_logger.info(f"  Mean Rule Metrics ({eval_set_name}): P={mean_prec:.3f}, C={mean_cov:.3f}, L={mean_comp:.2f}, F={mean_fid:.3f}")
    return model_accuracy, mean_prec, mean_cov, mean_comp, mean_fid


def build_centralized_concept_vectors_maxpool( 
    segment_infos_split_np,         # Flat numpy array of dicts for segments IN THIS DATA SPLIT
    embeddings_for_segments_in_split_np, # Embeddings for segments IN THIS SPLIT
    trained_concept_detectors_dict, # {original_kmeans_idx: (pipeline_obj, optimal_threshold_float)}
    ordered_final_concept_original_ids, # List mapping dense feature idx to original_kmeans_idx
    num_total_images_in_split,      # Total number of unique images in this data split
    image_base_ids_in_split,        # List of base_ids for images in this split
    config_dict, main_logger_passed
    ):
    """
    Build centralized concept vectors using MAX-POOLING strategy.
    This function computes concept vectors for a given split of data by applying
    max-pooling over the segment embeddings for each concept detector.
    It uses the trained concept detectors to predict probabilities for each segment
    and then aggregates these probabilities to form the final concept vectors.
    """
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
            logger_main.warning(f"Detector not found for concept orig_idx {original_kmeans_idx} (dense_idx {dense_feature_idx}). Filling with zeros.")
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
            
            if not global_indices_for_this_image_segments_in_split:
                continue
                
            probs_for_this_image_segments = all_split_seg_probs_for_concept[global_indices_for_this_image_segments_in_split]
            max_prob_for_image = np.max(probs_for_this_image_segments) if probs_for_this_image_segments.size > 0 else 0.0
            
            if max_prob_for_image >= optimal_threshold:
                concept_vecs_for_split[split_img_idx, dense_feature_idx] = 1.0
    
    return concept_vecs_for_split, list(image_base_ids_in_split)

def add_seg_crop_bgr_to_split_infos(segment_infos_split, images_rgb_split_list, masks_per_image_split_list, main_logger_passed_crop):
    """
    Adds 'seg_crop_bgr' field to each segment info in segment_infos_split.
    This field contains the cropped BGR image of the segment, or None if the segment is invalid.
    """
    processed_infos = []
    for seg_info_item in tqdm(segment_infos_split, desc="Preparing seg_crop_bgr", file=sys.stdout):
        if 'seg_crop_bgr' in seg_info_item and seg_info_item['seg_crop_bgr'] is not None:
            processed_infos.append(seg_info_item)
            continue

        new_seg_info_item = dict(seg_info_item)
        try:
            local_img_idx = new_seg_info_item['img_idx']
            seg_idx_in_img = new_seg_info_item['seg_idx']
            
            if local_img_idx >= len(images_rgb_split_list) or \
                local_img_idx >= len(masks_per_image_split_list) or \
                seg_idx_in_img >= len(masks_per_image_split_list[local_img_idx]):
                main_logger_passed_crop.warning(f"Index out of bounds for seg_crop_bgr creation: img_idx {local_img_idx}, seg_idx {seg_idx_in_img}")
                new_seg_info_item['seg_crop_bgr'] = None
                processed_infos.append(new_seg_info_item)
                continue

            original_img_rgb = images_rgb_split_list[local_img_idx]
            seg_mask_bool = masks_per_image_split_list[local_img_idx][seg_idx_in_img]

            ys, xs = np.where(seg_mask_bool)
            if len(ys) == 0: new_seg_info_item['seg_crop_bgr'] = None; processed_infos.append(new_seg_info_item); continue
            top, left, bottom, right = np.min(ys), np.min(xs), np.max(ys), np.max(xs)
            
            seg_crop_rgb_temp = original_img_rgb[top:bottom+1, left:right+1].copy()
            local_mask_temp = seg_mask_bool[top:bottom+1, left:right+1]
            seg_crop_rgb_temp[~local_mask_temp] = (0,0,0)
            new_seg_info_item['seg_crop_bgr'] = cv2.cvtColor(seg_crop_rgb_temp, cv2.COLOR_RGB2BGR)
        except Exception as e_crop:
            # main_logger_passed_crop.error(f"Error creating seg_crop_bgr for segment: {e_crop}", exc_info=True)
            new_seg_info_item['seg_crop_bgr'] = None
        processed_infos.append(new_seg_info_item)
    return np.array(processed_infos, dtype=object)

def get_all_leaf_paths_and_masks(
    tree_root, # Should be a LocalNode object (root of a tree from figs_model.trees_)
    feature_names: List[str], 
    X_df: pd.DataFrame, # DataFrame of X_eval_data_concepts
    n_samples: int,
    logger: logging.Logger
) -> List[Tuple[str, np.ndarray, np.ndarray]]: # (rule_str, leaf_value_vector, mask_for_rule)
    """
    Recursively traverses a FIGS tree to find all leaf nodes and extract:
    - The conjunctive rule string (path conditions) leading to the leaf.
    - The value vector of the leaf node.
    - A boolean mask indicating which samples in X_df satisfy the rule.
    """
    leaf_details = []
    
    def recurse(node, current_conditions_list: List[str]):
        if node is None: # Should not happen if called with a valid node
            return

        if node.left is None and node.right is None: # Leaf node
            rule_str = " & ".join(current_conditions_list) if current_conditions_list else "True"
            
            current_mask = np.ones(n_samples, dtype=bool)
            if rule_str != "True":
                try:
                    current_mask = X_df.eval(rule_str).to_numpy(dtype=bool)
                except Exception as e_eval: # Fallback manual parser if X_df.eval fails
                    # logger.debug(f"X_df.eval failed for rule '{rule_str[:100]}': {e_eval}. Using manual parser.")
                    current_mask.fill(True) # Start with all true for this path
                    parse_success = True
                    for cond_str_part in current_conditions_list: 
                        m = re.match(r"`(.+?)`\s*([><]=?)\s*([0-9eE.+-]+)", cond_str_part.strip()) # Improved regex for numbers
                        if not m:
                            logger.error(f"PathParseFail: Could not parse condition '{cond_str_part}' in rule '{rule_str[:100]}'")
                            current_mask.fill(False); parse_success = False; break 
                        
                        feat, op, val_str_parsed = m.groups()
                        try:
                            val_fl = float(val_str_parsed)
                        except ValueError:
                            logger.error(f"PathParseFail: Could not convert value '{val_str_parsed}' to float in condition '{cond_str_part}'")
                            current_mask.fill(False); parse_success = False; break

                        if feat not in X_df.columns:
                            logger.error(f"PathParseFail: Feature '{feat}' not in X_df columns for rule '{rule_str[:100]}'")
                            current_mask.fill(False); parse_success = False; break
                            
                        arr_feat_data = X_df[feat].to_numpy()
                        op_func = {'<=': operator.le, '>': operator.gt,
                                   '<': operator.lt, '>=': operator.ge,
                                   '==': operator.eq, '!=': operator.ne}.get(op)
                        
                        if not op_func:
                            logger.error(f"PathParseFail: Unknown operator '{op}' in condition '{cond_str_part}'")
                            current_mask.fill(False); parse_success = False; break
                        
                        try:
                            current_mask &= op_func(arr_feat_data, val_fl)
                        except Exception as e_op:
                            logger.error(f"PathParseFail: Error applying op '{op}' for condition '{cond_str_part}': {e_op}")
                            current_mask.fill(False); parse_success = False; break
                    if not parse_success:
                         logger.warning(f"Manual parse failed for rule: '{rule_str[:100]}'. Mask set to all False.")


            leaf_value = node.value # This should be (1, n_classes) or (n_classes,)
            if isinstance(leaf_value, np.ndarray):
                leaf_value_flat = leaf_value.flatten()
            else: # Fallback if not numpy array (should not happen with PatchedFIGS)
                logger.warning(f"Leaf value is not ndarray: {type(leaf_value)}. Using zeros.")
                leaf_value_flat = np.zeros(len(feature_names)) #  adjust num_outputs if available

            leaf_details.append( (rule_str, leaf_value_flat, current_mask) )
            return

        # Internal node
        if not hasattr(node, 'feature') or node.feature is None or \
           not hasattr(node, 'threshold') or node.threshold is None:
            # logger.warning(f"Internal node missing feature/threshold: {node}")
            return # Cannot proceed with this path

        f_idx = node.feature
        thr = node.threshold
        
        if not (0 <= f_idx < len(feature_names)):
            logger.error(f"Invalid feature index {f_idx} encountered during path extraction. Max index: {len(feature_names)-1}. Node: {node}")
            return 

        f_name = feature_names[f_idx] # Use the actual concept name for the rule string

        # Go left
        left_cond = f"`{f_name}` <= {thr:.6f}" # Using backticks for pandas.eval compatibility
        if node.left:
            recurse(node.left, current_conditions_list + [left_cond])
        
        # Go right
        right_cond = f"`{f_name}` > {thr:.6f}"
        if node.right:
            recurse(node.right, current_conditions_list + [right_cond])

    if tree_root is not None: # Ensure tree_root itself isn't None
        recurse(tree_root, [])
    return leaf_details

def calculate_metrics(
    figs_model, 
    X_eval: np.ndarray, # Raw concept vectors (e.g., X_test_concepts)
    y_eval: np.ndarray, # True 1D integer class labels
    feature_names: List[str], # List of concept names, e.g., ["concept_0", "concept_1", ...]
    logger: logging.Logger,
) -> Tuple[float, float, float]:
    """
    Calculates metrics for a AutoCore model
    - Model Accuracy: Standard predictive accuracy of the AutoCore model.
    - Rule Accuracy: Macro-averaged accuracy of DNF rules (OR of leaves) per class.
    - Rule Fidelity: Macro-averaged fidelity of DNF rules per class against AutoCore model predictions.

    Args:
        figs_model: Trained instance of PatchedFIGSClassifier.
        X_eval: Evaluation feature data (concept vectors), NumPy array.
        y_eval: True evaluation labels (1D integer labels), NumPy array.
        feature_names: List of names for the features in X_eval.
        logger: Logger instance.

    Returns:
        A tuple: (model_accuracy, mean_dnf_rule_accuracy, mean_dnf_rule_fidelity)
    """
    log = logger

    if not hasattr(figs_model, 'predict') or not hasattr(figs_model, 'classes_') or not hasattr(figs_model, 'trees_'):
        log.error("Provided figs_model is not a valid/fitted PatchedFIGSClassifier instance.")
        return 0.0, 0.0, 0.0

    # ---------- 1) Model predictions & Overall Accuracy ----------
    if X_eval.shape[0] == 0:
        log.warning("X_eval is empty. Cannot compute metrics.")
        return 0.0, 0.0, 0.0
        
    df_X_eval = pd.DataFrame(X_eval, columns=feature_names) # For X_df.eval()
    n_samples = df_X_eval.shape[0]

    try:
        y_hat_model_predictions = figs_model.predict(X_eval) # Should return 1D integer class labels
    except Exception as e_pred:
        log.error(f"Error during figs_model.predict(X_eval): {e_pred}")
        return 0.0, 0.0, 0.0
        
    model_overall_accuracy = accuracy_score(y_eval, y_hat_model_predictions)
    log.info(f"Overall Model Predictive Accuracy: {model_overall_accuracy:.4f}")

    # ---------- 2) Extract all leaf rules and their masks from all trees ----------
    # class_leaf_masks will store: {class_idx_predicted_by_leaf: [mask1, mask2, ...]}
    
    n_model_classes = len(figs_model.classes_) # Number of unique classes model was trained on
    class_leaf_masks: Dict[int, List[np.ndarray]] = defaultdict(list)

    for tree_idx, tree_root_node in enumerate(figs_model.trees_):
        if tree_root_node is None:
            log.warning(f"Tree {tree_idx} is None. Skipping.")
            continue

        all_paths_in_this_tree = get_all_leaf_paths_and_masks(
            tree_root_node, 
            feature_names, 
            df_X_eval, # Pass DataFrame for eval
            n_samples,
            logger
        )

        for _rule_str, leaf_value_vector, mask_for_rule in all_paths_in_this_tree:
            if mask_for_rule.sum() == 0: # Rule doesn't fire on any eval sample
                continue
            
            if leaf_value_vector.size != n_model_classes:
                if leaf_value_vector.size > 0:
                    predicted_class_idx_for_leaf = int(np.argmax(leaf_value_vector))
                    if not (0 <= predicted_class_idx_for_leaf < n_model_classes):
                        continue
                else:
                    # log.warning(f"Empty leaf value vector for rule '{_rule_str}'. Skipping leaf.")
                    continue
            else:
                predicted_class_idx_for_leaf = int(np.argmax(leaf_value_vector))
            
            class_leaf_masks[predicted_class_idx_for_leaf].append(mask_for_rule)

    # ---------- 3) Per-class DNF (OR of leaf rules) Evaluation ----------
    dnf_rule_accuracies_per_class = []
    dnf_rule_fidelities_per_class = []

    for class_idx_numeric in range(n_model_classes): # Iterate 0, 1, ..., n_classes-1
        actual_class_label_from_model = figs_model.classes_[class_idx_numeric] # Get the true label value (e.g., 7 for 'bedroom')

        masks_for_this_class = class_leaf_masks.get(class_idx_numeric) # Get masks for leaves predicting this numeric index

        if not masks_for_this_class: # No leaf rules primarily predict this class
            dnf_mask_for_this_class = np.zeros(n_samples, dtype=bool)
            # log.debug(f"No FIGS leaf rules found primarily predicting class '{actual_class_label_from_model}' (idx {class_idx_numeric}). DNF rule is effectively 'False'.")
        else:
            # Fc(x) = OR over all leaf masks that vote for this class_idx_numeric
            dnf_mask_for_this_class = np.any(np.array(masks_for_this_class), axis=0)
        

        current_class_dnf_accuracy = accuracy_score(y_eval == actual_class_label_from_model, dnf_mask_for_this_class)
        dnf_rule_accuracies_per_class.append(current_class_dnf_accuracy)

        # Fidelity of this DNF rule w.r.t. overall AutoCore model's prediction for this class
        # (y_hat_model_predictions == actual_class_label_from_model) is True where AutoCore model predicts this class
        current_class_dnf_fidelity = accuracy_score(y_hat_model_predictions == actual_class_label_from_model, dnf_mask_for_this_class)
        dnf_rule_fidelities_per_class.append(current_class_dnf_fidelity)
        
        # log.info(f"Class '{actual_class_label_from_model}' (Idx {class_idx_numeric}): DNF Rule Acc: {current_class_dnf_accuracy:.4f}, DNF Rule Fid: {current_class_dnf_fidelity:.4f}, DNF Coverage: {dnf_mask_for_this_class.mean():.4f}")


    mean_dnf_rule_accuracy = float(np.mean(dnf_rule_accuracies_per_class)) if dnf_rule_accuracies_per_class else 0.0
    mean_dnf_rule_fidelity = float(np.mean(dnf_rule_fidelities_per_class)) if dnf_rule_fidelities_per_class else 0.0
    
    log.info(
        f"Model Acc={model_overall_accuracy:.4f} | "
        f"Mean DNF Rule Accuracy (Macro Avg)={mean_dnf_rule_accuracy:.4f} | "
        f"Mean DNF Rule Fidelity (Macro Avg)={mean_dnf_rule_fidelity:.4f}"
    )
    return model_overall_accuracy, mean_dnf_rule_accuracy, mean_dnf_rule_fidelity

def setup_logging(log_dir="logs", run_id="run", log_level_str="INFO"):
    """
    Sets up logging to both a file and the console.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{run_id}.log")
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s [%(module)s.%(funcName)s:%(lineno)d]: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout) 
        ]
    )
    return logging.getLogger(f"Run_{run_id}")


def generate_run_id(method_name: str = "run") -> str:
    """
    Generates a unique run ID based on the current timestamp and an optional method name.
    Example: "MyMethod_20231027_153045"
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{method_name}_{timestamp}"


def save_config(config_dict: dict, config_path: str):
    """
    Saves a configuration dictionary to a JSON or YAML file.
    Determines format by file extension.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    try:
        if config_path.endswith(".json"):
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4, sort_keys=True)
        elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
        else:
            # Default to JSON if extension is unknown, or raise error
            base, ext = os.path.splitext(config_path)
            json_path = base + ".json"
            logging.warning(f"Unknown config file extension '{ext}'. Saving as JSON to '{json_path}'.")
            with open(json_path, 'w') as f:
                json.dump(config_dict, f, indent=4, sort_keys=True)
            config_path = json_path # Update path to where it was actually saved

        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save configuration to {config_path}: {e}")


def apply_label_noise_to_client_data(
    y_client_labels: np.ndarray, # 1D NumPy array of integer labels for a client
    noise_degree: float,         # Fraction of labels to shuffle (0.0 to 1.0)
    num_total_classes: int,      # Total number of unique classes in the dataset
    seed: int,
    logger: logging.Logger
) -> np.ndarray:
    """
    Applies label noise to a client's label set by shuffling a portion of them.
    Returns a new array with noisy labels.
    """
    if noise_degree == 0.0:
        return y_client_labels.copy() # No noise

    if y_client_labels.size == 0:
        return y_client_labels # Nothing to add noise to

    n_samples = len(y_client_labels)
    n_to_shuffle = int(n_samples * noise_degree)
    
    if n_to_shuffle == 0:
        return y_client_labels.copy()

    noisy_y = y_client_labels.copy()
    
    local_rng = np.random.default_rng(seed)

    indices_to_shuffle = local_rng.choice(np.arange(n_samples), size=n_to_shuffle, replace=False)
    
    # Create completely random labels for the selected indices
    # These random labels are chosen from the global set of possible classes (0 to num_total_classes-1)
    shuffled_labels_for_selected_indices = local_rng.integers(
        low=0, high=num_total_classes, size=n_to_shuffle
    )
    
    noisy_y[indices_to_shuffle] = shuffled_labels_for_selected_indices
    
    num_changed = np.sum(noisy_y != y_client_labels)
    logger.debug(f"Applied label noise: {noise_degree*100:.1f}% degree. "
                 f"{n_to_shuffle} labels targeted, {num_changed} actually changed (max possible if all random labels differ).")
    return noisy_y

def load_config(config_path: str) -> dict:
    """
    Loads a configuration dictionary from a JSON or YAML file.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            if config_path.endswith(".json"):
                config_dict = json.load(f)
            elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
                config_dict = yaml.safe_load(f)
            else:
                logging.error(f"Unsupported configuration file format: {config_path}. Only JSON and YAML are supported.")
                raise ValueError("Unsupported configuration file format.")
        logging.info(f"Configuration loaded from {config_path}")
        return config_dict
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_path}: {e}")
        raise


class SAM2Filter(logging.Filter):
    """
    Filters out common, verbose log messages from SAM2 processing if they are not desired.
    """
    def __init__(self, patterns_to_filter=None):
        super().__init__()
        if patterns_to_filter is None:
            self.patterns_to_filter = [
                "For numpy array image, we assume", 
                "Computing image embeddings", 
                "Image embeddings computed",    
                "Processing SAM Mask Generation"
            ]
        else:
            self.patterns_to_filter = patterns_to_filter

    def filter(self, record):
        message = record.getMessage()
        return not any(p in message for p in self.patterns_to_filter)

if __name__ == '__main__':
    print("--- Testing federated.utils ---")
    run_id1 = generate_run_id("MyTestFL")
    print(f"Generated Run ID: {run_id1}")
    temp_log_dir = "./temp_fl_logs"
    logger = setup_logging(log_dir=temp_log_dir, run_id=run_id1, log_level_str="DEBUG")
    logger.debug("This is a debug message from utils test.")
    logger.info("This is an info message from utils test.")
    sam_filter = SAM2Filter()
    logger.addFilter(sam_filter)
    logger.info("This message should appear.")
    logger.info("For numpy array image, we assume this should be filtered out by SAM2Filter.") 
    logger.removeFilter(sam_filter)
    logger.info("For numpy array image, we assume this should appear now (filter removed).")
    # Test config save/load
    test_config = {
        "learning_rate": 0.01,
        "batch_size": 32,
        "model_params": {"layers": [64, 32], "activation": "relu"},
        "dataset": "test_data_v1",
        "run_id_test": run_id1
    }
    json_config_path = os.path.join(temp_log_dir, "test_config.json")
    yaml_config_path = os.path.join(temp_log_dir, "test_config.yaml")
    print(f"\nSaving config to JSON: {json_config_path}")
    save_config(test_config, json_config_path)
    loaded_json_config = load_config(json_config_path)
    assert loaded_json_config["learning_rate"] == 0.01
    print("JSON Save/Load successful.")
    print(f"\nSaving config to YAML: {yaml_config_path}")
    save_config(test_config, yaml_config_path)
    loaded_yaml_config = load_config(yaml_config_path)
    assert loaded_yaml_config["batch_size"] == 32
    print("YAML Save/Load successful.")
    print(f"\nUtils test complete. Check logs in {temp_log_dir}")
