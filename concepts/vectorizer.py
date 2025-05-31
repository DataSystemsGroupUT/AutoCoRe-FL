
import numpy as np
from collections import defaultdict
import logging

logger_vec = logging.getLogger("Vectorizer")

def build_image_concept_vectors(
    filtered_segment_infos,
    linear_models: dict,
    optimal_thresholds: dict,
    final_embeddings: np.ndarray,
    target_num_features: int,
    config: dict
) -> tuple[np.ndarray, list]:
    min_activating_segments = config.get('vectorizer_min_activating_segments', 1)
    logger_vec.info(f"Building concept vectors with min_activating_segments: {min_activating_segments}")

    is_segment_infos_empty = False
    if isinstance(filtered_segment_infos, np.ndarray):
        # For numpy array, check its size or if its first dimension is 0
        if filtered_segment_infos.size == 0: # Covers empty array [] or array([{}]) where dict is empty.
            is_segment_infos_empty = True
    elif isinstance(filtered_segment_infos, list):
        if not filtered_segment_infos: # Standard check for empty list
            is_segment_infos_empty = True
    else: # Unexpected type
        logger_vec.warning(f"filtered_segment_infos is of unexpected type: {type(filtered_segment_infos)}. Assuming not empty for now, but this might cause issues.")

    # Check if final_embeddings is a NumPy array and has content
    is_final_embeddings_empty = True
    if isinstance(final_embeddings, np.ndarray):
        if final_embeddings.ndim > 0 and final_embeddings.shape[0] > 0:
            is_final_embeddings_empty = False
    
    if is_segment_infos_empty or is_final_embeddings_empty:
        logger_vec.warning(f"No segment_infos (empty: {is_segment_infos_empty}) or "
                           f"final_embeddings (empty: {is_final_embeddings_empty}) to build vectors from.")
        return np.empty((0, target_num_features), dtype=np.float32), []


    img_to_segs_indices = defaultdict(list)
    img_idx_to_base_id = {}
    

    for seg_global_idx, info in enumerate(filtered_segment_infos): 
        img_idx = info.get("img_idx")
        base_id = info.get("base_id")
        if img_idx is not None and base_id is not None:
            img_to_segs_indices[img_idx].append(seg_global_idx)
            if img_idx not in img_idx_to_base_id:
                img_idx_to_base_id[img_idx] = base_id
        else:
            logger_vec.warning(f"Segment info missing img_idx or base_id: {info}")

    if not img_to_segs_indices:
        logger_vec.warning("No valid segments mapped to images after processing segment_infos.")
        return np.empty((0, target_num_features), dtype=np.float32), []

    prob_cache = {}
    available_dense_indices = sorted([idx for idx in linear_models.keys() if idx < target_num_features])

    for dense_idx in available_dense_indices:
        model_pipeline = linear_models.get(dense_idx)
        if model_pipeline is None:
            logger_vec.warning(f"No model found for dense_idx {dense_idx} during vectorization.")
            continue
        try:
            if final_embeddings.ndim == 1: final_embeddings = final_embeddings.reshape(1, -1)
            if final_embeddings.shape[0] == 0 : # No embeddings to predict on
                logger_vec.warning(f"Final embeddings array is empty for dense_idx {dense_idx}. Skipping prob calculation.")
                continue

            if hasattr(model_pipeline, "predict_proba"):
                probs_all_segments = model_pipeline.predict_proba(final_embeddings)[:, 1]
            elif hasattr(model_pipeline, "decision_function"):
                scores_all_segments = model_pipeline.decision_function(final_embeddings)
                probs_all_segments = 1 / (1 + np.exp(-scores_all_segments))
            else:
                logger_vec.error(f"Model for dense_idx {dense_idx} has neither predict_proba nor decision_function.")
                continue
            prob_cache[dense_idx] = probs_all_segments
        except Exception as e:
            logger_vec.error(f"Error predicting probabilities for dense_idx {dense_idx} (Embeddings shape: {final_embeddings.shape}): {e}")

    kept_image_concept_vectors = []
    kept_image_base_ids = []
    
    sorted_image_indices = sorted(list(img_idx_to_base_id.keys()))

    for img_idx in sorted_image_indices:
        image_concept_vector = np.zeros((target_num_features,), dtype=np.float32)
        segment_indices_for_this_image = img_to_segs_indices.get(img_idx, [])

        if not segment_indices_for_this_image:
            continue

        for dense_concept_idx in range(target_num_features):
            if dense_concept_idx in prob_cache:
                threshold_for_concept = optimal_thresholds.get(dense_concept_idx)
                if threshold_for_concept is None:
                    continue

                all_segment_probs_for_concept = prob_cache[dense_concept_idx]
                
                # Ensure global segment indices are valid for the probability array
                valid_segment_indices_for_img = [s_idx for s_idx in segment_indices_for_this_image if s_idx < len(all_segment_probs_for_concept)]
                
                if not valid_segment_indices_for_img:
                    continue

                probs_of_concept_for_image_segments = all_segment_probs_for_concept[valid_segment_indices_for_img]
                activating_segment_count = np.sum(probs_of_concept_for_image_segments >= threshold_for_concept)

                if activating_segment_count >= min_activating_segments:
                    image_concept_vector[dense_concept_idx] = 1.0
        
        kept_image_concept_vectors.append(image_concept_vector)
        kept_image_base_ids.append(img_idx_to_base_id[img_idx])

    if not kept_image_concept_vectors:
        logger_vec.warning("No concept vectors generated after processing all images.")
        return np.empty((0, target_num_features), dtype=np.float32), []

    return np.array(kept_image_concept_vectors), kept_image_base_ids