import os
import sys
import pickle
import torch 
import logging 

# --- Add project root to sys.path ---
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
grandparent_dir = os.path.abspath(os.path.join(project_root_path, '..'))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)


from AutoCore_FL.embedding.compute_embeddings import compute_final_embeddings
from AutoCore_FL.federated.utils import setup_logging, add_seg_crop_bgr_to_split_infos

def _filter_segments_without_valid_crop(segment_infos_with_crops: list, logger_instance: logging.Logger) -> list:
    """Filters out segments where seg_crop_bgr is None or too small (e.g., all black)."""
    valid_segments = []
    removed_count = 0
    for seg_info in segment_infos_with_crops:
        if seg_info.get('seg_crop_bgr') is not None and seg_info['seg_crop_bgr'].shape[0] > 0 and seg_info['seg_crop_bgr'].shape[1] > 0:
            # Optional: check if crop is not all black (meaning original mask was tiny or problematic)
            # if np.any(seg_info['seg_crop_bgr']): # If any pixel is non-black
            valid_segments.append(seg_info)
        else:
            removed_count +=1
    if removed_count > 0:
        logger_instance.info(f"Removed {removed_count} segments that had invalid/missing 'seg_crop_bgr'.")
    return valid_segments


def _process_one_partition_from_loaded_data(
    partition_images_rgb: list,         # List of RGB np.array images for this partition
    partition_masks_per_image: list,    # List of lists of masks for this partition
    partition_segment_infos: list,      # List of seg_info dicts (local img_idx)
    partition_base_ids: list,           # List of base_ids for images in this partition
    partition_labels_int: list,         # List of integer labels for images in this partition
    unique_partition_id: str,           # E.g., "client_0", "server_val_set"
    config: dict,                       
    # Pre-loaded models
    dino_processor_model, dino_embedding_model,
    target_resnet_model # Can be None
):
    """
    Processes a single partition derived from pre-loaded full-dataset .npy data.
    Generates seg_crop_bgr, computes embeddings, saves caches, and manifest.
    """
    main_logger = logging.getLogger("UtilsLogger")
    logger = main_logger # Use the main script logger
    logger.info(f"Processing partition from loaded NPY: {unique_partition_id} with {len(partition_images_rgb)} images, {len(partition_segment_infos)} initial segments.")

    if not partition_images_rgb or not partition_segment_infos:
        logger.warning(f"Partition {unique_partition_id} has no images or segment infos. Saving empty manifest.")
        # Save empty manifest logic (as in previous script)
        partition_manifest_path = os.path.join(config['partition_manifest_dir'], unique_partition_id)
        os.makedirs(partition_manifest_path, exist_ok=True)
        manifest_file = os.path.join(partition_manifest_path, 'image_manifest.json')
        with open(manifest_file, 'w') as f: import json; json.dump([], f)
        # Save empty segment_infos_with_crops.pkl
        seg_info_cache_file = os.path.join(config['partition_segment_infos_cache_dir'], f"{unique_partition_id}_segment_infos_with_crops.pkl")
        with open(seg_info_cache_file, 'wb') as f: pickle.dump([], f)
        return

    # --- 1. Generate 'seg_crop_bgr' for segments in this partition ---
    # The 'img_idx' in partition_segment_infos is already local to this partition's image/mask lists.
    segment_infos_with_crops = add_seg_crop_bgr_to_split_infos(
        partition_segment_infos, partition_images_rgb, partition_masks_per_image, logger
    )

    # Filter out segments where seg_crop_bgr could not be made (e.g., mask was too small)
    final_segment_infos_for_embedding = _filter_segments_without_valid_crop(segment_infos_with_crops, logger)

    if not final_segment_infos_for_embedding:
        logger.warning(f"  No valid segments with crops remain for {unique_partition_id} after filtering. Embedding step will be skipped.")
    else:
        logger.info(f"  Generated/verified 'seg_crop_bgr' for {len(final_segment_infos_for_embedding)} segments in {unique_partition_id}.")

    # Save the (potentially filtered) segment_infos (now with 'seg_crop_bgr')
    # This is the primary "segment cache" for FL clients/server to load.
    os.makedirs(config['partition_segment_infos_cache_dir'], exist_ok=True)
    seg_info_cache_file = os.path.join(config['partition_segment_infos_cache_dir'], f"{unique_partition_id}_segment_infos_with_crops.pkl")
    with open(seg_info_cache_file, 'wb') as f:
        pickle.dump(final_segment_infos_for_embedding, f)
    logger.info(f"  Saved segment_infos_with_crops for {unique_partition_id} to {seg_info_cache_file}")


    # --- 2. Compute Embeddings (if there are segments to embed) ---
    if final_segment_infos_for_embedding:
        logger.info(f"  Computing embeddings for {unique_partition_id} (will use internal cache of compute_final_embeddings if available at {config['embedding_cache_dir']} based on partition ID)...")
        # compute_final_embeddings uses 'client_id' (here, unique_partition_id) for its own caching.
        # It needs config entries: 'embedding_cache_dir', 'embedding_type', 'embedding_dim', 'use_embedding_cache'.
        # It takes `filtered_segment_infos` (which must have 'seg_crop_bgr'), `filtered_images` (can be None if `seg_crop_bgr` is primary), etc.
        try:
            final_embeddings = compute_final_embeddings(
                filtered_segment_infos=final_segment_infos_for_embedding, # Must have 'seg_crop_bgr'
                filtered_images=None, # Not strictly needed if seg_crop_bgr is primary for embedding
                filtered_masks=None,  # Not strictly needed
                dino_processor=dino_processor_model,
                dino_model=dino_embedding_model,
                target_model=target_resnet_model,
                device=torch.device(config['device']),
                config=config, # Pass the main script config for sub-function caching
                client_id=unique_partition_id # For compute_final_embeddings internal cache key
            )
            if final_embeddings is not None and final_embeddings.shape[0] > 0:
                logger.info(f"  Embeddings computed for {unique_partition_id}. Shape: {final_embeddings.shape}")
                if final_embeddings.shape[0] != len(final_segment_infos_for_embedding):
                    logger.error(f"CRITICAL MISMATCH: Number of embeddings ({final_embeddings.shape[0]}) does not match number of final segments ({len(final_segment_infos_for_embedding)}) for {unique_partition_id}.")
            else:
                logger.warning(f"  No embeddings computed or embeddings empty for {unique_partition_id}.")
        except Exception as e_embed:
            logger.error(f"  Embedding computation failed for {unique_partition_id}: {e_embed}", exc_info=True)
    else:
        logger.info(f"  Skipping embedding for {unique_partition_id} as no valid segments with crops.")


    # --- 3. Save Partition Manifest (image base_ids and labels for this partition) ---
    partition_manifest_path_base = os.path.join(config['partition_manifest_dir'], unique_partition_id)
    os.makedirs(partition_manifest_path_base, exist_ok=True)
    
    manifest_content = []
    for i, base_id_val in enumerate(partition_base_ids): # Iterate over images in this partition
        label_int_val = partition_labels_int[i]
        # Original image path is not directly available unless reconstructed or stored.
        # For now, manifest focuses on base_id and label.
        manifest_content.append({
            'base_id': base_id_val,
            'label_int': int(label_int_val), # Ensure it's standard int for JSON
        })

    manifest_file_path = os.path.join(partition_manifest_path_base, 'image_manifest.json')
    with open(manifest_file_path, 'w') as f:
        import json
        json.dump(manifest_content, f, indent=2)
    logger.info(f"  Manifest for {unique_partition_id} saved to {manifest_file_path}")
    logger.info(f"  FL clients/server for {unique_partition_id} should load:\n    - Segment Infos (with crops): {seg_info_cache_file}\n    - Embeddings: from {config['embedding_cache_dir']} (e.g., embeddings_{config['embedding_type']}_{unique_partition_id}.pkl)\n    - Manifest: {manifest_file_path}")
