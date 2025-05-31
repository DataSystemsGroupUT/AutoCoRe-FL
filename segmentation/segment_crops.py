import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm

def generate_segments_and_masks(filtered_paths, mask_generator, config, client_id=None):
    """
    For each image in filtered_paths:
      - Generate SAM masks
      - Save each masked crop to disk
      - Store (img_idx, seg_idx) in all_segments
      - Store the boolean mask in all_masks[img_idx][seg_idx]
      - Store the original RGB image in all_images[img_idx]
    Returns:
      segment_infos, all_images, all_masks, all_segments
    """
    MIN_MASK_PIXELS = config.get('min_mask_pixels',100)
    CACHE_DIR = config.get('segment_cache_dir', './segment_cache')
    os.makedirs(CACHE_DIR, exist_ok=True)
    if client_id is not None:
        cache_file = os.path.join(CACHE_DIR, f"segments_{client_id}.pkl")
    else:
        cache_file = os.path.join(CACHE_DIR, f"segments_{client_id or 'server'}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            segment_infos, all_images, all_masks, all_segments = pickle.load(f)
        return segment_infos, all_images, all_masks, all_segments

    all_images = []
    all_masks = []
    all_segments = []
    segment_infos = []
    for img_idx, (base_id, img_path) in tqdm(enumerate(filtered_paths), total=len(filtered_paths), desc="Generating Segments"):
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        all_images.append(rgb)
        all_masks.append([])

        masks = mask_generator.generate(rgb)
        seg_count = 0
        for m in masks:
            seg_mask = m["segmentation"]
            ys, xs = np.where(seg_mask)
            if len(ys) == 0:
                continue
            top, left = np.min(ys), np.min(xs)
            bottom, right = np.max(ys), np.max(xs)
            mask_area = np.count_nonzero(seg_mask)
            if mask_area < MIN_MASK_PIXELS:
                continue
            seg_crop_rgb = rgb[top:bottom+1, left:right+1].copy()
            local_mask = seg_mask[top:bottom+1, left:right+1]
            seg_crop_rgb[~local_mask] = (0,0,0)
            save_name = f"{base_id}_seg{seg_count}.png"
            seg_crop_bgr = cv2.cvtColor(seg_crop_rgb, cv2.COLOR_RGB2BGR)
            segment_infos.append({
                "base_id": base_id,
                "img_idx": img_idx,
                "seg_idx": seg_count,
                "seg_crop_bgr": seg_crop_bgr
            })
            all_masks[img_idx].append(seg_mask)
            all_segments.append((img_idx, seg_count))
            seg_count += 1
    # Save to cache
    with open(cache_file, "wb") as f:
        pickle.dump((segment_infos, all_images, all_masks, all_segments), f)

    return segment_infos, all_images, all_masks, all_segments


def load_cached_segments(config, client_id=None):
    """
    Loads cached segments for a client or server.
    Returns: segment_infos, all_images, all_masks, all_segments
    """
    CACHE_DIR = config.get('segment_cache_dir', './segment_cache')
    os.makedirs(CACHE_DIR, exist_ok=True)
    if client_id is not None:
        cache_file = os.path.join(CACHE_DIR, f"segments_{client_id}.pkl")
    else:
        cache_file = os.path.join(CACHE_DIR, f"segments_{client_id or 'server'}.pkl")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"No cached segments found for client_id={client_id} at {cache_file}")
    with open(cache_file, "rb") as f:
        segment_infos, all_images, all_masks, all_segments = pickle.load(f)
    return segment_infos, all_images, all_masks, all_segments
def filter_zero_segment_images(all_images, all_masks, all_segments, segment_infos):
    """
    Filters out images with zero segments, updates indices, and returns filtered arrays.
    """
    valid_indices = []
    oldidx_to_newidx = {}
    new_idx = 0
    for old_idx, mask_list in enumerate(all_masks):
        if mask_list and len(mask_list) > 0:
            valid_indices.append(old_idx)
            oldidx_to_newidx[old_idx] = new_idx
            new_idx += 1

    filtered_images = [all_images[old_idx] for old_idx in valid_indices]
    filtered_masks = [all_masks[old_idx] for old_idx in valid_indices]

    new_all_segments = []
    for (old_img_idx, seg_idx) in all_segments:
        if old_img_idx in oldidx_to_newidx:
            new_img_idx = oldidx_to_newidx[old_img_idx]
            new_all_segments.append((new_img_idx, seg_idx))

    new_segment_infos = []
    for info in segment_infos:
        old_img_idx = info["img_idx"]
        if old_img_idx in oldidx_to_newidx:
            new_info = dict(info)
            new_info["img_idx"] = oldidx_to_newidx[old_img_idx]
            new_segment_infos.append(new_info)

    return (
        np.array(filtered_images, dtype=object),
        np.array(filtered_masks, dtype=object),
        np.array(new_all_segments, dtype=object),
        np.array(new_segment_infos, dtype=object)
    )