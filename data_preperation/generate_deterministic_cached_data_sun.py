import os
import sys
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch # For torch.device
import logging # For basic logging from this script
from tqdm import tqdm
from collections import defaultdict
from AutoCore_FL.segmentation.sam_loader import load_sam_model
from AutoCore_FL.segmentation.segment_crops import generate_segments_and_masks

# --- Add project root to sys.path ---
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
grandparent_dir = os.path.abspath(os.path.join(project_root_path, '..'))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# --- Imports from the project ---
from AutoCore_FL.data.ade20k_parition import (
    load_scene_categories, 
)
from AutoCore_FL.embedding.dino_loader import init_dino, init_target_model
from AutoCore_FL.embedding.compute_embeddings import compute_final_embeddings
from AutoCore_FL.federated.utils import setup_logging
from AutoCore_FL.data_preperation.utils import  _process_one_partition_from_loaded_data


sam_cfg_path = "configs/sam2.1/sam2.1_hiera_t.yaml" # Relative to project root or ensure absolute
sam_ckpt_path = "/gpfs/helios/home/soliman/logic_explained_networks/experiments/sam2.1_hiera_tiny.pt"
CHOSEN_CLASSES = ["bathroom", "bedroom", "bookstore"]
dataset_name = "sun"

# --- Configuration Section ---
DATA_GENERATION_CONFIG = {
    "data_root_for_npy": "/gpfs/helios/home/soliman/logic_explained_networks/data/sun_final/", # For sceneCategories.txt
    "scene_cat_file": "/gpfs/helios/home/soliman/logic_explained_networks/data/sun_final/sceneCategories.txt",
    "dataset_name" : dataset_name,
    # Paths to pre-generated full dataset .npy files
    "npy_full_dataset_base_path": "/gpfs/helios/home/soliman/logic_explained_networks/experiments/", # ADJUST THIS
    "chosen_classes": CHOSEN_CLASSES,
    "dino_model": "facebook/dinov2-base",
    "embedding_type": "dino_only",
    "embedding_dim": 768,
    "num_target_clients": 10,
    "server_val_fraction_total": 0.1,
    "server_test_fraction_total": 0.1,
    "random_seed": 42,
    "output_cache_main_dir": "./generated_autocore_fl_caches_reused_npy_10clients_sun",
    "run_tag_for_cache_paths": "sun_reused_npy",
    "sam_cfg": sam_cfg_path, "sam_ckpt": sam_ckpt_path,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "min_mask_pixels_for_crop": 100, # Min pixels for a segment to be considered for seg_crop_bgr
    "use_embedding_cache": True, # Let compute_final_embeddings use its cache

}

# Construct dynamic paths based on output_cache_main_dir and run_tag_for_cache_paths
base_output_path = os.path.join(DATA_GENERATION_CONFIG['output_cache_main_dir'], DATA_GENERATION_CONFIG['run_tag_for_cache_paths'])
DATA_GENERATION_CONFIG['partition_segment_infos_cache_dir'] = os.path.join(base_output_path, 'partition_segment_infos_with_crops')
DATA_GENERATION_CONFIG['embedding_cache_dir'] = os.path.join(base_output_path, 'partition_embedding_cache') # For compute_final_embeddings
DATA_GENERATION_CONFIG['partition_manifest_dir'] = os.path.join(base_output_path, 'partition_manifests')
DATA_GENERATION_CONFIG['log_dir'] = os.path.join(base_output_path, 'data_generation_logs')
DATA_GENERATION_CONFIG['run_id'] = DATA_GENERATION_CONFIG['run_tag_for_cache_paths']


# --- Basic Logger for this script ---
os.makedirs(DATA_GENERATION_CONFIG['log_dir'], exist_ok=True)
setup_logging(log_dir=DATA_GENERATION_CONFIG['log_dir'],run_id = DATA_GENERATION_CONFIG['run_id'])

main_logger = logging.getLogger("DataGen_Script")





def main_data_generation(config: dict):
    main_logger.info("=== Starting Deterministic Data Generation (Reusing NPY Segments) ===")
    main_logger.info(f"Outputting to base directory: {config['output_cache_main_dir']}/{config['run_tag_for_cache_paths']}")
    os.makedirs(config['partition_segment_infos_cache_dir'], exist_ok=True)
    os.makedirs(config['embedding_cache_dir'], exist_ok=True) # For compute_final_embeddings
    os.makedirs(config['partition_manifest_dir'], exist_ok=True)
    raw_image_paths_for_segmentation = [] # List of (base_id, full_path_to_image_for_sam)
    torch_device = torch.device(config['device'])

    # --- Deterministic Setup ---
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    if config['device'] == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_seed'])
    main_logger.info(f"Using device: {config['device']}")
    sunrgbd_prepared_images_root = os.path.join(config["data_root_for_npy"], "images")
    if not os.path.isdir(sunrgbd_prepared_images_root):
        main_logger.error(f"SUNRGBD (OTF): Prepared image directory not found: {sunrgbd_prepared_images_root}. "
                            "Run prepare_sunrgbd_subset_scenes.py first. Exiting.")
        return

    main_logger.info(f"SUNRGBD (OTF): Gathering image paths from prepared subset: {sunrgbd_prepared_images_root}")
    for scene_folder_name in config["chosen_classes"]: 
        scene_folder_path_actual = os.path.join(sunrgbd_prepared_images_root, scene_folder_name.replace(" ", "_").replace("/", "_"))
        if os.path.isdir(scene_folder_path_actual):
            for img_file in os.listdir(scene_folder_path_actual):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_full_path = os.path.join(scene_folder_path_actual, img_file)
                    # The 'base_id' for these images is their filename without extension,
                    # as used in sceneCategories_subset.txt
                    img_base_id = os.path.splitext(img_file)[0]
                    raw_image_paths_for_segmentation.append((img_base_id, img_full_path))
            
    # --- 1. Load Full Dataset from .npy files ---
    main_logger.info("Loading full dataset from .NPY files...")
    try:
        # --- Common On-the-Fly Segmentation (SAM) ---
        sam_model_otf, mask_gen_otf = load_sam_model(
            config['sam_cfg'], config['sam_ckpt'], torch_device,

        )

        full_dataset_segment_infos, full_dataset_all_images_rgb, full_dataset_all_masks, _ = generate_segments_and_masks(
            raw_image_paths_for_segmentation, 
            mask_gen_otf, config, 
            client_id=f"central_{config['dataset_name']}_otf_seg"
        )
        main_logger.info(f"Loaded .NPY data: {len(full_dataset_all_images_rgb)} images, {len(full_dataset_segment_infos)} total segments.")
    except FileNotFoundError as e:
        main_logger.error(f"CRITICAL: Could not load NPY files: {e}. Ensure paths in config are correct. Exiting.")
        return
    except Exception as e_load:
        main_logger.error(f"CRITICAL: Error loading NPY files: {e_load}. Exiting.", exc_info=True)
        return

    scene_map_full = load_scene_categories(config["scene_cat_file"])
    
    # Determine actual classes to use: either from config or all found in data if config is None
    target_scene_names_for_run = config.get("chosen_classes")
    if target_scene_names_for_run is None: # If None, derive from data
        main_logger.info("`chosen_classes` is None. Deriving classes from loaded segment_infos and scene_map.")
        scenes_found_in_data = set()
        for si in full_dataset_segment_infos:
            bid = si.get('base_id')
            if bid and bid in scene_map_full:
                scenes_found_in_data.add(scene_map_full[bid])
        if not scenes_found_in_data:
            main_logger.error("No scenes could be derived from segment_infos. Exiting.")
            return
        target_scene_names_for_run = sorted(list(scenes_found_in_data))
        main_logger.info(f"Using {len(target_scene_names_for_run)} classes derived from data: {target_scene_names_for_run[:5]}...")
    
    scene_to_idx_map_final = {name: i for i, name in enumerate(sorted(target_scene_names_for_run))}
    config['num_actual_classes'] = len(scene_to_idx_map_final)
    main_logger.info(f"Final class mapping created for {len(scene_to_idx_map_final)} classes.")
    class_map_file_path = os.path.join(config['partition_manifest_dir'], 'scene_to_idx_map.json')
    with open(class_map_file_path, 'w') as f: import json; json.dump(scene_to_idx_map_final, f, indent=2)
    main_logger.info(f"Saved class mapping to {class_map_file_path}")

    # Filter images based on target_scene_names_for_run and map base_ids
    # `original_global_img_idx_to_base_id` maps index in full_dataset_all_images_rgb to base_id
    original_global_img_idx_to_base_id = {}
    for si in full_dataset_segment_infos: # Iterate all loaded segment infos
        orig_glob_idx = si.get('img_idx')
        base_id_val = si.get('base_id')
        if orig_glob_idx is not None and base_id_val is not None and orig_glob_idx not in original_global_img_idx_to_base_id:
            original_global_img_idx_to_base_id[orig_glob_idx] = base_id_val
    
    valid_original_indices_for_split = [] # Indices into full_dataset_all_images_rgb
    labels_for_stratify_split = []      # Labels corresponding to valid_original_indices_for_split

    for orig_glob_idx, base_id_val in original_global_img_idx_to_base_id.items():
        scene_name = scene_map_full.get(base_id_val)
        if scene_name in scene_to_idx_map_final: # Check against the final map
            label_idx = scene_to_idx_map_final[scene_name]
            valid_original_indices_for_split.append(orig_glob_idx)
            labels_for_stratify_split.append(label_idx)
    
    if not valid_original_indices_for_split:
        main_logger.error("No images match the target classes after filtering. Exiting.")
        return
    
    main_logger.info(f"Identified {len(valid_original_indices_for_split)} images matching target classes for splitting.")
    labels_for_stratify_split_np = np.array(labels_for_stratify_split)

    # --- 3. Deterministic Split of Valid Images ---
    total_valid_images = len(valid_original_indices_for_split)
    server_val_size = int(total_valid_images * config['server_val_fraction_total'])
    server_test_size = int(total_valid_images * config['server_test_fraction_total'])
    server_pool_size = server_val_size + server_test_size

    # Split valid_original_indices_for_split
    indices_for_fl_pool_orig_global, indices_for_server_pool_orig_global = train_test_split(
        valid_original_indices_for_split,
        test_size=server_pool_size,
        random_state=config['random_seed'],
        stratify=labels_for_stratify_split_np if len(np.unique(labels_for_stratify_split_np)) > 1 else None
    )
    
    labels_for_server_pool_strat = labels_for_stratify_split_np[
        [valid_original_indices_for_split.index(i) for i in indices_for_server_pool_orig_global]
    ]

    indices_server_val_orig_global, indices_server_test_orig_global = train_test_split(
        indices_for_server_pool_orig_global,
        test_size=server_test_size / server_pool_size if server_pool_size > 0 else 0.0,
        random_state=config['random_seed'],
        stratify=labels_for_server_pool_strat if len(np.unique(labels_for_server_pool_strat)) > 1 else None
    )
    main_logger.info(f"Data split: FL Pool Original Indices ({len(indices_for_fl_pool_orig_global)}), "
                          f"Server Val Original Indices ({len(indices_server_val_orig_global)}), "
                          f"Server Test Original Indices ({len(indices_server_test_orig_global)}).")

    # --- 4. Initialize DINO Model (once) ---
    main_torch_dev = torch.device(config['device'])
    dino_proc, dino_mod = init_dino(config['dino_model'], main_torch_dev)
    target_res = init_target_model(main_torch_dev) if config['embedding_type'] == 'combined' else None
    main_logger.info("DINO and target (if any) models initialized.")


    # --- 5. Process Each Partition ---
    partitions_to_process = {
        "server_validation_set": indices_server_val_orig_global,
        "server_test_set": indices_server_test_orig_global
    }
    
    # Distribute FL pool indices to clients
    num_target_cls = config['num_target_clients']
    fl_pool_shuffled_orig_indices = random.sample(indices_for_fl_pool_orig_global, len(indices_for_fl_pool_orig_global))
    for i_client in range(num_target_cls):
        client_orig_indices = [fl_pool_shuffled_orig_indices[j] for j in range(i_client, len(fl_pool_shuffled_orig_indices), num_target_cls)]
        partitions_to_process[f"client_{i_client}"] = client_orig_indices

    for partition_name, list_of_orig_global_img_indices_for_part in partitions_to_process.items():
        main_logger.info(f"--- Processing partition: {partition_name} ({len(list_of_orig_global_img_indices_for_part)} images) ---")
        if not list_of_orig_global_img_indices_for_part:
            main_logger.warning(f"Partition {partition_name} is empty. Skipping actual processing, creating empty manifest.")
            _process_one_partition_from_loaded_data(
                [], [], [], [], [], partition_name, config, dino_proc, dino_mod, target_res
            )
            continue

        # a. Create partition-specific data views (images, masks, base_ids, labels)
        part_images_rgb = [full_dataset_all_images_rgb[orig_idx] for orig_idx in list_of_orig_global_img_indices_for_part]
        part_masks_per_img = [full_dataset_all_masks[orig_idx] for orig_idx in list_of_orig_global_img_indices_for_part]
        part_base_ids = [original_global_img_idx_to_base_id[orig_idx] for orig_idx in list_of_orig_global_img_indices_for_part]
        part_labels_int = [scene_to_idx_map_final[scene_map_full[bid]] for bid in part_base_ids] # Use final map

        # b. Re-index segment_infos for this partition
        #    Map: original_global_image_idx -> new_local_partition_image_idx
        orig_global_to_local_part_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(list_of_orig_global_img_indices_for_part)}
        
        part_segment_infos_list = []
        for seg_info_orig_global in full_dataset_segment_infos:
            orig_global_img_idx_of_seg = seg_info_orig_global.get('img_idx')
            if orig_global_img_idx_of_seg in orig_global_to_local_part_idx_map:
                new_local_part_img_idx = orig_global_to_local_part_idx_map[orig_global_img_idx_of_seg]
                new_seg_info_for_part = dict(seg_info_orig_global) # Copy
                new_seg_info_for_part['img_idx'] = new_local_part_img_idx # Re-index to local
                # 'seg_idx', 'base_id' remain original
                part_segment_infos_list.append(new_seg_info_for_part)
        
        # c. Process this partition's data
        _process_one_partition_from_loaded_data(
            part_images_rgb, part_masks_per_img, part_segment_infos_list,
            part_base_ids, part_labels_int,
            partition_name, config,
            dino_proc, dino_mod, target_res
        )

    main_logger.info("=== Deterministic Data Generation (Reusing NPY) FINISHED! ===")
    main_logger.info(f"Manifests in: {config['partition_manifest_dir']}")
    main_logger.info(f"Segment Infos (with crops) in: {config['partition_segment_infos_cache_dir']}")
    main_logger.info(f"Embedding caches (by compute_final_embeddings) in: {config['embedding_cache_dir']}")


if __name__ == "__main__":
    main_logger.info(f"Starting data generation with settings: {DATA_GENERATION_CONFIG}")
    main_data_generation(DATA_GENERATION_CONFIG)