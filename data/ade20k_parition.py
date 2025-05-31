from collections import defaultdict
import os
import random
import numpy as np

def get_filtered_image_paths(
    ade_root: str,
    scene_map: dict,
    chosen_classes: list,
    subset="training"
):
    """
    - ade_root: e.g. /path/to/ADE20K
    - scene_map: dictionary { "ADE_train_00000001": "airport_terminal", ...}
    - chosen_classes: e.g. CHOSEN_CLASSES
    - subset: "training" or "validation"
    Returns a list of (base_id, full_image_path) pairs
    that match the chosen classes.
    """
    images_dir = os.path.join(ade_root, "images", subset)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"{images_dir} does not exist.")

    # We will find images whose scene category is in chosen_classes
    filtered_paths = []
    for base_id, scene_str in scene_map.items():
        # base_id is like "ADE_train_00000001"
        if scene_str not in chosen_classes:
            continue
        # Construct the actual file path
        image_path_jpg = os.path.join(images_dir, f"{base_id}.jpg")
        image_path_png = os.path.join(images_dir, f"{base_id}.png")

        if os.path.isfile(image_path_jpg):
            filtered_paths.append((base_id, image_path_jpg))
        elif os.path.isfile(image_path_png):
            filtered_paths.append((base_id, image_path_png))
        # else no file found, skip

    return filtered_paths

def partition_dataset(filtered_paths, num_clients, seed=42):
    """
    Splits filtered_paths into num_clients partitions for federated learning.
    Each partition is a list of (image_id, image_path).
    """
    np.random.seed(seed)
    indices = np.arange(len(filtered_paths))
    np.random.shuffle(indices)
    partitions = np.array_split(indices, num_clients)
    client_partitions = []
    for part in partitions:
        client_partitions.append([filtered_paths[i] for i in part])
    return client_partitions

def stratified_partition(filtered_paths, scene_map, chosen_classes, num_clients, seed=0):
    random.seed(seed)
    # Group images by class
    class_to_images = defaultdict(list)
    for base_id, path in filtered_paths:
        scene = scene_map[base_id]
        if scene in chosen_classes:
            class_to_images[scene].append((base_id, path))
    # Distribute images from each class to clients
    partitions = [[] for _ in range(num_clients)]
    for images in class_to_images.values():
        random.shuffle(images)
        for i, img in enumerate(images):
            partitions[i % num_clients].append(img)
    return partitions
    
def load_scene_categories(scene_cat_file: str) -> dict:
    """
    Reads sceneCategories.txt lines like:
       ADE_train_00000001 airport_terminal
       ADE_train_00000002 airport_terminal
    Returns a dict: { "ADE_train_00000001": "airport_terminal", ... }
    """
    scene_map = {}
    with open(scene_cat_file, 'r') as f:
        for line in f:
            base, scene_str = line.strip().split()
            scene_map[base] = scene_str
    return scene_map