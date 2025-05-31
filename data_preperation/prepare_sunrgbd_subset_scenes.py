import os
import glob
import json
import shutil
import re
import urllib.request
import zipfile
import numpy as np
from tqdm import tqdm

# --- Configuration ---
SUNRGBD_DOWNLOAD_ROOT = "/gpfs/helios/home/soliman/logic_explained_networks/data/sunrgbd" 
SUNRGBD_MAIN_DATA_FOLDER = os.path.join(SUNRGBD_DOWNLOAD_ROOT, "SUNRGBD") 

TARGET_SUNRGBD_SCENES = sorted(["bathroom", "bedroom", "bookstore"])

ALLOWED_OBJECTS_SUNRGBD = sorted([
    "chair", "table", "sofa", "bed", "cabinet", "bookshelf", "desk", "door", "window",
    "toilet", "sink", "bathtub", "night_stand", "dresser", "lamp", "mirror",
    "books", "shelves" 
])

OUTPUT_DATASET_ROOT = "/gpfs/helios/home/soliman/logic_explained_networks/data/sun_final" 
OUTPUT_IMAGES_SUBDIR = os.path.join(OUTPUT_DATASET_ROOT, "images")

# Output file names (to match ADE20K style outputs)
OUTPUT_SUNRGBD_ATTRIBUTES_NPY = os.path.join(OUTPUT_DATASET_ROOT, "attributes.npy")
OUTPUT_SUNRGBD_SCENE_LABELS_NPY = os.path.join(OUTPUT_DATASET_ROOT, "scene_labels.npy")
OUTPUT_SUNRGBD_SCENE_CLASSES_TXT = os.path.join(OUTPUT_DATASET_ROOT, "scene_classes.txt")
OUTPUT_SUNRGBD_ATTRIBUTES_NAMES_TXT = os.path.join(OUTPUT_DATASET_ROOT, "attributes_names.txt")
OUTPUT_SUNRGBD_SCENE_CATEGORIES_SUBSET_TXT = os.path.join(OUTPUT_DATASET_ROOT, "sceneCategories.txt") # New file

# --- Helper functions ---
SUNRGBD_ZIP_URL = "http://rgbd.cs.princeton.edu/data/SUNRGBD.zip"

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path} ...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    else:
        print(f"'{dest_path}' already exists; skipping download.")

def extract_zip(zip_path, extract_to_parent_of_sunrgbd_folder):
    expected_extracted_path = os.path.join(extract_to_parent_of_sunrgbd_folder, "SUNRGBD")
    if not os.path.isdir(expected_extracted_path):
        print(f"Extracting {zip_path} into {extract_to_parent_of_sunrgbd_folder} (expecting it to create '{expected_extracted_path}')...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_parent_of_sunrgbd_folder)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print(f"ERROR: Failed to extract {zip_path}. It might be corrupted or incomplete. Please re-download.")
            return False
        if not os.path.isdir(expected_extracted_path):
            print(f"ERROR: Expected folder '{expected_extracted_path}' not found after extraction.")
            double_nested_check = os.path.join(extract_to_parent_of_sunrgbd_folder, "SUNRGBD", "SUNRGBD")
            if os.path.isdir(double_nested_check):
                print(f"Warning: Content might be in a double nested folder: {double_nested_check}. Manual check advised if script fails later.")
            return False
    else:
        print(f"'{expected_extracted_path}' already exists; skipping extraction of {zip_path}.")
    return True

def download_and_prepare_sunrgbd_raw_data(root_download_dir):
    os.makedirs(root_download_dir, exist_ok=True)
    zip_path = os.path.join(root_download_dir, "SUNRGBD.zip")
    download_file(SUNRGBD_ZIP_URL, zip_path)
    # We don't need SUNRGBDMeta2DBB_v2.mat for this script
    return extract_zip(zip_path, root_download_dir)

def normalize_name(name):
    return re.sub(r'\d+$', '', name.strip().lower())

def _find_all_record_dirs(root_dir_sunrgbd_data):
    valid_dirs = []
    if not os.path.isdir(root_dir_sunrgbd_data):
        print(f"ERROR: The root SUNRGBD data directory does not exist: {root_dir_sunrgbd_data}")
        return []
    for dirpath, dirnames, filenames in os.walk(root_dir_sunrgbd_data):
        if "image" in dirnames and "annotation2Dfinal" in dirnames:
            image_subdir_path = os.path.join(dirpath, "image")
            annotation_subdir_path = os.path.join(dirpath, "annotation2Dfinal")
            if os.path.isdir(image_subdir_path) and os.path.isdir(annotation_subdir_path):
                json_files = glob.glob(os.path.join(annotation_subdir_path, "*.json"))
                if json_files:
                    valid_dirs.append(dirpath)
    print(f"Found {len(valid_dirs)} potential SUNRGBD record directories using os.walk.")
    return list(set(valid_dirs))

def load_json_robust(pattern):
    try:
        files = glob.glob(pattern)
        if not files: return None
        ann_files = [f for f in files if "annotation" in f.lower() or "seg"in f.lower()]
        file_to_load = ann_files[0] if ann_files else files[0]
        with open(file_to_load, 'r', encoding='utf-8', errors='ignore') as f:
            return json.load(f)
    except Exception as e:
        # print(f"Warning: Error loading JSON {pattern}: {e}") # Can be very verbose
        return None

class SunRGBD_Record_Adapter:
    def __init__(self, record_dir_path):
        self.record_dir = record_dir_path
        self.annotation_data = load_json_robust(os.path.join(self.record_dir, "annotation2Dfinal", "*.json"))
        self.scene_name = self._load_scene_name_from_txt()
        
        img_dir_path = os.path.join(self.record_dir, "image")
        self.image_path = None
        self.image_base_name_original = None 

        if os.path.isdir(img_dir_path):
            img_files = sorted([f for f in os.listdir(img_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if img_files:
                self.image_path = os.path.join(img_dir_path, img_files[0])
                self.image_base_name_original = os.path.splitext(img_files[0])[0]

    def _load_scene_name_from_txt(self):
        scene_txt_path = os.path.join(self.record_dir, "scene.txt")
        if os.path.exists(scene_txt_path):
            try:
                with open(scene_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read().strip().lower()
            except Exception: pass # Silently pass if scene.txt is unreadable
        return None

    def get_present_object_names(self):
        if not self.annotation_data or not isinstance(self.annotation_data, dict): return []
        object_names_in_record = []
        for obj_info in self.annotation_data.get("objects", []):
            if isinstance(obj_info, dict):
                name_field = obj_info.get("name", obj_info.get("label"))
                if name_field:
                    object_names_in_record.append(normalize_name(str(name_field)))
        return list(set(object_names_in_record))

def make_scene_folders_for_output(output_images_root: str, scene_names_list: list) -> dict:
    os.makedirs(output_images_root, exist_ok=True)
    scene_to_folder_map = {}
    for scene_n in scene_names_list:
        folder_name_safe = scene_n.replace(" ", "_").replace("/", "_")
        scene_folder_path = os.path.join(output_images_root, folder_name_safe)
        os.makedirs(scene_folder_path, exist_ok=True)
        scene_to_folder_map[scene_n] = scene_folder_path
    return scene_to_folder_map

def build_sunrgbd_subset_for_pipeline():
    print(f"--- Preparing SUNRGBD data for {len(TARGET_SUNRGBD_SCENES)} scenes in ADE20K-like format ---")
    print(f"Target scenes: {TARGET_SUNRGBD_SCENES}")
    print(f"Object concept vocabulary ({len(ALLOWED_OBJECTS_SUNRGBD)} concepts): {ALLOWED_OBJECTS_SUNRGBD}")
    print(f"Output will be in: {OUTPUT_DATASET_ROOT}")

    os.makedirs(OUTPUT_DATASET_ROOT, exist_ok=True) # Ensure main output dir exists

    if not download_and_prepare_sunrgbd_raw_data(root_download_dir=SUNRGBD_DOWNLOAD_ROOT):
        print("Halting due to SUNRGBD raw data preparation issues.")
        return

    if not os.path.isdir(SUNRGBD_MAIN_DATA_FOLDER):
        print(f"ERROR: Main SUNRGBD data folder '{SUNRGBD_MAIN_DATA_FOLDER}' not found. Cannot proceed.")
        return

    all_record_dirs = _find_all_record_dirs(SUNRGBD_MAIN_DATA_FOLDER)
    if not all_record_dirs:
        print("ERROR: No SUNRGBD record directories found. Check data path and structure.")
        return

    scene_to_output_folder_map = make_scene_folders_for_output(OUTPUT_IMAGES_SUBDIR, TARGET_SUNRGBD_SCENES)
    scene_name_to_label_idx_map = {name: i for i, name in enumerate(TARGET_SUNRGBD_SCENES)}
    obj_name_to_concept_idx_map = {name: i for i, name in enumerate(ALLOWED_OBJECTS_SUNRGBD)}

    temp_image_data_map = {} 
    scene_categories_subset_map = {} 
    copied_image_counter = 0

    for record_dir_path in tqdm(all_record_dirs, desc="Processing SUNRGBD Records"):
        record = SunRGBD_Record_Adapter(record_dir_path)

        if not record.image_path or not record.scene_name or not record.image_base_name_original:
            continue
        if record.scene_name not in TARGET_SUNRGBD_SCENES:
            continue

        concept_vector = np.zeros(len(ALLOWED_OBJECTS_SUNRGBD), dtype=np.float32)
        present_objects = record.get_present_object_names()
        for obj_n in present_objects:
            if obj_n in obj_name_to_concept_idx_map:
                concept_vector[obj_name_to_concept_idx_map[obj_n]] = 1.0
        
        scene_label_idx = scene_name_to_label_idx_map[record.scene_name]

        path_parts = record.record_dir.replace(SUNRGBD_MAIN_DATA_FOLDER, "").strip(os.sep).split(os.sep)
        unique_prefix = "_".join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1] if path_parts else "rec"
        unique_prefix = re.sub(r'[^a-zA-Z0-9_]', '', unique_prefix)
        
        original_extension = os.path.splitext(record.image_path)[1]
        destination_filename_base = f"{unique_prefix}_{record.image_base_name_original}"
        destination_filename_with_ext = f"{destination_filename_base}{original_extension}"

        destination_folder = scene_to_output_folder_map[record.scene_name]
        destination_path = os.path.join(destination_folder, destination_filename_with_ext)
        
        try:
            if not os.path.exists(destination_path): # Avoid re-copying if script is run multiple times
                 shutil.copy2(record.image_path, destination_path)
            temp_image_data_map[destination_path] = (concept_vector, scene_label_idx)
            scene_categories_subset_map[destination_filename_base] = record.scene_name 
            copied_image_counter += 1 # Count successful considerations, not just copies
        except Exception as e_copy:
            print(f"Warning: Failed to copy or process {record.image_path} to {destination_path}: {e_copy}")
    
    # This count is now for images *considered* for target scenes, not just copied in this run.
    print(f"\nConsidered {copied_image_counter} images for new structure in '{OUTPUT_IMAGES_SUBDIR}'.")
    if copied_image_counter == 0:
        print("ERROR: No images were processed for target scenes. Check scene names and processing logic.")
        return

    final_attributes_list = []
    final_scene_labels_list = []
    ordered_scene_categories_subset_list = [] 
    
    print("Re-ordering data based on ImageFolder structure...")
    for scene_name_key in TARGET_SUNRGBD_SCENES:
        scene_output_folder_path = scene_to_output_folder_map[scene_name_key]
        if not os.path.isdir(scene_output_folder_path): continue

        for image_filename_in_class_folder in sorted(os.listdir(scene_output_folder_path)):
            file_base, file_ext = os.path.splitext(image_filename_in_class_folder)
            if not file_ext.lower() in ('.jpg', '.jpeg', '.png'):
                continue
            
            full_image_path_in_output = os.path.join(scene_output_folder_path, image_filename_in_class_folder)
            
            if full_image_path_in_output in temp_image_data_map:
                concept_vec, scene_idx = temp_image_data_map[full_image_path_in_output]
                final_attributes_list.append(concept_vec)
                final_scene_labels_list.append(scene_idx)
                ordered_scene_categories_subset_list.append((file_base, scene_name_key))


    if not final_attributes_list:
        print("ERROR: No attributes were collected after reordering. Check map population or reordering logic.")
        return

    final_attributes_np = np.array(final_attributes_list, dtype=np.float32)
    final_scene_labels_np = np.array(final_scene_labels_list, dtype=np.int64)

    print(f"Final reordered attributes shape: {final_attributes_np.shape}")
    print(f"Final reordered scene labels shape: {final_scene_labels_np.shape}")

    np.save(OUTPUT_SUNRGBD_ATTRIBUTES_NPY, final_attributes_np)
    np.save(OUTPUT_SUNRGBD_SCENE_LABELS_NPY, final_scene_labels_np)
    print(f"'attributes.npy' and 'scene_labels.npy' saved to '{OUTPUT_DATASET_ROOT}'.")

    with open(OUTPUT_SUNRGBD_SCENE_CLASSES_TXT, 'w') as f:
        for idx, scene_n_val in enumerate(TARGET_SUNRGBD_SCENES):
            f.write(f"{idx} {scene_n_val}\n")
    print(f"'scene_classes.txt' saved to '{OUTPUT_DATASET_ROOT}'.")

    # Format: JSON list of strings like "has_object::name" (without internal \n for cleaner JSON)
    concept_names_for_json = [f"has_object::{name}" for name in ALLOWED_OBJECTS_SUNRGBD]
    with open(OUTPUT_SUNRGBD_ATTRIBUTES_NAMES_TXT, 'w') as f:
        json.dump(concept_names_for_json, f, indent=2) 
    print(f"'attributes_names.txt' saved to '{OUTPUT_DATASET_ROOT}'.")

    with open(OUTPUT_SUNRGBD_SCENE_CATEGORIES_SUBSET_TXT, 'w') as f_sc_subset:
        for base_id, scene_name in ordered_scene_categories_subset_list:
            f_sc_subset.write(f"{base_id} {scene_name}\n")
    print(f"'sceneCategories_subset.txt' saved to '{OUTPUT_SUNRGBD_SCENE_CATEGORIES_SUBSET_TXT}'.")

    print("\n--- SUNRGBD Data Preparation for 3 Scenes (ADE20K-style) FINISHED! ---")
    print(f"Output dataset root: {OUTPUT_DATASET_ROOT}")

if __name__ == "__main__":
    build_sunrgbd_subset_for_pipeline()