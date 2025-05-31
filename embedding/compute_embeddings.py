import os
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2

def compute_final_embeddings(filtered_segment_infos, filtered_images, filtered_masks,
                             dino_processor, dino_model, target_model, # target_model can be None
                             device, config, client_id=None):
    embedding_type = config.get('embedding_type') # Get it directly
    if embedding_type not in ['dino_only', 'combined']:
        raise ValueError(f"Invalid embedding_type in config: '{embedding_type}'. Must be 'dino_only' or 'combined'.")

    CACHE_DIR = config.get('embedding_cache_dir', './embedding_cache')
    os.makedirs(CACHE_DIR, exist_ok=True)

    if client_id is not None:
        cache_file_name = f"embeddings_{embedding_type}_{client_id}.pkl"
    else:
        server_cache_id = config.get('run_id', 'server_default_run')
        cache_file_name = f"embeddings_{embedding_type}_{server_cache_id}.pkl" 
    
    cache_file = os.path.join(CACHE_DIR, cache_file_name)

    if config.get("use_embedding_cache", True) and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                final_embeddings = pickle.load(f)
            print(f"Loaded {embedding_type} embeddings for ID: {client_id or config.get('run_id', 'server')} from {cache_file}, shape: {final_embeddings.shape if isinstance(final_embeddings, np.ndarray) else 'Unknown'}")
            if not isinstance(final_embeddings, np.ndarray):
                raise ValueError("Cached embeddings are not a numpy array.")
            if final_embeddings.ndim != 2 or final_embeddings.shape[1] != config['embedding_dim']:
                 print(f"Warning: Cached embedding dim/shape issue. Shape: {final_embeddings.shape}, Expected dim: {config['embedding_dim']}. Recomputing.")
                 raise FileNotFoundError("Dimension/shape mismatch, recompute.")
            return final_embeddings
        except Exception as e:
            print(f"Error loading embeddings from cache or cache invalid ({e}), recomputing...")

    final_embeddings_list = []

    print(f"Computing {embedding_type} embeddings for ID: {client_id or config.get('run_id', 'server')}...")
    for info in tqdm(filtered_segment_infos, desc=f"Embeddings ({embedding_type}) for {client_id or 'server'}"):
        seg_crop_bgr = info.get("seg_crop_bgr")
        if seg_crop_bgr is None:
            print(f"Warning: seg_crop_bgr is None for segment. Appending zeros.")
            final_embeddings_list.append(np.zeros(config['embedding_dim'], dtype=np.float32))
            continue
        
        try:
            crop_pil_img = Image.fromarray(cv2.cvtColor(seg_crop_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e_pil:
            print(f"Error converting seg_crop_bgr to PIL: {e_pil}. Appending zeros.")
            final_embeddings_list.append(np.zeros(config['embedding_dim'], dtype=np.float32))
            continue
            
        inputs = dino_processor(images=crop_pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
            dino_feat = outputs.last_hidden_state[:, 0].cpu().numpy().squeeze()

        if embedding_type == 'dino_only':
            if dino_feat.shape[0] != config['embedding_dim']:
                 print(f"ERROR (dino_only): DINO feature dim {dino_feat.shape[0]} != config dim {config['embedding_dim']}.")
                 final_embeddings_list.append(np.zeros(config['embedding_dim'], dtype=np.float32))
                 continue
            final_embeddings_list.append(dino_feat)

    if not final_embeddings_list:
        print(f"Warning: No embeddings computed for ID: {client_id or config.get('run_id', 'server')}. Returning empty array.")
        return np.array([], dtype=np.float32).reshape(0, config['embedding_dim'])

    try:
        final_embeddings_array = np.array(final_embeddings_list, dtype=np.float32)
    except ValueError as e_stack: # If lists have different shapes, np.array will fail or create object array
        print(f"CRITICAL ERROR: Could not stack embeddings list into numpy array: {e_stack}. Individual shapes might be inconsistent.")
        # Try to find first non-matching shape for debugging
        ref_shape = None
        for i, emb_item in enumerate(final_embeddings_list):
            if isinstance(emb_item, np.ndarray):
                if ref_shape is None: ref_shape = emb_item.shape
                elif emb_item.shape != ref_shape:
                    print(f"  Mismatch found at index {i}: shape {emb_item.shape} vs ref_shape {ref_shape}")
                    break
            else:
                print(f"  Item at index {i} is not a numpy array: {type(emb_item)}")
                break
        return np.array([], dtype=np.float32).reshape(0, config['embedding_dim'])


    if final_embeddings_array.ndim == 2 and final_embeddings_array.shape[0] > 0 and final_embeddings_array.shape[1] != config['embedding_dim']:
        print(f"CRITICAL ERROR: Final array dim {final_embeddings_array.shape[1]} != config dim {config['embedding_dim']}.")
    elif final_embeddings_array.ndim != 2 and final_embeddings_array.size > 0 :
        print(f"CRITICAL ERROR: Final array not 2D. Shape: {final_embeddings_array.shape}")
        return np.array([], dtype=np.float32).reshape(0, config['embedding_dim'])

    try:
        with open(cache_file, "wb") as f: pickle.dump(final_embeddings_array, f)
        print(f"Saved {embedding_type} embeddings for ID: {client_id or config.get('run_id', 'server')} to {cache_file}, shape: {final_embeddings_array.shape}")
    except Exception as e_save:
        print(f"Error saving embeddings to {cache_file}: {e_save}")

    return final_embeddings_array