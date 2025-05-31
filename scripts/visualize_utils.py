import logging
import os
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments (e.g., servers)
import matplotlib.pyplot as plt
import cv2
from scipy.special import expit, softmax
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 



def find_segments_for_concept_in_image(
    image_idx_in_split,             # Index of the target image in its data split (e.g., test_split)
    target_concept_original_kmeans_idx, # The K-Means ID of the concept we're looking for
    trained_concept_detectors_map, # Dict: {original_kmeans_idx: (pipeline, threshold)}
    seg_infos_flat_for_split,      # Flat list of ALL segment_infos for the data split
    embeddings_flat_for_split,     # Flat list of ALL embeddings for the data split
    max_segments_to_return=2
):
    """
    Finds segment crops from a specific image that activate a given concept.
    Returns a list of seg_crop_bgr.
    """
    activating_segment_crops = []
    
    if target_concept_original_kmeans_idx not in trained_concept_detectors_map:
        return [] # Detector for this concept not available

    detector_pipeline, detector_threshold = trained_concept_detectors_map[target_concept_original_kmeans_idx]
    
    candidate_segment_indices_in_flat_list = []
    candidate_embeddings_list = []
    original_seg_info_for_candidates = []

    for global_seg_idx, seg_info in enumerate(seg_infos_flat_for_split):
        if seg_info.get('img_idx') == image_idx_in_split: # seg_info['img_idx'] is local to the split
            if embeddings_flat_for_split[global_seg_idx] is not None and \
               seg_info.get('seg_crop_bgr') is not None: # Ensure embedding and crop exist
                candidate_segment_indices_in_flat_list.append(global_seg_idx)
                candidate_embeddings_list.append(embeddings_flat_for_split[global_seg_idx])
                original_seg_info_for_candidates.append(seg_info)
    
    if not candidate_embeddings_list:
        return []

    candidate_embeddings_np = np.array(candidate_embeddings_list)
    
    # Get probabilities from the detector
    try:
        if hasattr(detector_pipeline, "predict_proba"):
            probs = detector_pipeline.predict_proba(candidate_embeddings_np)[:, 1]
        elif hasattr(detector_pipeline, "decision_function"):
            scores = detector_pipeline.decision_function(candidate_embeddings_np)
            probs = expit(scores) # Sigmoid
        else:
            return [] # Detector cannot give probabilities
    except Exception:
        return []

    activating_indices_within_candidates = np.where(probs >= detector_threshold)[0]
    
    # Select top segments (e.g., by highest probability or just first few)
    # For simplicity, just take the first few that activate
    count = 0
    for idx_in_candidates in activating_indices_within_candidates:
        if count < max_segments_to_return:
            seg_info_for_crop = original_seg_info_for_candidates[idx_in_candidates]
            activating_segment_crops.append(seg_info_for_crop.get('seg_crop_bgr'))
            count += 1
        else:
            break
            
    return [crop for crop in activating_segment_crops if crop is not None]


def get_random_segments_from_dataset(
    all_segment_infos_in_split, # Flat list of ALL segment_infos for the data split (e.g., test split)
    num_random_segments=2,
    exclude_image_idx=None # Optional: to avoid picking from the currently visualized image
):
    """Gets random segment crops from the entire provided segment pool."""
    
    candidate_indices = []
    for i, seg_info in enumerate(all_segment_infos_in_split):
        if seg_info.get('seg_crop_bgr') is not None:
            if exclude_image_idx is None or seg_info.get('img_idx') != exclude_image_idx:
                candidate_indices.append(i)
    
    if not candidate_indices:
        return []
        
    num_to_sample = min(num_random_segments, len(candidate_indices))
    if num_to_sample == 0:
        return []

    sampled_indices = np.random.choice(candidate_indices, num_to_sample, replace=False)
    return [all_segment_infos_in_split[i].get('seg_crop_bgr') for i in sampled_indices]


def visualize_centralized_decision(
    target_image_rgb_np,
    image_idx_in_test_split, 
    predicted_class_name_str,
    image_concept_vector_np,
    figs_model_instance,
    ordered_final_concept_original_ids,
    trained_concept_detectors_map_paper_viz,
    seg_infos_test_flat_paper_viz,     
    embeddings_test_flat_paper_viz,    
    feature_names_for_figs_paper_viz,
    config_paper_viz,
    main_logger_paper_viz,
    # These parameters define the TARGET content for the 2x2 panel
    target_num_top_concept_segments_from_image=2, 
    target_num_random_segments_from_dataset=2,
    output_filename_stem="paper_viz_decision"
):
    main_logger_paper_viz.info(f"Generating paper visualization for image_idx {image_idx_in_test_split} (local test split index), pred: {predicted_class_name_str}")

    # --- Plot 1: Original Image ---
    fig_orig, ax_orig = plt.subplots(1, 1, figsize=(6, 6))
    ax_orig.imshow(target_image_rgb_np)
    ax_orig.axis('off')
    save_plot(fig_orig, f"{output_filename_stem}_img{image_idx_in_test_split}_original", config_paper_viz, main_logger_paper_viz)

    # --- Prepare Segments for the 2x2 Panel ---
    panel_segment_crops = [None] * 4 # Initialize a list for 4 segment crops (for a 2x2 grid)
    current_panel_idx = 0

    # 1. Try to get target_num_top_concept_segments_from_image
    critical_dense_indices = get_critical_active_concepts_from_figs(
        figs_model_instance, image_concept_vector_np, feature_names_for_figs_paper_viz
    )
    main_logger_paper_viz.info(f"Image {image_idx_in_test_split}: Critical dense concept indices from FIGS: {critical_dense_indices}")

    num_concept_segments_found_from_image = 0
    if critical_dense_indices:
        # Iterate through critical concepts to find distinct segments from the target image
        # We want `segments_per_concept=1` essentially for this specific viz style if showing multiple concepts
        for dense_idx in critical_dense_indices:
            if num_concept_segments_found_from_image >= target_num_top_concept_segments_from_image:
                break # Found enough concept segments from the image

            if not (0 <= dense_idx < len(ordered_final_concept_original_ids)):
                continue
            original_kmeans_idx = ordered_final_concept_original_ids[dense_idx]
            concept_name_str = feature_names_for_figs_paper_viz[dense_idx]
            
            # Find ONE segment for this concept in this image
            crops = find_segments_for_concept_in_image(
                image_idx_in_test_split, original_kmeans_idx,
                trained_concept_detectors_map_paper_viz,
                seg_infos_test_flat_paper_viz, embeddings_test_flat_paper_viz,
                max_segments_to_return=1 
            )
            if crops and crops[0] is not None:
                if current_panel_idx < 4: # Check if there's space in the panel
                    panel_segment_crops[current_panel_idx] = crops[0]
                    current_panel_idx += 1
                    num_concept_segments_found_from_image += 1
                    main_logger_paper_viz.info(f"  Added segment for Top Concept ('{concept_name_str}') from target image to panel.")
            else:
                main_logger_paper_viz.info(f"  No segment found for Top Concept ('{concept_name_str}') in target image.")
    else:
        main_logger_paper_viz.warning(f"No critical active concepts by FIGS for image {image_idx_in_test_split}.")

    main_logger_paper_viz.info(f"Found {num_concept_segments_found_from_image} top concept segments from the target image.")

    # 2. Fill remaining slots with random segments from the dataset
    num_random_needed = 4 - current_panel_idx # How many more slots to fill in the 2x2 grid
    
    if num_random_needed > 0:
        main_logger_paper_viz.info(f"Attempting to find {num_random_needed} random segments from dataset...")
        random_crops_from_dataset = get_random_segments_from_dataset(
            seg_infos_test_flat_paper_viz, # All test segment infos
            num_random_segments=num_random_needed,
            exclude_image_idx=image_idx_in_test_split 
        )
        main_logger_paper_viz.info(f"  Found {len(random_crops_from_dataset)} random segments from dataset.")
        
        for rand_crop in random_crops_from_dataset:
            if current_panel_idx < 4: # Check if there's space in the panel
                if rand_crop is not None:
                    panel_segment_crops[current_panel_idx] = rand_crop
                    current_panel_idx += 1
                else: # Should not happen if get_random_segments_from_dataset filters Nones
                    panel_segment_crops[current_panel_idx] = None # Explicitly mark as unavailable
                    current_panel_idx += 1 
            else:
                break # Panel is full

    # At this point, panel_segment_crops has up to 4 BGR segment images (or Nones)

    # 3. Plotting the 2x2 Segment Panel
    if not any(crop is not None for crop in panel_segment_crops):
        main_logger_paper_viz.warning(f"No segments (concept or random) to plot for image {image_idx_in_test_split} panel.")
        return

    fig_segments_panel, axes_segments_panel = plt.subplots(2, 2, figsize=(5, 5), squeeze=False)
    axes_flat_panel = axes_segments_panel.flatten()

    for i_panel, crop_bgr_panel in enumerate(panel_segment_crops): # panel_segment_crops will have 4 elements
        ax_panel = axes_flat_panel[i_panel]
        if crop_bgr_panel is not None:
            try:
                ax_panel.imshow(cv2.cvtColor(crop_bgr_panel, cv2.COLOR_BGR2RGB))
            except Exception as e_plot_panel:
                main_logger_paper_viz.error(f"Error displaying panel segment crop: {e_plot_panel}")
                ax_panel.text(0.5,0.5, "PlotErr", ha='center',va='center')
        else:
            # If a slot is None, draw a placeholder or leave blank
            ax_panel.text(0.5,0.5, "N/A", ha='center',va='center', fontsize=8, color='grey')
        ax_panel.axis('off')
        
    plt.tight_layout(pad=0.1) # Minimal padding between subplots
    save_plot(fig_segments_panel, f"{output_filename_stem}_img{image_idx_in_test_split}_segment_panel_2x2", config_paper_viz, main_logger_paper_viz)



def load_labels_for_images(image_ids: list, scene_map: dict, scene_to_idx_map: dict, main_logger_passed) -> np.ndarray:
    """
    Load labels for a list of image IDs based on the provided scene map and scene-to-index mapping.
    """
    labels = []
    processed_ids_for_labeling = set()
    for img_id in image_ids:
        if img_id in processed_ids_for_labeling: continue
        scene = scene_map.get(img_id)
        if scene is None:
            main_logger_passed.warning(f"Image base_id '{img_id}' not found in scene_map during label loading.")
            continue
        idx = scene_to_idx_map.get(scene)
        if idx is not None:
            labels.append(idx)
            processed_ids_for_labeling.add(img_id)
        else:
            main_logger_passed.warning(f"Scene '{scene}' for image base_id '{img_id}' not in scene_to_idx_map.")
    return np.array(labels, dtype=np.int64)

def save_plot(fig, plot_name_stem, config, main_logger_passed):
    """
    Save the plot to a specified directory with a given name stem.
    """
    plots_dir = os.path.join(config.get('log_dir', './logs'), "plots_reused_segments")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"{plot_name_stem}_{config.get('run_id', 'run')}.png")
    try:
        fig.savefig(path, bbox_inches='tight', dpi=config.get('plot_dpi', 150))
        plt.close(fig) 
        main_logger_passed.info(f"Saved plot: {path}")
    except Exception as e:
        main_logger_passed.error(f"Failed to save plot {path}: {e}")

def visualize_random_segments_from_infos(segment_infos_list_of_dicts, num_samples, figsize,
                                          save_path_full_stem, dpi, title, config, main_logger_passed):
    """
    Visualizes random segments from a list of segment info dictionaries.
    """
    if not segment_infos_list_of_dicts or len(segment_infos_list_of_dicts) == 0 :
        main_logger_passed.warning(f"No segment_infos to visualize for '{title}'.")
        return

    actual_num_samples = min(num_samples, len(segment_infos_list_of_dicts))
    if actual_num_samples == 0:
        main_logger_passed.warning(f"Zero samples to visualize for '{title}'.")
        return

    sampled_indices = np.random.choice(len(segment_infos_list_of_dicts), actual_num_samples, replace=False)
    n_cols = min(5, actual_num_samples)
    n_rows = int(np.ceil(actual_num_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if actual_num_samples == 1: axes = np.array([axes]) 
    axes = axes.flatten()

    for i, ax_idx in enumerate(range(actual_num_samples)):
        ax = axes[ax_idx]
        info_dict = segment_infos_list_of_dicts[sampled_indices[i]]
        crop_path = info_dict.get('crop_path')
        seg_crop_bgr = info_dict.get('seg_crop_bgr')

        img_to_show = None
        if seg_crop_bgr is not None:
            try: img_to_show = cv2.cvtColor(seg_crop_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e_cvtc: main_logger_passed.debug(f"CVTColor error: {e_cvtc}")
        elif crop_path and os.path.exists(crop_path):
            try:
                img_bgr_loaded = cv2.imread(crop_path)
                if img_bgr_loaded is not None:
                    img_to_show = cv2.cvtColor(img_bgr_loaded, cv2.COLOR_BGR2RGB)
                else: main_logger_passed.warning(f"cv2.imread returned None for {crop_path}")
            except Exception as e_load: main_logger_passed.warning(f"Error loading crop {crop_path}: {e_load}")
        
        if img_to_show is not None:
            ax.imshow(img_to_show)
        else:
            ax.text(0.5, 0.5, "Img N/A", ha='center', va='center')
        ax.axis('off')

    for j in range(actual_num_samples, len(axes)): axes[j].axis('off')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, save_path_full_stem, config, main_logger_passed)

def visualize_cluster_segments_from_data(
    cluster_id_to_show, cluster_labels_for_all_segments,
    segment_infos_list_of_dicts, # Flat list for the current data split (train/test)
    all_masks_per_image_in_split, # Aligned with all_images_rgb_in_split
    all_images_rgb_in_split,      # Aligned with all_masks_per_image_in_split
    n_samples, grid_size, figsize, mask_alpha,
    save_path_full_stem, config, main_logger_passed
    ):
    """
    Visualizes segments from a specific cluster in the current data split.
    """
    indices_in_cluster = np.where(cluster_labels_for_all_segments == cluster_id_to_show)[0]
    if len(indices_in_cluster) == 0:
        main_logger_passed.info(f"Cluster {cluster_id_to_show} has no segments for visualization.")
        return

    actual_n_samples = min(n_samples, len(indices_in_cluster))
    if actual_n_samples == 0: return
    sampled_global_seg_indices_in_split = np.random.choice(indices_in_cluster, actual_n_samples, replace=False)
    
    n_rows, n_cols = grid_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if actual_n_samples == 1: axes = np.array([axes])
    axes = axes.flatten()

    for i, ax_idx in enumerate(range(actual_n_samples)):
        ax = axes[ax_idx]
        # global_seg_idx is an index into segment_infos_list_of_dicts (for current split)
        global_seg_idx_in_split = sampled_global_seg_indices_in_split[i] 
        
        seg_info_dict = segment_infos_list_of_dicts[global_seg_idx_in_split]
        # 'img_idx' in seg_info_dict is now LOCAL to the current split
        split_local_img_idx = seg_info_dict.get('img_idx')
        # 'seg_idx' in seg_info_dict is the index within that image's list of masks
        seg_idx_in_image = seg_info_dict.get('seg_idx')

        valid_indices = True
        if split_local_img_idx is None or split_local_img_idx >= len(all_images_rgb_in_split) or all_images_rgb_in_split[split_local_img_idx] is None:
            main_logger_passed.warning(f"Invalid local img_idx ({split_local_img_idx}) for cluster viz. Max: {len(all_images_rgb_in_split)}")
            valid_indices = False
        if seg_idx_in_image is None or split_local_img_idx is None or \
           (valid_indices and (split_local_img_idx >= len(all_masks_per_image_in_split) or \
                               seg_idx_in_image >= len(all_masks_per_image_in_split[split_local_img_idx]))):
            main_logger_passed.warning(f"Invalid seg_idx_in_image ({seg_idx_in_image}) for local_img_idx {split_local_img_idx} for cluster viz.")
            valid_indices = False
            
        if not valid_indices:
            ax.text(0.5, 0.5, "Data Err", ha='center', va='center'); ax.axis('off'); continue

        image_rgb = all_images_rgb_in_split[split_local_img_idx]
        mask_bool = all_masks_per_image_in_split[split_local_img_idx][seg_idx_in_image]
        if mask_bool.sum() == 0: ax.axis('off'); continue

        overlay_rgba = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
        # Consistent color for all segments from this cluster for this plot type
        color_for_mask = [0, 255, 0] # Green highlight
        for c_idx_loop in range(3): overlay_rgba[mask_bool, c_idx_loop] = color_for_mask[c_idx_loop]
        overlay_rgba[mask_bool, 3] = int(mask_alpha * 255)
        ax.imshow(image_rgb)
        ax.imshow(overlay_rgba)
        ax.axis('off')

    for j_ax in range(actual_n_samples, len(axes)): axes[j_ax].axis('off')
    fig.suptitle(f"Cluster {cluster_id_to_show} Examples (Overlay)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot(fig, save_path_full_stem, config, main_logger_passed)


def visualize_embedding_tsne(embeddings, labels, title_prefix, config, main_logger_passed, perplexity=30, n_iter=300):
    """
    Visualizes embeddings using t-SNE and saves the plot.
    """
    if embeddings is None or embeddings.shape[0] < max(2, perplexity + 1):
        main_logger_passed.warning(f"TSNE: Not enough samples for {title_prefix} ({embeddings.shape[0] if embeddings is not None else 'None'} samples)")
        return

    num_samples_for_tsne = min(2000, embeddings.shape[0]) # Limit t-SNE for performance
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
        main_logger_passed.error(f"t-SNE failed for {title_prefix}: {e_tsne}", exc_info=True) # Add exc_info
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Prepare labels for plotting
    labels_subset_for_plot = None
    cmap_to_use = 'viridis' 
    
    if labels is not None and len(labels) == embeddings.shape[0]:
        labels_subset_for_plot = labels[indices]
        unique_labels_in_subset = np.unique(labels_subset_for_plot)
        n_unique = len(unique_labels_in_subset)
        if 1 < n_unique <= 10: # If few discrete labels, a qualitative map might be better
            cmap_to_use = 'tab10' # Or 'Set1', 'Paired', etc.
    
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels_subset_for_plot, 
                         cmap=cmap_to_use,
                         alpha=0.6, s=10)
                         
    ax.set_title(f"{title_prefix} - t-SNE of Embeddings ({num_samples_for_tsne} samples)")
    
    # Add legend only if there are discrete labels and not too many
    if labels_subset_for_plot is not None:
        unique_labels_for_legend = np.unique(labels_subset_for_plot)
        if 1 < len(unique_labels_for_legend) <= 10: # Condition for adding a legend
            try:
                # Attempt to use legend_elements, good for discrete integer labels
                legend_elements = scatter.legend_elements(num=min(len(unique_labels_for_legend), 10))[0]
                ax.legend(handles=legend_elements, labels=[str(int(ul)) for ul in unique_labels_for_legend[:len(legend_elements)]], title="Clusters", loc="best")
            except Exception as e_legend:
                main_logger_passed.warning(f"Could not automatically create legend for t-SNE plot {title_prefix}: {e_legend}")
        elif len(unique_labels_for_legend) > 10:
             main_logger_passed.info(f"Too many unique labels ({len(unique_labels_for_legend)}) for t-SNE legend in {title_prefix}.")

    save_plot(fig, f"{title_prefix.replace(' ', '_')}_embeddings_tsne", config, main_logger_passed)


def visualize_decision_explanation( 
    image_idx_in_split,     
    images_split_rgb,           
    seg_infos_flat_split,      
    masks_split_per_image,     
    embeddings_flat_split_segments, 
    image_concept_vector_for_explanation_np,
    predicted_class_name,
    figs_model_instance, trained_detectors_map, ordered_final_concept_original_ids,
    feature_names_for_figs, config, main_logger_passed, title_extra=""
    ):
    """
    Visualizes the decision explanation for a specific image in the split.
    This function highlights critical active concepts and their segments.
    """
    main_logger = main_logger_passed
    main_logger.info(f"Visualizing decision for class {predicted_class_name}  {title_extra}")
    original_image_rgb = images_split_rgb[image_idx_in_split]

    critical_active_concept_dense_indices = get_critical_active_concepts_from_figs(
        figs_model_instance, image_concept_vector_for_explanation_np, feature_names_for_figs
    )
    if not critical_active_concept_dense_indices:
        main_logger.info("No critical *active* concepts for this image."); # Plot image only
        fig_nc, ax_nc = plt.subplots(figsize=(8,8))
        ax_nc.imshow(original_image_rgb)
        ax_nc.set_title(f"Pred: {predicted_class_name} - No Critical Active Concepts {title_extra}")
        ax_nc.axis('off')
        save_plot(fig_nc, f"decision_no_crit_concepts{title_extra.replace(' ','_')}", config, main_logger)
        return

    main_logger.info(f"Critical active concepts (dense indices): {critical_active_concept_dense_indices}")

    # Gather segments and their embeddings *only for this specific image*
    current_image_segment_embeddings_list_viz = []
    current_image_segment_full_masks_viz = [] # List of (H,W) bool masks

    # Iterate through `seg_infos_flat_split` to find segments belonging to `image_idx_in_split`
    # seg_infos_flat_split[k]['img_idx'] is the local index for images_split_rgb
    # seg_infos_flat_split[k]['seg_idx'] is the index into masks_split_per_image[local_img_idx]
    for global_seg_idx_in_flat_split, seg_info_dict in enumerate(seg_infos_flat_split):
        if seg_info_dict.get('img_idx') == image_idx_in_split:
            local_seg_idx_in_img_masks = seg_info_dict.get('seg_idx')
            if local_seg_idx_in_img_masks is not None and \
               local_seg_idx_in_img_masks < len(masks_split_per_image[image_idx_in_split]):
                
                current_image_segment_embeddings_list_viz.append(embeddings_flat_split_segments[global_seg_idx_in_flat_split])
                current_image_segment_full_masks_viz.append(masks_split_per_image[image_idx_in_split][local_seg_idx_in_img_masks])
            else: main_logger.warning(f"Seg info issue for img {image_idx_in_split}, seg_idx {local_seg_idx_in_img_masks}")
    
    if not current_image_segment_embeddings_list_viz:
        main_logger.warning(f"No segments re-assembled for image_idx_in_split {image_idx_in_split}. Cannot show detailed explanation."); return
    
    current_image_segment_embeddings_np_viz = np.array(current_image_segment_embeddings_list_viz)
    
    all_evidential_segment_masks_to_highlight = []
    for dense_idx in critical_active_concept_dense_indices:
        original_kmeans_idx = ordered_final_concept_original_ids[dense_idx]
        concept_name = feature_names_for_figs[dense_idx]
        if original_kmeans_idx not in trained_detectors_map: continue
        detector_pipeline, detector_threshold = trained_detectors_map[original_kmeans_idx]

        activating_indices_relative_to_image = find_activating_segments_for_concept_viz(
            current_image_segment_embeddings_np_viz, detector_pipeline, detector_threshold, main_logger
        )
        main_logger.debug(f"  Concept '{concept_name}': {len(activating_indices_relative_to_image)} activating segs in this image.")
        for rel_idx in activating_indices_relative_to_image:
            all_evidential_segment_masks_to_highlight.append(current_image_segment_full_masks_viz[rel_idx])

    fig, ax = plt.subplots(figsize=(10, 10)); ax.imshow(original_image_rgb)
    ax.set_title(f"Predicted: {predicted_class_name} - Evidential Segments {title_extra}", fontsize=10); ax.axis('off')
    if all_evidential_segment_masks_to_highlight:
        main_logger.info(f"Highlighting {len(all_evidential_segment_masks_to_highlight)} evidential segments.")
        combined_mask = np.zeros_like(all_evidential_segment_masks_to_highlight[0], dtype=bool)
        for m in all_evidential_segment_masks_to_highlight:
            if m.shape == combined_mask.shape: combined_mask = np.logical_or(combined_mask, m.astype(bool))
        overlay = np.zeros((original_image_rgb.shape[0], original_image_rgb.shape[1], 4), dtype=np.uint8)
        overlay[combined_mask, :3] = [255, 0, 0]; overlay[combined_mask, 3] = int(0.45 * 255)
        ax.imshow(overlay)
    else: main_logger.info("No specific evidential segments found from critical concepts.")
    save_plot(fig, f"decision_viz_cl{predicted_class_name}{title_extra.replace(' ','_')}", config, main_logger)



def visualize_concept_vs_random_segments(
    concept_cluster_id,                         # The K-Means cluster ID to visualize as the "concept"
    all_segment_infos_in_split,                 # Flat list of segment dicts for the current data split
                                                # Each dict must contain 'seg_crop_bgr' or 'crop_path'
    all_cluster_labels_for_segments_in_split,   # Flat array of cluster labels from K-Means
    num_segments_to_show_each_side=30,          # Number of concept segments AND random segments
    grid_cols=10,                               # How many segments per row in the plot
    figsize_per_row=20,                         # Figure width for a row of `grid_cols`
    config=None,                                # Main config dict
    main_logger_passed=None,                    # Main logger instance
    plot_title_prefix=""
    ):
    """
    Visualizes segments from a specific concept cluster against random segments from the same split.
    This function will plot segments from the specified concept cluster alongside random segments
    """

    if main_logger_passed is None: main_logger_passed = logging.getLogger("VisConceptVsRandom")
    if config is None: config = {} # Basic default

    # --- 1. Get segments belonging to the concept cluster ---
    concept_segment_indices_in_split = np.where(all_cluster_labels_for_segments_in_split == concept_cluster_id)[0]

    if len(concept_segment_indices_in_split) == 0:
        main_logger_passed.info(f"No segments found for concept cluster {concept_cluster_id}. Skipping visualization.")
        return

    num_concept_segments_to_plot = min(num_segments_to_show_each_side, len(concept_segment_indices_in_split))
    sampled_concept_indices = np.random.choice(
        concept_segment_indices_in_split,
        num_concept_segments_to_plot,
        replace=False
    )
    concept_segments_to_plot_infos = [all_segment_infos_in_split[i] for i in sampled_concept_indices]

    # --- 2. Get a random set of segments (that are NOT from the current concept_cluster_id) ---
    # Exclude the current concept's segments from the pool of random segments
    non_concept_segment_indices_in_split = np.where(all_cluster_labels_for_segments_in_split != concept_cluster_id)[0]
    
    if len(non_concept_segment_indices_in_split) == 0:
        main_logger_passed.warning(f"No segments found *outside* concept cluster {concept_cluster_id} for random sampling. Skipping random side.")
        # Proceed to plot only concept segments if desired, or return
        # make random_segments_to_plot_infos empty if this happens
        random_segments_to_plot_infos = []
        num_random_segments_to_plot = 0
    else:
        num_random_segments_to_plot = min(num_segments_to_show_each_side, len(non_concept_segment_indices_in_split))
        sampled_random_indices = np.random.choice(
            non_concept_segment_indices_in_split,
            num_random_segments_to_plot,
            replace=False
        )
        random_segments_to_plot_infos = [all_segment_infos_in_split[i] for i in sampled_random_indices]


    # --- 3. Plotting ---
    # We'll have two main sections in the plot: Concept Segments and Random Segments
    total_segments_to_plot = num_concept_segments_to_plot + num_random_segments_to_plot
    if total_segments_to_plot == 0:
        main_logger_passed.info(f"Nothing to plot for concept {concept_cluster_id}.")
        return

    # Calculate grid: plot concept segments first, then random ones
    # Each "side" gets its own rows if possible, or they flow if too many for one figure easily.
    # create one figure with two "halves"

    num_rows_concept = int(np.ceil(num_concept_segments_to_plot / grid_cols))
    num_rows_random = int(np.ceil(num_random_segments_to_plot / grid_cols))
    total_rows = num_rows_concept + num_rows_random + (1 if num_rows_concept > 0 and num_rows_random > 0 else 0) 

    fig_height_per_actual_row = figsize_per_row / grid_cols # Maintain aspect ratio of subplots
    fig_total_height = total_rows * fig_height_per_actual_row if total_rows > 0 else fig_height_per_actual_row

    if total_rows == 0: return # Should be caught by total_segments_to_plot == 0

    fig, axes = plt.subplots(total_rows, grid_cols, figsize=(figsize_per_row, fig_total_height), squeeze=False)
    
    current_ax_idx = 0

    def plot_segment_batch(segment_infos_batch, title):
        nonlocal current_ax_idx
        if not segment_infos_batch: return

        # Add title row for this batch if it's the first batch or if there's a previous batch
        if title:
            row_idx = current_ax_idx // grid_cols
            # Clear axes in this title row and add a centered title
            for c in range(grid_cols): axes[row_idx, c].axis('off')
            fig.text(0.5, axes[row_idx, 0].get_position().y0 + axes[row_idx,0].get_position().height / 2, 
                     title, ha='center', va='center', fontsize=16, weight='bold')
            current_ax_idx = (row_idx + 1) * grid_cols # Move to the next row for actual images

        for i, seg_info in enumerate(segment_infos_batch):
            row_idx = current_ax_idx // grid_cols
            col_idx = current_ax_idx % grid_cols
            
            if row_idx >= total_rows: # Should not happen with correct total_rows calculation
                main_logger_passed.warning("Exceeded calculated plot rows.")
                break
            
            ax = axes[row_idx, col_idx]
            
            img_to_show = None
            seg_crop_bgr = seg_info.get('seg_crop_bgr')
            crop_path = seg_info.get('crop_path')

            if seg_crop_bgr is not None:
                try: img_to_show = cv2.cvtColor(seg_crop_bgr, cv2.COLOR_BGR2RGB)
                except Exception as e_cvtc: main_logger_passed.debug(f"CVTColor error: {e_cvtc}")
            elif crop_path and os.path.exists(crop_path):
                try:
                    img_bgr_loaded = cv2.imread(crop_path)
                    if img_bgr_loaded is not None: img_to_show = cv2.cvtColor(img_bgr_loaded, cv2.COLOR_BGR2RGB)
                except Exception as e_load: main_logger_passed.debug(f"Error loading crop {crop_path}: {e_load}")
            
            if img_to_show is not None:
                ax.imshow(img_to_show)
            else:
                ax.text(0.5, 0.5, "Img N/A", ha='center', va='center')
            ax.axis('off')
            current_ax_idx += 1

    # Plot Concept Segments
    plot_segment_batch(concept_segments_to_plot_infos, f"Option A:")

    # Plot Random Segments (if any)
    if num_random_segments_to_plot > 0:
        plot_segment_batch(random_segments_to_plot_infos, f"Option B:")

    # Hide any remaining unused axes at the end
    final_row_plotted = (current_ax_idx -1) // grid_cols
    for r in range(total_rows):
        for c_ax in range(grid_cols):
            if r > final_row_plotted or (r == final_row_plotted and c_ax >= (current_ax_idx -1) % grid_cols + 1) :
                 if r < axes.shape[0] and c_ax < axes.shape[1]: # Check bounds before accessing
                    axes[r, c_ax].axis('off')


    plt.subplots_adjust(hspace=0.4, wspace=0.05) # Adjust spacing if titles overlap images
    
    plot_filename_stem = f"{plot_title_prefix}_concept_{concept_cluster_id}_vs_random"
    save_plot(fig, plot_filename_stem, config, main_logger_passed)


def visualize_concept_vs_random_six(
    concept_cluster_id,
    all_segment_infos,
    all_cluster_labels,
    num_each=6,
    config=None,
    logger=None,
    plot_prefix=""
):
    """
    Produce two separate figures:
      1. Concept segments (num_each) laid out in 2x3 grid
      2. Random segments (num_each) laid out in 2x3 grid
    No titles or text; axes are turned off. Clean whitespace.
    """
    if logger is None:
        logger = logging.getLogger("vis_six_simplified")
    if config is None:
        config = {}

    # Find indices
    idx_concept = np.where(all_cluster_labels == concept_cluster_id)[0]
    idx_random = np.where(all_cluster_labels != concept_cluster_id)[0]
    if len(idx_concept) == 0 or len(idx_random) == 0:
        logger.warning(f"Insufficient samples for cluster {concept_cluster_id}")
        return

    sel_concept = np.random.choice(idx_concept, min(num_each, len(idx_concept)), replace=False)
    sel_random = np.random.choice(idx_random, min(num_each, len(idx_random)), replace=False)

    # Helper to load image
    def load_rgb(info):
        bgr = info.get('seg_crop_bgr')
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        path = info.get('crop_path')
        if path and os.path.exists(path):
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    # --- Figure 1: Concept ---
    fig_con, axes_con = plt.subplots(2, 3, figsize=(6, 4), constrained_layout=True)
    axes_con = axes_con.flatten()
    for ax, idx in zip(axes_con, sel_concept):
        img = load_rgb(all_segment_infos[int(idx)])
        if img is not None:
            ax.imshow(img)
        ax.axis('off')

    stem_con = f"{plot_prefix}_cluster_{concept_cluster_id}_concept"
    dir_plots = os.path.join(config.get('log_dir', './logs'), "plots_reused_segments")
    os.makedirs(dir_plots, exist_ok=True)
    path_con = os.path.join(dir_plots, f"{stem_con}_{config.get('run_id','run')}.png")
    fig_con.savefig(path_con, bbox_inches='tight', dpi=config.get('plot_dpi', 150))
    plt.close(fig_con)
    logger.info(f"Saved concept-only plot: {path_con}")

    # --- Figure 2: Random ---
    fig_rand, axes_rand = plt.subplots(2, 3, figsize=(6, 4), constrained_layout=True)
    axes_rand = axes_rand.flatten()
    for ax, idx in zip(axes_rand, sel_random):
        img = load_rgb(all_segment_infos[int(idx)])
        if img is not None:
            ax.imshow(img)
        ax.axis('off')

    stem_rand = f"{plot_prefix}_cluster_{concept_cluster_id}_random"
    path_rand = os.path.join(dir_plots, f"{stem_rand}_{config.get('run_id','run')}.png")
    fig_rand.savefig(path_rand, bbox_inches='tight', dpi=config.get('plot_dpi', 150))
    plt.close(fig_rand)
    logger.info(f"Saved random-only plot: {path_rand}")

def get_critical_active_concepts_from_figs(figs_model_instance, image_concept_vector_np, feature_names_for_figs):
    """
    Extracts the critical active concepts from a fitted FIGS model for a given image concept vector.
    Returns a list of dense indices corresponding to the active concepts.
    """
    if image_concept_vector_np.ndim == 2: image_concept_vector_np = image_concept_vector_np.flatten()
    critical_active_concept_dense_indices = set()
    for tree_root in figs_model_instance.trees_:
        node = tree_root
        while node.left: 
            feature_idx, threshold_val = node.feature, node.threshold
            if feature_idx is None or not (0 <= feature_idx < len(image_concept_vector_np)): break
            concept_val_in_image = image_concept_vector_np[feature_idx]
            if concept_val_in_image <= threshold_val: node = node.left
            else: critical_active_concept_dense_indices.add(feature_idx); node = node.right
    return list(critical_active_concept_dense_indices)

def find_activating_segments_for_concept_viz(image_segments_embeddings_np_for_viz, concept_detector_pipeline_viz,concept_detector_threshold_viz, main_logger_passed_viz):
    """
    Finds segments that activate a specific concept in the visualization context.
    Returns a list of indices of segments that activate the concept.
    """
    if image_segments_embeddings_np_for_viz is None or image_segments_embeddings_np_for_viz.shape[0] == 0: return []
    try:
        if hasattr(concept_detector_pipeline_viz, "predict_proba"):
            segment_probs = concept_detector_pipeline_viz.predict_proba(image_segments_embeddings_np_for_viz)[:, 1]
        elif hasattr(concept_detector_pipeline_viz, "decision_function"):
            segment_scores = concept_detector_pipeline_viz.decision_function(image_segments_embeddings_np_for_viz)
            segment_probs = expit(segment_scores)
        else: return []
    except Exception as e: main_logger_passed_viz.error(f"Error predicting seg probs for viz: {e}"); return []
    return np.where(segment_probs >= concept_detector_threshold_viz)[0].tolist()

def visualize_concept_vectors_pca(concept_vectors, image_labels, title_prefix, config, main_logger_passed):
    """
    Visualizes concept vectors using PCA and saves the plot.
    """
    if concept_vectors is None or concept_vectors.shape[0] < 2:
        main_logger_passed.warning(f"PCA: Not enough concept vectors for {title_prefix} ({concept_vectors.shape[0] if concept_vectors is not None else 'None'} samples).")
        return

    num_components_for_pca = min(2, concept_vectors.shape[1])
    if num_components_for_pca == 0:
        main_logger_passed.warning(f"PCA: Concept vectors have 0 features for {title_prefix}.")
        return

    pca = PCA(n_components=num_components_for_pca, random_state=config['seed'])
    try:
        vectors_2d_or_1d = pca.fit_transform(concept_vectors)
    except Exception as e_pca:
        main_logger_passed.error(f"PCA transformation failed for {title_prefix}: {e_pca}", exc_info=True)
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    labels_for_plot = None
    cmap_to_use = 'viridis'
    
    if image_labels is not None and len(image_labels) == vectors_2d_or_1d.shape[0]:
        labels_for_plot = image_labels
        unique_labels_in_plot = np.unique(labels_for_plot)
        n_unique = len(unique_labels_in_plot)
        if 1 < n_unique <= 10: # If few discrete labels, a qualitative map might be better
            cmap_to_use = 'tab10' 
    elif image_labels is not None and len(image_labels) != vectors_2d_or_1d.shape[0]:
        main_logger_passed.warning(f"PCA Viz: Label length mismatch for {title_prefix}. "
                                   f"Vectors shape {vectors_2d_or_1d.shape[0]}, labels len {len(image_labels)}. Plotting without color labels.")


    if num_components_for_pca == 1:
        # Add some jitter on y-axis for 1D PCA plot to see point density
        y_jitter = np.random.rand(vectors_2d_or_1d.shape[0]) * 0.1 
        scatter = ax.scatter(vectors_2d_or_1d[:, 0], y_jitter, 
                             c=labels_for_plot, 
                             cmap=cmap_to_use, 
                             alpha=0.7, s=15)
    else: # num_components_for_pca == 2
        scatter = ax.scatter(vectors_2d_or_1d[:, 0], vectors_2d_or_1d[:, 1], 
                             c=labels_for_plot, 
                             cmap=cmap_to_use, 
                             alpha=0.7, s=15)
    
    ax.set_title(f"{title_prefix} - PCA of Image Concept Vectors")
    
    # Add legend only if there are discrete labels and not too many
    if labels_for_plot is not None:
        unique_labels_for_legend = np.unique(labels_for_plot)
        if 1 < len(unique_labels_for_legend) <= 10:
            try:
                legend_elements = scatter.legend_elements(num=min(len(unique_labels_for_legend), 10))[0]
                # Get class names if available from config, else use integer labels
                class_names_for_legend = config.get('chosen_classes', [str(int(ul)) for ul in unique_labels_for_legend])
                # Ensure class_names_for_legend matches the order and number of unique_labels_for_legend
                display_labels_for_legend = [class_names_for_legend[int(ul)] 
                                             if 0 <= int(ul) < len(class_names_for_legend) else str(int(ul))
                                             for ul in unique_labels_for_legend[:len(legend_elements)]]

                ax.legend(handles=legend_elements, labels=display_labels_for_legend, title="Image Classes", loc="best")
            except Exception as e_legend_pca:
                main_logger_passed.warning(f"Could not automatically create legend for PCA plot {title_prefix}: {e_legend_pca}")
        elif len(unique_labels_for_legend) > 10:
            main_logger_passed.info(f"Too many unique labels ({len(unique_labels_for_legend)}) for PCA legend in {title_prefix}.")

    save_plot(fig, f"{title_prefix.replace(' ', '_')}_concept_vectors_pca", config, main_logger_passed)