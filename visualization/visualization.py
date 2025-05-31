import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from collections import defaultdict
from sklearn.metrics import pairwise_distances
import cv2
def visualize_cluster_segments(
    cluster_id: int,
    cluster_labels: np.ndarray,
    all_segments: list,
    all_masks: list,
    all_images: list,
    n_samples=10,
    grid_size=(2, 5), 
    figsize=(12, 6),
    mask_alpha=0.8,
    save_path=None,
    dpi=600
):
    """
    visualization of cluster segments.

    Args:
        cluster_id: ID of the cluster to visualize.
        cluster_labels: (N,) cluster assignment array.
        all_segments: list of (img_id, seg_id).
        all_masks: list of lists of masks, indexed by [img_id][seg_id].
        all_images: list of RGB images.
        n_samples: Number of segments to visualize (max).
        grid_size: Tuple of (rows, columns) for subplot grid.
        figsize: Size of figure.
        mask_alpha: Transparency level for segment mask.
        save_path: Path to save the figure.
        dpi: Dots per inch for high-res output.
    """
    seg_indices = np.where(cluster_labels == cluster_id)[0]
    n_total = len(seg_indices)
    if n_total == 0:
        print(f"[Warning] Cluster {cluster_id} is empty.")
        return

    # Select samples to display
    seg_indices = seg_indices[:n_samples]
    n_display = len(seg_indices)

    n_rows, n_cols = grid_size
    if n_display < n_rows * n_cols:
        n_cols = min(n_cols, n_display)
        n_rows = int(np.ceil(n_display / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()

    # Plot each segment
    for ax in axs[n_display:]:  # Hide unused axes
        ax.axis('off')

    for idx, seg_idx in enumerate(seg_indices):
        base_id, img_id, seg_id, seg_crop_bgr = all_segments[seg_idx].values()
        print(f"[Visualizing] Segment {seg_id} from image {base_id} (ID: {img_id})")
        # 
        mask = all_masks[img_id][seg_id]  # (H, W), bool
        image_rgb = all_images[img_id].copy()  # (H, W, 3)

        if mask.sum() == 0 or image_rgb is None:
            axs[idx].axis('off')
            continue

        # Mask effect: transparency
        masked_image = image_rgb.astype(float) / 255.0
        background = np.ones_like(masked_image)  # white background

        # Apply mask as alpha blending
        mask_3d = np.stack([mask] * 3, axis=-1)
        blended = np.where(mask_3d, masked_image * mask_alpha + background * (1 - mask_alpha), background)

        # Ensure proper range
        blended = np.clip(blended, 0, 1)

        # Show image
        axs[idx].imshow(blended)
        axs[idx].axis('off')  # No axis
        axs[idx].set_aspect('equal')

    fig.suptitle(
        f"Cluster {cluster_id}: {n_display} samples (of {n_total})",
        fontsize=14,
        y=1.02
    )

    plt.tight_layout(pad=0.1)  # Minimal padding

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        plt.close(fig)
        print(f"[Saved] Figure saved to {save_path}")
    else:
        plt.show()

def visualize_random_segments(
    segment_infos: list,
    num_samples=10,
    grid_size=None,  # (rows, cols)
    figsize=(12, 6),
    save_path=None,
    dpi=600
):
    """
    visualization of random segment crops,
    with white background and no text/titles.

    Args:
        segment_infos: List of dicts with 'crop_path'.
        num_samples: Number of segments to display.
        grid_size: (rows, cols), auto-calculated if None.
        figsize: Figure size.
        save_path: Path to save output figure (high-res).
        dpi: DPI for high-quality export.
    """

    if len(segment_infos) == 0:
        print("[Warning] No segment infos to visualize.")
        return

    # Select random samples
    random_indices = np.random.choice(
        len(segment_infos),
        size=min(num_samples, len(segment_infos)),
        replace=False
    )
    n_display = len(random_indices)

    # Auto grid size if not specified
    if grid_size is None:
        n_cols = min(n_display, 5)
        n_rows = int(np.ceil(n_display / n_cols))
    else:
        n_rows, n_cols = grid_size

    # Setup figure and axes
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()

    # Hide extra axes
    for ax in axs[n_display:]:
        ax.axis('off')

    # Load and display each crop
    for ax, idx in zip(axs, random_indices):
        info = segment_infos[idx]
        crop_path = info["crop_path"]

        # Load image (RGBA or RGB)
        crop_rgba = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)  
        if crop_rgba is None:
            print(f"[Warning] Failed to load image: {crop_path}")
            ax.axis('off')
            continue

        # Handle alpha (mask) if available, otherwise treat black as background
        if crop_rgba.shape[-1] == 4:
            # Split channels
            b, g, r, a = cv2.split(crop_rgba)
            rgb = cv2.merge((r, g, b))
            mask = a > 0  # alpha mask
        else:
            b, g, r = cv2.split(crop_rgba)
            rgb = cv2.merge((r, g, b))
            mask = ~((r == 0) & (g == 0) & (b == 0))  # non-black pixels as object

        # Normalize to [0, 1]
        rgb = rgb.astype(float) / 255.0

        # Prepare white background
        white_bg = np.ones_like(rgb)

        # Blend: object where mask is True, white elsewhere
        mask_3d = np.stack([mask] * 3, axis=-1)
        blended = np.where(mask_3d, rgb, white_bg)

        # Ensure valid range
        blended = np.clip(blended, 0, 1)

        # Plot
        ax.imshow(blended)
        ax.axis('off')
        ax.set_aspect('equal')

    plt.tight_layout(pad=0.1, h_pad=0.2, w_pad=0.2)

    # Save or show figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
        plt.close(fig)
        print(f"[Saved] Figure saved to {save_path}")
    else:
        plt.show()
