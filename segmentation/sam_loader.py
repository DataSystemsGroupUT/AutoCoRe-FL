from sam2.build_sam import build_sam2 # type: ignore
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator # type: ignore

def load_sam_model(sam2_cfg, sam2_checkpoint, device, points_per_side=32, pred_iou_thresh=0.93, stability_score_thresh=0.92, min_mask_area=1000):
    """
    Loads the SAM-v2 model and mask generator with your parameters.
    """
    sam = build_sam2(sam2_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_area
    )
    return sam, mask_generator