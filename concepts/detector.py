import logging
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_curve

logger_detector = logging.getLogger("ConceptDetector")

def train_concept_detector(cluster_idx, X, y_cluster_labels, image_groups, config):
    """Train concept detector (LR or SVM) with image-level data splitting and optimal thresholding."""
    logger_detector.debug(f"Cluster {cluster_idx}: Training concept detector with {X.shape[0]} samples, {X.shape[1]} features.")
    logger_detector.debug(f"seed is,{config.get('seed')}")
    # Explicitly get the integer seed from the passed config dictionary
    current_random_state = config.get('seed')
    if not isinstance(current_random_state, int):
        logger_detector.warning(f"Cluster {cluster_idx}: Seed in config is not an int ({current_random_state}). Using default 42.")
        current_random_state = 42
    
    logger_detector.debug(f"Cluster {cluster_idx}: Using random_state={current_random_state} for detector training.")

    pos_mask = (y_cluster_labels == cluster_idx)
    neg_mask = ~pos_mask
    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    MIN_SAMPLES_PER_CLASS = config.get('detector_min_samples_per_class', 20)
    if len(pos_indices) < MIN_SAMPLES_PER_CLASS or len(neg_indices) < MIN_SAMPLES_PER_CLASS:
        logger_detector.debug(f"Cluster {cluster_idx}: Insufficient samples. Pos: {len(pos_indices)}, Neg: {len(neg_indices)}. Min req: {MIN_SAMPLES_PER_CLASS}")
        return cluster_idx, None, 0.0

    n_samples_balance = min(len(pos_indices), len(neg_indices))
    # Use the extracted integer seed for np.random.default_rng
    rng = np.random.default_rng(current_random_state) 
    pos_samples_balanced = rng.choice(pos_indices, n_samples_balance, replace=False)
    neg_samples_balanced = rng.choice(neg_indices, n_samples_balance, replace=False)

    X_balanced = np.vstack([X[pos_samples_balanced], X[neg_samples_balanced]])
    y_balanced = np.concatenate([np.ones(n_samples_balance), np.zeros(n_samples_balance)])
    groups_balanced = np.concatenate([image_groups[pos_samples_balanced], image_groups[neg_samples_balanced]])

    n_cv_splits = config.get('detector_cv_splits', 3)
    if len(np.unique(groups_balanced)) < n_cv_splits :
        n_cv_splits = max(2, len(np.unique(groups_balanced)))
        logger_detector.warning(f"Cluster {cluster_idx}: Reduced CV splits to {n_cv_splits} due to few groups.")

    sgkf = StratifiedGroupKFold(n_splits=n_cv_splits, shuffle=True, random_state=current_random_state)
    
    try:
        train_idx, val_idx = next(sgkf.split(X_balanced, y_balanced, groups_balanced))
    except ValueError as e:
        logger_detector.warning(f"Cluster {cluster_idx}: StratifiedGroupKFold failed ('{e}'). Using simple GroupShuffleSplit.")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=current_random_state)
        if X_balanced.shape[0] == 0: # Cannot split if empty
             logger_detector.error(f"Cluster {cluster_idx}: X_balanced is empty, cannot perform split.")
             return cluster_idx, None, 0.0
        train_idx, val_idx = next(gss.split(X_balanced, y_balanced, groups_balanced))

    X_train, X_val = X_balanced[train_idx], X_balanced[val_idx]
    y_train, y_val = y_balanced[train_idx], y_balanced[val_idx]

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        logger_detector.warning(f"Cluster {cluster_idx}: Train or Val set is empty after split. X_train: {X_train.shape}, X_val: {X_val.shape}")
        return cluster_idx, None, 0.0

    # PCA step initialization
    n_components_pca_config = config.get('pca_n_components', 128)

    max_possible_pca_components = min(X_train.shape[0], X_train.shape[1])
    
    actual_pca_n_components = min(n_components_pca_config, max_possible_pca_components)
    
    pca_pipeline_step = 'passthrough' # Default if PCA is not applicable
    if actual_pca_n_components >= 1: # Check if PCA is feasible and desired
        pca_pipeline_step = PCA(n_components=actual_pca_n_components, svd_solver='randomized', random_state=current_random_state)
        logger_detector.debug(f"Cluster {cluster_idx}: PCA using n_components={actual_pca_n_components}")
    else:
        logger_detector.warning(f"Cluster {cluster_idx}: PCA cannot be applied or n_components < 1 (Effective n_components: {actual_pca_n_components}). Skipping PCA.")


    detector_type = config.get('detector_type', 'lr')
    if detector_type == 'svm':
        classifier = SVC(class_weight='balanced', probability=True, random_state=current_random_state, kernel='linear', C=config.get('svm_C', 1.0))
    elif detector_type == 'lr':
        classifier = LogisticRegression(class_weight='balanced', solver='saga', max_iter=config.get('lr_max_iter', 10000), random_state=current_random_state)
    else:
        logger_detector.error(f"Unsupported detector_type: {detector_type} in config.")
        return cluster_idx, None, 0.0

    pipe = Pipeline([
        ('pca', pca_pipeline_step),
        ('clf', classifier)
    ])

    try:
        pipe.fit(X_train, y_train)
    except ValueError as e:
        logger_detector.error(f"Cluster {cluster_idx}: Fitting detector pipeline failed: {e}. X_train: {X_train.shape}, y_train unique: {np.unique(y_train, return_counts=True)}")
        return cluster_idx, None, 0.0

    if not hasattr(pipe, "predict_proba"): # Should not happen with SVC(probability=True) or LR
        logger_detector.error(f"Cluster {cluster_idx}: Fitted pipeline does not have predict_proba method.")
        return cluster_idx, None, 0.0

    val_probs = pipe.predict_proba(X_val)[:, 1]
    
    optimal_threshold = 0.5 # Default
    score = 0.0 # Default

    if len(np.unique(y_val)) < 2:
        logger_detector.warning(f"Cluster {cluster_idx}: Validation set has only one class ({np.unique(y_val)}). Score may be misleading. Using default threshold 0.5.")
        score = np.mean((val_probs >= optimal_threshold).astype(int) == y_val)
    else:
        fpr, tpr, thresholds_roc = roc_curve(y_val, val_probs)
        valid_thresholds = thresholds_roc[np.isfinite(thresholds_roc)] # Filter out non-finite thresholds
        if len(valid_thresholds) == 0:
             logger_detector.warning(f"Cluster {cluster_idx}: roc_curve returned no finite thresholds. Using default 0.5.")
             optimal_threshold = 0.5
        else:
            best_j_stat = -1
            for thr_candidate in sorted(np.unique(val_probs)): # Iterate unique predicted probabilities as thresholds
                j_stat = np.mean(val_probs[y_val==1] >= thr_candidate) - np.mean(val_probs[y_val==0] >= thr_candidate) # tpr - fpr
                if j_stat > best_j_stat:
                    best_j_stat = j_stat
                    optimal_threshold = thr_candidate
            if best_j_stat == -1 : # No valid threshold found, fallback
                optimal_threshold = 0.5


        val_preds_at_optimal = (val_probs >= optimal_threshold).astype(int)
        score = np.mean(val_preds_at_optimal == y_val)
        # score = roc_auc_score(y_val, val_probs) # Consider using AUC as the primary score for detector quality

    logger_detector.debug(f"Cluster {cluster_idx}: Detector trained. Score (Acc@OptThr): {score:.3f}, Optimal Threshold: {optimal_threshold:.3f}")
    return cluster_idx, (pipe, optimal_threshold), score