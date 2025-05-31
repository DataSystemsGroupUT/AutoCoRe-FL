from collections import defaultdict
import logging
import numpy as np
import operator
import os
import pickle
import pandas as pd
import re

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, check_array, check_X_y
import torch
from scipy.special import softmax ,expit


# --- FederatedClient class ---
from AutoCore_FL.segmentation.segment_crops import filter_zero_segment_images
try:
    from AutoCore_FL.segmentation.sam_loader import load_sam_model
    from AutoCore_FL.segmentation.segment_crops import generate_segments_and_masks, filter_zero_segment_images
    from AutoCore_FL.embedding.dino_loader import init_dino, init_target_model
    from AutoCore_FL.embedding.compute_embeddings import compute_final_embeddings
    from AutoCore_FL.concepts.detector import train_concept_detector
    from AutoCore_FL.clustering.federated_kmeans import FederatedKMeans
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Import Error for federated modules in client.py: {e}.")
    raise

try:
    from imodels.util.arguments import check_fit_arguments
    from imodels.util.data_util import encode_categories
except ImportError as e:
    logging.error(f"Failed to import utilities from imodels: {e}. Ensure imodels is installed or utils are available.")
    def check_fit_arguments(estimator, X, y, feature_names=None): 
        logging.warning("Using placeholder check_fit_arguments")
        X_out, y_out = check_X_y(X, y, multi_output=True if isinstance(estimator, ClassifierMixin) and hasattr(estimator, 'n_outputs_') and estimator.n_outputs_ > 1 else False) # Basic check
        if hasattr(X, 'columns') and feature_names is None: feature_names = X.columns.tolist()
        elif feature_names is None and X_out is not None: feature_names = [f'X{i}' for i in range(X_out.shape[1])]
        else: feature_names = []
            
        if isinstance(estimator, ClassifierMixin): 
            from sklearn.utils.multiclass import unique_labels
            estimator.classes_ = unique_labels(y) if y is not None else np.array([])
            estimator.n_outputs_ = len(estimator.classes_) if estimator.classes_.size > 0 else (1 if y_out.ndim == 1 or (y_out.ndim==2 and y_out.shape[1] == 1) else (y_out.shape[1] if y_out.ndim == 2 else 0) )
        if X_out is not None: estimator.n_features_in_ = X_out.shape[1]
        else: estimator.n_features_in_ = 0
        estimator.feature_names_ = feature_names
        return X_out, y_out, feature_names

    def encode_categories(X, categorical_features, encoder=None):
        logging.warning("Using placeholder encode_categories")
        if categorical_features: raise NotImplementedError("Placeholder encode_categories does not support categorical features.")
        return X, None

class LocalNode:
    def __init__(self, feature: int = None, threshold: float = None, value=None,
                 value_sklearn=None, idxs=None, is_root: bool = False, left=None,
                 impurity: float = None, impurity_reduction: float = None,
                 tree_num: int = None, node_id: int = None, right=None, N: int = None):
        self.is_root, self.idxs, self.tree_num = is_root, idxs, tree_num
        self.node_id, self.feature, self.impurity = node_id, feature, impurity
        self.impurity_reduction, self.value_sklearn = impurity_reduction, value_sklearn
        self.value, self.threshold, self.left, self.right = value, threshold, left, right
        self.left_temp, self.right_temp, self.N = None, None, N

    def setattrs(self, **kwargs): [setattr(self, k, v) for k, v in kwargs.items()]
    def __str__(self):
        if self.feature is not None: return f"X_{self.feature} <= {self.threshold:.3f} (N={self.N})"
        return f"Val: {self.value.flatten() if isinstance(self.value, np.ndarray) else self.value} (N={self.N})"
    def __repr__(self): return self.__str__()

class LocalFIGSBase(BaseEstimator):
    def __init__(self, max_rules: int = 12, max_trees: int = None, min_impurity_decrease: float = 0.0,
                 random_state=None, max_features=None):
        super().__init__()
        self.max_rules, self.max_trees, self.min_impurity_decrease = max_rules, max_trees, min_impurity_decrease
        self.random_state, self.max_features = random_state, max_features
        self._init_decision_function()

    def _init_decision_function(self):
        if isinstance(self, ClassifierMixin): self.decision_function = lambda x: self.predict_proba(x)[:, 1]
        elif isinstance(self, RegressorMixin): self.decision_function = self.predict
            
    def _get_logger(self): return getattr(self, 'logger', logging.getLogger(self.__class__.__name__))

    def _construct_node_with_stump(self, X, y, idxs, tree_num, sample_weight=None, max_features=None):
        logger = self._get_logger()
        N_unw = idxs.sum()
        sw_subset = sample_weight[idxs] if sample_weight is not None else None

        def make_leaf_node(val_arr_2d):
            return LocalNode(idxs=idxs, value=val_arr_2d, tree_num=tree_num, N=N_unw, impurity=0.0)

        if N_unw == 0 or (sw_subset is not None and sw_subset.size > 0 and np.sum(sw_subset) < 1e-9):
            val_shape = (1, y.shape[1]) if y.ndim == 2 else (1, 1)
            def_val = np.zeros(val_shape)
            if y.ndim > 0 and y[idxs].size > 0:
                def_val = np.mean(y[idxs], axis=0, keepdims=True) if y.ndim == 2 else np.array([[np.mean(y[idxs])]])
            return make_leaf_node(def_val)

        stump = DecisionTreeRegressor(max_depth=1, max_features=max_features, random_state=self.random_state)
        try: stump.fit(X[idxs], y[idxs], sample_weight=sw_subset)
        except ValueError as e:
            logger.error(f"Stump fit error T{tree_num}: {e}"); 
            val_shape = (1, y.shape[1]) if y.ndim == 2 else (1, 1)
            return make_leaf_node(np.zeros(val_shape))

        t_ = stump.tree_
        feat, thr, imp, val, n_w_s = t_.feature, t_.threshold, t_.impurity, t_.value, t_.weighted_n_node_samples
        if val.ndim == 3 and val.shape[2]==1: val = val.squeeze(axis=2)
        if val.ndim == 1: val = val.reshape(val.shape[0], -1)
        if val.ndim == 0: val = np.array([[val]])
        
        S, L, R = 0, 1, 2
        if feat[S] == -2: return LocalNode(idxs=idxs,value=val[S].reshape(1,-1),tree_num=tree_num,impurity=imp[S],N=N_unw)
        
        imp_r = (imp[S]*n_w_s[S] - imp[L]*n_w_s[L] - imp[R]*n_w_s[R])
        
        mask_split = X[:, feat[S]] <= thr[S]
        idxs_l_g, idxs_r_g = mask_split & idxs, ~mask_split & idxs
        
        node_s = LocalNode(idxs=idxs,value=val[S].reshape(1,-1),tree_num=tree_num,feature=feat[S],threshold=thr[S],impurity=imp[S],impurity_reduction=imp_r,N=N_unw)
        node_l = LocalNode(idxs=idxs_l_g,value=val[L].reshape(1,-1),impurity=imp[L],tree_num=tree_num,N=idxs_l_g.sum())
        node_r = LocalNode(idxs=idxs_r_g,value=val[R].reshape(1,-1),impurity=imp[R],tree_num=tree_num,N=idxs_r_g.sum())
        node_s.setattrs(left_temp=node_l, right_temp=node_r)
        return node_s

    def _encode_categories(self, X, cat_feats):
        enc = getattr(self, "_encoder_internal_figs", None)
        X_enc, new_enc = encode_categories(X, cat_feats, enc)
        if new_enc is not None: self._encoder_internal_figs = new_enc
        return X_enc, getattr(self, '_encoder_internal_figs', None)

    @property
    def feature_importances_(self):
        check_is_fitted(self)
        if not hasattr(self, 'importance_data_') or not self.importance_data_ or \
           not any(isinstance(arr,np.ndarray) and arr.size > 0 for arr in self.importance_data_):
            return np.zeros(getattr(self,'n_features_in_',0))
        valid_imp = [a for a in self.importance_data_ if isinstance(a,np.ndarray) and a.ndim==1 and a.size==self.n_features_in_]
        if not valid_imp: return np.zeros(self.n_features_in_)
        avg_imp = np.mean(valid_imp, axis=0); total_sum = np.sum(avg_imp)
        return avg_imp / total_sum if total_sum > 0 else avg_imp
    def __init__(
        self, feature: int = None, threshold: float = None, 
        value=None, value_sklearn=None, idxs=None, is_root: bool = False, left=None,
        impurity: float = None, impurity_reduction: float = None,
        tree_num: int = None, node_id: int = None, right=None, N: int = None,
    ):
        self.is_root = is_root; self.idxs = idxs; self.tree_num = tree_num;
        self.node_id = node_id; self.feature = feature; self.impurity = impurity;
        self.impurity_reduction = impurity_reduction; self.value_sklearn = value_sklearn;
        self.value = value; self.threshold = threshold; self.left = left;
        self.right = right; self.left_temp = None; self.right_temp = None;
        self.N = N # Number of unweighted samples in this node

    def setattrs(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)

    def __str__(self):
        node_type = ""
        if self.is_root: node_type += f"Tree #{self.tree_num} Root"
        
        if self.feature is not None and self.threshold is not None: # Split node
            feat_name = f"X_{self.feature}" 
            return f"{node_type} {feat_name} <= {self.threshold:.3f} (N={self.N if self.N is not None else 'NA'}, ImpRed={self.impurity_reduction:.3f} if self.impurity_reduction is not None else 'NA')"
        elif self.value is not None: # Leaf node
            val_str = f"{self.value.flatten() if isinstance(self.value, np.ndarray) else self.value}"
            return f"{node_type} Leaf Val: {val_str} (N={self.N if self.N is not None else 'NA'}, Imp={self.impurity:.3f} if self.impurity is not None else 'NA')"
        return f"{node_type} Undefined Node"

    def __repr__(self): return self.__str__()

class LocalFIGSBase(BaseEstimator):
    def __init__(self, max_rules: int = 12, max_trees: int = None,
                 min_impurity_decrease: float = 0.0, random_state=None, max_features=None):
        super().__init__()
        self.max_rules = max_rules; self.max_trees = max_trees;
        self.min_impurity_decrease = min_impurity_decrease;
        self.random_state = random_state; self.max_features = max_features;
        self._init_decision_function()

    def _init_decision_function(self):
        if isinstance(self, ClassifierMixin):
            def decision_function(x): 
                probas = self.predict_proba(x)
                return probas[:, 1] if probas.shape[1] >= 2 else probas.ravel() # Handle multi-class for predict_proba output
            self.decision_function = decision_function
        elif isinstance(self, RegressorMixin):
            self.decision_function = self.predict
            
    def _get_logger(self):
        return getattr(self, 'logger', logging.getLogger(f"{self.__class__.__name__}-{id(self)}"))

    def _construct_node_with_stump(self, X, y, idxs, tree_num, sample_weight=None, max_features=None):
        logger = self._get_logger()
        num_samples_in_node_unweighted = idxs.sum()

        sweight_subset = None
        if sample_weight is not None: sweight_subset = sample_weight[idxs]

        default_val_shape = (1, y.shape[1]) if y.ndim == 2 else (1, 1)
        default_value_arr = np.zeros(default_val_shape)

        if num_samples_in_node_unweighted == 0 or \
           (sweight_subset is not None and sweight_subset.size > 0 and np.sum(sweight_subset) < 1e-9):
            if y.ndim > 0 and y[idxs].size > 0:
                 if y.ndim == 2: default_value_arr = np.mean(y[idxs], axis=0, keepdims=True) 
                 else: default_value_arr = np.array([[np.mean(y[idxs])]])
            return LocalNode(idxs=idxs, value=default_value_arr, tree_num=tree_num, N=num_samples_in_node_unweighted)

        stump = DecisionTreeRegressor(max_depth=1, max_features=max_features, random_state=self.random_state)
        try:
            stump.fit(X[idxs], y[idxs], sample_weight=sweight_subset)
        except ValueError as e:
            logger.error(f"_CNS: Stump fit error T{tree_num} N_unw={num_samples_in_node_unweighted} Xsub={X[idxs].shape} ysub={y[idxs].shape if y is not None else 'None'}: {e}")
            return LocalNode(idxs=idxs, value=default_value_arr, tree_num=tree_num, N=num_samples_in_node_unweighted)

        tree_ = stump.tree_
        feature, thresh, imp, val_stump, n_w_samples = tree_.feature, tree_.threshold, tree_.impurity, tree_.value, tree_.weighted_n_node_samples

        if val_stump.ndim == 3 and val_stump.shape[2] == 1: val_stump = val_stump.squeeze(axis=2)
        if val_stump.ndim == 1: val_stump = val_stump.reshape(val_stump.shape[0], -1) 
        if val_stump.ndim == 0: val_stump = np.array([[val_stump]])

        SPLIT, LEFT, RIGHT = 0, 1, 2

        if feature[SPLIT] == -2:
            return LocalNode(idxs=idxs, value=val_stump[SPLIT].reshape(1,-1), tree_num=tree_num, impurity=imp[SPLIT], N=num_samples_in_node_unweighted)

        imp_reduc = (imp[SPLIT] * n_w_samples[SPLIT] - imp[LEFT] * n_w_samples[LEFT] - imp[RIGHT] * n_w_samples[RIGHT])
        
        # Child idxs relative to original X (full dataset)
        mask_feature_split_on_full_x = X[:, feature[SPLIT]] <= thresh[SPLIT]
        idxs_left_global = mask_feature_split_on_full_x & idxs
        idxs_right_global = ~mask_feature_split_on_full_x & idxs
        
        node_s = LocalNode(idxs=idxs, value=val_stump[SPLIT].reshape(1,-1), tree_num=tree_num, feature=feature[SPLIT],
                           threshold=thresh[SPLIT], impurity=imp[SPLIT], impurity_reduction=imp_reduc,
                           N=num_samples_in_node_unweighted)
        node_l = LocalNode(idxs=idxs_left_global, value=val_stump[LEFT].reshape(1,-1), impurity=imp[LEFT], tree_num=tree_num, N=idxs_left_global.sum())
        node_r = LocalNode(idxs=idxs_right_global, value=val_stump[RIGHT].reshape(1,-1), impurity=imp[RIGHT], tree_num=tree_num, N=idxs_right_global.sum())
        node_s.setattrs(left_temp=node_l, right_temp=node_r)
        return node_s

    def _encode_categories(self, X, categorical_features):
        encoder = getattr(self, "_encoder_internal_figs", None)
        X_encoded, new_encoder = encode_categories(X, categorical_features, encoder)
        if new_encoder is not None: self._encoder_internal_figs = new_encoder
        return X_encoded, getattr(self, '_encoder_internal_figs', None)

    @property
    def feature_importances_(self):
        check_is_fitted(self)
        if not hasattr(self, 'importance_data_') or not self.importance_data_ or \
           not any(isinstance(arr, np.ndarray) and arr.size > 0 for arr in self.importance_data_):
             return np.zeros(self.n_features_in_ if hasattr(self, 'n_features_in_') else 0)
        
        valid_imp_data = [arr for arr in self.importance_data_ if isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.size == self.n_features_in_]
        if not valid_imp_data: return np.zeros(self.n_features_in_)

        avg_imp = np.mean(valid_imp_data, axis=0, dtype=np.float64)
        sum_imp = np.sum(avg_imp)
        return avg_imp / sum_imp if sum_imp > 0 else avg_imp
    
    def _get_tree_complexity(self, node: LocalNode) -> int:
        if node is None: return 0
        if node.left is None and node.right is None: return 1
        return self._get_tree_complexity(node.left) + self._get_tree_complexity(node.right)

    def __str__(self):
        if not hasattr(self, "trees_") or not self.trees_:
            return f"{self.__class__.__name__}(max_rules={self.max_rules})"
        else:
            return f"{self.__class__.__name__} with {len(self.trees_)} trees and total rules/complexity {self.complexity_}"


class LocalFIGSClassifier(LocalFIGSBase, ClassifierMixin):
    pass


# --- Patched FIGS CLASSIFIER ---
class PatchedFIGSClassifier(LocalFIGSClassifier): 
    def __init__(self, max_rules: int = 12, max_trees: int = None,
                 min_impurity_decrease: float = 0.0, random_state=None,
                 max_features=None,
                 n_outputs_global: int = None):
        super().__init__(max_rules=max_rules, max_trees=max_trees,
                         min_impurity_decrease=min_impurity_decrease,
                         random_state=random_state,
                         max_features=max_features)
        self.logger = logging.getLogger(f"PatchedFIGS-{id(self)}")
        self._n_outputs_global_fixed = n_outputs_global # Store it

    def _predict_tree(self, root: LocalNode, X_data: np.ndarray) -> np.ndarray:
        logger = self.logger
        if not hasattr(self, 'n_outputs_') or self.n_outputs_ is None or self.n_outputs_ == 0:
             logger.error("Patched._predict_tree: n_outputs_ not properly set."); return np.zeros((X_data.shape[0], 1)) 
        preds = np.zeros((X_data.shape[0], self.n_outputs_))
        for i in range(X_data.shape[0]):
            node, sample = root, X_data[i,:]
            while node.left: 
                if node.feature is None or not (0 <= node.feature < X_data.shape[1]):
                    logger.error(f"Invalid feature {node.feature} in tree predict."); 
                    val = node.value if hasattr(node,'value') and isinstance(node.value,np.ndarray) and node.value.size==self.n_outputs_ else np.zeros((1,self.n_outputs_))
                    preds[i,:] = val.reshape(self.n_outputs_); break 
                if sample[node.feature] <= node.threshold: node = node.left
                else: node = node.right
            else: 
                if hasattr(node,'value') and isinstance(node.value,np.ndarray) and node.value.size==self.n_outputs_:
                    preds[i,:]=node.value.reshape(self.n_outputs_)
                else: logger.warning(f"Leaf value issue: {node.value}, expected size {self.n_outputs_}"); preds[i,:]=np.zeros(self.n_outputs_)
        return preds

    def fit(self, X, y=None, feature_names=None, verbose=False, sample_weight=None, categorical_features=None, _y_fit_override=None):
        logger = getattr(self, 'logger', logging.getLogger(f"PatchedFIGS-{id(self)}"))
        original_y_for_class_determination = y.copy() if y is not None else None

        if categorical_features is not None:
            X, self._encoder_internal_figs = self._encode_categories(X, categorical_features)
        
        # check_fit_arguments will still set self.classes_ based on original_y_for_class_determination
        X_processed, y_processed_1d_for_class_setup, current_feature_names = check_fit_arguments(
            self, X, original_y_for_class_determination, feature_names
        )
        self.feature_names_ = current_feature_names 
        self.n_features_in_ = X_processed.shape[1]
        
        # Set self.classes_ based on the actual y provided to this fit call 
        if original_y_for_class_determination is not None:
            from sklearn.utils.multiclass import unique_labels
            self.classes_ = unique_labels(original_y_for_class_determination)
        else:
            # If y is None, but we have a global n_outputs, create dummy classes_
            if self._n_outputs_global_fixed is not None and self._n_outputs_global_fixed > 0:
                self.classes_ = np.arange(self._n_outputs_global_fixed)
            elif isinstance(self, ClassifierMixin): # Should not happen if y is always passed for classifier
                 raise ValueError("Cannot get classes_ as y is None and n_outputs_global not set.")
            else:
                self.classes_ = np.array([])

        if self._n_outputs_global_fixed is not None and self._n_outputs_global_fixed > 0:
            self.n_outputs_ = self._n_outputs_global_fixed # Use the globally consistent number of outputs
            logger.info(f"PatchedFIGS.fit: self.n_outputs_ OVERRIDDEN to global value: {self.n_outputs_}")
        elif not hasattr(self, 'n_outputs_') or self.n_outputs_ is None or self.n_outputs_ == 0 : # Fallback to inferring if global not set
            if isinstance(self, ClassifierMixin): 
                self.n_outputs_ = len(self.classes_) if self.classes_ is not None and self.classes_.size > 0 else 0
                target_for_shape = _y_fit_override if _y_fit_override is not None else y_processed_1d_for_class_setup
                self.n_outputs_ = 1 if target_for_shape.ndim == 1 else target_for_shape.shape[1]
            logger.warning(f"PatchedFIGS.fit: self.n_outputs_ was inferred to {self.n_outputs_} (global n_outputs not provided).")
        
        if isinstance(self, ClassifierMixin) and (not hasattr(self, 'n_outputs_') or self.n_outputs_ == 0):
            raise ValueError(f"PatchedFIGS.fit: Classifier has n_outputs_ = {getattr(self, 'n_outputs_', 'NotSet/Zero')}. Cannot proceed.")

        # --- Determine self.Y_ (the actual target for fitting the trees) ---
        if _y_fit_override is not None:
            #logger.info(f"PatchedFIGS.fit: Using _y_fit_override (shape: {_y_fit_override.shape}) as self.Y_.")
            if _y_fit_override.shape[0] != X_processed.shape[0]:
                raise ValueError(f"Shape mismatch: _y_fit_override samples ({_y_fit_override.shape[0]}) != X_processed samples ({X_processed.shape[0]})")
            
            # Now the check uses the (potentially overridden) self.n_outputs_
            if _y_fit_override.ndim == 1 and self.n_outputs_ > 1 : # Should not happen if residuals are (N, num_global_classes)
                 if self.n_outputs_ == 1:
                     self.Y_ = _y_fit_override.reshape(-1, 1)
                 else: # Problem
                    raise ValueError(f"Shape mismatch: _y_fit_override is 1D but n_outputs_ is {self.n_outputs_}.")

            elif _y_fit_override.ndim == 2 and _y_fit_override.shape[1] != self.n_outputs_:
                 raise ValueError(f"Shape mismatch: _y_fit_override features ({_y_fit_override.shape[1]}) != n_outputs_ ({self.n_outputs_})")
            else:
                self.Y_ = _y_fit_override

        elif isinstance(self, ClassifierMixin) and self.n_outputs_ > 1: # Multi-class, no override, use one-hot of y
            self.Y_ = np.zeros((X_processed.shape[0], self.n_outputs_))
            
            unique_labels_in_y = np.unique(y_processed_1d_for_class_setup)
            class_to_global_idx_map = {label_val: int(label_val) for label_val in unique_labels_in_y if 0 <= int(label_val) < self.n_outputs_}

            for original_idx, val_in_y in enumerate(y_processed_1d_for_class_setup):
                if val_in_y in class_to_global_idx_map:
                    global_idx = class_to_global_idx_map[val_in_y]
                    self.Y_[original_idx, global_idx] = 1.0            
        elif isinstance(self, ClassifierMixin) and self.n_outputs_ == 1: # Binary, no override
             self.Y_ = y_processed_1d_for_class_setup.reshape(-1,1)
        elif isinstance(self, RegressorMixin): # Regressor, no override
             self.Y_ = y_processed_1d_for_class_setup.reshape(-1,1) if y_processed_1d_for_class_setup.ndim == 1 else y_processed_1d_for_class_setup
        else: 
            logger.error(f"Could not determine self.Y_. n_outputs_: {self.n_outputs_}, type: {type(self)}")
            self.Y_ = y_processed_1d_for_class_setup

        # Final check on self.Y_ shape against the now fixed self.n_outputs_
        if self.Y_.ndim == 1 and self.n_outputs_ == 1:
            self.Y_ = self.Y_.reshape(-1,1)
        elif self.Y_.ndim != 2 or self.Y_.shape[1] != self.n_outputs_ :
            logger.error(f"CRITICAL: self.Y_ final shape {self.Y_.shape} not compatible with self.n_outputs_ ({self.n_outputs_})")
            raise ValueError(f"self.Y_ shape {self.Y_.shape} incompatible with n_outputs_ {self.n_outputs_}")

        if sample_weight is not None: self.sample_weight_ = _check_sample_weight(sample_weight, X_processed)
        else: self.sample_weight_ = None

        # --- Fitting loop from LocalFIGSBase/imodels.FIGS ---
        # These are the variables used in the loop, derived from self attributes
        X_to_fit = X_processed 
        y_target_for_loop = self.Y_ # This is THE target for the entire fitting process (one-hot labels or residuals)
        sample_weight_for_loop = self.sample_weight_

        self.trees_ = []
        self.complexity_ = 0 
        y_predictions_per_tree_in_ensemble = {} 
        y_residuals_for_stump_fitting = {} 
        idxs_all_samples_mask = np.ones(X_to_fit.shape[0], dtype=bool)
        
        # Initial stump is fit on y_target_for_loop
        node_init = self._construct_node_with_stump(
            X_to_fit, y_target_for_loop, idxs_all_samples_mask, 
            tree_num=-1, # For a potential new tree
            sample_weight=sample_weight_for_loop, 
            max_features=self.max_features
        )
        
        potential_splits = []
        if node_init and hasattr(node_init, 'impurity_reduction') and node_init.impurity_reduction is not None:
            node_init.setattrs(is_root=True)
            potential_splits.append(node_init)
        
        potential_splits = sorted(potential_splits, key=lambda x: x.impurity_reduction if hasattr(x, 'impurity_reduction') and x.impurity_reduction is not None else -np.inf, reverse=True)

        finished = False
        while len(potential_splits) > 0 and not finished:
            split_node = potential_splits.pop(0)

            # Check impurity reduction of the chosen split
            if not hasattr(split_node, 'impurity_reduction') or \
               split_node.impurity_reduction is None or \
               split_node.impurity_reduction < (self.min_impurity_decrease - 1e-9):
                # If best split is not good enough, check if any other potential splits are
                if not any(p.impurity_reduction is not None and p.impurity_reduction >= (self.min_impurity_decrease - 1e-9) for p in potential_splits):
                    finished = True; break # No good splits left
                else: continue # Skip this bad split, try next from sorted list

            if split_node.is_root and self.max_trees is not None and len(self.trees_) >= self.max_trees:
                # Already have max_trees, and this split would start a new one. Skip.
                # (But allow splits that grow existing trees)
                # Remove other potential root nodes to prevent new trees if max_trees is met
                if len(self.trees_) >= self.max_trees:
                    potential_splits = [ps for ps in potential_splits if not ps.is_root or ps.tree_num != -1]
                continue

            if verbose: logger.info(f"Adding split: {split_node}")
            self.complexity_ += 1

            if split_node.is_root:
                self.trees_.append(split_node)
                current_tree_idx = len(self.trees_) - 1
                split_node.tree_num = current_tree_idx
                if split_node.left_temp: split_node.left_temp.tree_num = current_tree_idx
                if split_node.right_temp: split_node.right_temp.tree_num = current_tree_idx
                
                if self.max_trees is None or len(self.trees_) < self.max_trees:
                    # N is set for consistency, though this node is temporary until its stump is constructed.
                    # It will be re-evaluated with residuals.
                    potential_splits.append(LocalNode(is_root=True, idxs=idxs_all_samples_mask, tree_num=-1, N=idxs_all_samples_mask.sum()))

            split_node.setattrs(left=split_node.left_temp, right=split_node.right_temp)
            if split_node.left: potential_splits.append(split_node.left)
            if split_node.right: potential_splits.append(split_node.right)

            # Update predictions P_t = h_t(X) for all existing trees
            for i_tree in range(len(self.trees_)):
                y_predictions_per_tree_in_ensemble[i_tree] = self._predict_tree(self.trees_[i_tree], X_to_fit) 
            
            y_predictions_per_tree_in_ensemble[-1] = np.zeros((X_to_fit.shape[0], self.n_outputs_)) # For potential new tree

            # Update residuals r_t = Y_target_overall - sum_{j != t} P_j for fitting/evaluating stump t
            for target_tree_idx_for_residual in list(range(len(self.trees_))) + [-1]: 
                y_residuals_for_stump_fitting[target_tree_idx_for_residual] = y_target_for_loop.copy() # Start from overall target
                for other_tree_idx_effect in range(len(self.trees_)):
                    if other_tree_idx_effect != target_tree_idx_for_residual: 
                        y_residuals_for_stump_fitting[target_tree_idx_for_residual] -= y_predictions_per_tree_in_ensemble[other_tree_idx_effect]
            
            new_potential_splits_list = []
            for ps_node_to_re_evaluate in potential_splits:
                # Get the correct residual target for this node's (potential) tree
                residual_target_for_this_stump_fit = y_residuals_for_stump_fitting.get(ps_node_to_re_evaluate.tree_num)
                if residual_target_for_this_stump_fit is None: 
                     logger.warning(f"Residual target not found for node {ps_node_to_re_evaluate} in tree {ps_node_to_re_evaluate.tree_num}. Skipping re-eval.")
                     continue
                
                re_evaluated_stump_node = self._construct_node_with_stump(
                    X_to_fit, residual_target_for_this_stump_fit, ps_node_to_re_evaluate.idxs, 
                    ps_node_to_re_evaluate.tree_num, sample_weight_for_loop, self.max_features
                )
                
                if re_evaluated_stump_node and hasattr(re_evaluated_stump_node, 'impurity_reduction') and re_evaluated_stump_node.impurity_reduction is not None:
                    ps_node_to_re_evaluate.setattrs(
                        feature=re_evaluated_stump_node.feature, threshold=re_evaluated_stump_node.threshold,
                        impurity=re_evaluated_stump_node.impurity, impurity_reduction=re_evaluated_stump_node.impurity_reduction,
                        left_temp=re_evaluated_stump_node.left_temp, right_temp=re_evaluated_stump_node.right_temp,
                        value=re_evaluated_stump_node.value, N=re_evaluated_stump_node.N 
                    )
                    new_potential_splits_list.append(ps_node_to_re_evaluate)
            
            potential_splits = sorted(new_potential_splits_list, key=lambda x:x.impurity_reduction if hasattr(x, 'impurity_reduction') and x.impurity_reduction is not None else -np.inf, reverse=True)

            if verbose: logger.info(str(self)) # Print current model state
            if self.max_rules is not None and self.complexity_ >= self.max_rules:
                finished = True; break
        
        # --- Populate self.importance_data_ ---
        self.importance_data_ = []
        for tree_final_root_node in self.trees_:
            node_id_iter = iter(range(int(1e7))) 
            def _annotate(node, X_sub, y_sub_target): 
                if node is None: return
                node.N = X_sub.shape[0] 
                if isinstance(self, ClassifierMixin) and _y_fit_override is None: # Standard classification
                    node.value_sklearn = np.sum(y_sub_target, axis=0) if y_sub_target.shape[0] > 0 else np.zeros(self.n_outputs_)
                else: # Regression or GBM residual fitting
                    node.value_sklearn = np.mean(y_sub_target, axis=0) if y_sub_target.shape[0] > 0 else np.zeros(self.n_outputs_)
                node.node_id = next(node_id_iter)
                if node.left:
                    if node.feature is None or not (0 <= node.feature < X_sub.shape[1]): return
                    mask = X_sub[:, node.feature] <= node.threshold
                    _annotate(node.left, X_sub[mask], y_sub_target[mask])
                    _annotate(node.right, X_sub[~mask], y_sub_target[~mask])
            
            if hasattr(tree_final_root_node, 'feature') and tree_final_root_node.feature is not None: 
                 _annotate(tree_final_root_node, X_to_fit, y_target_for_loop) 

            tree_imp_calculated = np.zeros(self.n_features_in_)
            def _calc_imp_recursive(node): 
                if node is None or node.left is None or not hasattr(node, 'feature') or node.feature is None or node.feature < 0 or \
                   not all(hasattr(n, attr) and getattr(n,attr) is not None for n in [node, node.left, node.right] for attr in ['N', 'impurity']):
                    return
                if node.N > 0 and hasattr(node.left, 'N') and node.left.N > 0 and hasattr(node.right, 'N') and node.right.N > 0:
                    decrease = (node.N * node.impurity - 
                                node.left.N * node.left.impurity - 
                                node.right.N * node.right.impurity)
                    if node.feature < self.n_features_in_: tree_imp_calculated[node.feature] += max(0, decrease)
                _calc_imp_recursive(node.left); _calc_imp_recursive(node.right)
            
            if hasattr(tree_final_root_node, 'feature') and tree_final_root_node.feature is not None: 
                _calc_imp_recursive(tree_final_root_node)
            self.importance_data_.append(tree_imp_calculated)
            
        self._is_fitted = True
        return self

    def predict(self, X, categorical_features=None): 
        logger = self.logger
        if hasattr(self, "_encoder_internal_figs") and categorical_features is not None:
            X, _ = self._encode_categories(X, categorical_features=categorical_features)
        
        # Ensure X is a NumPy array for consistent processing
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        elif isinstance(X, np.ndarray):
            X_np = X
        else:
            X_np = check_array(X, ensure_2d=True, accept_sparse=False) # Fallback

        if X_np.shape[0] == 0: # Handle empty input
            return np.array([], dtype=self.classes_.dtype if hasattr(self, 'classes_') and self.classes_.size > 0 else int)

        sum_preds = np.zeros((X_np.shape[0], self.n_outputs_)) 
        if not self.trees_:
            logger.warning("Model has no trees; predict method returning default (first class or zeros).")
            if isinstance(self, ClassifierMixin) and self.n_outputs_ > 0 :
                 return self.classes_[np.zeros(X_np.shape[0], dtype=int)] if hasattr(self, 'classes_') and self.classes_.size > 0 else np.zeros(X_np.shape[0], dtype=int)
            # For regressor, it would return zeros based on sum_preds initialization
            return sum_preds.squeeze() if self.n_outputs_ == 1 else sum_preds

        for tree_root in self.trees_:
            sum_preds += self._predict_tree(tree_root, X_np) # _predict_tree expects NumPy array
        
        if isinstance(self, RegressorMixin):
            return sum_preds.squeeze(axis=1) if self.n_outputs_ == 1 else sum_preds
        elif isinstance(self, ClassifierMixin):
            if not hasattr(self, 'classes_') or self.classes_ is None or self.classes_.size == 0:
                logger.error("Classifier has no classes_ defined. Cannot map predictions.")
                # Fallback to raw class indices if classes_ is not available
                if self.n_outputs_ == 1:
                    return (expit(sum_preds.squeeze(axis=1)) > 0.5).astype(int)
                else:
                    probabilities = softmax(sum_preds, axis=1)
                    return np.argmax(probabilities, axis=1)

            # Proceed with mapping to actual class labels
            if self.n_outputs_ == 1: # Binary classification
                 # sum_preds are logits for class 1. Convert to probability then threshold.
                 probabilities_class1 = expit(sum_preds.squeeze(axis=1)) 
                 predicted_indices = (probabilities_class1 > 0.5).astype(int)
                 # Ensure self.classes_ has at least two elements for binary if mapping
                 if self.classes_.size >= 2:
                    return self.classes_[predicted_indices]
                 else: # Should not happen if classes_ was set correctly from 2 unique y values
                    logger.warning("Binary classifier has < 2 classes in self.classes_. Returning raw 0/1 predictions.")
                    return predicted_indices
            else: # Multi-class
                # Convert summed scores/logits to probabilities using softmax
                probabilities = softmax(sum_preds, axis=1)
                # Get class indices from probabilities
                predicted_indices = np.argmax(probabilities, axis=1)
                return self.classes_[predicted_indices]
    def predict_proba(self, X, categorical_features=None): # Override for multi-class from LocalFIGSClassifier
        logger = self.logger
        if hasattr(self, "_encoder_internal_figs") and categorical_features is not None:
            X, _ = self._encode_categories(X, categorical_features=categorical_features)
        X = check_array(X, ensure_2d=True, accept_sparse=False)
        if isinstance(self, RegressorMixin): raise AttributeError("predict_proba for Regressor.")

        sum_preds = np.zeros((X.shape[0], self.n_outputs_))
        if not self.trees_:
            logger.warning("Model has no trees; predict_proba returning uniform probabilities.")
            return np.ones((X.shape[0], self.n_outputs_)) / self.n_outputs_ if self.n_outputs_ > 0 else np.array([[]])


        for tree_root in self.trees_:
            sum_preds += self._predict_tree(tree_root, X)

        if self.n_outputs_ == 1: # Binary
            probs_class1 = expit(sum_preds.squeeze(axis=1)) # Sigmoid
            return np.vstack((1 - probs_class1, probs_class1)).T
        else: # Multi-class
            return softmax(sum_preds, axis=1)

class FederatedClient:
    def __init__(self, client_id, config, data_partition, scene_to_idx, 
                 skip_model_initialization=False): # New parameter
        self.client_id = client_id
        self.config = config.copy() 
        self.data_partition = data_partition # Used by original run_local_pipeline
        self.scene2idx = scene_to_idx     # Used by original run_local_pipeline
        self.logger = logging.getLogger(f"Client-{self.client_id}")
        self.device = self.config.get('device', 'cpu')
        
        self.torch_device = torch.device(self.device)
        self.sam_model, self.mask_generator = None, None
        self.dino_processor, self.dino_model = None, None
        self.target_model = None # For 'combined' embeddings

        if not skip_model_initialization: # Conditional initialization
            self._initialize_models() 
        else:
            self.logger.info(f"Client {self.client_id}: Skipping SAM/DINO model initialization as requested.")


        self.state = {
            'linear_models': {}, 'optimal_thresholds': {}, 'detector_scores': {},
            'final_concept_indices_ordered': None, 'final_original_to_dense_map': None,
            'concept_vecs': None, 'feature_names_for_figs': None, 'kept_image_ids_for_model': [],
            'figs_model_trained_instance': None, 'figs_extracted_terms': [],
            'filtered_segment_infos': None, # Not used in ResNet concepts run
            'final_embeddings': None,      # Not used in ResNet concepts run (for K-Means)
            'cluster_labels': None,        # Not used in ResNet concepts run
            'filtered_images': None,       # Not used in ResNet concepts run
            'filtered_masks': None,        # Not used in ResNet concepts run
            "local_model_mse": 0.0,
            'feature_names_fitted_': None, 
            'sample_weights': None,
            'global_figs_terms_from_server': [],
            'canonical_detector_params_by_original_idx': {}, # Store received canonical params
            'accumulated_global_model_Fm_terms': [], 
            'learning_rate_gbm': self.config.get('learning_rate_gbm', 0.1),
        }
        self.logger.info(f"Client {self.client_id} initialized. Device: {self.device}")

    def _initialize_models(self):
        self.logger.info(f"Client {self.client_id}: Initializing segmentation/embedding models...")
        # Check if necessary config keys exist before trying to load
        sam_cfg = self.config.get('sam_cfg')
        sam_ckpt = self.config.get('sam_ckpt')
        dino_model_name_cfg = self.config.get('dino_model') 
        if sam_cfg and sam_ckpt:
            try:
                self.sam_model, self.mask_generator = load_sam_model(sam_cfg, sam_ckpt, self.torch_device)
                self.logger.info(f"Client {self.client_id}: SAM model initialized.")
            except Exception as e:
                self.logger.error(f"Client {self.client_id}: FAILED to initialize SAM model: {e}. Features requiring SAM will fail.")
        else:
            self.logger.warning(f"Client {self.client_id}: 'sam_cfg' or 'sam_ckpt' not in config. SAM model not initialized.")

        if dino_model_name_cfg: 
            try:
                self.dino_processor, self.dino_model = init_dino(dino_model_name_cfg, self.torch_device)
                self.logger.info(f"Client {self.client_id}: DINO model initialized.")
            except Exception as e:
                self.logger.error(f"Client {self.client_id}: FAILED to initialize DINO model: {e}. Features requiring DINO will fail.")
        else:
            self.logger.warning(f"Client {self.client_id}: 'dino_model' (or 'dino_model_name') not in config. DINO model not initialized.")
        
        if self.config.get('embedding_type') == 'combined':
            if self.dino_model is None: # DINO is needed for combined
                 self.logger.warning(f"Client {self.client_id}: DINO model not initialized, cannot proceed with 'combined' embeddings if target_model also needs it.")
            try:
                self.target_model = init_target_model(self.torch_device) # init_target_model is ResNet50
                self.logger.info(f"Client {self.client_id}: Target model (ResNet50) for 'combined' embeddings initialized.")
            except Exception as e:
                 self.logger.error(f"Client {self.client_id}: FAILED to initialize target_model (ResNet50): {e}")
        else:
            self.target_model = None
            self.logger.info(f"Client {self.client_id}: Target model not initialized (embedding_type: {self.config.get('embedding_type')}).")

    def _load_cached_data(self, component_name, expected_type=tuple, expected_len=None):
        cache_dir_key = f"{component_name}_cache_dir"
        cache_dir = self.config.get(cache_dir_key)
        if not cache_dir:
            self.logger.debug(f"No cache dir key '{cache_dir_key}' for {component_name}. Cache disabled.") 
            raise FileNotFoundError("Cache directory not configured.")

        fname_suffix = ""
        if component_name == "embedding":
            fname_suffix = f"_{self.config.get('embedding_type', 'combined')}"
        
        client_id_str = str(self.client_id)
        cache_file = os.path.join(cache_dir, f"{component_name}{fname_suffix}_{client_id_str}.pkl")
        
        if not self.config.get(f"use_{component_name}_cache", True):
            self.logger.info(f"Caching disabled for {component_name} for client {client_id_str}.")
            raise FileNotFoundError("Caching disabled by config.")

        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"{component_name.capitalize()} cache not found for client {client_id_str} at {cache_file}")

        self.logger.info(f"Loading {component_name} from cache: {cache_file}")
        with open(cache_file, "rb") as f: data = pickle.load(f)
        
        if not isinstance(data, expected_type):
            raise ValueError(f"Cache type error for {component_name}. Expected {expected_type}, got {type(data)}")
        if expected_len is not None and isinstance(data, tuple) and len(data) != expected_len:
             raise ValueError(f"Cache length error for {component_name}. Expected {expected_len}, got {len(data)}")
        return data

    def _save_cache(self, data, component_name):
        if not self.config.get(f"use_{component_name}_cache", True): return
        cache_dir_key = f"{component_name}_cache_dir"
        cache_dir = self.config.get(cache_dir_key)
        if not cache_dir: return
        
        os.makedirs(cache_dir, exist_ok=True)
        fname_suffix = ""
        if component_name == "embedding":
            fname_suffix = f"_{self.config.get('embedding_type', 'combined')}"
        client_id_str = str(self.client_id)
        cache_file = os.path.join(cache_dir, f"{component_name}{fname_suffix}_{client_id_str}.pkl")
        
        try:
            with open(cache_file, "wb") as f: pickle.dump(data, f)
            self.logger.info(f"Saved {component_name} cache for client {client_id_str} to {cache_file}")
        except Exception as e:
            self.logger.error(f"Failed to save {component_name} cache for client {client_id_str}: {e}")

    def run_local_pipeline(self, global_centroids=None, current_config=None): # For K-Means stats
        effective_config = current_config if current_config is not None else self.config
        self.logger.info(f"Client {self.client_id}: Preparing for K-Means statistics computation (Embed type: {effective_config.get('embedding_type')}).")


        if self.state.get('final_embeddings') is not None and \
           self.state.get('filtered_segment_infos') is not None and \
           isinstance(self.state['final_embeddings'], np.ndarray) and \
           self.state['final_embeddings'].size > 0 and \
           isinstance(self.state['filtered_segment_infos'], (list, np.ndarray)) and \
           len(self.state['filtered_segment_infos']) > 0 and \
           self.state['final_embeddings'].shape[0] == len(self.state['filtered_segment_infos']):
            self.logger.info(f"Client {self.client_id}: Using pre-loaded segment infos and embeddings for K-Means.")
            # Data is already in self.state, no on-the-fly generation needed here.
        else:
            # This block is a fallback if data wasn't pre-loaded.
            # For the cached data workflow, this block should ideally NOT be hit.
            self.logger.warning(f"Client {self.client_id}: Pre-loaded data not found or invalid in state. "
                                f"Attempting on-the-fly segmentation/embedding. This is NOT the expected path for V2 cached data workflow.")            
            try:
                s_infos_orig, all_img_orig, all_msk_orig, all_seg_orig = self._load_cached_data('segment', tuple, 4)
                self.logger.info(f"Client {self.client_id}: (Fallback) Loaded segments from client's own segment cache.")
            except FileNotFoundError:
                self.logger.info(f"Client {self.client_id}: (Fallback) Generating segments as not pre-loaded and not in client cache.")
                if not self.mask_generator: # Critical check
                    self.logger.error(f"Client {self.client_id}: (Fallback) SAM Mask Generator not initialized. Cannot generate segments.")
                    return None, None
                s_infos_orig, all_img_orig, all_msk_orig, all_seg_orig = generate_segments_and_masks(
                    self.data_partition, self.mask_generator, effective_config, client_id=self.client_id)
                if s_infos_orig is not None: self._save_cache((s_infos_orig, all_img_orig, all_msk_orig, all_seg_orig), 'segment')
            
            if not s_infos_orig or (isinstance(s_infos_orig, np.ndarray) and s_infos_orig.size == 0): 
                self.logger.error(f"Client {self.client_id}: (Fallback) No segment infos. Cannot proceed."); return None, None
            
            self.state['filtered_images'], self.state['filtered_masks'], _, self.state['filtered_segment_infos'] = \
                filter_zero_segment_images(all_img_orig, all_msk_orig, all_seg_orig, s_infos_orig)

            if self.state['filtered_segment_infos'] is None or \
               (isinstance(self.state['filtered_segment_infos'], np.ndarray) and self.state['filtered_segment_infos'].size == 0) or \
               (isinstance(self.state['filtered_segment_infos'], list) and not self.state['filtered_segment_infos']):
                self.logger.error(f"Client {self.client_id}: (Fallback) No segments after filter. Cannot proceed."); return None, None
            
            try:
                self.state['final_embeddings'] = self._load_cached_data('embedding', np.ndarray) # Checks client's own embedding cache
                self.logger.info(f"Client {self.client_id}: (Fallback) Loaded embeddings from client's own embedding cache.")
            except FileNotFoundError:
                self.logger.info(f"Client {self.client_id}: (Fallback) Computing embeddings as not pre-loaded and not in client cache.")
                if not self.dino_processor or not self.dino_model: # Critical check
                    self.logger.error(f"Client {self.client_id}: (Fallback) DINO models not initialized. Cannot compute embeddings.")
                    return None, None
                self.state['final_embeddings'] = compute_final_embeddings(
                    self.state['filtered_segment_infos'], self.state['filtered_images'], self.state['filtered_masks'],
                    self.dino_processor, self.dino_model, self.target_model,
                    self.torch_device, effective_config, client_id=self.client_id)
                if self.state['final_embeddings'] is not None and self.state['final_embeddings'].size > 0:
                    self._save_cache(self.state['final_embeddings'], 'embedding')
                else: 
                    self.logger.error(f"Client {self.client_id}: (Fallback) Embedding computation returned None or empty."); return None, None

        # Ensure final_embeddings are ready for K-Means (either pre-loaded or generated by fallback)
        if self.state.get('final_embeddings') is None or not isinstance(self.state['final_embeddings'], np.ndarray) or self.state['final_embeddings'].size == 0:
            self.logger.error(f"Client {self.client_id}: Final embeddings are None or empty after all checks. Cannot run KMeans.")
            return None, None
        
        current_segment_infos = self.state.get('filtered_segment_infos')
        if current_segment_infos is None or \
           (not isinstance(current_segment_infos, (list, np.ndarray))) or \
           (isinstance(current_segment_infos, np.ndarray) and current_segment_infos.size == 0) or \
           (isinstance(current_segment_infos, list) and not current_segment_infos) or \
           len(current_segment_infos) != self.state['final_embeddings'].shape[0]:
            self.logger.error(f"Client {self.client_id}: Mismatch or invalid segment_infos vs final_embeddings before KMeans. "
                              f"Embeddings shape: {self.state['final_embeddings'].shape}, "
                              f"SegInfos length: {len(current_segment_infos) if current_segment_infos is not None else 'None'}.")
            return None, None

        # Proceed with K-Means statistics computation
        num_k = global_centroids.shape[0] if global_centroids is not None else effective_config['num_clusters']
        kmeans_algo = FederatedKMeans(num_clusters=num_k, client_id=self.client_id) 
        if global_centroids is not None: 
            kmeans_algo.set_centroids(global_centroids)
        
        try:
            sums, counts, labels = kmeans_algo.compute_local_stats(self.state['final_embeddings'])
            self.state['cluster_labels'] = labels # Store cluster labels for detector training
            return sums, counts
        except Exception as e: 
            self.logger.exception(f"Client {self.client_id}: KMeans stats computation error: {e}")
            return None, None
    def receive_final_concepts_and_detectors(self, final_concept_indices_ordered: list,
                                             final_original_to_dense_map: dict,
                                             canonical_detector_params_by_original_idx: dict):
        """ Stores final concept info AND the parameters for the canonical detectors. """
        self.state['final_concept_indices_ordered'] = final_concept_indices_ordered
        self.state['final_original_to_dense_map'] = final_original_to_dense_map
        num_final_concepts = len(final_concept_indices_ordered)
        self.state['feature_names_for_figs'] = [f"concept_{i}" for i in range(num_final_concepts)]
        self.state['canonical_detector_params_by_original_idx'] = canonical_detector_params_by_original_idx
        self.logger.info(f"Client {self.client_id}: Received final {num_final_concepts} concepts and "
                         f"{len(canonical_detector_params_by_original_idx)} canonical detector params.")


    def _get_scores_using_accumulated_global_model(self, concept_vecs_df: pd.DataFrame) -> np.ndarray:
        """ 
        Calculates raw summed scores from the client's current accumulated global model F_m.
        (This was previously _get_scores_using_global_terms, but now specifically uses the accumulated model)
        Returns (n_samples, n_classes). 
        """
        logger = self.logger
        # Use the client's accumulated model terms
        accumulated_terms = self.state.get('accumulated_global_model_Fm_terms', [])
        
        if concept_vecs_df is None or concept_vecs_df.empty:
            logger.debug(f"Client {self.client_id}: _get_scores_accumulated: concept_vecs_df empty.")
            num_classes_fallback = self.config.get('num_classes', 1)
            return np.empty((0, num_classes_fallback))
        
        num_classes = self.config.get('num_classes', 1)
        if not accumulated_terms: # F_0 case or if accumulation is empty
            logger.debug(f"Client {self.client_id}: _get_scores_accumulated: accumulated_terms empty. Returning zero scores for F_0.")
            return np.zeros((concept_vecs_df.shape[0], num_classes))

        # Sanity check num_classes from terms if available
        if accumulated_terms:
            first_term_val_arr = accumulated_terms[0].get('aggregated_value_array')
            if isinstance(first_term_val_arr, list) and len(first_term_val_arr) > 0:
                if num_classes != len(first_term_val_arr):
                    logger.warning(f"Client {self.client_id}: Mismatch num_classes in config ({num_classes}) vs accumulated term ({len(first_term_val_arr)}). Using config's.")

        summed_scores = np.zeros((concept_vecs_df.shape[0], num_classes))
        cond_regex = re.compile(r"`(.+?)`\s*([><]=?)\s*([0-9.-]+)")

        for i in range(concept_vecs_df.shape[0]):
            sample_series = concept_vecs_df.iloc[i]
            sample_total_value = np.zeros(num_classes)
            for term in accumulated_terms: # Iterate over terms in F_m
                rule_ok = True
                if term['rule_str'] != "True":
                    for cond_str in term['rule_str'].split(' & '):
                        match = cond_regex.match(cond_str.strip())
                        if not match: rule_ok = False; break
                        feat, op, val_s = match.groups()
                        if feat not in sample_series.index: rule_ok = False; break
                        s_val, cond_v = sample_series[feat], float(val_s)
                        op_fn = {'<=':operator.le,'>':operator.gt,'<':operator.lt,'>=':operator.ge,'==':operator.eq}.get(op)
                        if not (op_fn and op_fn(s_val, cond_v)): rule_ok = False; break
                if rule_ok:
                    term_va = term.get('aggregated_value_array', [])
                    if isinstance(term_va, list) and len(term_va) == num_classes:
                        sample_total_value += np.array(term_va)
            summed_scores[i] = sample_total_value
        return summed_scores
    
    def _reconstruct_and_predict_with_canonical_detector(self, detector_params: dict, embeddings_for_concept: np.ndarray) -> np.ndarray:
        logger = self.logger
        # logger.debug(f"Applying canonical detector. Type: {detector_params.get('type')}, PCA output comps: {detector_params.get('n_pca_components_out')}")

        current_embeddings = embeddings_for_concept

        # 1. Manual PCA Transformation (if params exist)
        if detector_params.get('n_pca_components_out', 0) > 0 and \
           detector_params.get('pca_mean') is not None and \
           detector_params.get('pca_components') is not None:
            
            pca_mean = np.array(detector_params['pca_mean'])
            pca_components = np.array(detector_params['pca_components']) # Shape (n_components_out, n_features_in)
            
            if pca_mean.shape[0] == embeddings_for_concept.shape[1] and \
               pca_components.shape[1] == embeddings_for_concept.shape[1]:
                try:
                    X_centered = embeddings_for_concept - pca_mean
                    current_embeddings = np.dot(X_centered, pca_components.T)
                except Exception as e_pca_manual:
                    logger.error(f"Error during manual PCA transform: {e_pca_manual}. Using original embeddings for classifier.")
                    # current_embeddings remains original embeddings_for_concept
            else:
                logger.error(f"PCA param mismatch for manual transform. Mean shape {pca_mean.shape}, Comps shape {pca_components.shape}, Input feats {embeddings_for_concept.shape[1]}. Using original embeddings.")


        # 2. Manual Linear Model Application & Sigmoid
        detector_type_str = detector_params.get('type', 'unknown')
        # Initialize probabilities to 0.5 (uncertain) if anything fails
        probs = np.full(embeddings_for_concept.shape[0], 0.5, dtype=float) 

        if 'lr_manual' in detector_type_str or 'svm_linear_manual' in detector_type_str:
            if 'clf_coef' not in detector_params or 'clf_intercept' not in detector_params:
                logger.error(f"Manual {detector_type_str}: coef/intercept missing."); return probs
            
            coef = np.array(detector_params['clf_coef']) # Expected (1, n_features_after_pca)
            intercept = np.array(detector_params['clf_intercept']) # Expected (1,)

            if coef.ndim == 2 and coef.shape[0] == 1: coef = coef.flatten() # Make it 1D: (n_features_after_pca,)
            if intercept.ndim > 0 and intercept.size == 1: intercept = intercept[0] # Make it scalar
            else: intercept = 0.0 # Fallback if intercept is not scalar

            if coef.shape[0] != current_embeddings.shape[1]:
                logger.error(f"Manual {detector_type_str}: Coef shape {coef.shape} mismatch with transformed embedding features {current_embeddings.shape[1]}.")
                return probs
            
            try:
                decision_values = np.dot(current_embeddings, coef) + intercept
                probs = expit(decision_values) # Sigmoid function
            except Exception as e_manual_pred:
                logger.error(f"Error during manual prediction for {detector_type_str}: {e_manual_pred}")
        else:
            logger.error(f"Unsupported or unknown canonical detector type for manual application: {detector_type_str}")

        return probs
    def train_concept_detectors(self, config_from_main_script):
        self.logger.info(f"Client {self.client_id}: Training local detectors (type: {config_from_main_script.get('detector_type','lr')})...")
        if any(key not in self.state or self.state[key] is None for key in ['final_embeddings', 'cluster_labels', 'filtered_segment_infos']):
             self.logger.error(f"Client {self.client_id}: State not ready for detector training."); return
        if not (isinstance(self.state['final_embeddings'], np.ndarray) and self.state['final_embeddings'].size > 0 and \
                isinstance(self.state['cluster_labels'], np.ndarray) and self.state['cluster_labels'].size > 0):
             self.logger.warning(f"Client {self.client_id}: Embeddings or cluster_labels empty/invalid for detector training."); return
        if len(self.state['filtered_segment_infos']) != self.state['final_embeddings'].shape[0]:
             self.logger.error(f"Client {self.client_id}: SegInfo/Embedding mismatch for detector training."); return

        image_groups = np.array([info["img_idx"] for info in self.state['filtered_segment_infos']])
        results = []
        unique_labels = np.unique(self.state['cluster_labels'])
        self.logger.info(f"Client {self.client_id}: Training detectors for {len(unique_labels)} unique local cluster labels.")

        for cluster_idx_float in unique_labels:
            cluster_idx = int(cluster_idx_float)
            try:
                _, model_info, score = train_concept_detector(
                    cluster_idx, self.state['final_embeddings'], self.state['cluster_labels'], 
                    image_groups, config_from_main_script 
                )
                if model_info: results.append((cluster_idx, model_info, score))
                else: results.append((cluster_idx, None, 0.0)) 
            except TypeError as te: # Catch specific errors if needed
                self.logger.error(f"Client {self.client_id}: TypeError training detector for local cluster_idx {cluster_idx}: {te}")
                results.append((cluster_idx, None, 0.0))
            except Exception as e_train:
                self.logger.error(f"Client {self.client_id}: Exception training detector for local cluster_idx {cluster_idx}: {e_train}")
                results.append((cluster_idx, None, 0.0))
        
        self.state['linear_models'].clear(); self.state['optimal_thresholds'].clear(); self.state['detector_scores'].clear()
        min_score = config_from_main_script.get('min_detector_score', 0.60)
        kept_count = 0
        for c_idx, m_info, scr in results:
            if m_info and scr is not None and scr >= min_score : 
                self.state['linear_models'][c_idx], self.state['optimal_thresholds'][c_idx] = m_info[0], m_info[1]
                self.state['detector_scores'][c_idx] = scr; kept_count += 1
        
        self.state['linear_models'] = dict(sorted(self.state['linear_models'].items()))
        self.state['optimal_thresholds'] = {k: self.state['optimal_thresholds'][k] for k in self.state['linear_models']}
        self.state['detector_scores'] = {k: self.state['detector_scores'][k] for k in self.state['linear_models']}
        self.logger.info(f"Client {self.client_id}: Kept {kept_count} valid local detectors (score >= {min_score:.2f}).")

    def get_detector_update(self, required_original_indices: list) -> dict:
        # This sends client's *locally trained* detector objects/params.
        # The server then picks the canonical ones.
        update = {}
        local_models = self.state.get('linear_models', {})
        local_thresholds = self.state.get('optimal_thresholds', {})
        local_scores = self.state.get('detector_scores', {})
        
        for original_idx in required_original_indices:
            if original_idx in local_models:
                update[original_idx] = {
                    'model': local_models[original_idx], # The Pipeline object
                    'threshold': local_thresholds[original_idx],
                    'score': local_scores.get(original_idx, 0.0),
                    'client_id': self.client_id 
                }
        self.logger.info(f"Client {self.client_id}: Providing {len(update)}/{len(required_original_indices)} local detector updates.")
        return update

    # receive_final_concepts_and_detectors is crucial
    def receive_final_concepts_and_detectors(self, final_concept_indices_ordered: list,
                                             final_original_to_dense_map: dict,
                                             canonical_detector_params_by_original_idx: dict):
        self.state['final_concept_indices_ordered'] = final_concept_indices_ordered
        self.state['final_original_to_dense_map'] = final_original_to_dense_map
        num_final_concepts = len(final_concept_indices_ordered)
        self.state['feature_names_for_figs'] = [f"concept_{i}" for i in range(num_final_concepts)]
        
        # Store the *parameters* of the canonical detectors received from the server
        self.state['canonical_detector_params_by_original_idx'] = canonical_detector_params_by_original_idx
        self.logger.info(f"Client {self.client_id}: Received final {num_final_concepts} concepts and "
                         f"{len(canonical_detector_params_by_original_idx)} canonical detector parameters.")

    def build_concept_vectors(self, config_from_main_script): 
        self.logger.info(f"Client {self.client_id}: Building concept vectors using CANONICAL detector parameters.")
        final_concept_indices_ordered = self.state.get('final_concept_indices_ordered') # Original K-Means IDs
        final_original_to_dense_map = self.state.get('final_original_to_dense_map')
        # Use the CANONICAL detector parameters received from the server
        canonical_detector_params_by_original_idx = self.state.get('canonical_detector_params_by_original_idx', {})
        
        final_embeddings_all_segments = self.state.get('final_embeddings')

        if not all([final_concept_indices_ordered is not None, 
                    final_original_to_dense_map is not None, 
                    final_embeddings_all_segments is not None,
                    canonical_detector_params_by_original_idx is not None]): # Check all are populated
            self.logger.error(f"Client {self.client_id}: Missing state for building concept vectors with canonical detectors."); return None, []
        
        num_final_features = len(final_concept_indices_ordered)
        if num_final_features == 0:
            self.state['concept_vecs'], self.state['kept_image_ids_for_model'] = np.empty((0,0)), []
            return self.state['concept_vecs'], self.state['kept_image_ids_for_model']

        img_to_segs_indices_in_embeddings = defaultdict(list)
        img_idx_to_base_id = {}
        for global_seg_idx, info in enumerate(self.state['filtered_segment_infos']):
            img_idx, base_id = info.get("img_idx"), info.get("base_id")
            if img_idx is not None and base_id is not None:
                img_to_segs_indices_in_embeddings[img_idx].append(global_seg_idx)
                if img_idx not in img_idx_to_base_id: img_idx_to_base_id[img_idx] = base_id

        if not img_to_segs_indices_in_embeddings:
            self.logger.warning(f"Client {self.client_id}: No valid segments mapped to images for vectorization."); return np.empty((0, num_final_features)), []

        # --- Predict with Canonical Detectors ---
        prob_cache_canonical = {} # Store probs: {dense_idx: array_of_probs_for_all_segments}
        for dense_idx in range(num_final_features):
            original_kmeans_idx = final_concept_indices_ordered[dense_idx]
            detector_params = canonical_detector_params_by_original_idx.get(original_kmeans_idx)

            if detector_params:
                if final_embeddings_all_segments.shape[0] > 0: # Only predict if there are embeddings
                    probs_for_this_concept = self._reconstruct_and_predict_with_canonical_detector(
                        detector_params, final_embeddings_all_segments
                    )
                    prob_cache_canonical[dense_idx] = probs_for_this_concept
                else: # No embeddings to predict on
                    prob_cache_canonical[dense_idx] = np.array([]) # Empty probs
            else:
                self.logger.warning(f"Client {self.client_id}: No canonical detector params for original_idx {original_kmeans_idx} (dense_idx {dense_idx}). This concept feature will be 0.")
        
        min_activating_segments = config_from_main_script.get('vectorizer_min_activating_segments', 1)
        kept_vecs, kept_ids = [], []
        sorted_client_img_indices = sorted(list(img_idx_to_base_id.keys()))

        for client_img_idx in sorted_client_img_indices:
            vec = np.zeros((num_final_features,), dtype=np.float32)
            segment_indices_global_for_this_image = img_to_segs_indices_in_embeddings.get(client_img_idx, [])
            if not segment_indices_global_for_this_image: continue

            for dense_idx in range(num_final_features):
                if dense_idx in prob_cache_canonical:
                    all_segment_probs = prob_cache_canonical[dense_idx]
                    if all_segment_probs.size == 0 : continue # No probs computed for this concept

                    original_kmeans_idx = final_concept_indices_ordered[dense_idx]
                    detector_params = canonical_detector_params_by_original_idx.get(original_kmeans_idx, {})
                    thr = detector_params.get('optimal_threshold') # Threshold comes from server params
                    if thr is None: self.logger.warning(f"No optimal_threshold for concept dense_idx {dense_idx}."); continue

                    valid_indices_for_probs = [s_idx for s_idx in segment_indices_global_for_this_image if s_idx < len(all_segment_probs)]
                    if not valid_indices_for_probs: continue
                    
                    current_image_segment_probs = all_segment_probs[valid_indices_for_probs]
                    activating_segment_count = np.sum(current_image_segment_probs >= thr)
                    if activating_segment_count >= min_activating_segments:
                        vec[dense_idx] = 1.0
            
            kept_vecs.append(vec)
            kept_ids.append(img_idx_to_base_id[client_img_idx])

        if not kept_vecs:
            self.logger.warning(f"Client {self.client_id}: No concept vectors generated using canonical detectors.");
            self.state['concept_vecs'], self.state['kept_image_ids_for_model'] = np.empty((0, num_final_features)), []
            return self.state['concept_vecs'], self.state['kept_image_ids_for_model']
            
        self.state['concept_vecs'] = np.array(kept_vecs) 
        self.state['kept_image_ids_for_model'] = kept_ids
        self.logger.info(f"Client {self.client_id}: Built concept vectors using CANONICAL detectors. Shape: {self.state['concept_vecs'].shape}")
        return self.state['concept_vecs'], self.state['kept_image_ids_for_model']
    def get_current_concept_vectors_and_ids(self):
        return self.state.get('concept_vecs'), self.state.get('kept_image_ids_for_model', [])

    def _traverse_figs_tree_to_extract_terms(self, node: LocalNode, current_path_conditions, feature_names_for_tree, X_df_for_support_calc):
        terms = []
        if node is None: return terms
        if node.left is None and node.right is None: # Leaf node
            rule_str = " & ".join(sorted(list(set(current_path_conditions)))) if current_path_conditions else "True"
            local_support = 0
            if X_df_for_support_calc is not None and not X_df_for_support_calc.empty:
                try:
                    local_support = X_df_for_support_calc.query(rule_str).shape[0] if rule_str != "True" else X_df_for_support_calc.shape[0]
                except Exception as e_query: 
                    self.logger.warning(f"Client {self.client_id}: Query failed for support: Rule='{rule_str[:100]}', Err={e_query}. Support=0.")
                    local_support = 0
            
            leaf_val_raw = node.value # Should be (1, K_outputs) or (K_outputs,)
            model_instance = self.state.get('figs_model_trained_instance')
            num_model_outputs = getattr(model_instance, 'n_outputs_', self.config.get('num_classes',1))
            
            processed_value_list = np.zeros(num_model_outputs).tolist() # Default
            if isinstance(leaf_val_raw, np.ndarray):
                processed_value_list = leaf_val_raw.flatten().tolist()
            elif isinstance(leaf_val_raw, (list, tuple)):
                processed_value_list = list(leaf_val_raw)

            if len(processed_value_list) != num_model_outputs:
                correct_length_value_list = np.zeros(num_model_outputs)
                len_to_copy = min(len(processed_value_list), num_model_outputs)
                correct_length_value_list[:len_to_copy] = processed_value_list[:len_to_copy]
                processed_value_list = correct_length_value_list.tolist()

            terms.append({'rule_str': rule_str, 'value_array': processed_value_list, 'local_support': int(local_support)})
            return terms

        if not hasattr(node, 'feature') or node.feature is None or \
           node.feature < 0 or node.feature >= len(feature_names_for_tree):
            return terms 

        feat_name = feature_names_for_tree[node.feature]
        threshold_val = node.threshold
        left_condition = f"`{feat_name}` <= {threshold_val:.6f}"
        right_condition = f"`{feat_name}` > {threshold_val:.6f}"

        if node.left: terms.extend(self._traverse_figs_tree_to_extract_terms(node.left, current_path_conditions + [left_condition], feature_names_for_tree, X_df_for_support_calc))
        if node.right: terms.extend(self._traverse_figs_tree_to_extract_terms(node.right, current_path_conditions + [right_condition], feature_names_for_tree, X_df_for_support_calc))
        return terms

    def _predict_proba_using_global_terms(self, concept_vecs_df, global_terms_from_server):
        if concept_vecs_df is None or concept_vecs_df.empty: return np.array([])
        if not global_terms_from_server:
            num_s, num_c_fallback = concept_vecs_df.shape[0], self.config.get('num_classes',1)
            return np.ones((num_s, num_c_fallback)) / max(1, num_c_fallback)

        num_c = 0
        for t in global_terms_from_server:
            va = t.get('aggregated_value_array')
            if isinstance(va, list) and va: num_c = len(va); break
        if num_c == 0: num_c = self.config.get('num_classes',1)
        
        scores = np.zeros((concept_vecs_df.shape[0], num_c))
        cond_regex = re.compile(r"`(.+?)`\s*([><]=?)\s*([0-9.-]+)")

        for i in range(concept_vecs_df.shape[0]):
            sample_series = concept_vecs_df.iloc[i]
            sample_val = np.zeros(num_c)
            for term in global_terms_from_server:
                rule_ok = True
                if term['rule_str'] != "True":
                    for cond_str in term['rule_str'].split(' & '):
                        match = cond_regex.match(cond_str.strip())
                        if not match: rule_ok = False; break
                        feat, op, val_s = match.groups()
                        if feat not in sample_series.index: rule_ok = False; break
                        s_val, cond_v = sample_series[feat], float(val_s)
                        op_fn = {'<=': operator.le, '>': operator.gt, '<': operator.lt, '>=': operator.ge, '==': operator.eq}.get(op)
                        if not (op_fn and op_fn(s_val, cond_v)): rule_ok = False; break
                if rule_ok:
                    term_va = term.get('aggregated_value_array', [])
                    if isinstance(term_va, list) and len(term_va) == num_c: sample_val += np.array(term_va)
            scores[i] = sample_val
        
        return softmax(scores, axis=1)


    def _get_scores_using_global_terms(self, concept_vecs_df: pd.DataFrame, global_terms_from_server: list) -> np.ndarray:
        """ Calculates raw summed scores from global FIGS terms. Returns (n_samples, n_classes). """
        logger = self.logger
        if concept_vecs_df is None or concept_vecs_df.empty:
            logger.debug(f"Client {self.client_id}: _get_scores: concept_vecs_df is empty.")
            return np.array([]) 
        
        num_classes = self.config.get('num_classes', 1) # Get from client's base config
        if not global_terms_from_server: # If no global model (e.g., first round)
            logger.debug(f"Client {self.client_id}: _get_scores: global_terms empty. Returning zero scores.")
            return np.zeros((concept_vecs_df.shape[0], num_classes))

        # Determine num_classes from first valid term if possible, as a sanity check
        first_term_val_arr = global_terms_from_server[0].get('aggregated_value_array')
        if isinstance(first_term_val_arr, list) and len(first_term_val_arr) > 0:
            if num_classes != len(first_term_val_arr):
                logger.warning(f"Client {self.client_id}: Mismatch num_classes in config ({num_classes}) vs global term ({len(first_term_val_arr)}). Using config.")
        # else: (no valid terms or first term malformed, num_classes from config will be used)

        summed_scores = np.zeros((concept_vecs_df.shape[0], num_classes))
        cond_regex = re.compile(r"`(.+?)`\s*([><]=?)\s*([0-9.-]+)")

        for i in range(concept_vecs_df.shape[0]):
            sample_series = concept_vecs_df.iloc[i]
            sample_total_value = np.zeros(num_classes)
            for term in global_terms_from_server:
                rule_ok = True
                if term['rule_str'] != "True":
                    for cond_str in term['rule_str'].split(' & '):
                        match = cond_regex.match(cond_str.strip())
                        if not match: rule_ok = False; break
                        feat, op, val_s = match.groups()
                        if feat not in sample_series.index: rule_ok = False; break
                        s_val, cond_v = sample_series[feat], float(val_s)
                        op_fn = {'<=':operator.le,'>':operator.gt,'<':operator.lt,'>=':operator.ge,'==':operator.eq}.get(op)
                        if not (op_fn and op_fn(s_val, cond_v)): rule_ok = False; break
                if rule_ok:
                    term_va = term.get('aggregated_value_array', [])
                    if isinstance(term_va, list) and len(term_va) == num_classes:
                        sample_total_value += np.array(term_va)
            summed_scores[i] = sample_total_value
        return summed_scores



    def receive_residual_model_update(self, residual_model_hm_terms: list):
        logger = self.logger
        learning_rate = self.state.get('learning_rate_gbm', 0.1)
        # F_{m-1} terms
        current_Fm_minus_1_terms = self.state.get('accumulated_global_model_Fm_terms', []) 
        
        logger.info(f"Client {self.client_id}: Received {len(residual_model_hm_terms)} terms for h^(m). Current F_m-1 has {len(current_Fm_minus_1_terms)} terms. LR: {learning_rate}")

        # Create a new dictionary for F_m based on F_{m-1}
        new_Fm_terms_dict = {term['rule_str']: term.copy() for term in current_Fm_minus_1_terms}

        for term_h in residual_model_hm_terms: # Terms from h^(m)
            rule_str_h = term_h['rule_str']
            value_array_h = np.array(term_h['aggregated_value_array'])
            scaled_value_h_contribution = value_array_h * learning_rate # nu * h_m(x)

            if rule_str_h in new_Fm_terms_dict:
                # Rule exists in F_{m-1}, add the scaled residual prediction from h^(m)
                current_main_value = np.array(new_Fm_terms_dict[rule_str_h]['aggregated_value_array'])
                new_Fm_terms_dict[rule_str_h]['aggregated_value_array'] = (current_main_value + scaled_value_h_contribution).tolist()
                new_Fm_terms_dict[rule_str_h]['global_support_samples'] = term_h.get('global_support_samples', new_Fm_terms_dict[rule_str_h].get('global_support_samples',0))
            else:
                # If rule is new from h^(m), add it (scaled) to F_m
                new_term_for_Fm = term_h.copy() 
                new_term_for_Fm['aggregated_value_array'] = scaled_value_h_contribution.tolist()
                new_Fm_terms_dict[rule_str_h] = new_term_for_Fm
        
        self.state['accumulated_global_model_Fm_terms'] = list(new_Fm_terms_dict.values())
        logger.info(f"Client {self.client_id}: Updated accumulated global model F_m. Now has {len(self.state['accumulated_global_model_Fm_terms'])} terms.")


    def train_figs(self, image_labels_for_concepts: np.ndarray, current_config_from_main: dict):
        logger = self.logger
        current_round_idx = current_config_from_main.get('current_round', 0) # 0-indexed
        current_round_display = current_round_idx + 1
        logger.info(f"Client {self.client_id}: Prep FIGS training (GBM-style). Round: {current_round_display}")
        
        vecs_np = self.state.get('concept_vecs')
        feat_names = self.state.get('feature_names_for_figs')
        
        if vecs_np is None or vecs_np.size == 0: logger.error(f"Client {self.client_id}: No concept vecs."); return None, []
        if feat_names is None or len(feat_names) != vecs_np.shape[1]: logger.error(f"Client {self.client_id}: Feat names mismatch."); return None, []

        num_samples = vecs_np.shape[0]
        num_classes = current_config_from_main.get('num_classes')

        Y_true_one_hot = np.zeros((num_samples, num_classes))
        if image_labels_for_concepts.size > 0 and image_labels_for_concepts.max() < num_classes :
            Y_true_one_hot[np.arange(num_samples), image_labels_for_concepts.astype(int)] = 1.0
        else: 
            logger.error(f"Client {self.client_id}: image_labels empty or out of bounds for num_classes={num_classes}. Labels unique: {np.unique(image_labels_for_concepts)}")
            return None,[]

        df_for_scores = pd.DataFrame(vecs_np, columns=feat_names)
        
        # Scores from F_{m-1} (client's current accumulated model)
        # This uses self.state['accumulated_global_model_Fm_terms']
        scores_Fm_minus_1 = self._get_scores_using_accumulated_global_model(df_for_scores) 
        
        if scores_Fm_minus_1.shape != (num_samples, num_classes):
            logger.error(f"Client {self.client_id}: Scores from F_m-1 shape mismatch. Expected {(num_samples, num_classes)}, got {scores_Fm_minus_1.shape}. Defaulting residuals to Y_true_one_hot.")
            # This means F_0 essentially, so residuals are just Y_true
            scores_Fm_minus_1 = np.zeros((num_samples, num_classes)) 
        
        # Calculate pseudo-residuals: r = Y_true_one_hot - Scores(F_{m-1})
        # These are the targets for the current round's local FIGS model h_k^(m)
        residual_targets_for_hk = Y_true_one_hot - scores_Fm_minus_1 
        
        sample_weights_for_fit = None # Uniform weights for standard GBM tree fitting
        self.state['sample_weights'] = None # Explicitly clear any old boosting weights

        figs_params = current_config_from_main.get('figs_params', {})
        logger.info(f"Client {self.client_id}: Init PatchedFIGSClassifier for residuals with params: {figs_params}")
        init_kwargs = {
                    'random_state': current_config_from_main.get('seed', 42),
                    'n_outputs_global': num_classes, 
                    **figs_params
                }
        try: figs_model_for_residuals = PatchedFIGSClassifier(**init_kwargs) 
        except TypeError as e_init: logger.error(f"Client {self.client_id}: TypeError init FIGS: {e_init}. Kwargs: {init_kwargs}"); return None, []

        if vecs_np.shape[0] < num_classes * 2 : logger.warning(f"Client {self.client_id}: Not enough samples for FIGS fit."); return None, []

        try:
            df_fit = pd.DataFrame(vecs_np, columns=feat_names)
            logger.info(f"Client {self.client_id}: Fitting PatchedFIGS on RESIDUALS. X:{df_fit.shape}, y_residuals:{residual_targets_for_hk.shape}, w:None")
            
            figs_model_for_residuals.fit(X=df_fit, y=image_labels_for_concepts, 
                                         feature_names=feat_names, 
                                         sample_weight=sample_weights_for_fit,
                                         _y_fit_override=residual_targets_for_hk) 
            
            self.state['figs_model_trained_instance'] = figs_model_for_residuals
            self.state['feature_names_fitted_'] = getattr(figs_model_for_residuals, 'feature_names_', feat_names) 
            check_is_fitted(figs_model_for_residuals)
            logger.info(f"Client {self.client_id}: PatchedFIGS (for residuals) fit done. Complexity: {getattr(figs_model_for_residuals, 'complexity_', 'N/A')}, Trees: {len(getattr(figs_model_for_residuals,'trees_',[]))}")
        except Exception as e: logger.exception(f"Client {self.client_id}: PatchedFIGS fitting (for residuals) fail: {e}"); return None, []

        extracted_terms = []
        model_to_traverse = self.state['figs_model_trained_instance']
        if hasattr(model_to_traverse, 'trees_') and model_to_traverse.trees_:
             actual_feat_names = self.state['feature_names_fitted_']
             X_df_for_support = pd.DataFrame(vecs_np, columns=actual_feat_names)
             for tree_idx, root_node in enumerate(model_to_traverse.trees_):
                 if root_node is not None:
                    extracted_terms.extend(self._traverse_figs_tree_to_extract_terms(root_node, [], actual_feat_names, X_df_for_support))
        self.state['figs_extracted_terms'] = extracted_terms
        logger.info(f"Client {self.client_id}: Extracted {len(extracted_terms)} residual terms for h_k^(m).")
        
        # report R^2 or MSE of the residuals
        self.state['local_model_mse'] = np.mean((residual_targets_for_hk - scores_Fm_minus_1) ** 2) if residual_targets_for_hk.size > 0 else 0.0
        
        logger.info(f"Client {self.client_id}: Local FIGS (residual) training done.")
        return model_to_traverse, extracted_terms
    
    
    # get_model_update should still send terms of h_k^(m) (the residual model)
    def get_model_update(self):
        terms = self.state.get('figs_extracted_terms', [])
        valid_terms = []
        num_c = self.config.get('num_classes',1)
        for term in terms:
            if isinstance(term, dict) and 'rule_str' in term and \
               'value_array' in term and isinstance(term['value_array'], list) and \
               len(term['value_array']) == num_c and \
               'local_support' in term and isinstance(term['local_support'], int):
                valid_terms.append(term)
        
        support = len(self.state.get('concept_vecs', [])) if self.state.get('concept_vecs') is not None else 0
        mse = self.state.get('local_model_mse', 0.0) 
        self.logger.info(f"Client {self.client_id}: Sending {len(valid_terms)} residual terms (h_k^(m)). Support: {support}, Reported MSE on residual: {mse:.4f}")
        return {'figs_terms': valid_terms, 'support': support, 'mse': mse}