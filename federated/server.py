from collections import defaultdict, Counter
import numpy as np
import logging
import pandas as pd
import copy
import operator
import re
from scipy.special import softmax 
from sklearn.pipeline import Pipeline 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from AutoCore_FL.federated.aggregation import pareto_select as pareto_select_func

try:
    from AutoCore_FL.federated.aggregation import aggregate_kmeans_centroids
except ImportError:
    try:
        from .aggregation import aggregate_kmeans_centroids 
    except ImportError:
        logging.error("Failed to import aggregate_kmeans_centroids. Ensure it's in the correct path.")
        def aggregate_kmeans_centroids(client_stats):
            logging.error("aggregate_kmeans_centroids placeholder called. Aggregation will likely fail.")
            sums, counts = zip(*client_stats)
            return np.mean(np.stack(sums), axis=0), np.array([False]*len(counts[0]))


class FederatedServer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"Server-{self.config.get('run_id', 'DefaultRun')}")
        self.centroids = None 

        # Stores actual model objects for server's own use (e.g., for test set vectorization)
        self.canonical_detector_model_objects = {} 
        self.canonical_thresholds = {} 
        self.global_figs_model_terms = [] 
        self._prev_figs_model_term_repr_for_convergence = "[]" 
        self.feature_names_for_figs = [] 
        self.accumulated_Fm_global_terms = [] # Stores F_0, F_1, ..., F_M
        self.server_X_test_concepts_for_validation = None 
        self.server_y_test_labels_for_validation = None 
        self.server_feature_names_for_validation = None 
        self.logger.info(f"FederatedServer initialized for FIGS. Run ID: {self.config.get('run_id')}")

    def initialize_centroids(self, embedding_dim: int, num_clusters: int):
        self.logger.info(f"Initializing {num_clusters} centroids with dimension {embedding_dim}.")
        self.centroids = np.random.randn(num_clusters, embedding_dim).astype(np.float32)
        return self.centroids

    def aggregate_kmeans(self, client_stats: list):
        if not client_stats:
            self.logger.error("KMeans agg: empty client_stats."); 
            num_k = self.config.get('num_clusters',1)
            return self.centroids if self.centroids is not None else None, np.array([False]*num_k)
        try:
            new_cents, live_mask = aggregate_kmeans_centroids(client_stats)
            if new_cents is not None and self.centroids is not None and new_cents.shape == self.centroids.shape:
                self.centroids = new_cents
            elif new_cents is not None and self.centroids is None: 
                self.centroids = new_cents
            elif new_cents is not None:
                 self.logger.warning(f"KMeans agg shape mismatch. New: {new_cents.shape}, Old: {self.centroids.shape if self.centroids is not None else 'None'}.")
            else: self.logger.warning("KMeans agg returned None for new_centroids.")
            return self.centroids, live_mask
        except Exception as e:
            self.logger.exception(f"KMeans agg error: {e}"); 
            num_k = self.config.get('num_clusters',1)
            return self.centroids if self.centroids is not None else None, np.array([True]*(self.centroids.shape[0] if self.centroids is not None else num_k))
    
    
    def _add_residual_terms_to_main_model(self, main_model_terms: list, residual_terms: list, learning_rate: float) -> list:
        """
        Adds scaled residual terms to a list of main model terms.
        If rule_str overlaps, values are added. Otherwise, new rule is appended.
        """
        main_model_dict = {term['rule_str']: term.copy() for term in main_model_terms}

        for res_term in residual_terms:
            rule_str = res_term['rule_str']
            res_value_array = np.array(res_term['aggregated_value_array'])
            scaled_res_value = res_value_array * learning_rate

            if rule_str in main_model_dict:
                # Rule exists, add residual contribution to its value array
                current_main_value = np.array(main_model_dict[rule_str]['aggregated_value_array'])
                main_model_dict[rule_str]['aggregated_value_array'] = (current_main_value + scaled_res_value).tolist()
                # Optionally update other metrics like support (e.g., take latest or max)
                main_model_dict[rule_str]['global_support_samples'] = res_term.get('global_support_samples', main_model_dict[rule_str].get('global_support_samples',0))
                main_model_dict[rule_str]['num_contributing_clients'] = res_term.get('num_contributing_clients', main_model_dict[rule_str].get('num_contributing_clients',0))

            else:
                # New rule from residual model, add it (scaled)
                new_term = res_term.copy()
                new_term['aggregated_value_array'] = scaled_res_value.tolist()
                main_model_dict[rule_str] = new_term
        
        return list(main_model_dict.values())

    def _extract_detector_parameters(self, model_pipeline_object, original_idx_for_logging):
        """ Extracts key serializable parameters for manual application on client. """
        params = {'type': 'unknown', 'optimal_threshold': None} # optimal_threshold added later
        logger = self.logger

        if not isinstance(model_pipeline_object, Pipeline):
            logger.error(f"Detector {original_idx_for_logging} not Pipeline. Type: {type(model_pipeline_object)}"); return params

        pca_step = model_pipeline_object.named_steps.get('pca')
        clf_step = model_pipeline_object.named_steps.get('clf')

        if clf_step is None: logger.error(f"Detector {original_idx_for_logging}: Pipeline missing 'clf'."); return params

        if isinstance(pca_step, PCA) and hasattr(pca_step, 'mean_') and pca_step.mean_ is not None and \
           hasattr(pca_step, 'components_') and pca_step.components_ is not None:
            params['pca_mean'] = pca_step.mean_.tolist()
            params['pca_components'] = pca_step.components_.tolist()
            params['n_pca_components_out'] = pca_step.components_.shape[0]
            logger.debug(f"Detector {original_idx_for_logging}: PCA params extracted. Output dim: {params['n_pca_components_out']}")
        else:
            params['n_pca_components_out'] = 0 # Indicates no PCA or passthrough
            params['pca_mean'], params['pca_components'] = None, None
            logger.debug(f"Detector {original_idx_for_logging}: No PCA params extracted or PCA is passthrough.")

        if isinstance(clf_step, LogisticRegression):
            params['type'] = 'lr_manual' 
            if hasattr(clf_step, 'coef_') and hasattr(clf_step, 'intercept_'):
                params['clf_coef'] = clf_step.coef_.tolist() # Shape (1, n_features_after_pca) for binary
                params['clf_intercept'] = clf_step.intercept_.tolist() # Shape (1,)
            else: logger.error(f"LR for {original_idx_for_logging} unfitted."); return {'type':'unknown'}
        
        elif isinstance(clf_step, SVC):
            if clf_step.kernel != 'linear':
                logger.error(f"SVM for {original_idx_for_logging} has non-linear kernel ('{clf_step.kernel}'). Manual reconstruction only supports linear SVM. Skipping."); 
                return {'type':'unknown'}
            params['type'] = 'svm_linear_manual'
            if hasattr(clf_step, '_dual_coef_') and hasattr(clf_step, 'support_vectors_') and hasattr(clf_step, '_intercept_'):

                if clf_step._dual_coef_.shape[0] == 1: # Binary case
                    coef = np.dot(clf_step._dual_coef_[0], clf_step.support_vectors_) # Result is (n_features,)
                    params['clf_coef'] = coef.reshape(1, -1).tolist() # Ensure (1, n_features)
                    params['clf_intercept'] = clf_step._intercept_.tolist() # Shape (1,)
                else:
                    logger.error(f"Linear SVM for {original_idx_for_logging} dual_coef_ shape {clf_step._dual_coef_.shape} not suitable for binary concept detector. Skipping."); return {'type':'unknown'}
            else: logger.error(f"Linear SVM for {original_idx_for_logging} unfitted or params missing."); return {'type':'unknown'}
        else:
            logger.error(f"Detector {original_idx_for_logging}: CLF step UNKNOWN type {type(clf_step)}."); return {'type':'unknown'}
        
        logger.debug(f"Detector {original_idx_for_logging}: Parameters extracted for type {params['type']}")
        return params


    def aggregate_detectors(self, all_client_detector_updates: list, shared_concept_indices: list) -> tuple[list, dict]:
        self.logger.info(f"Aggregating detectors for {len(shared_concept_indices)} shared concepts.")
        detectors_by_original_idx = {idx: [] for idx in shared_concept_indices}
        received_valid_detector_updates_count = 0

        for client_update_dict in all_client_detector_updates:
            if not isinstance(client_update_dict, dict): continue
            for original_idx, detector_info in client_update_dict.items():
                 if original_idx in detectors_by_original_idx and isinstance(detector_info, dict) and \
                    all(k in detector_info for k in ['model', 'threshold', 'score']):
                     detectors_by_original_idx[original_idx].append(detector_info)
                     received_valid_detector_updates_count += 1
        
        self.logger.info(f"Received {received_valid_detector_updates_count} individual detector updates.")

        self.canonical_detector_model_objects.clear() 
        self.canonical_thresholds.clear() # This clears it before repopulating
        final_kept_original_indices = []
        canonical_detector_params_for_broadcast = {} 

        for original_idx in sorted(shared_concept_indices):
            detectors_for_concept = detectors_by_original_idx.get(original_idx, [])
            if not detectors_for_concept:
                self.logger.warning(f"No detector for shared concept (orig_idx) {original_idx}. Dropping."); continue
            
            # Sort by score to get the best one
            best_detector_info = sorted(detectors_for_concept, key=operator.itemgetter('score'), reverse=True)[0]
            model_obj = best_detector_info['model'] 
            threshold = best_detector_info['threshold'] 

            extracted_params = self._extract_detector_parameters(model_obj, original_idx)
            if extracted_params.get('type', 'unknown') != 'unknown':
                extracted_params['optimal_threshold'] = threshold
                
                self.canonical_detector_model_objects[original_idx] = model_obj 
                self.canonical_thresholds[original_idx] = threshold
                canonical_detector_params_for_broadcast[original_idx] = extracted_params
                final_kept_original_indices.append(original_idx)
            else:
                self.logger.error(f"Could not extract valid parameters for canonical detector original_idx {original_idx}. Concept dropped.")

        self.logger.info(f"Finalized {len(final_kept_original_indices)} concepts with extractable canonical detector params.")
        return final_kept_original_indices, canonical_detector_params_for_broadcast


    def phase2_prep_broadcast(self, clients, final_concept_indices_ordered, final_original_to_dense_map, canonical_detector_params):
        self.logger.info(f"Broadcasting final concepts and {len(canonical_detector_params)} canonical detector params to clients...")
        for client in clients:
            # This method name matches what was defined in client.py
            client.receive_final_concepts_and_detectors( 
                final_concept_indices_ordered,
                final_original_to_dense_map,
                copy.deepcopy(canonical_detector_params) 
            )

    def _evaluate_rule_on_server_data(self, rule_str: str, X_data_df: pd.DataFrame, y_data: np.ndarray, rule_intended_class: int):
        if X_data_df is None or X_data_df.empty or y_data is None or y_data.size == 0: return 0.0, 0
        mask = np.zeros(X_data_df.shape[0], dtype=bool)
        try:
            if rule_str == "True": mask = np.ones(X_data_df.shape[0], dtype=bool)
            else:
                try: mask = X_data_df.eval(rule_str).to_numpy(dtype=bool)
                except Exception:
                    current_mask = np.ones(X_data_df.shape[0], dtype=bool); parsed = True
                    for cond in rule_str.split(' & '):
                        m = re.match(r"`(.+?)`\s*([><]=?)\s*([0-9.-]+)", cond.strip())
                        if not m: parsed=False; break
                        feat, op, val_s = m.groups(); val = float(val_s)
                        if feat not in X_data_df.columns: parsed=False; break
                        op_fn = {'<=':operator.le,'>':operator.gt,'<':operator.lt,'>=':operator.ge,'==':operator.eq}.get(op)
                        if not op_fn: parsed=False; break
                        current_mask = current_mask & op_fn(X_data_df[feat].to_numpy(), val)
                    if parsed: mask = current_mask
                    else: self.logger.warning(f"Rule parse fail server_eval: {rule_str[:100]}"); mask.fill(False)
        except Exception as e: self.logger.error(f"Rule eval error: {rule_str[:100]}, {e}"); mask.fill(False)
        
        n_cov = np.sum(mask)
        if n_cov == 0: return 0.0, 0
        prec = np.mean(y_data[mask] == rule_intended_class) if rule_intended_class != -1 else 0.0
        return prec, n_cov

    def aggregate_figs_models(self, client_updates: list):
        logger = self.logger
        logger.info("Aggregating FIGS model terms...")
        num_classes = self.config.get('num_classes')
        if not isinstance(num_classes, int) or num_classes <= 0:
            logger.error(f"Invalid num_classes: {num_classes}."); self.global_figs_model_terms = []; return []

        pooled_terms = defaultdict(lambda: {'v_arrs':[],'w_avg':[],'supps':[],'c_ids':[],'c_preds':[]})
        use_acc_w = self.config.get('use_accuracy_weighting_server', False)

        for client_upd in client_updates:
            c_id, c_mse = client_upd.get('client_id','Unk'), client_upd.get('mse',0.0)
            for term_info in client_upd.get('figs_terms',[]):
                r_str, v_list, l_supp = term_info.get('rule_str'), term_info.get('value_array'), term_info.get('local_support',0)
                if not r_str or not isinstance(v_list,list) or len(v_list)!=num_classes: continue
                
                canon_r = self._canonicalize_figs_rule_string(r_str); v_np = np.array(v_list,float)
                weight = float(l_supp) * (max(c_mse,0.01) if use_acc_w else 1.0); weight = max(weight, 1e-6)

                entry = pooled_terms[canon_r]
                entry['v_arrs'].append(v_np); entry['w_avg'].append(weight)
                entry['supps'].append(l_supp); entry['c_ids'].append(c_id)
                entry['c_preds'].append(np.argmax(v_np))
        
        validated_terms = []
        min_clients = self.config.get('min_clients_for_figs_term',1)
        s_min_prec, s_min_cov = self.config.get('server_rule_validation_min_precision',0.0), self.config.get('server_rule_min_coverage_count',0)
        
        s_X_df = None
        if self.server_X_test_concepts_for_validation is not None and \
           self.server_y_test_labels_for_validation is not None and \
           self.server_feature_names_for_validation and \
           self.server_X_test_concepts_for_validation.shape[0]>0 and \
           self.server_X_test_concepts_for_validation.shape[1]==len(self.server_feature_names_for_validation):
            s_X_df = pd.DataFrame(self.server_X_test_concepts_for_validation, columns=self.server_feature_names_for_validation)
            logger.info(f"Server-side rule validation WILL BE USED with {s_X_df.shape[0]} samples.")
        else: logger.info("Server validation data not available/misconfigured; server-side rule filtering will be skipped.")

        for r_str, data in pooled_terms.items():
            if len(data['c_ids']) < min_clients: continue
            total_w = np.sum(data['w_avg'])
            agg_v = np.average(data['v_arrs'],axis=0,weights=data['w_avg']) if total_w > 1e-7 else np.mean(data['v_arrs'],axis=0)
            
            counts = Counter(data['c_preds']); 
            dom_cls, dom_cnt = counts.most_common(1)[0] if counts else (-1, 0)
            c_agree = dom_cnt / len(data['c_ids']) if len(data['c_ids']) > 0 else 0.0
            
            rule_pred_cls_agg = np.argmax(agg_v) if agg_v.size > 0 else -1
            
            s_prec, s_cov_cnt = -1.0, -1 # Defaults if not validated
            if s_X_df is not None: # Perform server validation
                s_prec, s_cov_cnt = self._evaluate_rule_on_server_data(r_str, s_X_df, self.server_y_test_labels_for_validation, rule_pred_cls_agg)
                if s_cov_cnt < s_min_cov or s_prec < s_min_prec: 
                    #logger.info(f"Rule '{r_str}' dropped: server_prec={s_prec:.2f} (min {s_min_prec}), server_cov_cnt={s_cov_cnt} (min {s_min_cov})")
                    continue
            
            validated_terms.append({
                'rule_str':r_str, 'aggregated_value_array':agg_v.tolist(), 
                'global_support_samples':int(round(np.sum(data['supps']))), 
                'num_contributing_clients':len(data['c_ids']), 'client_agreement_score':c_agree,
                'server_val_precision':s_prec, 'server_val_coverage_count':s_cov_cnt })

            temp_df_for_pareto_list = []
            for term in validated_terms:
                complexity = term['rule_str'].count('&') + 1 if term['rule_str'] != "True" else 0

                temp_df_for_pareto_list.append({
                    'rule': term['rule_str'],
                    'class': np.argmax(term['aggregated_value_array']), 
                    'support': term['global_support_samples'],
                    'precision': term['server_val_precision'],
                    'coverage': term['server_val_coverage_count'] / (s_X_df.shape[0] if s_X_df is not None and s_X_df.shape[0] > 0 else 1), #Approx coverage
                    'complexity': complexity,
                    'aggregated_value_array': term['aggregated_value_array'],
                    'num_contributing_clients': term['num_contributing_clients'],
                    'client_agreement_score': term['client_agreement_score']
                })

            if temp_df_for_pareto_list:
                df_for_pareto = pd.DataFrame(temp_df_for_pareto_list)
                
                
                #logger.info(f"Applying Pareto selection to {len(df_for_pareto)} candidate global rules.")
                pareto_selected_df = pareto_select_func(df_for_pareto.copy()) # Pass a copy
                # Convert pareto_selected_df back to list of dicts for self.global_figs_model_terms
                validated_terms_after_pareto = []
                for _idx, row in pareto_selected_df.iterrows():
                    validated_terms_after_pareto.append({
                        'rule_str': row['rule'],
                        'aggregated_value_array': row['aggregated_value_array'],
                        'global_support_samples': int(row['support']),
                        'num_contributing_clients': row['num_contributing_clients'],
                        'client_agreement_score': row['client_agreement_score'],
                        'server_val_precision': row['precision'],
                        'server_val_coverage_count': int(row['coverage'] * (s_X_df.shape[0] if s_X_df is not None and s_X_df.shape[0] > 0 else 1) ),
                    })
                validated_terms = validated_terms_after_pareto
                #logger.info(f"Number of terms after Pareto selection: {len(validated_terms)}")
            else:
                logger.info("No terms to apply Pareto selection to.")
                validated_terms = [] # Ensure it's an empty list

            # The existing sort and truncation can still apply after Pareto
            validated_terms.sort(key=lambda x:(x['server_val_precision'], x['client_agreement_score'], x['global_support_samples']), reverse=True)
            max_terms = self.config.get('max_global_figs_terms')
            if max_terms is not None and len(validated_terms) > max_terms:
                logger.info(f"Pruning global terms from {len(validated_terms)} to {max_terms} after Pareto and sort.")
                validated_terms = validated_terms[:max_terms]
                
            self.global_figs_model_terms = validated_terms
        logger.info(f"Aggregation complete. Finalized {len(self.global_figs_model_terms)} global terms.")
        return self.global_figs_model_terms

    def _canonicalize_figs_rule_string(self, rule_str: str) -> str:
        if not isinstance(rule_str, str) or rule_str == "True": return rule_str
        if " & " not in rule_str:
             if not re.match(r"`(.+?)`\s*([><]=?)\s*([0-9.-]+)", rule_str.strip()): pass
             return rule_str.strip()
        return " & ".join(sorted([c.strip() for c in rule_str.split(" & ")]))

    def has_converged(self, new_figs_model_terms_repr: str) -> bool:
        model_same = (self._prev_figs_model_term_repr_for_convergence == new_figs_model_terms_repr)
        self.logger.info(f"Convergence check: Global FIGS same as prev: {model_same}")
        self._prev_figs_model_term_repr_for_convergence = new_figs_model_terms_repr
        return model_same


    def phase2_aggregate(self, client_updates: list, current_round_num: int):
        """
        Aggregates client's residual FIGS models (h_k) into a global residual model (h_m).
        Updates the server's accumulated F_m model.
        Checks for convergence of h_m.
        """
        # 1. Aggregate client residual contributions to get h^(m)
        # self.global_figs_model_terms will temporarily store terms of h^(m) for broadcasting
        self.global_figs_model_terms = self.aggregate_figs_models(client_updates) 
        
        # 2. Server updates its own accumulated global model F_m = F_{m-1} + nu * h^(m)
        learning_rate_gbm = self.config.get('learning_rate_gbm', 0.1)
        self.accumulated_Fm_global_terms = self._add_residual_terms_to_main_model(
            self.accumulated_Fm_global_terms, # This is F_{m-1}
            self.global_figs_model_terms,    # This is h^(m)
            learning_rate_gbm
        )
        self.logger.info(f"Server updated its accumulated global model F_m. Now has {len(self.accumulated_Fm_global_terms)} total effective terms.")
        
        # 3. Check convergence based on the stability of h^(m)
        sorted_hm_terms_for_repr = sorted(
            [{**term, 'aggregated_value_array': [round(v, 5) for v in term['aggregated_value_array']]} 
             for term in self.global_figs_model_terms], # Use terms of h^(m)
            key=lambda x: x['rule_str']
        )
        current_hm_terms_repr_str = str(sorted_hm_terms_for_repr)
        converged = self.has_converged(current_hm_terms_repr_str) # Compares h^(m) with h^(m-1)
        
        self.logger.info(f"Round {current_round_num} convergence status for residual model h^(m): {converged}")
        
        # Returns the terms of h^(m) for broadcasting, and convergence status
        return self.global_figs_model_terms, converged

    def broadcast_model(self, clients: list, is_residual_model=True):
        # This method now always broadcasts self.global_figs_model_terms,
        # which in the GBM context (after phase2_aggregate) are the terms for h^(m).
        num_terms_to_broadcast = len(self.global_figs_model_terms)
        model_type_str = "residual model h^(m)" if is_residual_model else "full global model F^(m) (ERROR: Should be residual)"
        
        self.logger.info(f"Broadcasting {num_terms_to_broadcast} terms of {model_type_str} to {len(clients)} clients.")
        
        for client_obj in clients:
            if hasattr(client_obj, 'receive_residual_model_update'):
                client_obj.receive_residual_model_update(copy.deepcopy(self.global_figs_model_terms))
            else:
                self.logger.error(f"Client {getattr(client_obj,'client_id','Unk')} missing 'receive_residual_model_update'. Cannot update client's F_m.")