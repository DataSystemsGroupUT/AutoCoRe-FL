import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch 
import random
from AutoCore_FL.federated.client import PatchedFIGSClassifier
from AutoCore_FL.federated.utils import setup_logging, calculate_metrics

FEATURES_LOAD_DIR_RESNET =os.path.join("/gpfs/helios/home/soliman/logic_explained_networks/experiments", "features", "sunrgbd_3scenes_predicted_concepts")

MODEL_ARCH_FN_PART_RESNET = "resnet18_tl_sunrgbd3cls" # From Stage 1 that generated these features

# This list MUST match the one used in 0_generate_ground_truth_attributes_for_stage1.py
# and 2_generate_predicted_concept_features.py to ensure correct Y labels and scene names
SCENES_FOR_PIPELINE_CONSISTENCY = ['bathroom', 'bedroom', 'bookstore']
SCENES_FOR_PIPELINE_CONSISTENCY = sorted(list(set(SCENES_FOR_PIPELINE_CONSISTENCY)))
scene_fn_part_resnet = '_'.join([s.replace("_", "") for s in SCENES_FOR_PIPELINE_CONSISTENCY])
dataset_name = "sun" # For logging and results naming

RESNET_PRED_CONCEPTS_X_NPY_NAME = f"{MODEL_ARCH_FN_PART_RESNET}_binary_X_scenes_{scene_fn_part_resnet}.npy"
RESNET_PRED_CONCEPTS_Y_NPY_NAME = f"{MODEL_ARCH_FN_PART_RESNET}_binary_Y_scenes_{scene_fn_part_resnet}.npy"
RESNET_SCENE_NAMES_NPY_NAME = f"{MODEL_ARCH_FN_PART_RESNET}_binary_scene_names_{scene_fn_part_resnet}.npy"
RESNET_OBJECT_NAMES_NPY_NAME = f"{MODEL_ARCH_FN_PART_RESNET}_binary_object_names.npy"

RESNET_PRED_X_PATH = os.path.join(FEATURES_LOAD_DIR_RESNET, RESNET_PRED_CONCEPTS_X_NPY_NAME)
RESNET_PRED_Y_PATH = os.path.join(FEATURES_LOAD_DIR_RESNET, RESNET_PRED_CONCEPTS_Y_NPY_NAME)
RESNET_SCENE_NAMES_PATH = os.path.join(FEATURES_LOAD_DIR_RESNET, RESNET_SCENE_NAMES_NPY_NAME)
RESNET_OBJECT_NAMES_PATH = os.path.join(FEATURES_LOAD_DIR_RESNET, RESNET_OBJECT_NAMES_NPY_NAME)

METHOD_NAME = f"AutoCore_Cent_ResNetConcepts_{dataset_name}"
SEED = 42
TEST_SPLIT_RATIO_CENT = 0.2

def generate_config_AutoCore_resnet(run_id_base=f"AutoCore_Cent_ResNetConcepts_{dataset_name}"):
    effective_run_id = f"{run_id_base}_{MODEL_ARCH_FN_PART_RESNET}" # More specific run_id
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, f"experiment_results_centralized/{METHOD_NAME.lower()}_run_{effective_run_id}")
    os.makedirs(base_dir, exist_ok=True)
    log_dir_path = os.path.join(base_dir, "logs")
    os.makedirs(log_dir_path, exist_ok=True)
    
    # AutoCore parameters will be iterated in the main loop
    config = {
        "seed": SEED,
        "test_split_ratio": TEST_SPLIT_RATIO_CENT,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_dir": log_dir_path,
        "run_id": effective_run_id,
        "method_name": METHOD_NAME,
        "plot_dpi": 100,
        "figs_params_base": {"min_impurity_decrease": 0.0, "random_state": SEED}, # max_rules, max_trees, max_features added in sweep
        "chosen_classes_for_metrics": SCENES_FOR_PIPELINE_CONSISTENCY, # For y_hat mapping
        "num_classes": len(SCENES_FOR_PIPELINE_CONSISTENCY), # Number of scene classes
    }
    return config

def main_AutoCore_centralized_resnet():
    config = generate_config_AutoCore_resnet()
    setup_logging(log_dir=config['log_dir'],run_id = config['run_id'])
    main_logger = logging.getLogger(f"MainAutoCoreCentResNet_{dataset_name}_{config['run_id']}")
    
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    random.seed(config['seed']) 
    main_logger.info(f"======== Starting Centralized AutoCore with ResNet18 Concepts - dataset: {dataset_name} Run ID: {config['run_id']} ========")
    main_logger.info(f"Full Config: {config}")

    # --- 1. Load Pre-computed ResNet18 Concept Features and Labels ---
    main_logger.info("--- Phase 1: Loading Pre-computed ResNet18 Concept Data ---")
    if not all(os.path.exists(p) for p in [RESNET_PRED_X_PATH, RESNET_PRED_Y_PATH, RESNET_SCENE_NAMES_PATH, RESNET_OBJECT_NAMES_PATH]):
        main_logger.error("One or more .npy files for ResNet18 predicted concepts not found. "
                          "Ensure 2_generate_predicted_concept_features.py was run successfully and paths are correct.")
        return

    X_all_resnet_concepts = np.load(RESNET_PRED_X_PATH) # Shape: (N_images, 150 object concepts) - BINARY
    y_all_scene_labels = np.load(RESNET_PRED_Y_PATH)    # Shape: (N_images,) - integer scene labels
    scene_class_names_loaded = np.load(RESNET_SCENE_NAMES_PATH, allow_pickle=True).tolist() # List of scene names
    object_concept_names_loaded = np.load(RESNET_OBJECT_NAMES_PATH, allow_pickle=True).tolist()

    main_logger.info(f"Loaded ResNet18 concept data: X shape {X_all_resnet_concepts.shape}, Y shape {y_all_scene_labels.shape}")
    main_logger.info(f"Number of scene classes loaded: {len(scene_class_names_loaded)}")
    main_logger.info(f"Number of object concept features: {len(object_concept_names_loaded)}")

    if X_all_resnet_concepts.shape[1] != len(object_concept_names_loaded):
        main_logger.error("Mismatch between number of features in X_all_resnet_concepts and length of object_concept_names_loaded.")
        return
    if len(scene_class_names_loaded) != config.get("num_classes", len(SCENES_FOR_PIPELINE_CONSISTENCY)):
         main_logger.warning(f"Mismatch in num_classes. Config: {config.get('num_classes')}, Loaded scene names: {len(scene_class_names_loaded)}")
         config["num_classes"] = len(scene_class_names_loaded)
         config["chosen_classes"] = scene_class_names_loaded 

    # Store for y_hat mapping in metrics
    config['sorted_chosen_classes_for_mapping'] = scene_class_names_loaded # Already sorted from generation script

    # --- 2. Split Data (Consistent with other centralized runs) ---
    main_logger.info(f"--- Phase 2: Splitting Data (Test Ratio: {config['test_split_ratio']}) ---")
    try:
        X_train_concepts, X_test_concepts, y_train_labels, y_test_labels_final = train_test_split(
            X_all_resnet_concepts,
            y_all_scene_labels,
            test_size=config['test_split_ratio'],
            random_state=config['seed'],
            stratify=y_all_scene_labels
        )
    except ValueError:
        main_logger.warning("Stratification failed during train/test split. Using non-stratified split.")
        X_train_concepts, X_test_concepts, y_train_labels, y_test_labels_final = train_test_split(
            X_all_resnet_concepts, y_all_scene_labels, test_size=config['test_split_ratio'], random_state=config['seed']
        )
    
    main_logger.info(f"Train data: X_shape={X_train_concepts.shape}, Y_shape={y_train_labels.shape}")
    main_logger.info(f"Test data: X_shape={X_test_concepts.shape}, Y_shape={y_test_labels_final.shape}")

    num_actual_features = X_train_concepts.shape[1]
    figs_feature_names = object_concept_names_loaded[:num_actual_features] # Use loaded object names

    # --- 3. AutoCore Model Training & Evaluation Sweep (using PatchedFIGSClassifier) ---
    main_logger.info("--- Phase 3: AutoCore Model Training & Evaluation Sweep ---")

    max_rules_values = config.get("figs_max_rules_sweep", [20, 30, 40, 60, 80, 100, 110, 120, 130, 150, 180])
    max_trees_values = config.get("figs_max_trees_sweep", [1, 3, 5,7, None]) # None means max_rules constraint dominates
    max_features_values = config.get("figs_max_features_sweep", [None, 'sqrt'])
    
    # Prepare y_train_labels_1d for PatchedFIGSClassifier.fit()
    if y_train_labels.ndim > 1 and y_train_labels.shape[1] == 1:
        y_train_labels_1d_for_figs = y_train_labels.ravel()
    elif y_train_labels.ndim == 1:
        y_train_labels_1d_for_figs = y_train_labels
    else:
        main_logger.error(f"AtuoCore ResNet: y_train_labels is not 1D. Shape: {y_train_labels.shape}.")
        return

    df_train_concepts_for_figs = pd.DataFrame(X_train_concepts, columns=figs_feature_names)
    
    results_log = []
    best_overall_accuracy = -1.0
    best_params = {}

    for rules_val in max_rules_values:
        for trees_val in max_trees_values:
            for features_val in max_features_values:
                current_figs_params = config['figs_params_base'].copy()
                current_figs_params['max_rules'] = rules_val
                current_figs_params['max_trees'] = trees_val
                current_figs_params['max_features'] = features_val
                
                main_logger.info(f"Testing FIGS with params: {current_figs_params}")
                figs_model = PatchedFIGSClassifier(**current_figs_params, n_outputs_global=config['num_classes']) # Uses seed from figs_params_base

                try:
                    figs_model.fit(
                        df_train_concepts_for_figs,
                        y_train_labels_1d_for_figs,
                        feature_names=figs_feature_names,
                        _y_fit_override=None
                    )
                    main_logger.info(f"AutoCore trained. Complexity: {getattr(figs_model, 'complexity_', 'N/A')}, Trees: {len(getattr(figs_model,'trees_',[]))}")

                    if X_test_concepts.shape[0] > 0:
                        y_test_labels_1d_for_eval = y_test_labels_final.ravel() if y_test_labels_final.ndim > 1 else y_test_labels_final
                        
                        # Use your LR-XFL style metrics function
                        model_acc, dnf_rule_acc, dnf_rule_fid = calculate_metrics(
                            figs_model, X_test_concepts, y_test_labels_1d_for_eval,
                            figs_feature_names, main_logger
                        )
                        main_logger.info(f"Results for {current_figs_params}: ModelAcc={model_acc:.4f}, DNF_RuleAcc={dnf_rule_acc:.4f}, DNF_RuleFid={dnf_rule_fid:.4f}")
                        
                        results_log.append({
                            **current_figs_params,
                            "ModelAcc": model_acc,
                            "DNF_RuleAcc": dnf_rule_acc,
                            "DNF_RuleFid": dnf_rule_fid
                        })
                        if model_acc > best_overall_accuracy:
                            best_overall_accuracy = model_acc
                            best_params = current_figs_params.copy()
                    else:
                        main_logger.warning("  Skipping evaluation: No test data.")
                except Exception as e_figs_fit:
                    main_logger.error(f"  Error training/evaluating FIGS with {current_figs_params}: {e_figs_fit}", exc_info=True)
                main_logger.info("-" * 30)

    main_logger.info(f"--- AutoCore Hyperparameter Sweep Complete ---")
    main_logger.info(f"Best Overall Model Accuracy: {best_overall_accuracy:.4f} with params: {best_params}")

    # Save all results
    results_df = pd.DataFrame(results_log)
    results_filename = os.path.join(config['log_dir'], f"AutoCore_cent_resnet_hyperparam_sweep_results_{dataset_name}_{config['run_id']}.csv")
    results_df.to_csv(results_filename, index=False)
    main_logger.info(f"Hyperparameter sweep results saved to {results_filename}")
    
    main_logger.info(f"======== Centralized AutoCore with ResNet18 Concepts -  dataset: {dataset_name} Run ID: {config['run_id']} Complete ========")

if __name__ == "__main__":
        main_AutoCore_centralized_resnet()