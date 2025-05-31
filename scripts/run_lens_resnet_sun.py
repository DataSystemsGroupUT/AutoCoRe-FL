import json
import os
import numpy as np
import pandas as pd
import torch
import sys
from torch.utils.data import Dataset 

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
autocore_fl_package_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if autocore_fl_package_root not in sys.path: sys.path.insert(0, autocore_fl_package_root)
logic_explained_networks_base_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

lens_project_grandparent_dir = "/gpfs/helios/home/soliman/logic_explained_networks" 

if lens_project_grandparent_dir not in sys.path:
    sys.path.insert(0, lens_project_grandparent_dir)
    print(f"Added to sys.path: {lens_project_grandparent_dir}")
lrxfl_project_root = "/gpfs/helios/home/soliman/logic_explained_networks/lens"
print(f"Adding LENS project root to sys.path: {lrxfl_project_root}")
lrxfl_experiments_submodule_path = lrxfl_project_root
original_sys_path = list(sys.path) 
if os.path.isdir(lrxfl_experiments_submodule_path):
    if lrxfl_experiments_submodule_path not in sys.path: sys.path.insert(0, lrxfl_experiments_submodule_path)
else: print(f"FATAL: lens experiments path not found: {lrxfl_experiments_submodule_path}."); sys.exit(1)
if os.path.isdir(lrxfl_project_root):
    if lrxfl_project_root not in sys.path: sys.path.insert(0, lrxfl_project_root)
else: print(f"FATAL: lens project root not found: {lrxfl_project_root}."); sys.exit(1)

from lens.utils.data import get_splits_train_val_test 

import os

import numpy as np
import pandas as pd
import time
import json

from tqdm import trange

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from lens.models.tree import XDecisionTreeClassifier
from lens.models.mu_nn import XMuNN
from lens.models.random_forest import XRandomForestClassifier 
from lens.utils.base import set_seed, ClassifierNotTrainedError, IncompatibleClassifierError
from lens.utils.metrics import Accuracy, F1Score
from lens.utils.data import get_splits_train_val_test
from lens.logic.eval import test_explanation
from lens.logic.metrics import complexity, fidelity, formula_consistency



# --- 1. Configuration ---
dataset_name = 'sunrgbd_3cls_gt_concepts' # New specific name

# Path to your pre-processed SUNRGBD 3-class data
SUNRGBD_PREPARED_DATA_ROOT = "/gpfs/helios/home/soliman/logic_explained_networks/data/sun_final"

# results_dir: Update this to a new location for these specific experiments
results_dir = f'/gpfs/helios/home/soliman/logic_explained_networks/experiments/results/logic_xai/{dataset_name}'
os.makedirs(results_dir, exist_ok=True)

print(f"--- Loading Pre-processed SUNRGBD 3-Class Data from: {SUNRGBD_PREPARED_DATA_ROOT} ---")

# File paths based on prepare_sunrgbd_subset_scenes.py outputs
IMAGES_ROOT_SUNRGBD = os.path.join(SUNRGBD_PREPARED_DATA_ROOT, "images")
ATTRIBUTES_NPY_SUNRGBD = os.path.join(SUNRGBD_PREPARED_DATA_ROOT, "attributes.npy")
SCENE_LABELS_NPY_SUNRGBD = os.path.join(SUNRGBD_PREPARED_DATA_ROOT, "scene_labels.npy")
SCENE_CLASSES_TXT_SUNRGBD = os.path.join(SUNRGBD_PREPARED_DATA_ROOT, "scene_classes.txt")
CONCEPT_NAMES_TXT_SUNRGBD = os.path.join(SUNRGBD_PREPARED_DATA_ROOT, "attributes_names.txt") # This is the JSON list

if not all(os.path.exists(p) for p in [IMAGES_ROOT_SUNRGBD, ATTRIBUTES_NPY_SUNRGBD, SCENE_LABELS_NPY_SUNRGBD, SCENE_CLASSES_TXT_SUNRGBD, CONCEPT_NAMES_TXT_SUNRGBD]):
    raise FileNotFoundError("One or more required SUNRGBD pre-processed files not found. Ensure prepare_sunrgbd_subset_scenes.py ran successfully.")

# Load the data
concept_vectors_X = np.load(ATTRIBUTES_NPY_SUNRGBD) # These are your features (ground truth concepts)
scene_labels_Y = np.load(SCENE_LABELS_NPY_SUNRGBD)   # These are your targets

with open(SCENE_CLASSES_TXT_SUNRGBD, 'r') as f:
    # Format: "0 bathroom", "1 bedroom", ...
    loaded_scene_classes = [line.strip().split(maxsplit=1)[1] for line in f if line.strip()]

with open(CONCEPT_NAMES_TXT_SUNRGBD, 'r') as f:
    # Format: JSON list of strings like ["has_object::bathtub", ...]
    loaded_concept_names_with_prefix = json.load(f)
    # Clean for LENS if needed (LENS might expect just "bathtub", not "has_object::bathtub")
    # This depends on how your X*Classifier models expect feature_names.
    # For now, let's assume clean names are preferred.
    concept_names_for_lens = [name.replace("has_object::", "").strip() for name in loaded_concept_names_with_prefix]

print(f"Loaded data: X_concepts shape: {concept_vectors_X.shape}, Y_labels shape: {scene_labels_Y.shape}")
print(f"Scene classes: {loaded_scene_classes}")
print(f"Concept names (first 5): {concept_names_for_lens[:5]}")


class DirectConceptDataset(Dataset):
    def __init__(self, x_data, y_data, dataset_root_path, dataset_name_str): # Added parameters
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.LongTensor(y_data)
        self.root = dataset_root_path  # LENS expects this
        self.dataset_name = dataset_name_str # LENS expects this for naming split files

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# When creating the dataset instance:
dataset_for_lens = DirectConceptDataset(
    concept_vectors_X, 
    scene_labels_Y,
    dataset_root_path=SUNRGBD_PREPARED_DATA_ROOT, # Pass the root path
    dataset_name_str=dataset_name                 # Pass the specific dataset name string
)
# LENS needs these attributes on the dataset object:
dataset_for_lens.n_attributes = concept_vectors_X.shape[1]
dataset_for_lens.attribute_names = np.array(concept_names_for_lens) 
dataset_for_lens.n_classes = len(loaded_scene_classes)
dataset_for_lens.classes = loaded_scene_classes 
dataset_for_lens.class_to_idx = {name: i for i, name in enumerate(loaded_scene_classes)}
# dataset_for_lens.dataset_name is now set in __init__
# dataset_for_lens.root is now set in __init__
dataset_for_lens.targets = scene_labels_Y # LENS utils might check this too for one-hot conversion logic inside get_splits

# dataset_for_lens.targets = scene_labels_Y # If LENS utils need this separately

print(f"Dataset for LENS: {len(dataset_for_lens)} samples.")
print(f"n_attributes: {dataset_for_lens.n_attributes}, n_classes: {dataset_for_lens.n_classes}")


# --- LENS Model Training Loop (This section should largely remain the same) ---
# The variables like `concept_names`, `n_features`, `n_classes`, `class_names`
# should now be correctly populated from the loaded SUNRGBD data.

# Assign to variables LENS loop expects:
concept_names = dataset_for_lens.attribute_names # This is now the list of clean concept names
n_features = dataset_for_lens.n_attributes
n_classes = dataset_for_lens.n_classes
class_names = dataset_for_lens.classes # This is now ["bathroom", "bedroom", "bookstore"]


print(f"Starting LENS training loop with: n_features={n_features}, n_classes={n_classes}")

loss = CrossEntropyLoss() 
metric = Accuracy()   
expl_metric = F1Score()   

# Your existing method_list and LENS training loop
method_list = ["DTree",  "General", "RandomForest"]
print("Methods to run:", method_list)

epochs = 1000 
# timeout = 60 * 60 
l_r = 1e-3
lr_scheduler = False # Or True with a scheduler
top_k_explanations = None # Or an integer
simplify = True # For LENS explanations
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device for LENS:", device)

all_methods_results_summary = []

for method in method_list:
    methods_log = []
    splits_log = []
    model_explanations_log = []
    model_accuracies_log = []
    explanation_accuracies_log = []
    elapsed_times_log = []
    explanation_fidelities_log = []
    explanation_complexities_log = []

    # Adjust seeds or number of runs if needed
    seeds = [*range(1)] if method != "BRL" else [*range(1)] # Reduced seeds for speed
    print(f"\nRunning Method: {method} with seeds: {seeds}")

    for seed in seeds:
        set_seed(seed)
        # Construct a unique name for this run, incorporating dataset and concept type (which is GT here)
        concept_type_for_filename = "gt_concepts" 
        run_name = os.path.join(results_dir, f"{method}_{seed}_concept_type_{concept_type_for_filename}")

        # Split dataset_for_lens
        # Corrected call to get_splits_train_val_test
        train_data, val_data, test_data = get_splits_train_val_test(
            dataset_for_lens, 
            val_split=0.1,  # Corresponds to your previous val_size=0.1
            test_split=0.2, # Corresponds to your previous test_size=0.2
            load=False,     # To generate new splits based on the current seed
            test_transform=None # Assuming no separate transform for val/test in this specific pipeline
        )        

        x_test = test_data.dataset.x_data[test_data.indices]
        y_test = test_data.dataset.y_data[test_data.indices]
        x_train = train_data.dataset.x_data[train_data.indices] # Needed for some explanation methods
        y_train = train_data.dataset.y_data[train_data.indices]
        x_val = val_data.dataset.x_data[val_data.indices]
        y_val = val_data.dataset.y_data[val_data.indices]


        print(f"Seed {seed}: Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
        print(f"Training {run_name} classifier...")
        start_time = time.time()

        model = None # Initialize model
        accuracy = 0.0
        explanations_for_method, exp_accuracies, exp_fidelities, exp_complexities = [""]*n_classes, [0.0]*n_classes, [0.0]*n_classes, [0.0]*n_classes


        if method == 'DTree':
            max_depth = 10 # Adjusted for potentially smaller dataset
            run_name_dt = f"{run_name}_depth{max_depth}" # More specific name
            model = XDecisionTreeClassifier(name=run_name_dt, n_classes=n_classes,
                                            n_features=n_features, max_depth=max_depth)
            try:
                model.load(device)
                print(f"Model {run_name_dt} loaded from cache.")
            except (ClassifierNotTrainedError, IncompatibleClassifierError, FileNotFoundError):
                print(f"Training DTree model: {run_name_dt}")
                model.fit(train_data, val_data, metric=metric, save=True)
            
            outputs, labels = model.predict(test_data, device=device)
            accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
            
            for i in trange(n_classes, desc=f"{method} explanations (seed {seed})", leave=False):
                explanation = model.get_global_explanation(i, concept_names)
                exp_accuracy, exp_predictions = test_explanation(explanation, i, x_test, y_test, metric=expl_metric,
                                                                 concept_names=concept_names, inequalities=True)
                exp_fidelity = 1.0 # DTree rules are the model
                explanation_complexity = complexity(explanation)
                explanations_for_method[i] = explanation; exp_accuracies[i] = exp_accuracy
                exp_fidelities[i] = exp_fidelity; exp_complexities[i] = explanation_complexity


        elif method == 'General':
            l1_weight = 1e-4; hidden_neurons = [20]; fan_in_mu = 10 # Example params
            model = XMuNN(n_classes=n_classes, n_features=n_features, hidden_neurons=hidden_neurons,
                               loss=loss, name=run_name, l1_weight=l1_weight, fan_in=fan_in_mu)
            try: model.load(device); print(f"Model {run_name} loaded.")
            except (ClassifierNotTrainedError, IncompatibleClassifierError, FileNotFoundError):
                print(f"Training General model: {run_name}")
                model.fit(train_data, val_data, epochs=epochs, l_r=l_r, metric=metric,
                          lr_scheduler=lr_scheduler, device=device, save=True, verbose=False) # verbose False for FL speed
            outputs, labels = model.predict(test_data, device=device)
            accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
            for i in trange(n_classes, desc=f"{method} explanations (seed {seed})", leave=False):
                explanation = model.get_global_explanation(x_train, y_train, i, top_k_explanations=top_k_explanations,
                                                           concept_names=concept_names, simplify=simplify,
                                                           metric=expl_metric, x_val=x_val, y_val=y_val)
                exp_accuracy, exp_predictions = test_explanation(explanation, i, x_test, y_test, metric=expl_metric, concept_names=concept_names)
                class_output = (outputs.argmax(dim=1) == i) if outputs is not None else torch.zeros_like(y_test, dtype=torch.bool)
                exp_fidelity = fidelity(torch.as_tensor(exp_predictions), class_output.to(exp_predictions.device), expl_metric) if exp_predictions is not None else 0.0
                explanation_complexity = complexity(explanation)
                explanations_for_method[i] = explanation; exp_accuracies[i] = exp_accuracy
                exp_fidelities[i] = exp_fidelity; exp_complexities[i] = explanation_complexity
        
        elif method == 'RandomForest':
            model = XRandomForestClassifier(name=run_name, n_classes=n_classes, n_features=n_features, random_state=seed)
            try: model.load(device); print(f"Model {run_name} loaded.")
            except (ClassifierNotTrainedError, IncompatibleClassifierError, FileNotFoundError):
                print(f"Training RandomForest model: {run_name}")

                try:
                    model.fit(train_data, val_data, metric=metric, save=True) # LENS wrapper handles it
                except: # Fallback if it expects raw X,y
                     model.fit(x_train.numpy(), y_train.numpy(), val_data=(x_val.numpy(), y_val.numpy()), metric=metric, save=True)
            accuracy = model.evaluate(test_data, metric=metric) # LENS wrapper

            explanations_for_method = ["RF: Global explanation not standard"]*n_classes; exp_accuracies = [0.0]*n_classes
            exp_fidelities = [0.0]*n_classes; exp_complexities = [n_features]*n_classes # Placeholder complexity

        # ... (Add similar blocks for Psi, Relu, BRL, ensuring they use the correct data formats)
        else:
            print(f"Method {method} not fully implemented in this condensed example. Skipping training details.")
            # continue # Skip to next method if not implemented here

        elapsed_time = time.time() - start_time
        if model and model.time is None : model.time = elapsed_time # Save time if model has attribute
        if model and hasattr(model, 'save') and callable(model.save): model.save(device) # Ensure model is saved

        methods_log.append(method)
        splits_log.append(seed)
        # For multi-class, log the first explanation or a summary
        model_explanations_log.append(explanations_for_method[0] if explanations_for_method else "") 
        model_accuracies_log.append(accuracy)
        elapsed_times_log.append(elapsed_time)
        explanation_accuracies_log.append(np.mean(exp_accuracies) if exp_accuracies else 0.0)
        explanation_fidelities_log.append(np.mean(exp_fidelities) if exp_fidelities else 0.0)
        explanation_complexities_log.append(np.mean(exp_complexities) if exp_complexities else 0.0)
        
        print(f"Method {method}, Seed {seed}: TestAcc={accuracy:.4f}, Time={elapsed_time:.2f}s, "
              f"AvgExpAcc={np.mean(exp_accuracies):.4f}, AvgExpFid={np.mean(exp_fidelities):.4f}, AvgExpComp={np.mean(exp_complexities):.2f}")

    # After all seeds for a method
    if methods_log: # If any seed ran
        explanation_consistency_method = formula_consistency(model_explanations_log) # Consistency across seeds for first class's rule
        print(f'Consistency of explanations for {method} (class 0 rules across seeds): {explanation_consistency_method:.4f}')

        method_results_df = pd.DataFrame({
            'method': methods_log, 'split': splits_log, 'explanation_class0': model_explanations_log,
            'model_accuracy': model_accuracies_log, 'explanation_accuracy_avg': explanation_accuracies_log,
            'explanation_fidelity_avg': explanation_fidelities_log, 'explanation_complexity_avg': explanation_complexities_log,
            'explanation_consistency_class0_rules': [explanation_consistency_method] * len(seeds),
            'elapsed_time': elapsed_times_log,
        })
        method_results_df.to_csv(os.path.join(results_dir, f'results_{method}_concept_type_{concept_type_for_filename}.csv'), index=False)
        all_methods_results_summary.append(method_results_df) # Append DataFrame to list

# After all methods
if all_methods_results_summary:
    full_results_df = pd.concat(all_methods_results_summary, ignore_index=True)
    full_results_df.to_csv(os.path.join(results_dir, f'RESULTS_ALL_METHODS_concept_type_{concept_type_for_filename}.csv'), index=False)
    
    # Create summary (mean and sem over seeds for each method)
    summary_list = []
    for method_df_iter in all_methods_results_summary:
        if not method_df_iter.empty:
            method_name_iter = method_df_iter['method'].iloc[0]
            cols_to_summarize = ['model_accuracy', 'explanation_accuracy_avg', 'explanation_fidelity_avg', 
                                 'explanation_complexity_avg', 'elapsed_time', 'explanation_consistency_class0_rules']
            mean_series = method_df_iter[cols_to_summarize].mean()
            sem_series = method_df_iter[cols_to_summarize].sem()
            summary_entry = {'method': method_name_iter}
            for col in cols_to_summarize:
                summary_entry[f'{col}_mean'] = mean_series[col]
                summary_entry[f'{col}_sem'] = sem_series[col]
            summary_list.append(summary_entry)
    
    if summary_list:
        summary_df_final = pd.DataFrame(summary_list)
        summary_df_final.to_csv(os.path.join(results_dir, f'SUMMARY_ALL_METHODS_concept_type_{concept_type_for_filename}.csv'), index=False)
        print("\n--- Overall Summary ---")
        print(summary_df_final)

print(f"\nLENS pipeline for SUNRGBD 3-Class GT Concepts finished. Results in {results_dir}")

