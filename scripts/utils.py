import logging
import os
import json
import hashlib
from datetime import datetime
import yaml 
import sys
import torch 
from collections import OrderedDict 

def setup_logging(log_dir="logs", run_id="run", log_level_str="INFO"):
    """
    Sets up logging to both a file and the console.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{run_id}.log")
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s [%(module)s.%(funcName)s:%(lineno)d]: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout) 
        ]
    )
    return logging.getLogger(f"Run_{run_id}")

def generate_run_id(prefix="exp"):
    """
    Generates a unique run ID based on the current date and time.
    This is useful for logging and tracking experiments.
    """
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def save_config(config_dict, path):
    """
    Saves the configuration dictionary to a YAML file.
    If the directory does not exist, it will be created.
    """
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        logger.info(f"Configuration used for this run saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {path}: {e}")

def load_config(path):
    logger = logging.getLogger(__name__)
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(f"Config file not found: {path}")
    
    config_dict = None
    with open(path, 'r') as f:
        file_content = f.read()
        if not file_content.strip():
            logger.error(f"Config file is empty or contains only whitespace: {path}")
            raise ValueError(f"Config file is empty: {path}")
        try:
            config_dict = yaml.safe_load(file_content)
            logger.info(f"Configuration successfully loaded as YAML from {path}")
        except yaml.YAMLError as yaml_e:
            logger.error(f"Failed to parse config file as YAML: {yaml_e}")
            logger.error(f"Content preview (first 200 chars): '{file_content[:200]}'")
            raise ValueError(f"Could not parse config file as YAML: {path}.") from yaml_e

    if config_dict is None:
        raise ValueError(f"Config dictionary is None after attempting to load {path}")
    return config_dict

def get_static_cache_path(static_cache_base_dir: str, component_group: str, operation_name: str, params_for_hash: dict, file_extension="pkl"):
    """
    Generates a static cache path for a given operation and parameters.
    """
    component_cache_dir = os.path.join(static_cache_base_dir, component_group)
    os.makedirs(component_cache_dir, exist_ok=True)
    try:
        identifier_str = json.dumps(params_for_hash, sort_keys=True, default=str)
    except TypeError as e:
        logging.error(f"Error serializing params_for_hash for caching {operation_name}: {e}. Params: {params_for_hash}")
        identifier_str = json.dumps({"error": "params_not_serializable", "op": operation_name}, sort_keys=True)
    identifier_hash = hashlib.md5(identifier_str.encode()).hexdigest()[:12]
    filename = f"{operation_name}_{identifier_hash}.{file_extension}"
    return os.path.join(component_cache_dir, filename)

class SAM2Filter(logging.Filter):
    """
    Custom logging filter to exclude SAM2 specific messages.
    This is useful to avoid cluttering logs with SAM2-specific messages
    that are not relevant to the AutoCore flow.
    """
    def filter(self, record):
        sam2_patterns = ["For numpy array image, we assume", "Computing image embeddings", "Image embeddings computed"]
        return not any(p in record.getMessage() for p in sam2_patterns)

def average_weights(local_weights_dict: dict):
    """
    Computes the average of the weights from a dictionary of local model weights.
    local_weights_dict: {client_id: state_dict}
    """
    if not local_weights_dict:
        return None
    
    # Get the first state_dict to initialize the average
    avg_weights = OrderedDict()
    first_client_id = list(local_weights_dict.keys())[0]
    
    for key in local_weights_dict[first_client_id].keys():
        # Sum weights from all clients for the current key
        # Ensure tensors are on CPU for aggregation if they were on GPU
        sum_layer_weights = torch.stack([local_weights_dict[cid][key].cpu().float() for cid in local_weights_dict if key in local_weights_dict[cid]], dim=0).sum(dim=0)
        avg_weights[key] = sum_layer_weights / len(local_weights_dict)
        
    return avg_weights

def weighted_weights(local_weights_dict: dict, client_weights_coeff: dict):
    """
    Computes the weighted average of local model weights.
    local_weights_dict: {client_id: state_dict}
    client_weights_coeff: {client_id: coefficient_float}
    """
    if not local_weights_dict or not client_weights_coeff:
        logging.warning("weighted_weights: Empty local_weights_dict or client_weights_coeff. Returning None.")
        return None

    avg_weights = OrderedDict()
    total_weight = sum(client_weights_coeff.get(cid, 0) for cid in local_weights_dict.keys())

    if total_weight == 0:
        logging.warning("weighted_weights: Total weight is zero. Using simple averaging if possible, else None.")
        # Fallback to simple average if total weight is zero but there are weights
        if local_weights_dict:
            return average_weights(local_weights_dict)
        return None

    first_client_id_with_weights = None
    for cid in local_weights_dict:
        if cid in client_weights_coeff and client_weights_coeff[cid] > 0:
            first_client_id_with_weights = cid
            break
    
    if first_client_id_with_weights is None:
        logging.warning("weighted_weights: No clients with positive weights. Using simple average.")
        return average_weights(local_weights_dict)

    for key in local_weights_dict[first_client_id_with_weights].keys():
        weighted_sum_layer = torch.zeros_like(local_weights_dict[first_client_id_with_weights][key].cpu().float())
        actual_total_weight_for_key = 0
        
        for client_id, state_dict in local_weights_dict.items():
            client_coeff = client_weights_coeff.get(client_id, 0)
            if client_coeff > 0 and key in state_dict:
                weighted_sum_layer += state_dict[key].cpu().float() * client_coeff
                actual_total_weight_for_key += client_coeff
        
        if actual_total_weight_for_key > 0:
            avg_weights[key] = weighted_sum_layer / actual_total_weight_for_key
        else: # Should not happen if first_client_id_with_weights was found
            avg_weights[key] = local_weights_dict[first_client_id_with_weights][key].cpu().float() 
            logging.warning(f"weighted_weights: Zero total weight for key {key}, using first client's weights for this key.")
    return avg_weights