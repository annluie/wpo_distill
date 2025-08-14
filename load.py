# loading functions

###################
# import dependencies
###################
# ------------------- ENV & PATH SETUP -------------------
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:32'

# ------------------- TIME & LOGGING -------------------
import logging
# ------------------- PYTORCH -------------------
import torch
# ------------------- PROJECT MODULES -------------------
from plots import *
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import toy_data
#from WPO_SGM import function_cpu as LearnCholesky

###################
# loading functions
###################

def load_model(model, centers, load_model_path, load_centers_path, device='cpu'):   
    """
    Loads model weights from the specified path.
    """
    if load_model_path is not None and os.path.exists(load_model_path):
        state_dict = torch.load(load_model_path, map_location=device)
        
        # Strip "module." prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            state_dict = new_state_dict
        
        # Strip "_orig_mod." prefix if present
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace("_orig_mod.", "")] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        logging.info(f"Loaded model weights from {load_model_path}")
    
    if load_centers_path is not None and os.path.exists(load_centers_path):
        centers = torch.load(load_centers_path, map_location=device)
        logging.info(f"Loaded centers from {load_centers_path}")
    else:
        print(f"No model loaded. Path does not exist: {load_model_path}")
    
    return model, centers

def load_centers(path, device='cpu'):
    if path is None:
        logging.warning("No path provided for centers loading.")
        return None
    if not os.path.exists(path):
        logging.warning(f"Centers file not found at {path}")
        return None
    try:
        centers = torch.load(path, map_location=device)
        logging.info(f"Loaded centers from {path}")
        return centers
    except Exception as e:
        logging.error(f"Error loading centers from {path}: {e}")
        return None

def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    if path is None or not os.path.exists(path):
        logging.warning(f"Checkpoint path {path} does not exist.")
        return model, optimizer, scheduler, 0

    checkpoint = torch.load(path, map_location=device)

    # Handle possible prefixes like "module." or "_orig_mod."
    state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', {}))
    if any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        state_dict = new_state_dict
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("_orig_mod.", "")] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

    # Load optimizer state if present
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if present
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    elif scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    step = checkpoint.get('step', 0)
    logging.info(f"✅ Loaded checkpoint from {path} at step {step}")
    return model, optimizer, scheduler, step


def load_latest_step(save_dir):
    """Load the latest step number"""
    step_file = os.path.join(save_dir, "latest_step.txt")
    if os.path.exists(step_file):
        with open(step_file, "r") as f:
            step = int(f.read().strip())
        print(f"Resuming from step {step}")
        return step
    return 0
