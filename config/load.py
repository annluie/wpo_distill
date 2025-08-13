# loading functions

###################
# import dependencies
###################
# ------------------- ENV & PATH SETUP -------------------
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:32'
import argparse
# ------------------- TIME & LOGGING -------------------
import logging
# ------------------- PYTORCH -------------------
import torch
# ------------------- PROJECT MODULES -------------------
from utilities.plots import *
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import toy_data
#from WPO_SGM import function_cpu as LearnCholesky

###################
# loading functions
###################
# argument parser
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WPO-SGM Model Parallel Training")
    
    # Data and model parameters
    parser.add_argument('--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 
                                          'moons', '2spirals', 'checkerboard', 'rings',
                                          'swissroll_6D_xy1', 'cifar10'], 
                       default='cifar10', help='Dataset to use')
    parser.add_argument('--depth', type=int, default=5, 
                       help='Number of hidden layers in score network')
    parser.add_argument('--hiddenunits', type=int, default=64, 
                       help='Number of nodes per hidden layer')
    
    # Training parameters
    parser.add_argument('--niters', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--stability', type=float, default=0.01, help='Stability parameter')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_type', type=str, default='one_cycle',
                       choices=['reduce_on_plateau', 'cosine_annealing', 'one_cycle', 'step'],
                       help='Type of LR scheduler to use')
    
    # Data sizes
    parser.add_argument('--train_kernel_size', type=int, default=50, help='Number of centers')
    parser.add_argument('--train_samples_size', type=int, default=500, help='Training sample size')
    parser.add_argument('--test_samples_size', type=int, default=5, help='Test sample size')
    
    # I/O parameters
    parser.add_argument('--save', type=str, default='cifar10_experiments/', 
                       help='Save directory')
    parser.add_argument('--load_model_path', type=str, default=None, 
                       help='Path to load model checkpoint')
    parser.add_argument('--load_centers_path', type=str, default=None, 
                       help='Path to load centers')
    
    return parser.parse_args()

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
   
def load_model_parallel_checkpoint(checkpoint_path, model_parallel, optimizer=None, scheduler=None):
    """
    Load checkpoint for model parallel training
    """
    if not os.path.exists(checkpoint_path):
        return None, None, None, 0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state dicts for each device
        if 'model_state_dicts' in checkpoint:
            model_state_dicts = checkpoint['model_state_dicts']
            device_ids = checkpoint.get('device_ids', list(range(len(model_state_dicts))))
            
            # Ensure we have the right number of models
            if len(model_state_dicts) != len(model_parallel.factornets):
                print(f"Warning: Checkpoint has {len(model_state_dicts)} models, but current setup has {len(model_parallel.factornets)}")
                return None, None, None, 0
            
            # Load state dict for each device model
            for i, (factornet, state_dict) in enumerate(zip(model_parallel.factornets, model_state_dicts)):
                factornet.load_state_dict(state_dict)
                print(f"✅ Loaded model state for device {device_ids[i]}")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Loaded optimizer state")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ Loaded scheduler state")
        
        step = checkpoint.get('step', 0)
        print(f"✅ Loaded checkpoint from step {step}")
        
        return model_parallel, optimizer, scheduler, step
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None, None, None, 0

def load_latest_step(save_dir):
    """Load the latest step number"""
    step_file = os.path.join(save_dir, "latest_step.txt")
    if os.path.exists(step_file):
        with open(step_file, "r") as f:
            step = int(f.read().strip())
        print(f"Resuming from step {step}")
        return step
    return 0
