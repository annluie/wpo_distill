# diagnostic functions

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
from utilities.plots import *
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import toy_data
#from WPO_SGM import function_cpu as LearnCholesky

###################
# functions
###################
#-----------------------checking functions-----------------------
def check_model_gradients(model):
    total_params = 0
    trainable_params = 0
    
    # Handle model parallel case
    if hasattr(model, 'factornets'):
        for i, factornet in enumerate(model.factornets):
            print(f"Checking gradients for device {i}:")
            for name, param in factornet.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                else:
                    print(f"Non-trainable parameter: {name}, shape: {param.shape}")
    else:
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                print(f"Non-trainable parameter: {name}, shape: {param.shape}")
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return trainable_params > 0

#-----------------------Print functions---------------------------
def print_memory_usage(step):
    """Print detailed memory usage for all GPUs"""
    num_gpus = torch.cuda.device_count()
    print(f"Memory usage at step {step}:")
    for device_id in range(num_gpus):
        allocated = torch.cuda.memory_allocated(device_id) / 2**30  # GB
        reserved = torch.cuda.memory_reserved(device_id) / 2**30   # GB
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 2**30  # GB
        print(f"GPU {device_id} Step {step:04d} | "
          f"Allocated: {allocated:.2f}GB | "
          f"Reserved: {reserved:.2f}GB | "
          f"Max: {max_allocated:.2f}GB")

