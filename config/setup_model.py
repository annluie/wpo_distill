# setup for neural network

###################
# import dependencies
###################
# ------------------- ENV & PATH SETUP -------------------
import sys
import os

# ------------------- TIME & LOGGING -------------------
import logging
# ------------------- PYTORCH -------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
# ------------------- PROJECT MODULES -------------------
from utilities.plots import *
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import toy_data
#from WPO_SGM import function_cpu as LearnCholesky

###################
# setup functions
###################
# ------------------- MODEL -------------------
## Cholesky factor model
def construct_factor_model(dim:int, depth:int, hidden_units:int):
    '''
    Initializes neural network that models the Cholesky factor of the precision matrix # For nD examples (in theory)
    '''
    chain = []
    chain.append(nn.Linear(dim,int(hidden_units),bias =True)) 
    chain.append(nn.GELU())


    for _ in range(depth-1):
        chain.append(nn.Linear(int(hidden_units),int(hidden_units),bias = True))
        chain.append(nn.GELU())
    
    # Final layer - this is crucial for stability
    final_layer = nn.Linear(hidden_units, int(dim*(dim+1)/2), bias=True)
    
    # Initialize final layer to produce identity-like matrices
    with torch.no_grad():
        final_layer.weight.data.fill_(0.0)
        final_layer.bias.data.fill_(0.0)
        
        diagonal_indices = []
        k = 0
        for i in range(dim):
            for j in range(i + 1):  # i >= j for lower triangle
                if i == j:
                    diagonal_indices.append(k)
                k += 1
        final_layer.bias.data[diagonal_indices] = 0.1
    chain.append(final_layer)
    
    #chain.append(nn.Linear(int(hidden_units),int(dim*(dim+1)/2),bias = True)) 
    return nn.Sequential(*chain)

def setup_devices(self):
        """Setup and validate GPU devices for model parallel training."""
        if not torch.cuda.is_available():
            print("❌ ERROR: Model parallel training requires CUDA")
            sys.exit(1)
            
        devices = list(range(torch.cuda.device_count()))
        if len(devices) < 2:
            print("⚠️ Warning: Model parallel training works best with multiple GPUs")
        
        print(f"✅ Using {len(devices)} GPUs: {devices}")
        return devices

# ------------------- OPTIMIZER/SCHEDULER -------------------
def setup_optimizer_scheduler(args,model_params):
        """Setup optimizer and learning rate scheduler."""
        optimizer = optim.AdamW(
            model_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        scheduler = create_scheduler(args, optimizer)
        return optimizer, scheduler

def create_scheduler(args, optimizer):
        """Create appropriate learning rate scheduler."""
        scheduler_type = args.scheduler_type
        
        if scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5,
                threshold=1e-4, cooldown=2, min_lr=args.lr * 1e-3
            )
        elif scheduler_type == 'cosine_annealing':
            return CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2, eta_min=args.lr * 1e-4
            )
        elif scheduler_type == 'one_cycle':
            return OneCycleLR(
                optimizer, max_lr=args.lr, total_steps=args.total_steps,
                pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
                base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                final_div_factor=1e4
            )
        else:
            return optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
