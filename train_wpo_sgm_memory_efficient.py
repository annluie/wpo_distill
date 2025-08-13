## Memory-efficient version of WPO-SGM training

import os
import sys
import argparse
import time
import gc
import logging
from tqdm import trange
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR

# Import your existing modules
from utilities.plots import *
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import toy_data
import config.load as load
from train_wpo_sgm_model_parallel import (
    setup_model_parallel_training, 
    setup_chunked_training, 
    setup_gradient_accumulation_training,
    train_with_model_parallel,
    train_with_chunked_approach,
    train_with_gradient_accumulation
)

def get_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

def estimate_memory_requirements(num_centers, dim, batch_size, model_params_count, chunk_size=100):
    """
    Estimate GPU memory usage (in GB) for training step based on input sizes.
    Assumes float32 tensors.
    
    Parameters:
    - num_centers: total number of centers (N_centers)
    - dim: feature dimension (D)
    - batch_size: number of samples in batch
    - model_params_count: total number of model parameters
    - chunk_size: chunk size used in precision matrix computation
    
    Returns:
    - total estimated memory in GB
    """
    bytes_per_float32 = 4
    gb_conversion = 1024 ** 3
    
    # Input tensors (already on device)
    centers_memory = num_centers * dim * bytes_per_float32 / gb_conversion
    samples_memory = batch_size * dim * bytes_per_float32 / gb_conversion
    
    # Model parameters + gradients
    model_memory = model_params_count * bytes_per_float32 / gb_conversion
    gradients_memory = model_memory  # approx same size
    
    # Precision matrix peak memory per chunk:
    # Each chunk allocates L and C: 2 * chunk_size * dim^2 floats
    precision_chunk_memory = 2 * chunk_size * dim * dim * bytes_per_float32 / gb_conversion
    
    # Intermediate computations for batch (gradient/laplacian)
    # Roughly batch_size * dim floats for gradient_eval_log + some extra
    intermediate_batch_memory = batch_size * dim * bytes_per_float32 / gb_conversion * 3  # factor 3 for other intermediates
    
    # Total forward memory (excluding model) is dominated by:
    # - precision_chunk_memory (peak at chunk computation)
    # - intermediate batch computations
    forward_memory = precision_chunk_memory + intermediate_batch_memory + centers_memory + samples_memory
    
    # Total memory including model params and gradients
    total_memory_no_overhead = forward_memory + model_memory + gradients_memory
    
    # Factor in autograd overhead and fragmentation (e.g., 2.5x)
    safety_factor = 2.5
    total_memory = total_memory_no_overhead * safety_factor
    
    print(f"Memory estimates (GB):")
    print(f"  Centers tensor:           {centers_memory:.3f}")
    print(f"  Samples tensor:           {samples_memory:.3f}")
    print(f"  Precision matrices (chunk): {precision_chunk_memory:.3f} ⚠️ MAIN BOTTLENECK")
    print(f"  Batch intermediates:      {intermediate_batch_memory:.3f}")
    print(f"  Model parameters:         {model_memory:.3f}")
    print(f"  Gradients (same size):    {gradients_memory:.3f}")
    print(f"  Forward pass total (excl model): {forward_memory:.3f}")
    print(f"  Total w/ backward & overhead (x{safety_factor}): {total_memory:.3f}")
    
    return total_memory

def choose_training_strategy(num_centers, available_gpus, memory_per_gpu_gb, data_dim):
    """
    Automatically choose the best training strategy based on resources
    """
    # Use actual data dimension instead of hardcoded value
    
    # Estimate memory requirements
    total_memory_needed = estimate_memory_requirements(num_centers, data_dim, 8, 307021632)
    
    print(f"\nAvailable resources:")
    print(f"  GPUs: {available_gpus}")
    print(f"  Memory per GPU: {memory_per_gpu_gb:.1f} GB")
    print(f"  Total GPU memory: {available_gpus * memory_per_gpu_gb:.1f} GB")
    
    if total_memory_needed <= memory_per_gpu_gb * 0.8:  # Single GPU with 80% utilization
        print("\n✅ Recommendation: Use single GPU with standard training")
        return "single_gpu", {}
    
    elif available_gpus > 1 and total_memory_needed <= available_gpus * memory_per_gpu_gb * 0.6:
        print("\n✅ Recommendation: Use model parallel training")
        return "model_parallel", {"device_ids": list(range(available_gpus))}
    
    elif total_memory_needed <= memory_per_gpu_gb * 1.5:  # Can fit with chunking
        chunk_size = max(10, int(num_centers * memory_per_gpu_gb * 0.6 / total_memory_needed))
        print(f"\n✅ Recommendation: Use chunked training with chunk_size={chunk_size}")
        return "chunked", {"chunk_size": chunk_size}
    
    else:  # Need gradient accumulation
        accumulation_steps = max(2, int(total_memory_needed / (memory_per_gpu_gb * 0.6)))
        print(f"\n✅ Recommendation: Use gradient accumulation with {accumulation_steps} steps")
        return "gradient_accumulation", {"accumulation_steps": accumulation_steps}

def setup_training_strategy(strategy, factornet, centers, training_samples, args, **kwargs):
    """
    Setup the chosen training strategy
    """
    if strategy == "single_gpu":
        print("Setting up standard single GPU training...")
        if torch.cuda.device_count() > 1:
            factornet = nn.DataParallel(factornet)
        
        optimizer = optim.AdamW(
            factornet.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, 'weight_decay', 1e-4)
        )
        
        return factornet, optimizer, None
    
    elif strategy == "model_parallel":
        print("Setting up model parallel training...")
        device_ids = kwargs.get("device_ids", [0, 1])
        model_parallel, optimizer = setup_model_parallel_training(
            factornet, centers, device_ids, args
        )
        return model_parallel, optimizer, None
    
    elif strategy == "chunked":
        print("Setting up chunked training...")
        chunk_size = kwargs.get("chunk_size", 50)
        chunked_trainer = setup_chunked_training(factornet, centers, chunk_size)
        
        optimizer = optim.AdamW(
            chunked_trainer.factornet.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, 'weight_decay', 1e-4)
        )
        
        return chunked_trainer, optimizer, None
    
    elif strategy == "gradient_accumulation":
        print("Setting up gradient accumulation training...")
        accumulation_steps = kwargs.get("accumulation_steps", 4)
        grad_accum_trainer = setup_gradient_accumulation_training(
            factornet, centers, accumulation_steps
        )
        
        optimizer = optim.AdamW(
            grad_accum_trainer.factornet.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, 'weight_decay', 1e-4)
        )
        
        return grad_accum_trainer, optimizer, None
    
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")

def run_training_loop(strategy, model_or_trainer, training_samples, optimizer, scheduler, args, centers=None):
    """
    Run the appropriate training loop based on strategy
    """
    print(f"\nStarting training with {strategy} strategy...")
    
    if strategy == "single_gpu":
        # Standard training loop
        model_or_trainer.train()
        
        for step in trange(args.niters, desc="Training"):
            # Sample batch
            randind = torch.randint(0, training_samples.shape[0], [args.batch_size])
            samples = training_samples[randind, :]
            
            # Forward pass
            optimizer.zero_grad()
            loss = LearnCholesky.score_implicit_matching_stable(
                model_or_trainer, samples, centers, args.stability
            )
            loss.backward()
            
            # Optimization
            torch.nn.utils.clip_grad_norm_(model_or_trainer.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")
                allocated, reserved = get_memory_info()
                print(f"Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Memory cleanup
            del samples, loss
            if step % 50 == 0:
                torch.cuda.empty_cache()
    
    elif strategy == "model_parallel":
        train_with_model_parallel(model_or_trainer, training_samples, optimizer, scheduler, args)
    
    elif strategy == "chunked":
        train_with_chunked_approach(model_or_trainer, training_samples, optimizer, scheduler, args)
    
    elif strategy == "gradient_accumulation":
        train_with_gradient_accumulation(model_or_trainer, training_samples, optimizer, scheduler, args)

def main():
    """
    Main training function with automatic strategy selection
    """
    # Parse arguments (reuse from original script)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','swissroll_6D_xy1', 'cifar10'], type=str, default='cifar10')
    parser.add_argument('--depth', help='number of hidden layers of score network', type=int, default=5)
    parser.add_argument('--hiddenunits', help='number of nodes per hidden layer', type=int, default=64)
    parser.add_argument('--niters', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save', type=str, default='memory_efficient_experiments/')
    parser.add_argument('--train_kernel_size', type=int, default=50)
    parser.add_argument('--train_samples_size', type=int, default=500)
    parser.add_argument('--test_samples_size', type=int, default=5)
    parser.add_argument('--stability', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler_type', type=str, default='one_cycle')
    parser.add_argument('--strategy', type=str, default='auto', 
                       choices=['auto', 'single_gpu', 'model_parallel', 'chunked', 'gradient_accumulation'],
                       help='Training strategy to use')
    parser.add_argument('--chunk_size', type=int, default=None, help='Chunk size for chunked training')
    parser.add_argument('--accumulation_steps', type=int, default=None, help='Steps for gradient accumulation')
    
    args = parser.parse_args()
    
    # Setup device and check resources
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device('cuda:0')
        memory_per_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"Found {num_gpus} GPUs with {memory_per_gpu:.1f} GB memory each")
    else:
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
        num_gpus = 0
        memory_per_gpu = 8.0  # Assume 8GB RAM for CPU
    
    # Initialize data
    print(f"Initializing data: {args.data}")
    means = toy_data.inf_train_gen(args.data, batch_size=args.train_kernel_size)
    if not torch.is_tensor(means):
        means = torch.tensor(means, dtype=torch.float32, device=device)
    else:
        means = means.clone().detach().to(dtype=torch.float32, device=device)
    data_dim = means.shape[1]
    del means
    torch.cuda.empty_cache()
    
    # Generate training samples and centers
    p_samples = toy_data.inf_train_gen(args.data, batch_size=args.train_samples_size)
    if not torch.is_tensor(p_samples):
        training_samples = torch.tensor(p_samples, dtype=torch.float32, device=device)
    else:
        training_samples = p_samples.clone().detach().to(dtype=torch.float32, device=device)
    centers = training_samples[:args.train_kernel_size]
    del p_samples
    
    print(f"Data dimension: {data_dim}")
    print(f"Number of centers: {args.train_kernel_size}")
    print(f"Training samples: {args.train_samples_size}")
    
    # Choose training strategy
    if args.strategy == 'auto':
        strategy, strategy_kwargs = choose_training_strategy(
            args.train_kernel_size, num_gpus, memory_per_gpu, data_dim
        )
    else:
        strategy = args.strategy
        strategy_kwargs = {}
        if args.chunk_size:
            strategy_kwargs['chunk_size'] = args.chunk_size
        if args.accumulation_steps:
            strategy_kwargs['accumulation_steps'] = args.accumulation_steps
        if strategy == 'model_parallel':
            strategy_kwargs['device_ids'] = list(range(min(num_gpus, 4)))  # Use up to 4 GPUs
    
    # Initialize model (copied from train_wpo_sgm_stable to avoid import issues)
    def construct_factor_model(dim: int, depth: int, hidden_units: int):
        '''
        Initializes neural network that models the Cholesky factor of the precision matrix
        '''
        chain = []
        chain.append(nn.Linear(dim, int(hidden_units), bias=True)) 
        chain.append(nn.GELU())

        for _ in range(depth-1):
            chain.append(nn.Linear(int(hidden_units), int(hidden_units), bias=True))
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
        
        return nn.Sequential(*chain)
    
    factornet = construct_factor_model(data_dim, args.depth, args.hiddenunits).to(device).to(dtype=torch.float32)
    factornet.dim = data_dim
    factornet.depth = args.depth
    factornet.hidden_units = args.hiddenunits
    
    # Setup training strategy
    model_or_trainer, optimizer, scheduler = setup_training_strategy(
        strategy, factornet, centers, training_samples, args, **strategy_kwargs
    )
    
    # Setup scheduler
    if args.scheduler_type == 'one_cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.niters,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    elif args.scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Create save directory
    save_dir = os.path.join(args.save, f"strategy_{strategy}", f"centers_{args.train_kernel_size}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(save_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    print(f"Saving results to: {save_dir}")
    
    # Run training
    try:
        run_training_loop(
            strategy, model_or_trainer, training_samples, 
            optimizer, scheduler, args, centers
        )
        print("✅ Training completed successfully!")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ Out of memory error: {e}")
            print("💡 Try reducing --train_kernel_size or --batch_size")
            print("💡 Or use --strategy chunked or --strategy gradient_accumulation")
        else:
            print(f"❌ Training failed: {e}")
            raise e
    
    # Save final model
    if hasattr(model_or_trainer, 'factornet'):
        # For chunked or gradient accumulation trainers
        model_to_save = model_or_trainer.factornet
    elif hasattr(model_or_trainer, 'factornets') and hasattr(model_or_trainer.factornets, '__getitem__'):
        # For model parallel trainer
        model_to_save = model_or_trainer.factornets[0]  # Save first device model
    else:
        # For standard single GPU training
        model_to_save = model_or_trainer
    
    # Ensure we have a proper model with state_dict
    if hasattr(model_to_save, 'state_dict'):
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    else:
        print("Warning: Could not save model - no state_dict found")
    torch.save(centers, os.path.join(save_dir, 'centers.pth'))
    
    print(f"Model and centers saved to {save_dir}")

if __name__ == "__main__":
    main()
