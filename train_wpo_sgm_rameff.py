## Pretrain model with DDP (Distributed Data Parallel)

###################
# Setup
###################
import os
import sys
import argparse
import time
import gc
import logging
import math
from tqdm import trange
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.checkpoint as cp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR

# Set CUDA memory configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# FIX 1: Set matplotlib backend before importing any plotting modules
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Project modules
from utilities.plots import *
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import toy_data
import config.load as load

###################
# DDP Setup Functions
###################
def setup_ddp(rank, world_size, backend='nccl'):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def cleanup_ddp():
    """Clean up DDP"""
    dist.destroy_process_group()

###################
# Neural Network Models
###################
class CheckpointedSequential(nn.Sequential):
    def __init__(self, *modules, chunks=2):
        super().__init__(*modules)
        self.chunks = chunks

    def forward(self, x):
        return cp.checkpoint_sequential(self, self.chunks, x)

def construct_factor_model(dim: int, depth: int, hidden_units: int, dropout: float = 0.1, use_checkpointing: bool = False):
    """
    Initializes a neural network that models the Cholesky factor of the precision matrix.
    Includes dropout for regularization to mitigate memorization.
    """
    layers = []
    layers.append(nn.Linear(dim, hidden_units))
    layers.append(nn.GELU())
    layers.append(nn.Dropout(dropout))

    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden_units, hidden_units))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

    final_layer = nn.Linear(hidden_units, int(dim * (dim + 1) / 2), bias=True)

    # Initialize final layer to produce near-identity precision matrices
    with torch.no_grad():
        final_layer.weight.fill_(0.0)
        final_layer.bias.fill_(0.0)
        diag_indices = []
        k = 0
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    diag_indices.append(k)
                k += 1
        final_layer.bias[diag_indices] = 0.1

    layers.append(final_layer)

    if use_checkpointing:
        return CheckpointedSequential(*layers, chunks=2)
    else:
        return nn.Sequential(*layers)

###################
# Model Loading Functions
###################
def load_model_weights(model, load_path, device):
    """Load model weights and handle common state dict prefixes"""
    if load_path is None or not os.path.exists(load_path):
        return model
    
    state_dict = torch.load(load_path, map_location=device)
    
    # Handle common prefixes
    prefixes_to_remove = ["module.", "_orig_mod."]
    for prefix in prefixes_to_remove:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace(prefix, "")] = v
            state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    if dist.get_rank() == 0:
        logging.info(f"Loaded model weights from {load_path}")
    
    return model

def setup_optimizer_and_scheduler(model, args, total_steps=None):
    """Setup optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 1e-4),
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler_type = getattr(args, 'scheduler_type', 'reduce_on_plateau')
    
    if scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=2,
            min_lr=args.lr * 1e-3
        )
    elif scheduler_type == 'cosine_annealing':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=50,
            T_mult=2,
            eta_min=args.lr * 1e-4
        )
    elif scheduler_type == 'one_cycle' and total_steps:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1e4
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=500,
            gamma=0.5
        )
    
    return optimizer, scheduler

###################
# Training Functions
###################
def evaluate_model(factornet, kernel_centers, num_test_sample, num_batches=2, stab=0.001, dataset='cifar10'):
    """Evaluate the model by computing average loss over test batches"""
    device = kernel_centers.device
    total_loss_sum = 0.0

    factornet.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            p_samples = toy_data.inf_train_gen(dataset, batch_size=num_test_sample)
            if not torch.is_tensor(p_samples):
                p_samples = torch.tensor(p_samples, device=device, dtype=torch.float32)
            else:
                p_samples = p_samples.to(device=device, dtype=torch.float32, non_blocking=True)

            loss = LearnCholesky.score_implicit_matching_stable(
                factornet, p_samples, kernel_centers, stab
            )
            total_loss_sum += loss.item()

    return total_loss_sum / num_batches

def compute_loss_and_backward(factornet, samples, centers, optimizer, scheduler=None,
                              scheduler_type='one_cycle', stab=1e-6, accumulation_steps=1, do_step=True):
    """Compute loss and perform backward pass with gradient accumulation"""
    loss = LearnCholesky.score_implicit_matching_stable(factornet, samples, centers, stab)
    
    if torch.isnan(loss) or torch.isinf(loss) or not loss.requires_grad:
        if dist.get_rank() == 0:
            print(f"⚠️ Loss issue detected: {loss}")
        return loss
    
    if loss.requires_grad and loss.grad_fn is not None:
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        
        if do_step:
            torch.nn.utils.clip_grad_norm_(factornet.parameters(), max_norm=100.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None and scheduler_type != 'reduce_on_plateau':
                scheduler.step()
    else:
        if dist.get_rank() == 0:
            print("❌ Gradient flow broken!")
        
    return loss

def check_model_gradients(model):
    """Check if model has trainable parameters"""
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            if dist.get_rank() == 0:
                print(f"Non-trainable parameter: {name}, shape: {param.shape}")
    
    if dist.get_rank() == 0:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    return trainable_params > 0

def print_memory_stats(rank, world_size, step):
    """Print memory usage statistics for all GPUs"""
    allocated = torch.cuda.memory_allocated(rank) / 2**30
    reserved = torch.cuda.memory_reserved(rank) / 2**30
    max_allocated = torch.cuda.max_memory_allocated(rank) / 2**30
    
    allocated_tensor = torch.tensor([allocated], device=f'cuda:{rank}')
    reserved_tensor = torch.tensor([reserved], device=f'cuda:{rank}')
    max_allocated_tensor = torch.tensor([max_allocated], device=f'cuda:{rank}')
    
    if rank == 0:
        allocated_list = [torch.zeros_like(allocated_tensor) for _ in range(world_size)]
        reserved_list = [torch.zeros_like(reserved_tensor) for _ in range(world_size)]
        max_allocated_list = [torch.zeros_like(max_allocated_tensor) for _ in range(world_size)]
        
        dist.gather(allocated_tensor, allocated_list, dst=0)
        dist.gather(reserved_tensor, reserved_list, dst=0)
        dist.gather(max_allocated_tensor, max_allocated_list, dst=0)
        
        print(f"\n=== Step {step:04d} GPU Memory Summary ===")
        total_allocated = sum(t.item() for t in allocated_list)
        total_reserved = sum(t.item() for t in reserved_list)
        total_max = sum(t.item() for t in max_allocated_list)
        
        for gpu_id in range(world_size):
            print(f"GPU {gpu_id}: {allocated_list[gpu_id].item():.2f}GB | "
                  f"Reserved: {reserved_list[gpu_id].item():.2f}GB | "
                  f"Max: {max_allocated_list[gpu_id].item():.2f}GB")
        
        print(f"TOTAL: {total_allocated:.2f}GB | {total_reserved:.2f}GB | {total_max:.2f}GB")
        print("=" * 50)
    else:
        dist.gather(allocated_tensor, dst=0)
        dist.gather(reserved_tensor, dst=0)
        dist.gather(max_allocated_tensor, dst=0)

###################
# Save/Load Functions
###################
def create_save_dir(save, args):
    """Create a subfolder to save all the outputs"""
    if save is not None:
        subfolder = os.path.join(
            save,
            f"sample_size{args.train_samples_size}",
            f"centers{args.train_kernel_size}",
            f"batch_size{args.batch_size}_epochs{args.niters}_dropout{args.dropout}",
            f"lr{args.lr}_scheduler{args.scheduler_type}_stab{args.stability}_stabver_ddp"
        )
    else:
        subfolder = os.path.join(
            f"sample_size{args.train_samples_size}",
            f"centers{args.train_kernel_size}",
            f"batch_size{args.batch_size}_epochs{args.niters}",
            f"lr{args.lr}_scheduler{args.scheduler_type}_stab{args.stability}_stabver_ddp"
        )
    
    os.makedirs(subfolder, exist_ok=True)
    return subfolder
def save_checkpoint(path, model, optimizer, scheduler, step):
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
    }
    torch.save(checkpoint, path)
    if dist.get_rank() == 0:
        logging.info(f"Saved checkpoint at step {step}")

def save_model_weights(factornet, step, save_dir):
    """Save model weights (only on rank 0)"""
    if dist.get_rank() == 0 and save_dir is not None:
        filename = os.path.join(save_dir, f"step{step:05d}_factornet.pth")
        state_dict = factornet.module.state_dict() if hasattr(factornet, 'module') else factornet.state_dict()
        torch.save(state_dict, filename)
        
        # Save latest checkpoint
        latest_path = os.path.join(save_dir, "latest_factornet.pth")
        torch.save(state_dict, latest_path)
        
        logging.info(f"Saved model weights to {filename}")

def log_training_step(step, iter_time, loss_time, loss_value, max_mem, save_dir):
    """Log training step information"""
    if dist.get_rank() == 0 and save_dir is not None:
        logging.info(f"Step {step:04d} | Iter: {iter_time:.4f}s | "
                    f"Loss: {loss_time:.4f}s | Value: {loss_value:.6f} | "
                    f"Max Mem: {max_mem:.2f}GB")

###################
# Main Training Function
###################
def main_worker(rank, world_size, args):
    """Main training function for each process"""
    setup_ddp(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    
    # Training parameters
    # set parameters from args
    train_kernel_size = args.train_kernel_size
    train_samples_size = args.train_samples_size
    test_samples_size = args.test_samples_size
    dataset = args.data 
    epochs = args.niters
    batch_size = args.batch_size // world_size
    lr = args.lr
    hidden_units = args.hiddenunits
    depth = args.depth
    save_directory = args.save
    load_model_path = args.load_model_path
    load_centers_path = args.load_centers_path
    stab = args.stability
    weight_decay = args.weight_decay
    scheduler_type = args.scheduler_type
    accum_steps = 1
    dropout = args.dropout  
    
    # Setup device optimizations
    torch.set_float32_matmul_precision('high')
    LearnCholesky.setup_memory_efficient_settings()
    
    # Create save directory
    if rank == 0:
        save_directory = create_save_dir(save_directory, args)
        print('save_directory',save_directory)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print('Created directory ' + save_directory)

        logging.basicConfig(
            filename=os.path.join(save_directory, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info(f"---------------------------------------------------------------------------------------------")
        logging.info("Training started")
    
    # Broadcast save directory to all processes
    save_dir_list = [save_directory if rank == 0 else None]
    dist.broadcast_object_list(save_dir_list, src=0)
    save_directory = save_dir_list[0]
    
    # Initialize data and model
    data_sample = toy_data.inf_train_gen(dataset, batch_size=train_kernel_size)
    data_dim = data_sample.shape[1]
    
    factornet = construct_factor_model(data_dim, depth, hidden_units, dropout=dropout).to(device, dtype=torch.float32)
    
    # Generate training data
    p_samples = toy_data.inf_train_gen(dataset, batch_size=train_samples_size)
    training_samples = p_samples.clone().detach().to(device, dtype=torch.float32)
    
    # Calculate total steps for scheduler
    steps_per_epoch = max(1, train_samples_size // batch_size)
    total_steps = epochs * steps_per_epoch
    
    # Initialize optimizer and scheduler BEFORE loading checkpoint
    optimizer, scheduler = setup_optimizer_and_scheduler(factornet, args, total_steps)
    start_step = 0
    
    # Load or create checkpoint
    latest_checkpoint_path = os.path.join(save_directory, "latest_checkpoint.pth")
    if args.load_model_path is None and os.path.exists(latest_checkpoint_path):
        args.load_model_path = latest_checkpoint_path
    
    # Load checkpoint if available (this will update optimizer and scheduler states)
    if args.load_model_path and os.path.exists(args.load_model_path):
        try:
            factornet, optimizer, scheduler, start_step = load.load_checkpoint(
                args.load_model_path, factornet, optimizer, scheduler, device=device
            )
            if rank == 0:
                print(f"✅ Loaded checkpoint from {args.load_model_path}")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ Failed to load checkpoint: {e}")
                print("Continuing with fresh optimizer and scheduler...")
    
    # Load or generate centers
    centers_path = args.load_centers_path or os.path.join(save_directory, "latest_centers.pth")
    centers = load.load_centers(centers_path, device=device)
    
    if start_step == 0 or centers is None:
        centers = training_samples[:train_kernel_size]
        if rank == 0:
            torch.save(centers, centers_path)
            plot_and_save_centers(centers, os.path.join(save_directory, "centers.png"))
    
    # Setup DDP
    if world_size > 1:
        factornet = DDP(factornet, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # Adjust scheduler for resumption
    if hasattr(scheduler, '_step_count'):
        scheduler._step_count = start_step
    if hasattr(scheduler, 'last_epoch'):
        scheduler.last_epoch = start_step - 1
    
    # Check model gradients
    if not check_model_gradients(factornet):
        if rank == 0:
            print("ERROR: No trainable parameters found!")
        cleanup_ddp()
        return
    
    # Training loop
    gc.collect()
    torch.cuda.empty_cache()
    
    pbar = trange(start_step, total_steps, desc="Training") if rank == 0 else range(start_step, total_steps)
    
    for step in pbar:
        torch.cuda.reset_peak_memory_stats()
        iter_start = time.time()
        
        # Training step with gradient accumulation
        total_loss = 0.0
        for accum_idx in range(accum_steps):
            randind = torch.randint(0, train_samples_size, [batch_size // accum_steps], device=device)
            samples = training_samples[randind, :]
            
            if accum_idx < accum_steps - 1:
                with factornet.no_sync():
                    loss = compute_loss_and_backward(
                        factornet, samples, centers, optimizer,
                        scheduler=scheduler, scheduler_type=scheduler_type,
                        stab=stab, accumulation_steps=accum_steps, do_step=False
                    )
            else:
                loss = compute_loss_and_backward(
                    factornet, samples, centers, optimizer,
                    scheduler=scheduler, scheduler_type=scheduler_type,
                    stab=stab, accumulation_steps=accum_steps, do_step=True
                )
            
            total_loss += loss.item()
        
        iter_end = time.time()
        iter_time = iter_end - iter_start
        max_mem = torch.cuda.max_memory_allocated(rank) / 2**30
        
        # Check for invalid loss
        if torch.isnan(torch.tensor(total_loss)) or torch.isinf(torch.tensor(total_loss)):
            if rank == 0:
                print(f"Invalid loss at step {step}. Exiting.")
            cleanup_ddp()
            return
        
        # Logging and evaluation
        if step % 100 == 0 and rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}, Loss: {total_loss:.3e}, LR: {current_lr:.2e}")
            
            with open(os.path.join(save_directory, "loss_log.csv"), "a") as f:
                f.write(f"{step},{total_loss},{current_lr}\n")
        
        if step % 200 == 0:
            print_memory_stats(rank, world_size, step)
            
            loss_start = time.time()
            with torch.no_grad():
                eval_loss = evaluate_model(factornet, centers, test_samples_size, stab=stab, dataset=dataset)
            loss_time = time.time() - loss_start
            
            if rank == 0:
                log_training_step(step, iter_time, loss_time, eval_loss, max_mem, save_directory)
                with torch.no_grad():
                    generated = sample_from_model(factornet, centers, sample_number=10, eps=stab)
                    l2 = torch.mean((generated - training_samples[:10])**2).item()
                    print(f"[Step {step}] L2 to training data: {l2:.2f}")
                    logging.info(f"L2 {l2:.2f} | ")
                # Step scheduler for reduce_on_plateau
                if scheduler_type == 'reduce_on_plateau' and not (math.isnan(eval_loss) or math.isinf(eval_loss)):
                    scheduler.step(eval_loss)
                
                # Save checkpoint
                checkpoint_path = os.path.join(save_directory, "latest_checkpoint.pth")
                save_checkpoint(checkpoint_path, factornet, optimizer, scheduler, step)
        
        if step % 200 == 0 and rank == 0:
            with torch.no_grad():
                filename = os.path.join(save_directory, f"step{step:05d}")
                #plot_images_with_model(factornet, centers, plot_number=10, eps=stab, save_path=filename)
                plot_images_with_model_and_nn(factornet, centers, training_samples, plot_number=10, eps=stab, save_path=filename)
        if step % 1000 == 0:
            save_model_weights(factornet, step, save_directory)
        
        # Cleanup
        if step < total_steps - 1:
            del samples, loss
            gc.collect()
            torch.cuda.empty_cache()
    
    # Final evaluation and saving
    if rank == 0:
        final_loss = evaluate_model(factornet, centers, test_samples_size, stab=stab, dataset=dataset)
        save_model_weights(factornet, epochs, save_directory)
        logging.info(f'Final loss: {final_loss:.3e}')
        
        with torch.no_grad():
            plot_images_with_model(factornet, centers, plot_number=10, eps=stab, save_path=save_directory)
    
    cleanup_ddp()
def main():
    """Main function to launch DDP training"""
    parser = argparse.ArgumentParser(description='DDP Training Script')
    parser.add_argument('--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', 
                                          '2spirals', 'checkerboard', 'rings', 'swissroll_6D_xy1', 'cifar10'], 
                       default='cifar10', help='Dataset to use')
    parser.add_argument('--depth', type=int, default=5, help='Number of hidden layers')
    parser.add_argument('--hiddenunits', type=int, default=64, help='Hidden units per layer')
    parser.add_argument('--niters', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save', type=str, default='cifar10_experiments/', help='Save directory')
    parser.add_argument('--train_kernel_size', type=int, default=50, help='Number of kernel centers')
    parser.add_argument('--train_samples_size', type=int, default=500, help='Training sample size')
    parser.add_argument('--test_samples_size', type=int, default=5, help='Test sample size')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load model')
    parser.add_argument('--load_centers_path', type=str, default=None, help='Path to load centers')
    parser.add_argument('--stability', type=float, default=0.01, help='Stability parameter')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--scheduler_type', type=str, default='one_cycle',
                        choices=['reduce_on_plateau', 'cosine_annealing', 'one_cycle', 'step'],
                        help='Type of LR scheduler')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for DDP training")
    
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()