## Pretrain model (DDP version of Example_WPO_SGM.ipynb)

###################
# setup
###################
# ------------------- ENV & PATH SETUP -------------------
import os
import sys
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Add these environment variables for better DDP performance
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
# ------------------- TIME & LOGGING -------------------
import time
import gc
import logging
from tqdm import trange
from memory_profiler import profile

# ------------------- MATH -------------------
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix as pdsm
import matplotlib.pyplot as plt
import math

# ------------------- PYTORCH -------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# ------------------- PROJECT MODULES -------------------
from utilities.plots import *
from utilities.plots import *
#from WPO_SGM import functions_WPO_SGM as LearnCholesky
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import toy_data
###################
# DDP functions
###################
def setup_ddp(rank, world_size, backend='nccl'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes for averaging."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

###################
# functions
###################
#----------------------- NEURAL NETWORK -------------------
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
    
    return nn.Sequential(*chain)

def load_model(model, centers, load_model_path, load_centers_path, device):   
    """
    Loads model weights from the specified path.
    """
    if load_model_path is not None and os.path.exists(load_model_path):
        state_dict = torch.load(load_model_path, map_location=device)
        
        # Strip various prefixes if present
        prefixes_to_remove = ["module.", "_orig_mod."]
        for prefix in prefixes_to_remove:
            if any(k.startswith(prefix) for k in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k.replace(prefix, "")] = v
                state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        if dist.get_rank() == 0:
            logging.info(f"Loaded model weights from {load_model_path}")
    
    if load_centers_path is not None and os.path.exists(load_centers_path):
        centers = torch.load(load_centers_path, map_location=device)
        if dist.get_rank() == 0:
            logging.info(f"Loaded centers from {load_centers_path}")
    else:
        if dist.get_rank() == 0:
            print(f"No model loaded. Path does not exist: {load_model_path}")
    
    return model, centers

    
#-----------------------  HELPER FUNCTIONS -------------------
def print_memory_usage(rank, step):
    """Print detailed memory usage for current GPU"""
    allocated = torch.cuda.memory_allocated(rank) / 2**30  # GB
    reserved = torch.cuda.memory_reserved(rank) / 2**30   # GB
    max_allocated = torch.cuda.max_memory_allocated(rank) / 2**30  # GB
    
    print(f"GPU {rank} Step {step:04d} | "
          f"Allocated: {allocated:.2f}GB | "
          f"Reserved: {reserved:.2f}GB | "
          f"Max: {max_allocated:.2f}GB")

def gather_and_print_all_gpu_memory(rank, world_size, step):
    """Gather memory stats from all GPUs and print on rank 0"""
    # Get current GPU's memory stats
    allocated = torch.cuda.memory_allocated(rank) / 2**30
    reserved = torch.cuda.memory_reserved(rank) / 2**30
    max_allocated = torch.cuda.max_memory_allocated(rank) / 2**30
    
    # Create tensors for gathering
    allocated_tensor = torch.tensor([allocated], device=f'cuda:{rank}')
    reserved_tensor = torch.tensor([reserved], device=f'cuda:{rank}')
    max_allocated_tensor = torch.tensor([max_allocated], device=f'cuda:{rank}')
    
    # Gather all stats to rank 0
    if rank == 0:
        allocated_list = [torch.zeros_like(allocated_tensor) for _ in range(world_size)]
        reserved_list = [torch.zeros_like(reserved_tensor) for _ in range(world_size)]
        max_allocated_list = [torch.zeros_like(max_allocated_tensor) for _ in range(world_size)]
        
        dist.gather(allocated_tensor, allocated_list, dst=0)
        dist.gather(reserved_tensor, reserved_list, dst=0)
        dist.gather(max_allocated_tensor, max_allocated_list, dst=0)
        
        print(f"\n=== Step {step:04d} GPU Memory Summary ===")
        total_allocated = 0
        total_reserved = 0
        total_max = 0
        
        for gpu_id in range(world_size):
            alloc = allocated_list[gpu_id].item()
            res = reserved_list[gpu_id].item()
            max_alloc = max_allocated_list[gpu_id].item()
            
            print(f"GPU {gpu_id}: Allocated: {alloc:.2f}GB | Reserved: {res:.2f}GB | Max: {max_alloc:.2f}GB")
            
            total_allocated += alloc
            total_reserved += res
            total_max += max_alloc
        
        print(f"TOTAL: Allocated: {total_allocated:.2f}GB | Reserved: {total_reserved:.2f}GB | Max: {total_max:.2f}GB")
        print("=" * 50)
    else:
        dist.gather(allocated_tensor, dst=0)
        dist.gather(reserved_tensor, dst=0)
        dist.gather(max_allocated_tensor, dst=0)

compiled_loss = torch.compile(
        LearnCholesky.score_implicit_matching_stable, 
        LearnCholesky.score_implicit_matching_stable, 
        mode="reduce-overhead"  # Better for training loops
    )
def evaluate_model(factornet, kernel_centers, num_test_sample, dataset, stab, world_size): 
    '''
    Evaluate the model by computing the average total loss over 10 batch of testing samples
    '''
    with torch.no_grad():
        total_loss_sum = 0
        device = kernel_centers.device
        for i in range(10):
            p_samples = toy_data.inf_train_gen(dataset, batch_size=num_test_sample)
            testing_samples = torch.as_tensor(p_samples, dtype=torch.float32, device=device)
            #total_loss = LearnCholesky.score_implicit_matching_stable(factornet, testing_samples, kernel_centers, stab)
            #total_loss = LearnCholesky.score_implicit_matching_ddp_optimized(factornet, testing_samples, kernel_centers, stab)
            total_loss = compiled_loss(factornet, testing_samples, kernel_centers, stab)

            # Reduce loss across all processes
            reduced_loss = reduce_tensor(total_loss, world_size)
            total_loss_sum += reduced_loss.item()
            
            # Free up memory
            del p_samples, testing_samples, total_loss
            gc.collect()
            torch.cuda.empty_cache()
        average_total_loss = total_loss_sum / 10
    return average_total_loss

def opt_check(factornet, samples, centers, optimizer, stab):
    '''
    Optimization function that computes the loss and performs backpropagation using mixed precision
    '''
    optimizer.zero_grad(set_to_none=True)
    #loss = LearnCholesky.score_implicit_matching_stable(factornet, samples, centers, stab)
    loss = LearnCholesky.score_implicit_matching_stable(factornet, samples, centers, stab)
    loss = LearnCholesky.score_implicit_matching_stable(factornet, samples, centers, stab)
    torch.autograd.set_detect_anomaly(True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(factornet.parameters(), max_norm=100.0) #gradient clipping
    optimizer.step()
    return loss

# Pre-generate training data to avoid repeated generation
class DataBuffer:
    def __init__(self, dataset, total_samples, device, world_size, rank):
        self.dataset = dataset
        self.device = device
        self.world_size = world_size
        self.rank = rank
        
        # Pre-generate all training data
        torch.manual_seed(42)  # Consistent across ranks
        all_samples = toy_data.inf_train_gen(dataset, batch_size=total_samples)
        self.training_samples = all_samples.clone().detach().to(dtype=torch.float32, device=device)
        
        # Pre-compute random indices for each epoch to avoid repeated randint calls
        self.precomputed_indices = {}
        
    def get_batch(self, epoch, batch_size):
        if epoch not in self.precomputed_indices:
            torch.manual_seed(epoch * self.world_size + self.rank)
            self.precomputed_indices[epoch] = torch.randint(
                0, len(self.training_samples), [batch_size]
            )
        
        indices = self.precomputed_indices[epoch]
        return self.training_samples[indices]

#----------------------- SAVE FUNCTIONS -------------------
def create_save_dir(save, train_samples_size, test_samples_size, total_batch_size, train_kernel_size, lr, hidden_units, stab):
    '''
    Create a subfolder to save all the outputs
    '''
    if save is not None:
        subfolder = os.path.join(
            save,
            f"sample_size{train_samples_size}",
            f"test_size{test_samples_size}",
            f"batch_size{total_batch_size}",
            f"centers{train_kernel_size}",
            f"lr{lr}_hu{hidden_units}_stab{stab}_ddp"
        )
        os.makedirs(subfolder, exist_ok=True)
    else:
        subfolder = os.path.join(
            f"sample_size{train_samples_size}",
            f"test_size{test_samples_size}",
            f"batch_size{total_batch_size}",
            f"centers{train_kernel_size}",
            f"lr{lr}_hu{hidden_units}_stab{stab}_ddp"
        )
        os.makedirs(subfolder, exist_ok=True)
    return subfolder

def save_training_slice_cov(factornet, means, epoch, save, rank):
    '''
    Save the training slice of the NN in parameter-based subfolders,
    with epoch appended to the filename. Only save from rank 0.
    '''
    if save is not None and rank == 0:
        # Create filename with epoch appended
        filename = os.path.join(save, f"epoch{epoch:04d}_factornet.pth")

        # Save model weights (unwrap DDP wrapper)
        state_dict = factornet.module.state_dict()
        torch.save(state_dict, filename)
        logging.info(f"Saved model checkpoint to {filename}")

def save_training_slice_log(iter_time, loss_time, epoch, max_mem, loss_value, save, rank, world_size):
    '''
    Save the training log for the slice. Only log from rank 0.
    '''
    if save is not None and rank == 0:
        logging.info(f"Training started for epoch {epoch}")
        
        # Log memory for all GPUs
        memory_info = f"Memory per GPU: "
        for gpu_id in range(world_size):
            if gpu_id < torch.cuda.device_count():
                gpu_mem = torch.cuda.max_memory_allocated(gpu_id) / 2**30
                memory_info += f"GPU{gpu_id}:{gpu_mem:.2f}GB "
        
        logging.info(
        f"Step {epoch:04d} | Iter Time: {iter_time:.4f}s | "
        f"Loss Time: {loss_time:.4f}s | Loss: {loss_value:.6f} | "
        f"{memory_info}"
        )   

###################
# Main training function
###################
def main_worker(rank, world_size, args):
    """Main training function for each process."""
    
    # Initialize DDP
    setup_ddp(rank, world_size)
    
    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    
    # ------------------- SET PARAMETERS -------------------
    torch.set_float32_matmul_precision('high')
    LearnCholesky.setup_optimal_device_settings()
    LearnCholesky.setup_optimal_device_settings()
    
    # Extract parameters from args
    train_kernel_size = args.train_kernel_size
    train_samples_size = args.train_samples_size
    test_samples_size = args.test_samples_size
    dataset = args.data 
    epochs = args.niters
    #batch_size = args.batch_size #need to distribute batches over ddp to prevent OOM (DP does this automatic)
    total_batch_size = args.batch_size
    batch_size = args.batch_size // world_size
    lr = args.lr
    hidden_units = args.hiddenunits
    depth = args.depth
    save_directory = args.save
    load_model_path = args.load_model_path
    load_centers_path = args.load_centers_path
    stab = args.stability

    #-------------------- Initialize Data -------------------
    if dataset not in ['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','swissroll_6D_xy1', 'cifar10']:
        dataset = 'cifar10'
    
    # Generate initial data to get dimensions
    temp_data = toy_data.inf_train_gen(dataset, batch_size=train_kernel_size)
    data_dim = temp_data.shape[1]
    del temp_data
    torch.cuda.empty_cache()

    #-------------------- Create Save Directory (only on rank 0) -------------------
    if rank == 0:
        save_directory = create_save_dir(save_directory, train_samples_size, test_samples_size, 
                                       total_batch_size, train_kernel_size, lr, hidden_units, stab)
        print('save_directory', save_directory)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print('Created directory ' + save_directory)

        # Configure the logger
        log_filename = os.path.join(save_directory, 'training.log')
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info(f"---------------------------------------------------------------------------------------------")
    
    # Broadcast save directory to all processes
    if rank == 0:
        save_dir_tensor = torch.tensor([ord(c) for c in save_directory], dtype=torch.long, device=device)
        save_dir_len = torch.tensor([len(save_directory)], dtype=torch.long, device=device)
    else:
        save_dir_len = torch.tensor([0], dtype=torch.long, device=device)
    
    dist.broadcast(save_dir_len, src=0)
    
    if rank != 0:
        save_dir_tensor = torch.zeros(save_dir_len.item(), dtype=torch.long, device=device)
    
    dist.broadcast(save_dir_tensor, src=0)
    
    if rank != 0:
        save_directory = ''.join([chr(c.item()) for c in save_dir_tensor])

    #######################
    # Construct the model
    #######################
    factornet = construct_factor_model(data_dim, depth, hidden_units).to(device).to(dtype=torch.float32)
    
    # Generate centers (same on all processes for consistency)
    torch.manual_seed(42)  # Ensure same centers across all processes
    centers = toy_data.inf_train_gen(dataset, batch_size=train_kernel_size).clone().detach().to(dtype=torch.float32, device=device)
    
    # Load model and centers if specified
    if load_model_path or load_centers_path:
        if rank == 0:
            print("loading model")
        factornet, centers = load_model(factornet, centers, load_model_path, load_centers_path, device)
    
    # Wrap model in DDP
    factornet = DDP(
        factornet, 
        device_ids=[rank],
        find_unused_parameters=False,  # Important optimization
        gradient_as_bucket_view=True,  # Memory optimization
        broadcast_buffers=False,       # Skip if no buffers need syncing
        static_graph=True             # Graph doesn't change
    )
    
    # Optional: compile the model (after DDP wrapping)
    #factornet = torch.compile(factornet, mode="max-autotune")

    #------------------------ Initialize the optimizer -------------------
    optimizer = optim.Adam(factornet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=10.0,
        threshold_mode='abs',
        cooldown=2,
        min_lr=1e-7,
        #verbose=(rank == 0)  # Only verbose on rank 0
    )
    '''
    # Generate training samples (same seed for consistency)
    torch.manual_seed(42)
    p_samples = toy_data.inf_train_gen(dataset, batch_size=train_samples_size)
    training_samples = p_samples.clone().detach().to(dtype=torch.float32, device=device)
    '''
    data_buffer = DataBuffer(dataset, train_samples_size, device, world_size, rank)
    training_samples = data_buffer.get_batch(0, train_samples_size)
    # Save centers (only on rank 0)
    '''
    if rank == 0:
        filename_final = os.path.join(save_directory, 'centers.pt')
        centers_to_save = training_samples[torch.randperm(training_samples.size(0))[:train_kernel_size]]
        torch.save(centers_to_save, filename_final)
        filename_final = os.path.join(save_directory, 'centers.png')
        LearnCholesky.plot_and_save_centers(centers_to_save, filename_final)
    #del p_samples
    '''
    if rank == 0:
        centers_file = os.path.join(save_directory, 'centers.pt')
        torch.save(centers, centers_file)
        filename_final = os.path.join(save_directory, 'centers.png')
        plot_and_save_centers(centers, filename_final)
        plot_and_save_centers(centers, filename_final)
    ###########################
    # Training loop
    ###########################
    gc.collect()
    torch.cuda.empty_cache()
    #compiled_opt_check = torch.compile(opt_check, mode="reduce-overhead")
    compiled_opt_check = opt_check
    # Use trange only on rank 0
    if rank == 0:
        pbar = trange(epochs, desc="Training")
    else:
        pbar = range(epochs)

    for step in pbar:
        torch.cuda.reset_peak_memory_stats()
        
        '''
        # Generate different random indices for each process
        torch.manual_seed(step * world_size + rank)  # Different seed per process
        randind = torch.randint(0, train_samples_size, [batch_size,])
        samples = training_samples[randind, :]
        '''
        # Get batch from buffer (much faster than repeated generation)
        samples = data_buffer.get_batch(step, batch_size)

        iter_start = time.time()
        #loss = opt_check(factornet, samples, centers, optimizer, stab)
        loss = compiled_opt_check(factornet, samples, centers, optimizer, stab)
        loss_value = loss.item()
        
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0:
                print(f"Invalid loss (NaN or Inf) at step {step}. Exiting.")
            cleanup_ddp()
            sys.exit(1)
            
        iter_end = time.time()
        iter_time = iter_end - iter_start
        max_mem = torch.cuda.max_memory_allocated() / 2**30  # in GiB

        if step % 100 == 0:
            # Print memory for each GPU
            print_memory_usage(rank, step)
            
            if rank == 0:
                print(f"Step {step} started")
                with open(os.path.join(save_directory, "loss_log.csv"), "a") as f:
                    f.write(f"{step},{loss_value}\n")
                    
        if step % 500 == 0:
            # Print comprehensive memory summary across all GPUs
            gather_and_print_all_gpu_memory(rank, world_size, step)
                
        if step % 200 == 0:
            loss_start = time.time()
            loss0 = evaluate_model(factornet, centers, test_samples_size, dataset, stab, world_size)
            loss_end = time.time()
            loss_time = loss_end - loss_start
            
            save_training_slice_log(iter_time, loss_time, step, max_mem, loss0, save_directory, rank, world_size)
            
            if not math.isnan(loss0) and not math.isinf(loss0):
                old_lrs = [group['lr'] for group in optimizer.param_groups]
                scheduler.step(loss0)
                new_lrs = [group['lr'] for group in optimizer.param_groups]

                if rank == 0:
                    for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                        if old_lr != new_lr:
                            logging.info(f"LR changed for param group {i} from {old_lr:.2e} to {new_lr:.2e} at step {step}")
            else:
                if rank == 0:
                    print("⚠️ Warning: Skipping LR scheduler step due to invalid val_loss:", loss0)
            
            # Sample and save generated images at intermediate steps (only on rank 0)
            if rank == 0:
                with torch.no_grad():
                    filename_step_sample = os.path.join(save_directory, f"step{step:05d}")
                    plot_images_with_model(factornet, centers, plot_number=10, eps=stab, save_path=filename_step_sample)
                    plot_images_with_model(factornet, centers, plot_number=10, eps=stab, save_path=filename_step_sample)
                    logging.info(f"Saved samples at step {step} to {filename_step_sample}")
                    generated = sample_from_model(factornet, training_samples, sample_number=10, eps=stab)
                    generated = sample_from_model(factornet, training_samples, sample_number=10, eps=stab)
                    l2 = torch.mean((generated - training_samples[:10])**2).item()
                    print(f"[Step {step}] L2 to training data: {l2:.2f}")
                    logging.info(f"L2 {l2:.2f} | ")
                    
        if step % 1000 == 0:
            save_training_slice_cov(factornet, centers, step, save_directory, rank)
            
        if step < epochs - 1:
            del samples
            gc.collect()
            torch.cuda.empty_cache()

    ###############################
    # Evaluate the final model
    ################################
    gc.collect()
    torch.cuda.empty_cache()

    loss0 = evaluate_model(factornet, centers, test_samples_size, dataset, stab, world_size)    
    save_training_slice_cov(factornet, centers, step, save_directory, rank)
    
    if rank == 0:
        formatted_loss = f'{loss0:.3e}'
        logging.info(f'After train, Average total_loss: {formatted_loss}')

        #---------------------------- Sample and save -------------------
        with torch.no_grad():
            plot_images_with_model(factornet, centers, plot_number=10, save_path=save_directory)
            plot_images_with_model(factornet, centers, plot_number=10, save_path=save_directory)
            logging.info(f'Sampled images saved to {save_directory}_sampled_images.png')
    
    # Clean up DDP
    cleanup_ddp()

def main():
    """Main function to setup and launch DDP training."""
    # setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','swissroll_6D_xy1', 'cifar10'], type=str, default='cifar10')
    parser.add_argument('--depth', help='number of hidden layers of score network', type=int, default=5)
    parser.add_argument('--hiddenunits', help='number of nodes per hidden layer', type=int, default=64)
    parser.add_argument('--niters', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-3) 
    parser.add_argument('--save', type=str, default='cifar10_experiments/')
    parser.add_argument('--train_kernel_size', type=int, default=50)
    parser.add_argument('--train_samples_size', type=int, default=500)
    parser.add_argument('--test_samples_size', type=int, default=5)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--load_centers_path', type=str, default=None)
    parser.add_argument('--stability', type=float, default=0.01)
    args = parser.parse_args()

    # Check CUDA availability and get world size
    if not torch.cuda.is_available():
        print("CUDA not available. DDP requires CUDA.")
        sys.exit(1)
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs with DistributedDataParallel")
    
    # Launch training processes
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
