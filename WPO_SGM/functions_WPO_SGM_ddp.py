#stable version of code optimized for DistributedDataParallel (DDP)
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributions.multivariate_normal import MultivariateNormal
import lib.toy_data as toy_data
import numpy as np
import argparse
from memory_profiler import profile
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch.linalg as LA
import torchvision.utils as vutils
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#----------------------------------------------#
#### DDP SETUP AND UTILITIES ####
#----------------------------------------------#

def setup_ddp(rank, world_size, backend='nccl'):
    """
    Initialize the distributed process group for DDP
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set the GPU for this process
    torch.cuda.set_device(rank)
    
    print(f"DDP initialized for rank {rank}/{world_size}")

def cleanup_ddp():
    """
    Clean up the distributed process group
    """
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size, reduction='mean'):
    """
    Reduce tensor across all processes in DDP
    """
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    
    if reduction == 'mean':
        reduced_tensor /= world_size
    
    return reduced_tensor

def all_gather_tensor(tensor, world_size):
    """
    Gather tensors from all processes
    """
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def sync_across_processes(func):
    """
    Decorator to ensure synchronization across processes before function execution
    """
    def wrapper(*args, **kwargs):
        if dist.is_initialized():
            dist.barrier()
        return func(*args, **kwargs)
    return wrapper

#----------------------------------------------#
#### STABILITY ENHANCEMENT FUNCTIONS ####
#----------------------------------------------#

def adaptive_regularization(matrices, base_epsilon=1e-4, max_cond=1e12):
    """
    Compute adaptive regularization based on condition numbers
    DDP-optimized: reduced memory footprint and better GPU utilization
    """
    try:
        # Use more memory-efficient condition number estimation
        batch_size = matrices.shape[0]
        device = matrices.device
        dtype = matrices.dtype
        
        # Fast condition estimation using eigenvalues of symmetric matrices
        eigenvals = torch.linalg.eigvals(matrices)
        eigenvals_real = torch.real(eigenvals)  # Take real part for numerical stability
        eigenvals_pos = torch.clamp(eigenvals_real, min=1e-12)  # Ensure positivity
        
        cond_estimates = torch.max(eigenvals_pos, dim=-1)[0] / torch.min(eigenvals_pos, dim=-1)[0]
        cond_clamped = torch.clamp(cond_estimates, 1.0, max_cond)
        
        adaptive_eps = base_epsilon * torch.sqrt(cond_clamped / 1e6)
        return adaptive_eps.view(-1, 1, 1)
        
    except:
        # Fallback to base epsilon if condition number computation fails
        batch_size = matrices.shape[0]
        return torch.full((batch_size, 1, 1), base_epsilon, device=matrices.device, dtype=matrices.dtype)

def adaptive_regularization_fast_ddp(matrices, base_epsilon=1e-4, max_cond=1e12):
    """
    DDP-optimized faster adaptive regularization
    Reduced memory allocations and improved GPU memory access patterns
    """
    batch_size = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype
    
    try:
        # More efficient norm computations for DDP
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for stability
            diag_elements = torch.diagonal(matrices, dim1=-2, dim2=-1)
            diag_norm = torch.norm(diag_elements, dim=-1)
            
            # Use view operations instead of reshape for better memory access
            matrices_flat = matrices.view(batch_size, -1)
            frob_norm = torch.norm(matrices_flat, dim=-1)
            
            cond_estimate = torch.clamp(frob_norm / (diag_norm + 1e-8), 1.0, max_cond)
            adaptive_eps = base_epsilon * torch.sqrt(cond_estimate / 1e6)
            
        return adaptive_eps.view(-1, 1, 1)
        
    except:
        return torch.full((batch_size, 1, 1), base_epsilon, device=device, dtype=dtype)

def stable_logdet_ddp(matrices, eps=1e-6, max_attempts=3):
    """
    DDP-optimized log determinant computation with reduced memory usage
    """
    batch_size = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype
    
    # Pre-allocate identity matrix to avoid repeated allocations
    eye = torch.eye(matrices.shape[-1], device=device, dtype=dtype)
    
    for attempt in range(max_attempts):
        current_eps = eps * (10 ** attempt)
        
        try:
            # In-place regularization to save memory
            regularized = matrices + current_eps * eye
            
            # Use Cholesky decomposition for PD matrices (most stable)
            chol = torch.linalg.cholesky(regularized)
            diag_chol = torch.diagonal(chol, dim1=-2, dim2=-1)
            logdet = 2.0 * torch.sum(torch.log(torch.clamp(diag_chol, min=1e-8)), dim=-1)
            
            # Check for valid results
            if torch.all(torch.isfinite(logdet)):
                return logdet
                
        except RuntimeError:
            continue
    
    # Final fallback: SVD-based method with increased regularization
    try:
        regularized = matrices + eps * 100 * eye
        return torch.logdet(regularized)
    except:
        # Ultimate fallback: return safe default values
        return torch.full((batch_size,), -10.0, device=device, dtype=dtype)

def vectors_to_precision_ddp_optimized(vectors, dim, base_epsilon=1e-4):
    """
    DDP-optimized precision matrix computation with memory efficiency
    """
    if torch.isnan(vectors).any():
        print("❌ NaNs in vectors before conversion to precision!")
    
    batch_size = vectors.shape[0]
    device = vectors.device
    dtype = vectors.dtype
    
    # Use memory-efficient indexing
    L = torch.zeros(batch_size, dim, dim, dtype=dtype, device=device)
    
    # Create triangular indices once and reuse
    tril_indices = torch.tril_indices(dim, dim, device=device)
    L[:, tril_indices[0], tril_indices[1]] = vectors
    
    # Vectorized diagonal processing with in-place operations where possible
    diag_mask = torch.eye(dim, dtype=torch.bool, device=device)
    L[:, diag_mask] = F.softplus(L[:, diag_mask]) + 1e-6
    
    # Use more memory-efficient matrix multiplication
    with torch.cuda.amp.autocast(enabled=False):  # Ensure numerical stability
        C = torch.bmm(L, L.transpose(-2, -1))
    
    # Efficient regularization
    adaptive_eps = adaptive_regularization_fast_ddp(C, base_epsilon)
    eye = torch.eye(dim, device=device, dtype=dtype)
    precision = C + adaptive_eps * eye
    
    return precision

def stable_softmax_ddp(logits, temperature=1.0, dim=-1):
    """
    DDP-optimized numerically stable softmax
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    # More aggressive clamping for DDP stability
    logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
    logits_stable = torch.clamp(logits - logits_max, min=-100, max=50)
    
    return F.softmax(logits_stable, dim=dim)

def grad_and_laplacian_mog_density_ddp_optimized(x, means, precisions, temperature=1.0):
    """
    DDP-optimized gradient and Laplacian computation with reduced memory usage
    """
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, dim)
    dim = x.size(-1)
    batch_size, num_components = x.size(0), means.size(0)

    # Memory-efficient computation with autocast disabled for critical operations
    with torch.cuda.amp.autocast(enabled=False):
        x_mean = x - means  # Shape: (batch_size, num_components, dim)
        
        # Efficient einsum computation
        x_mean_cov = torch.einsum('kij,bkj->bki', precisions, x_mean)
        
        # Squared terms computation
        squared_linear = torch.sum(x_mean_cov * x_mean_cov, dim=-1)
        
        # Stable log determinant
        logdet = stable_logdet_ddp(precisions)
        logdet = logdet.unsqueeze(0).expand(batch_size, -1)

        # Log probabilities with enhanced stability
        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
        log_probs = 0.5 * logdet - 0.5 * squared_linear - 0.5 * dim * log_2pi
        log_probs = torch.clamp(log_probs, min=-100, max=50)
        
        # Stable softmax
        softmax_probs = stable_softmax_ddp(log_probs, temperature=temperature, dim=1)
        
        # Gradient computation
        gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)
        
        # Laplacian computation
        trace_precision = torch.diagonal(precisions, dim1=-2, dim2=-1).sum(dim=-1)
        trace_precision = trace_precision.unsqueeze(0).expand(batch_size, -1)
        
        laplacian_component = softmax_probs * (squared_linear - trace_precision)
        laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    # Enhanced numerical health check for DDP
    if torch.any(torch.isnan(gradient)) or torch.any(torch.isnan(laplacian_over_density)):
        print(f"Warning: NaN detected in gradient/Laplacian on rank {dist.get_rank() if dist.is_initialized() else 0}")
        gradient = torch.zeros_like(gradient)
        laplacian_over_density = torch.zeros_like(laplacian_over_density)
    
    return gradient, laplacian_over_density

def grad_and_laplacian_mog_density_ddp_chunked(x, means, precisions, 
                                              batch_chunk_size=32, 
                                              component_chunk_size=32,
                                              temperature=1.0):
    """
    DDP-optimized chunked computation with better memory management
    """
    gradients = []
    laplacians = []
    
    # Process in smaller chunks for DDP efficiency
    for start in range(0, x.size(0), batch_chunk_size):
        end = min(start + batch_chunk_size, x.size(0))
        x_chunk = x[start:end]
        
        if means.size(0) > component_chunk_size:
            grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_component_chunked_ddp(
                x_chunk, means, precisions, chunk_size=component_chunk_size, temperature=temperature)
        else:
            grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_ddp_optimized(
                x_chunk, means, precisions, temperature=temperature)
        
        gradients.append(grad_chunk)
        laplacians.append(laplacian_chunk)
        
        # Explicit memory cleanup for DDP
        del grad_chunk, laplacian_chunk, x_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return torch.cat(gradients, dim=0), torch.cat(laplacians, dim=0)

def grad_and_laplacian_mog_density_component_chunked_ddp(x, means, precisions, 
                                                        chunk_size=64, temperature=1.0):
    """
    DDP-optimized component-wise chunked computation
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    
    all_log_probs = []
    all_weighted_gradients = []
    all_squared_linear = []
    all_trace_precision = []
    
    # Process components in chunks
    for start in range(0, num_components, chunk_size):
        end = min(start + chunk_size, num_components)
        
        means_chunk = means[start:end]
        precisions_chunk = precisions[start:end]
        
        with torch.cuda.amp.autocast(enabled=False):
            x_expanded = x.unsqueeze(1)
            x_mean = x_expanded - means_chunk
            
            x_mean_cov = torch.einsum("kij,bkj->bki", precisions_chunk, x_mean)
            squared_linear = torch.sum(x_mean_cov * x_mean_cov, dim=-1)
            
            log_det = stable_logdet_ddp(precisions_chunk)
            log_det = log_det.unsqueeze(0).expand(batch_size, -1)
            
            log_2pi = torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
            log_probs_chunk = 0.5 * log_det - 0.5 * squared_linear - 0.5 * dim * log_2pi
            log_probs_chunk = torch.clamp(log_probs_chunk, min=-100, max=50)
            
            weighted_gradient_chunk = -x_mean_cov
            
            trace_chunk = torch.diagonal(precisions_chunk, dim1=-2, dim2=-1).sum(dim=-1)
            trace_chunk = trace_chunk.unsqueeze(0).expand(batch_size, -1)
        
        all_log_probs.append(log_probs_chunk)
        all_weighted_gradients.append(weighted_gradient_chunk)
        all_squared_linear.append(squared_linear)
        all_trace_precision.append(trace_chunk)
        
        # Aggressive memory cleanup for DDP
        del x_mean, x_mean_cov, squared_linear, log_det, log_probs_chunk, weighted_gradient_chunk, trace_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate results
    log_probs = torch.cat(all_log_probs, dim=1)
    weighted_gradients = torch.cat(all_weighted_gradients, dim=1)
    squared_linear_full = torch.cat(all_squared_linear, dim=1)
    trace_precision_full = torch.cat(all_trace_precision, dim=1)
    
    # Final computations with stability
    softmax_probs = stable_softmax_ddp(log_probs, temperature=temperature, dim=1)
    gradient = torch.sum(softmax_probs.unsqueeze(-1) * weighted_gradients, dim=1)
    laplacian_component = softmax_probs * (squared_linear_full - trace_precision_full)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    # Cleanup
    del log_probs, weighted_gradients, squared_linear_full, trace_precision_full, softmax_probs, laplacian_component
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return gradient, laplacian_over_density

def numerical_health_check_ddp(tensor, name="tensor", rank=None):
    """
    DDP-aware numerical health check with rank information
    """
    if rank is None and dist.is_initialized():
        rank = dist.get_rank()
    
    rank_str = f"[Rank {rank}] " if rank is not None else ""
    
    if torch.any(torch.isnan(tensor)):
        print(f"{rank_str}Warning: NaN detected in {name}")
        return False
    if torch.any(torch.isinf(tensor)):
        print(f"{rank_str}Warning: Inf detected in {name}")
        return False
    if torch.any(torch.abs(tensor) > 1e10):
        max_val = torch.max(torch.abs(tensor))
        print(f"{rank_str}Warning: Very large values detected in {name} (max: {max_val})")
        return False
    return True

@sync_across_processes
def score_implicit_matching_ddp_optimized(factornet, samples, centers, base_epsilon=1e-4, 
                                         temperature=1.0, max_attempts=3, rank=None):
    """
    DDP-optimized score matching with synchronization and reduced memory usage
    """
    if rank is None and dist.is_initialized():
        rank = dist.get_rank()
    
    torch.cuda.reset_peak_memory_stats()
    dim = centers.shape[-1]
    
    centers = centers.clone().detach().requires_grad_(True)

    for attempt in range(max_attempts):
        try:
            if attempt != 0 and (rank is None or rank == 0):
                print(f"\n⏱️ Attempt {attempt + 1}/{max_attempts}")
            
            t0 = time.time()
            current_epsilon = base_epsilon * (2 ** attempt)

            # === FACTORNET FORWARD ===
            with torch.cuda.amp.autocast():
                factor_eval = factornet(centers)
            factor_eval = factor_eval.float()  # <-- cast back to float32 here

            if not numerical_health_check_ddp(factor_eval, "factor_eval", rank):
                if attempt < max_attempts - 1:
                    continue
                if rank is None or rank == 0:
                    print("⚠️ Proceeding with potentially unstable factor network output")

            # === PRECISION CONSTRUCTION ===
            precisions = vectors_to_precision_ddp_optimized(factor_eval, dim, current_epsilon)
            del factor_eval
            torch.cuda.empty_cache()

            if not numerical_health_check_ddp(precisions, "precisions", rank):
                if attempt < max_attempts - 1:
                    continue
                if rank is None or rank == 0:
                    print("⚠️ Proceeding with potentially unstable precision matrices")

            # === GRAD & LAPLACIAN ===
            gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_ddp_chunked(
                samples, centers, precisions, temperature=temperature)
            del precisions
            torch.cuda.empty_cache()

            if not numerical_health_check_ddp(gradient_eval_log, "gradient_eval_log", rank):
                if attempt < max_attempts - 1:
                    continue
            if not numerical_health_check_ddp(laplacian_over_density, "laplacian_over_density", rank):
                if attempt < max_attempts - 1:
                    continue

            # === FINAL LOSS ===
            gradient_eval_log_squared = torch.sum(gradient_eval_log * gradient_eval_log, dim=1)
            del gradient_eval_log
            loss = 2 * laplacian_over_density - gradient_eval_log_squared
            del laplacian_over_density, gradient_eval_log_squared
            torch.cuda.empty_cache()

            # === RETURN ===
            total_time = time.time() - t0
            if numerical_health_check_ddp(loss, "loss", rank):
                '''
                if rank is None or rank == 0:
                    print(f"🎯 Total score matching pass time: {total_time:.4f} sec")
                '''
                return loss.mean(dim=0)
            elif attempt < max_attempts - 1:
                if rank is None or rank == 0:
                    print(f"Retrying due to bad loss at attempt {attempt + 1}")
                continue
            else:
                if rank is None or rank == 0:
                    print("⚠️ Returning clamped fallback loss")
                return torch.clamp(loss, min=-1e6, max=1e6).mean(dim=0)

        except Exception as e:
            if rank is None or rank == 0:
                print(f"❌ Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_attempts - 1:
                if rank is None or rank == 0:
                    print("Retrying with increased regularization...")

def setup_optimal_device_settings_ddp():
    """
    Configure optimal device settings for DDP performance
    """
    if torch.cuda.is_available():
        # Enable tensor cores and optimizations for DDP
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # DDP-specific optimizations
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Memory allocation strategy for DDP
        torch.cuda.empty_cache()
        
        if dist.is_initialized():
            rank = dist.get_rank()
            print(f"Optimal CUDA settings enabled for rank {rank}")
        else:
            print("Optimal CUDA settings enabled")
    else:
        # CPU optimizations for DDP
        torch.set_num_threads(max(1, torch.get_num_threads() // torch.cuda.device_count() if torch.cuda.is_available() else torch.get_num_threads()))
        print(f"Using {torch.get_num_threads()} CPU threads per process")

def create_ddp_dataloader(dataset, batch_size, num_workers=4, shuffle=True):
    """
    Create a DataLoader compatible with DDP
    """
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(shuffle and sampler is None)
    )
    return dataloader, sampler

def ddp_main_worker(rank, world_size, main_func, *args, **kwargs):
    """
    Main worker function for DDP training
    """
    try:
        setup_ddp(rank, world_size)
        setup_optimal_device_settings_ddp()
        
        # Run the main training function
        main_func(rank, world_size, *args, **kwargs)
        
    finally:
        cleanup_ddp()

def launch_ddp_training(main_func, world_size=None, *args, **kwargs):
    """
    Launch DDP training across multiple processes
    """
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    if world_size > 1:
        mp.spawn(ddp_main_worker, args=(world_size, main_func) + args, 
                 nprocs=world_size, join=True)
    else:
        # Single GPU training
        main_func(0, 1, *args, **kwargs)

# Example usage pattern for DDP training:
def example_ddp_training_loop(rank, world_size, model, dataset, epochs=100):
    """
    Example of how to structure a DDP training loop
    """
    # Move model to GPU and wrap with DDP
    model = model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Create DDP-compatible dataloader
    dataloader, sampler = create_ddp_dataloader(dataset, batch_size=32)
    
    # Setup optimizer and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)  # Important for proper shuffling
        
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            
            with autocast():
                # Your training logic here using the optimized functions
                loss = score_implicit_matching_ddp_optimized(model, data, target, rank=rank)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Synchronize loss across processes for logging
            if dist.is_initialized():
                loss = reduce_tensor(loss, world_size)
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Optional: synchronize at end of epoch
        if dist.is_initialized():
            dist.barrier()

# Usage:
# launch_ddp_training(example_ddp_training_loop, model=your_model, dataset=your_dataset)



#----------------------------------------------#
#### plotting functions ####
#----------------------------------------------#
def sample_from_model(factornet, means, sample_number, eps):
    num_components, dim = means.shape
    comp_num = torch.randint(0, num_components, (sample_number,), device=means.device)
    samples = torch.empty(sample_number, dim, device=means.device)

    unique_indices = comp_num.unique()

    for i in unique_indices:
        idx = (comp_num == i).nonzero(as_tuple=True)[0]
        n_i = idx.shape[0]
        centers_i = means[i].unsqueeze(0).expand(n_i, -1)  # [n_i, dim]

        # Get model output vectors (flattened Cholesky)
        vectors = factornet(centers_i)  # [n_i, d*(d+1)//2]

        # Use your existing function to get precision matrix with stabilization
        precision = vectors_to_precision_ddp_optimized(vectors, dim, eps)  # [n_i, dim, dim]

        # Create multivariate normal with precision matrix
        mvn = MultivariateNormal(loc=centers_i, precision_matrix=precision)

        # Sample from this distribution
        samples_i = mvn.rsample()  # [n_i, dim]

        samples[idx] = samples_i

    return samples


def plot_images(means, precisions, epoch = 0, plot_number = 10, save_path=None):
    # plots plot_number samples from the trained model for image data
    num_components = means.shape[0]
    # sample from the multivariate normal distribution
    comp_num = torch.randint(0, num_components, (1,plot_number)) #shape: [1, plot_number]
    comp_num = comp_num.squeeze(0)  # shape: [plot_number]
    multivariate_normal = torch.distributions.MultivariateNormal(means[comp_num], precision_matrix=precisions[comp_num])
    samples = multivariate_normal.rsample()
    # transform images back to original data 
    samples = samples.view(-1, 3, 32, 32)
    #samples = samples * 0.5 + 0.5
     #   Undo CIFAR-10 normalization
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=samples.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=samples.device).view(1, 3, 1, 1)
    samples = samples * std + mean
    fig, axs = plt.subplots(1, plot_number, figsize=(15, 2))
    for i in range(plot_number):
        img = samples[i].permute(1, 2, 0).cpu().numpy()  # change from [C, H, W] to [H, W, C]
        axs[i].imshow(img)
        axs[i].axis('off')
    if save_path is not None:
        save_path = save_path + 'epoch'+ str(epoch) + '.png'
        plt.savefig(save_path)
    
    plt.close(fig)

    return None

def plot_images_with_model(factornet, means, plot_number = 10, eps=1e-4, save_path=None):
    # plots plot_number samples from the trained model for image data
    num_components = means.shape[0]
    dim = means.shape[-1]
    # sample from the multivariate normal distribution
    comp_num = torch.randint(0, num_components, (1,plot_number)) #shape: [1, plot_number]
    comp_num = comp_num.squeeze(0)  # shape: [plot_number]
    samples = torch.empty(plot_number, dim, device=means.device)  # shape: [plot_number, d]
    samples = sample_from_model(factornet, means, plot_number, eps)
    # transform images back to original data 
    samples = samples.view(-1, 3, 32, 32)
    samples = samples * 0.5 + 0.5
    samples = torch.clamp(samples, 0, 1)  # clip to [0, 1] to avoid warnings
    fig, axs = plt.subplots(1, plot_number, figsize=(15, 2))
    for i in range(plot_number):
        img = samples[i].permute(1, 2, 0).cpu().numpy()  # change from [C, H, W] to [H, W, C]
        axs[i].imshow(img)
        axs[i].axis('off')
    if save_path is not None:
        save_path = save_path + '_sampled_images.png'
        plt.savefig(save_path)
    plt.close(fig)
    return None
def denormalize_cifar10(tensor):
    """
    Denormalizes a batch of CIFAR-10 images from standardized (mean=0, std=1)
    back to [0, 1] for visualization.
    
    Args:
        tensor: (N, 3, 32, 32) tensor, normalized using CIFAR-10 mean/std

    Returns:
        Denormalized tensor (still in [0, 1] range)
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

import torchvision.utils as vutils

def plot_and_save_centers(centers, save_path, nrow=10):
    centers = centers.view(-1, 3, 32, 32)  # reshape
    centers = denormalize_cifar10(centers).clamp(0, 1)  # denormalize and clip to [0, 1]

    grid = vutils.make_grid(centers, nrow=nrow, padding=2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vutils.save_image(grid, save_path)
    print(f"Saved centers to {save_path}")

    # Optional preview
    plt.figure(figsize=(nrow, nrow))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
