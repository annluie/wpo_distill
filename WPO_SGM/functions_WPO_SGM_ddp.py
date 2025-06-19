import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import lib.toy_data as toy_data
import numpy as np
import argparse
from memory_profiler import profile
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
import gc


## vector to precision matrix function
def vectors_to_precision(vectors,dim):
    """
    Maps an array of 1xdim vectors into Cholesky factors and returns an array of precision matrices.
    
    Args:
    vectors (torch.Tensor): A tensor of shape (batch_size, 1, dim), where each 1xdim tensor represents the
                            lower triangular part of the Cholesky factor.
    
    Returns:
    torch.Tensor: A tensor of shape (batch_size, dim, dim), containing the corresponding precision matrices.
    """
    batch_size = vectors.shape[0]
    # Reshape the input vectors into lower triangular matrices
    L = torch.zeros(batch_size, dim, dim, dtype=vectors.dtype, device=vectors.device)
    indices = torch.tril_indices(dim, dim)
    L[:, indices[0], indices[1]] = vectors.squeeze(1)
    
    # Construct the precision matrices using Cholesky factorization
    C = torch.matmul(L, L.transpose(1, 2)) + 0.01 * torch.eye(dim).to(vectors.device) # (add identity matrix to maintain positive definiteness)
    
    return C
    
def vectors_to_precision_chunked(vectors, dim, chunk_size=2):
    batch_size = vectors.shape[0]
    precisions = []
    tril_indices = torch.tril_indices(dim, dim, device=vectors.device)

    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk = vectors[start:end].squeeze(1)

        L = torch.zeros((end - start, dim, dim), dtype=vectors.dtype, device=vectors.device)
        L[:, tril_indices[0], tril_indices[1]] = chunk

        eye = torch.eye(dim, device=vectors.device).expand(end - start, -1, -1)
        C_chunk = torch.baddbmm(0.01 * eye, L, L.transpose(1, 2))

        precisions.append(C_chunk)

        # Optional: explicitly free GPU memory in loop
        del chunk, L, C_chunk
        torch.cuda.empty_cache()

    return torch.cat(precisions, dim=0)

#----------------------------------------------#
#### Mixture of Gaussians functions ####
#----------------------------------------------#

# Compute MoG density for each data point
def mog_density(x, means, precisions):
    device = x.device
    num_components = means.shape[0]
    
    # Expand the dimensions of x for broadcasting
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, dim)
    
    # Create a batch of Multivariate Normal distributions for each component
    mvns = MultivariateNormal(loc=means, precision_matrix=precisions) 
    
    # Calculate the log probabilities for each component
    log_component_probs = mvns.log_prob(x) - torch.log(torch.tensor(num_components, dtype=x.dtype, device=x.device))  # Shape: (batch_size, num_components)
    
    # Use logsumexp to calculate the log of the mixture density
    log_densities = torch.logsumexp(log_component_probs, dim=1)  # Shape: (batch_size,)
    
    return log_densities.exp()


# Compute gradient 
def gradient_mog_density(x, means, precisions):
    """
    Computes the gradient of the MoG density function with respect to the input x.
    x: (batch_size, dim)
    means: (num_components, dim)
    precisions: (num_components, dim, dim)

    output: (batch_size, dim)
    """
    dim = x.size(-1)
    batch_size, num_components = x.size(0), means.size(0)

    # Expand dimensions for broadcasting
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, dim)
    
    # Expand dimensions of means and precisions
    means_expanded = means.unsqueeze(0)  # Shape: (1, num_components, 1, 2)
    precisions_expanded = precisions.unsqueeze(0)  # Shape: (1, num_components, 2, 2)

    # Calculate the difference between x and means for all components
    x_mean = x - means_expanded  # Shape: (batch_size, num_components, 1, 2)

    # Reshape x_mean to match the shape of precisions for matrix multiplication
    x_mean_reshaped = x_mean.view(batch_size, num_components, dim, 1)

    # Create Multivariate Normal distributions for all components
    mvns = MultivariateNormal(loc=means_expanded.squeeze(2), precision_matrix=precisions_expanded)

    # Calculate log_prob for all components
    log_prob = mvns.log_prob(x.squeeze(2)).unsqueeze(-1)  # Shape: (batch_size, num_components, 1)

    # Calculate the gradient components using matrix multiplication
    x_mean_cov = torch.matmul(precisions_expanded, x_mean_reshaped).squeeze(-1)

    # Multiply by prob and sum over components to get the gradient
    gradient = (-log_prob.exp() * x_mean_cov).sum(dim=1)  # Shape: (batch_size, 2)
    
    return gradient / num_components

# Compute grad log pi
def grad_log_mog_density(x, means, precisions):
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, 2)
    dim = x.size(-1)
    batch_size, num_components = x.size(0), means.size(0)

    # Create a batch of Multivariate Normal distributions for each component
    mvns = MultivariateNormal(loc=means, precision_matrix=precisions)

    # Calculate the log probabilities for each component
    log_probs = mvns.log_prob(x)  # Shape: (batch_size, num_components)
    del mvns #clear memory
    # Use torch.logsumexp to compute the log of the sum of exponentiated log probabilities
    # log_sum_exp = torch.logsumexp(log_probs, dim=1, keepdim=True)  # Shape: (batch_size, 1) #i don't think this is used anywhere

    # Calculate the softmax probabilities along the components dimension
    
    softmax_probs = torch.softmax(log_probs, dim=1)  # Shape: (batch_size, num_components)
    x_mean = x - means  # Shape: (batch_size, num_components, 2)
    """
    x_mean_reshaped = x_mean.view(batch_size, num_components, dim, 1)
    precision_matrix = precisions.unsqueeze(0)  # Shape: (1, num_components, 2, 2)
    precision_matrix = precision_matrix.expand(x.shape[0], -1, -1, -1)  # Shape: (batch_size, num_components, 2, 2)

    x_mean_cov = torch.matmul(precision_matrix, x_mean_reshaped).squeeze(dim = -1)
    print(f"matmul mean cov: {x_mean_cov}")
    del x_mean_cov
    gc.collect()
    torch.cuda.empty_cache()
    """
    # calculate without resizing (more memory efficient)
    x_mean_cov = torch.einsum('kij,bkj->bki', precisions, x_mean)
    
    # Calculate the gradient of log density with respect to x

    gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)
    del log_probs, softmax_probs, x_mean_cov
    #gc.collect()
    torch.cuda.empty_cache()
    return gradient

# Compute Laplacian

def laplacian_mog_density(x, means, precisions):
    """
    Computes the gradient of the MoG density function with respect to the input x.
    x: (batch_size, dim)
    means: (num_components, dim)
    precisions: (num_components, dim, dim)

    output: (batch_size)
    """
    batch_size, num_components = x.size(0), means.size(0)
    
    dim = x.size(-1)
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, 2)
    means = means.unsqueeze(0)  # Shape: (1, num_components, 2)

    x_mean = x - means  # Shape: (batch_size, num_components, 2)
    x_mean_reshaped = x_mean.view(batch_size, num_components, dim, 1)
    precision_matrix = precisions.unsqueeze(0)  # Shape: (1, num_components, 2, 2)
    precision_matrix = precision_matrix.expand(x.shape[0], -1, -1, -1)  # Shape: (batch_size, num_components, 2, 2)

    mvn = MultivariateNormal(means, precision_matrix=precision_matrix)
    
    log_prob = mvn.log_prob(x) # Shape: (batch_size, num_components)
    
    prob = log_prob.exp() # Shape: (batch_size, num_components)

    # Calculate the gradient components using matrix multiplication
    x_mean_cov = torch.matmul(precision_matrix, x_mean_reshaped).squeeze(-1)

    squared_linear = torch.sum(x_mean_cov * x_mean_cov, dim=2) # Shape: (batch_size, num_components)

    trace_precision = precision_matrix.view(x.shape[0], num_components, -1)[:, :, ::3].sum(dim=-1)  # Shape: (batch_size, num_components)

    laplacian_component = prob * (squared_linear - trace_precision)  # Shape: (batch_size, num_components)
    laplacian = torch.mean(laplacian_component, dim=1)  # Shape: (batch_size,)
    
    return laplacian

#@profile
def laplacian_mog_density_div_density(x, means, precisions):
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, 2)
    batch_size, num_components = x.size(0), means.size(0)

    # Create a batch of Multivariate Normal distributions for each component
    mvns = MultivariateNormal(loc=means, precision_matrix=precisions)

    # Calculate the log probabilities for each component
    log_probs = mvns.log_prob(x) #.contiguous()  # Shape: (batch_size, num_components)

    # Use torch.logsumexp to compute the log of the sum of exponentiated log probabilities
    log_sum_exp = torch.logsumexp(log_probs, dim=1, keepdim=True)  # Shape: (batch_size, 1)

    # Calculate the softmax probabilities along the components dimension
    softmax_probs = torch.softmax(log_probs, dim=1)  # Shape: (batch_size, num_components)

    x_mean = x - means  # Shape: (batch_size, num_components, 2)

    # Calculate the covariance matrix term
    #cov_term = torch.matmul(precisions, x_mean.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, num_components, 2)
    cov_term = torch.einsum("kij,bkj->bki", precisions, x_mean)
    
    # Calculate the squared linear term
    squared_linear = torch.sum(cov_term * cov_term, dim=-1)  # Shape: (batch_size, num_components)

    # Calculate the trace of the precision matrix term
    trace_precision = torch.diagonal(precisions, dim1=-2, dim2=-1)  # Shape: (num_components, 2)
    trace_precision = trace_precision.sum(dim=-1)  # Shape: (num_components,)

    # Expand dimensions for broadcasting
    trace_precision = trace_precision.unsqueeze(0)  # Shape: (1, num_components)

    # Calculate the Laplacian component
    laplacian_component = softmax_probs * (squared_linear - trace_precision)

    # Sum over components to obtain the Laplacian of the density over the density
    laplacian_over_density = torch.sum(laplacian_component, dim=1)  # Shape: (batch_size,)
    
    #free up memory 
    del mvns, log_probs, log_sum_exp, softmax_probs, x_mean, cov_term
    #gc.collect()
    torch.cuda.empty_cache()
    return laplacian_over_density

#----------------------------------------------#
# BATCHED VERSIONS
#----------------------------------------------#

def laplacian_mog_density_div_density_chunked(x, means, precisions, chunk_size=4):
    """
    Compute Laplacian/density ratio with component-wise chunking for memory efficiency.
    Reduced default chunk_size for better memory management in DDP.
    """
    device = x.device
    dtype = x.dtype
    batch_size = x.shape[0]
    num_components = means.shape[0]
    
    # Pre-allocate result tensor
    total_result = torch.zeros(batch_size, device=device, dtype=dtype)
    
    # Process components in chunks
    for start_idx in range(0, num_components, chunk_size):
        end_idx = min(start_idx + chunk_size, num_components)
        
        # Extract chunk without creating unnecessary references
        means_chunk = means[start_idx:end_idx]
        precisions_chunk = precisions[start_idx:end_idx]
        
        # Compute chunk contribution
        chunk_result = laplacian_mog_density_div_density(x, means_chunk, precisions_chunk)
        
        # Accumulate results in-place
        total_result.add_(chunk_result)
        
        # Clean up chunk result immediately
        del chunk_result
        
    return total_result


def grad_log_mog_density_chunked(x, means, precisions, chunk_size=16):
    """
    Compute gradient of log density with batch chunking.
    Improved for DDP stability - no unnecessary detach operations.
    """
    batch_size = x.size(0)
    
    # Handle case where chunk_size >= batch_size
    if chunk_size >= batch_size:
        return grad_log_mog_density(x, means, precisions)
    
    # Pre-allocate output list with known size
    num_chunks = (batch_size + chunk_size - 1) // chunk_size
    outputs = []
    outputs.reserve(num_chunks) if hasattr(outputs, 'reserve') else None
    
    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        x_chunk = x[i:end_idx]
        
        # Preserve gradient requirements from original tensor
        if x.requires_grad:
            x_chunk.requires_grad_(True)
        
        result = grad_log_mog_density(x_chunk, means, precisions)
        
        # Only detach if we're not in the middle of backprop
        if not torch.is_grad_enabled():
            result = result.detach()
            
        outputs.append(result)
    
    return torch.cat(outputs, dim=0)


def grad_log_mog_density_double_chunked(x, means, precisions, 
                                       batch_chunk_size=32, 
                                       component_chunk_size=8):
    """
    Double chunking for both batch and components.
    Optimized for DDP with better memory management and gradient flow.
    """
    batch_size = x.size(0)
    num_components = means.size(0)
    
    # Handle small batches efficiently
    if batch_chunk_size >= batch_size and component_chunk_size >= num_components:
        return grad_log_mog_density(x, means, precisions)
    
    total_results = []
    
    for i in range(0, batch_size, batch_chunk_size):
        end_batch = min(i + batch_chunk_size, batch_size)
        x_chunk = x[i:end_batch]
        
        # Preserve gradient flow
        if x.requires_grad:
            x_chunk.requires_grad_(True)
        
        # Initialize accumulator for this batch chunk
        batch_result = torch.zeros_like(x_chunk)
        
        # Process components in chunks
        for j in range(0, num_components, component_chunk_size):
            end_comp = min(j + component_chunk_size, num_components)
            
            means_chunk = means[j:end_comp]
            precisions_chunk = precisions[j:end_comp]
            
            # Compute gradient for this component chunk
            comp_grad = grad_log_mog_density(x_chunk, means_chunk, precisions_chunk)
            
            # Accumulate in-place
            batch_result.add_(comp_grad)
            
            # Clean up immediately
            del comp_grad
        
        # Only detach if not training
        if not torch.is_grad_enabled():
            batch_result = batch_result.detach()
            
        total_results.append(batch_result)
    
    return torch.cat(total_results, dim=0)


def score_implicit_matching(factornet, samples, centers, 
                          use_amp=True, 
                          batch_chunk_size=32,
                          component_chunk_size=8,
                          precision_chunk_size=4):
    """
    Stable and memory-efficient score computation for DDP training.
    
    Args:
        factornet: Network to evaluate
        samples: Input samples
        centers: Center points
        use_amp: Whether to use automatic mixed precision
        batch_chunk_size: Chunk size for batch processing
        component_chunk_size: Chunk size for component processing  
        precision_chunk_size: Chunk size for precision computation
    """
    
    # Get dimensions
    dim = centers.shape[-1]
    device = centers.device
    
    # Ensure centers require gradients for DDP
    if not centers.requires_grad:
        centers = centers.clone().detach().requires_grad_(True)
    
    # Use gradient checkpointing for memory efficiency
    # But be careful with DDP - use_reentrant=False is important
    with autocast(device_type='cuda', enabled=use_amp):
        factor_eval = checkpoint(factornet, centers, use_reentrant=False)
        
        # Compute precisions with chunking
        precisions = vectors_to_precision_chunked(
            factor_eval, dim, chunk_size=precision_chunk_size
        )
        
        # Clear intermediate results
        del factor_eval
        
        # Compute Laplacian/density ratio
        laplacian_over_density = laplacian_mog_density_div_density_chunked(
            samples, centers, precisions, chunk_size=component_chunk_size
        )
        
        # Compute gradient of log density
        gradient_eval_log = grad_log_mog_density_double_chunked(
            samples, centers, precisions,
            batch_chunk_size=batch_chunk_size,
            component_chunk_size=component_chunk_size
        )
        
        # Clear precisions after use
        del precisions
        
        # Compute gradient norm squared efficiently
        gradient_eval_log_squared = torch.sum(
            gradient_eval_log.square(), dim=1
        )
        
        # Clear gradient evaluation
        del gradient_eval_log
        
        # Compute final loss
        loss = 2.0 * laplacian_over_density - gradient_eval_log_squared
        
        # Clear intermediate tensors
        del laplacian_over_density, gradient_eval_log_squared
        
        # Return mean loss - this will properly sync gradients in DDP
        return loss.mean()


# Additional utility functions for better DDP compatibility

def sync_batch_norm_if_distributed(tensor):
    """Synchronize batch statistics across DDP processes if needed."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Only sync if we have multiple processes
        if torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            tensor.div_(torch.distributed.get_world_size())
    return tensor


def clean_memory_if_needed(threshold_mb=1000):
    """Clean GPU memory if usage exceeds threshold."""
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        if current_memory > threshold_mb:
            # Force garbage collection
            gc.collect()
            # Only empty cache if really needed (can hurt performance)
            if current_memory > threshold_mb * 1.5:
                torch.cuda.empty_cache()


# Wrapper function for training loop integration
def compute_score_with_ddp_safety(factornet, samples, centers, **kwargs):
    """
    Wrapper that handles DDP-specific concerns and memory management.
    """
    try:
        # Clean memory before computation if needed
        clean_memory_if_needed()
        
        # Compute score
        score = score_implicit_matching(factornet, samples, centers, **kwargs)
        
        # Ensure proper gradient synchronization for DDP
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # The loss.backward() call will handle gradient synchronization
            # No manual synchronization needed here
            pass
            
        return score
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Handle OOM gracefully
            print(f"OOM encountered, cleaning memory and retrying with smaller chunks...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Retry with smaller chunk sizes
            new_kwargs = kwargs.copy()
            new_kwargs['batch_chunk_size'] = max(8, kwargs.get('batch_chunk_size', 32) // 2)
            new_kwargs['component_chunk_size'] = max(4, kwargs.get('component_chunk_size', 8) // 2)
            
            return score_implicit_matching(factornet, samples, centers, **new_kwargs)
        else:
            raise e
        
#----------------------------------------------#
#### plotting functions ####
#----------------------------------------------#

def sample_from_model(factornet, means, sample_number):
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
        precision = vectors_to_precision(vectors, dim)  # [n_i, dim, dim]

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
    samples = samples * 0.5 + 0.5
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

def plot_images_with_model(factornet, means, plot_number = 10, save_path=None):
    # plots plot_number samples from the trained model for image data
    num_components = means.shape[0]
    dim = means.shape[-1]
    # sample from the multivariate normal distribution
    comp_num = torch.randint(0, num_components, (1,plot_number)) #shape: [1, plot_number]
    comp_num = comp_num.squeeze(0)  # shape: [plot_number]
    samples = torch.empty(plot_number, dim, device=means.device)  # shape: [plot_number, d]
    samples = sample_from_model(factornet, means, plot_number)
    # transform images back to original data 
    samples = samples.view(-1, 3, 32, 32)
    samples = samples * 0.5 + 0.5
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
