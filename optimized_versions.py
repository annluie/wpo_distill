# optimized versions

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
import torch.nn.functional as F
import torch.linalg as LA
import torchvision.utils as vutils
import time

def vectors_to_precision_optimized(vectors, dim, base_epsilon=1e-3, max_cond=1e5):
    """
    Converts a batch of lower-triangular Cholesky vectors into precision matrices,
    compatible with shape (batch_size, 1, D_tri) like the old version.
    
    Args:
        vectors (Tensor): (batch_size, 1, D_tri) or (batch_size, D_tri)
        dim (int): dimension of square matrix
        base_epsilon (float): base regularization added to the diagonal
        max_cond (float): max allowed condition number to adapt epsilon
    
    Returns:
        Tensor: (batch_size, dim, dim) precision matrices
    """
    if vectors.ndim == 3 and vectors.shape[1] == 1:
        vectors = vectors.squeeze(1)
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)

    batch_size = vectors.shape[0]
    device, dtype = vectors.device, vectors.dtype

    tril_indices = torch.tril_indices(dim, dim, device=device)
    L = torch.zeros(batch_size, dim, dim, dtype=dtype, device=device)
    L[:, tril_indices[0], tril_indices[1]] = vectors

    # Ensure diagonal is strictly positive
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    diag_soft = F.softplus(diag).clamp(min=1e-2)
    L = L + torch.diag_embed(diag_soft - diag)

    # Compute precision matrix: L @ Lᵀ
    C = torch.bmm(L, L.transpose(-1, -2))

    # Add scalar regularization
    eps = adaptive_regularization_fast(C, base_epsilon, max_cond)
    C = C + torch.eye(dim, dtype=dtype, device=device) * eps

    return C

'''
def adaptive_regularization_fast(matrices, base_epsilon, max_cond):
    """
    Faster scalar epsilon based on condition number estimation using norm ratio.
    """
    batch_size = matrices.size(0)
    device, dtype = matrices.device, matrices.dtype

    # Subsample matrices
    sample_size = min(4, batch_size)
    sample = matrices[torch.randperm(batch_size, device=device)[:sample_size]]

    try:
        # Use Frobenius norm as a cheap approximation of operator norm
        norm = torch.linalg.norm(sample, ord='fro', dim=(1, 2))  # shape: (sample_size,)
        
        # Estimate inverse norm using Cholesky + triangular solve (cheaper than SVD)
        inv_norms = []
        for i in range(sample_size):
            try:
                L = torch.linalg.cholesky(sample[i])
                eye = torch.eye(L.size(-1), dtype=dtype, device=device)
                inv = torch.cholesky_solve(eye, L)
                inv_norm = torch.linalg.norm(inv, ord='fro')
                inv_norms.append(inv_norm)
            except RuntimeError:
                inv_norms.append(torch.tensor(1e6, device=device, dtype=dtype))  # fallback

        cond = norm * torch.stack(inv_norms)
        worst = cond.max()

        if worst > max_cond:
            scale = worst / max_cond
            adaptive_eps = base_epsilon * scale
        else:
            adaptive_eps = base_epsilon

    except RuntimeError:
        adaptive_eps = base_epsilon * 10

    return float(adaptive_eps)
'''
def adaptive_regularization_fast(matrices, base_epsilon, max_cond):
    """
    Fast adaptive epsilon scaling using condition number estimation on a matrix subset.
    """
    batch_size = matrices.size(0)
    device, dtype = matrices.device, matrices.dtype

    sample_size = min(4, batch_size)
    sample = matrices[torch.randperm(batch_size, device=device)[:sample_size]]

    try:
        s = torch.linalg.svdvals(sample)  # shape: (sample_size, dim)
        cond = s[:, 0] / s[:, -1].clamp(min=1e-12)
        worst = cond.max()

        if worst > max_cond:
            scale = worst / max_cond
            adaptive_eps = base_epsilon * scale
        else:
            adaptive_eps = base_epsilon

    except RuntimeError:
        adaptive_eps = base_epsilon * 10  # fallback if SVD fails

    return float(adaptive_eps)


def stable_softmax_optimized(logits, temperature=1.0, dim=-1):
    """
    Optimized stable softmax with fewer operations
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    # Combined max subtraction and clamping
    logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
    logits_stable = torch.clamp(logits - logits_max, min=-50, max=50)
    
    return F.softmax(logits_stable, dim=dim)

def precompute_precision_terms(precisions):
    """
    Precompute terms that are reused multiple times
    """
    # Compute log determinant once
    logdet = stable_logdet_fast(precisions)
    
    # Compute trace once
    trace_precision = torch.diagonal(precisions, dim1=-2, dim2=-1).sum(dim=-1)
    
    return logdet, trace_precision

def stable_logdet_fast(matrices):
    """
    Faster stable log determinant computation
    """
    try:
        # Try Cholesky first (fastest for PSD matrices)
        L = torch.linalg.cholesky(matrices)
        logdet = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
        return logdet
    except RuntimeError:
        # Fallback to LU decomposition
        try:
            LU, pivots = torch.linalg.lu_factor(matrices)
            logdet = torch.sum(torch.log(torch.abs(torch.diagonal(LU, dim1=-2, dim2=-1))), dim=-1)
            # Adjust for pivot sign
            det_P = torch.det(torch.eye(matrices.shape[-1], device=matrices.device)[pivots])
            logdet += torch.log(torch.abs(det_P))
            return logdet
        except RuntimeError:
            # Conservative fallback
            return torch.zeros(matrices.shape[0], device=matrices.device)

def grad_and_laplacian_mog_density_optimized(x, means, precisions, temperature=1.0, 
                                           precomputed_terms=None):
    """
    Optimized version with precomputed terms and reduced allocations
    Fixed dtype consistency for mixed precision
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    device = x.device
    dtype = x.dtype
    
    # Use precomputed terms if available
    if precomputed_terms is not None:
        logdet, trace_precision = precomputed_terms
        # Ensure precomputed terms have correct dtype
        logdet = logdet.to(dtype=dtype, device=device)
        trace_precision = trace_precision.to(dtype=dtype, device=device)
    else:
        logdet, trace_precision = precompute_precision_terms(precisions)
    
    # Reshape for efficient batch operations
    x_expanded = x.unsqueeze(1)  # (B, 1, D)
    means_expanded = means.unsqueeze(0).to(dtype=dtype, device=device)  # (1, K, D)
    x_mean = x_expanded - means_expanded  # (B, K, D)
    
    # Optimized einsum with explicit dtype casting
    precisions_typed = precisions.to(dtype=dtype, device=device)
    x_mean_cov = torch.einsum('bkd,kdi->bki', x_mean, precisions_typed)  # (B, K, D)
    
    # Compute squared terms efficiently
    squared_linear = torch.sum(x_mean * x_mean_cov, dim=-1)  # (B, K)
    
    # Vectorized log probability computation with proper dtype
    pi_const = torch.tensor(2 * torch.pi, dtype=dtype, device=device)
    log_probs = (0.5 * logdet.unsqueeze(0) - 0.5 * squared_linear - 
                0.5 * dim * torch.log(pi_const))
    
    # Stable softmax
    softmax_probs = stable_softmax_optimized(log_probs, temperature, dim=1)
    
    # Efficient gradient computation
    gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)
    
    # Efficient Laplacian computation
    laplacian_component = softmax_probs * (squared_linear - trace_precision.unsqueeze(0))
    laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    return gradient, laplacian_over_density

def grad_and_laplacian_mog_density_mega_optimized(x, means, precisions, temperature=1.0):
    """
    Maximum optimization version for when you need every bit of speed
    """
    # Precompute all reusable terms
    precomputed = precompute_precision_terms(precisions)
    
    # Use fused operations where possible
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        return grad_and_laplacian_mog_density_optimized(
            x, means, precisions, temperature, precomputed
        )

def score_implicit_matching_optimized(factornet, samples, centers, base_epsilon=1e-4, 
                                    temperature=1.0, use_mixed_precision=False):
    """
    Optimized version with better memory management and fewer operations
    Mixed precision disabled by default due to dtype issues
    """
    dim = centers.shape[-1]
    centers = centers.clone().detach().requires_grad_(True)
    
    # Disable mixed precision by default to avoid dtype issues
    # Can be enabled if you ensure all your model weights are in float16
    autocast_enabled = use_mixed_precision and torch.cuda.is_available()
    
    try:
        #start = time.time()
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            # Forward pass
            #t0 = time.time()
            factor_eval = factornet(centers)
            print("factor_eval shape:", factor_eval.shape)  # Should be (K, D_tri)
            #print("Factornet time:", time.time() - t0)
            
            # Ensure consistent dtypes
            if autocast_enabled:
                factor_eval = factor_eval.float()
                centers = centers.float()
                samples = samples.float()
            
            # Optimized precision computation
            #t0 = time.time()
            precisions = vectors_to_precision_optimized(factor_eval, dim, base_epsilon)
            print("precisions shape:", precisions.shape)
            eigvals = torch.linalg.eigvalsh(precisions)
            print("Min eig:", eigvals.min().item(), "Max eig:", eigvals.max().item())
            print("Any NaNs in C?", torch.isnan(precisions).any().item())
            #print("Precision time:", time.time() - t0)
            del factor_eval  # Early cleanup
            
            # Precompute terms for reuse
            #t0 = time.time()
            precomputed_terms = precompute_precision_terms(precisions)
            #print("Precompute time:", time.time() - t0)

            # Optimized gradient and Laplacian computation
            #t0 = time.time()
            gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_optimized(
                samples, centers, precisions, temperature, precomputed_terms
            )
            #print("lapl, grad time:", time.time() - t0)

            # Compute final loss efficiently
            gradient_norm_squared = torch.sum(gradient_eval_log * gradient_eval_log, dim=1)
            loss = 2 * laplacian_over_density - gradient_norm_squared
            
            return loss.mean()
            
    except RuntimeError as e:
        print(f"Optimization failed: {e}, falling back to stable version")
        # Fallback to your original stable version
        '''
        return score_implicit_matching_stable(factornet, samples, centers, 
                                            base_epsilon, temperature)
                                            '''
        


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
