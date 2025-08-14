#stable version of code
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.distributed as dist
#import lib.toy_data as toy_data
import numpy as np
import argparse
from memory_profiler import profile
# Mixed precision removed for compatibility with large center counts
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch.linalg as LA
import torchvision.utils as vutils
from torch.jit import script
import time
import gc
from contextlib import contextmanager
from WPO_SGM.utilities import *
from torch import compile
# ===================== #
# Global Variables Config
# ===================== #
DEFAULT_EPSILON = 1e-4
DEFAULT_MAX_COND = 1e12
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CENTERS_CHUNK_SIZE = 100
DEFAULT_BATCH_CHUNK_SIZE = 64
DEFAULT_TEMPERATURE = 1.0
DEFAULT_LOGPROB_CLAMP = 200
DEFAULT_LOGITS_CLAMP = 50  # Reduced for stability
DEFAULT_CLEAR_CACHE_FREQUENCY = 10  # Reduced frequency for aggressive cache clearing

# ===================== #
# Precompute and Cache reused values
# ===================== #
PRECISION_EYE = None  # Will be initialized in score function
TRIL_INDICES = None  # Will be initialized in score function

# ===================== #
# JIT-Compiled Core Functions
# ===================== #
@script
def _fast_condition_estimate_jit(matrices):
    """
    More robust condition number estimation
    """
    batch_size = matrices.shape[0]
    
    # Multiple condition estimates for robustness
    diag_vals = torch.diagonal(matrices, dim1=-2, dim2=-1)
    min_diag = torch.min(diag_vals, dim=-1)[0]
    max_diag = torch.max(diag_vals, dim=-1)[0]
    
    # Estimate 1: Diagonal ratio
    diag_ratio = torch.clamp(max_diag / (min_diag + 1e-8), 1.0, 1e12)
    
    # Estimate 2: Frobenius norm ratio
    frob_norm = torch.norm(matrices.view(batch_size, -1), dim=-1)
    trace = torch.trace(matrices.view(-1, matrices.shape[-1])).view(batch_size)
    frob_ratio = torch.clamp(frob_norm / (trace + 1e-8), 1.0, 1e12)
    
    # Take the maximum of both estimates for safety
    return torch.max(diag_ratio, frob_ratio)

@script
def stable_softmax_jit(logits, temperature: float = 1.0)-> torch.Tensor:
    """JIT-compiled stable softmax"""
    if temperature != 1.0:
        logits = logits / temperature
    
    logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_stable = logits - logits_max
    logits_clamped = torch.clamp(logits_stable, min=-100, max=100)
    
    return F.softmax(logits_clamped, dim=-1)

#----------------------------------------------#
#### Precision Matrix Functions ####
#----------------------------------------------#
def vectors_to_precision_optimized(vectors, dim, base_epsilon=DEFAULT_EPSILON):
    """
    Highly optimized version with minimal redundant operations
    """
    if torch.isnan(vectors).any():
        print("❌ NaNs in vectors before conversion to precision!")
    
    batch_size = vectors.shape[0]
    device = vectors.device
    dtype = vectors.dtype

    if not hasattr(vectors_to_precision_optimized, 'tril_indices'): #only compute indices once to save memory
        vectors_to_precision_optimized.tril_indices = torch.tril_indices(dim, dim, device=device)
    # Pre-allocate and use advanced indexing for better memory efficiency
    L = torch.zeros(batch_size, dim, dim, dtype=dtype, device=device)
    
    # Create triangular indices once
    tril_indices = vectors_to_precision_optimized.tril_indices
    L[:, tril_indices[0], tril_indices[1]] = vectors
    
    # Faster diagonal processing
    diag_slice = slice(None), torch.arange(dim, device=device), torch.arange(dim, device=device)
    L[diag_slice] = F.softplus(L[diag_slice]) + 1e-6
    
    # Use bmm for better batched matrix multiplication
    C = torch.bmm(L, L.transpose(-2, -1))
    
    # Single-pass regularization
    adaptive_eps = adaptive_regularization_fast_new(C, base_epsilon)
    eye = torch.eye(dim, device=device, dtype=dtype)
    precision = C + adaptive_eps * eye
    
    # Skip expensive eigendecomposition and use simpler validation
    # Most matrices will be positive definite after proper regularization
    return precision

def vectors_to_precision_chunked_optimized(vectors, dim, base_epsilon=DEFAULT_EPSILON, chunk_size=DEFAULT_CENTERS_CHUNK_SIZE):
    results = []
    for i in range(0, vectors.size(0), chunk_size):
        chunk = vectors[i:i+chunk_size]
        result = vectors_to_precision_optimized(chunk, dim, base_epsilon)
        results.append(result)
        # KEEP - Major chunk processing
        del chunk, result
        if i % (chunk_size * 4) == 0:  # Less frequent cache clearing
            torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

def vectors_to_precision_optimized_new(vectors, dim, base_epsilon=DEFAULT_EPSILON):
    """
    Highly optimized version with minimal redundant operations
    """
    if torch.isnan(vectors).any():
        print("❌ NaNs in vectors before conversion to precision!")
        vectors = torch.nan_to_num(vectors, nan=0.0, posinf=1.0, neginf=-1.0)
    
    batch_size = vectors.shape[0]
    device = vectors.device
    dtype = vectors.dtype

    # Pre-allocate all tensors at once
    L = torch.empty(batch_size, dim, dim, dtype=dtype, device=device)
    
    # Use torch.triu_indices for vectorized assignment (faster than tril)
    triu_indices = torch.triu_indices(dim, dim, device=device)
    
    # Zero out and fill lower triangular part
    L.zero_()
    L[:, triu_indices[1], triu_indices[0]] = vectors  # Note: swapped indices for lower tri
    
    # Vectorized diagonal softplus
    diag_mask = torch.eye(dim, dtype=torch.bool, device=device)
    L[:, diag_mask] = F.softplus(L[:, diag_mask]) + 1e-6
    
    # Single matrix multiplication
    Lt = L.transpose(-2, -1)
    C = torch.bmm(L, Lt)

    # Single-pass regularization
    adaptive_eps = adaptive_regularization_fast_new(C, base_epsilon)
    adaptive_eps = torch.clamp(adaptive_eps, min=base_epsilon * 5)  # More aggressive minimum
    eye = torch.eye(dim, device=device, dtype=dtype)
    precision = C + adaptive_eps * eye
    
    # Skip expensive eigendecomposition and use simpler validation
    # Most matrices will be positive definite after proper regularization
    return precision

def vectors_to_precision_chunked_optimized_new(vectors, dim, base_epsilon=DEFAULT_EPSILON, chunk_size=DEFAULT_CENTERS_CHUNK_SIZE):
    results = []
    for i in range(0, vectors.size(0), chunk_size):
        chunk = vectors[i:i+chunk_size]
        result = vectors_to_precision_optimized_new(chunk, dim, base_epsilon)
        results.append(result)
        # KEEP - Major chunk processing
        del chunk, result
        if i % (chunk_size * DEFAULT_CLEAR_CACHE_FREQUENCY) == 0:  # Less frequent cache clearing
            torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

def vectors_to_precision_precomputed(vectors, dim, base_epsilon=DEFAULT_EPSILON,
                                     eye=PRECISION_EYE, tril_indices=TRIL_INDICES):
    batch_size = vectors.shape[0]
    device = vectors.device
    dtype = vectors.dtype

    # Allocate L
    L = torch.zeros(batch_size, dim, dim, dtype=dtype, device=device)

    # Fill lower-triangular part
    L[:, tril_indices[0], tril_indices[1]] = vectors

    # Out-of-place softplus on diagonal
    diag = F.softplus(L.diagonal(dim1=-2, dim2=-1)) + 1e-6
    L = L + torch.diag_embed(diag - L.diagonal(dim1=-2, dim2=-1))

    # Covariance
    C = torch.bmm(L, L.transpose(-2, -1))

    # Adaptive regularization
    adaptive_eps = adaptive_regularization_fast_new(C, base_epsilon)
    adaptive_eps = torch.clamp(adaptive_eps, min=base_epsilon*5)

    # Precision
    precision = C + adaptive_eps * eye.to(device=C.device, dtype=C.dtype)

    return precision

def vectors_to_precision_precomputed_chunked(vectors, dim, base_epsilon=DEFAULT_EPSILON, chunk_size=DEFAULT_CENTERS_CHUNK_SIZE, eye=PRECISION_EYE, tril_indices=TRIL_INDICES):
    results = []
    for i in range(0, vectors.size(0), chunk_size):
        chunk = vectors[i:i+chunk_size]
        result = vectors_to_precision_precomputed(chunk, dim, base_epsilon, eye, tril_indices)
        results.append(result)
        # KEEP - Major chunk processing
        del chunk, result
        if i % (chunk_size * DEFAULT_CLEAR_CACHE_FREQUENCY) == 0:  # Less frequent cache clearing
            torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

#----------------------------------------------#
#### Computation Functions ####
#----------------------------------------------#
def adaptive_regularization_fast_new(matrices, base_epsilon=DEFAULT_EPSILON, max_cond=DEFAULT_MAX_COND):
    """
    Faster adaptive regularization using JIT-compiled condition estimates
    """
    batch_size = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype
    
    try:
        # Reuse the compiled condition estimate function
        cond_estimate = _fast_condition_estimate_jit(matrices)  # (B,)
        cond_estimate = torch.clamp(cond_estimate, 1.0, max_cond)
        
        # Adaptive epsilon: scale with sqrt of condition number (adjust the 1e6 as needed)
        adaptive_eps = base_epsilon * torch.sqrt(cond_estimate / 1e6)
        return adaptive_eps.view(-1, 1, 1)
    
    except Exception:
        return torch.full((batch_size, 1, 1), base_epsilon, device=device, dtype=dtype)

def stable_logdet(matrices, eps=1e-6, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """
    Compute log determinant with multiple fallback methods for stability
    """
    batch_size = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype
    
    for attempt in range(max_attempts):
        current_eps = eps * (10 ** attempt)  # Increase regularization with each attempt
        
        try:
            # Method 1: Cholesky-based (most stable for PD matrices)
            regularized = matrices + current_eps * torch.eye(matrices.shape[-1], device=device, dtype=dtype)
            chol = torch.linalg.cholesky(regularized)
            logdet = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)
            
            # Check for valid results
            if torch.all(torch.isfinite(logdet)):
                return logdet
                
        except RuntimeError:
            pass
    
    # Final fallback: SVD-based method
    try:
        regularized = matrices + eps * 10 * torch.eye(matrices.shape[-1], device=device, dtype=dtype)
        return torch.logdet(regularized)
    except:
        # Ultimate fallback: return safe default values
        return torch.full((batch_size,), -10.0, device=device, dtype=dtype)

def stable_logdet_hybrid(matrices, eps=1e-6, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """
    Fast and stable log-determinant computation for small matrices.
    Uses per-matrix Cholesky when possible, with fallback to slogdet.S
    
    matrices: (B, D, D) tensor
    eps: base regularization
    max_attempts: number of regularization retries
    """
    batch_size, dim, _ = matrices.shape
    device, dtype = matrices.device, matrices.dtype
    I = torch.eye(dim, device=device, dtype=dtype)
    logdet = torch.empty(batch_size, device=device, dtype=dtype)

    for i in range(batch_size):
        for attempt in range(max_attempts):
            current_eps = eps * (10 ** attempt)
            try:
                L = torch.linalg.cholesky(matrices[i] + current_eps * I)
                logdet[i] = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
                break  # success, exit attempt loop
            except RuntimeError:
                # fallback to slogdet
                sign, ld = torch.linalg.slogdet(matrices[i] + current_eps * I)
                if sign > 0:
                    logdet[i] = ld
                    break
                elif attempt == max_attempts - 1:
                    # final fallback for pathological matrices
                    logdet[i] = -1e10
    return logdet

def stable_softmax(logits, temperature=1.0, dim=-1, eps=1e-8):
    """
    Numerically stable softmax with temperature scaling
    """
    # Scale by temperature
    logits_scaled = logits / temperature
    
    # Subtract max for numerical stability
    logits_max = torch.max(logits_scaled, dim=dim, keepdim=True)[0]
    logits_stable = logits_scaled - logits_max
    
    # Clamp to prevent extreme values
    logits_clamped = torch.clamp(logits_stable, min=-DEFAULT_LOGITS_CLAMP, max=DEFAULT_LOGITS_CLAMP)
    
    return F.softmax(logits_clamped, dim=dim)

def stable_softmax_optimized(logits, temperature=1.0, dim=-1):
    """
    Optimized stable softmax with fewer operations
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    # Combined max subtraction and clamping
    logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
    logits_stable = torch.clamp(logits - logits_max, min=-DEFAULT_LOGITS_CLAMP, max=DEFAULT_LOGITS_CLAMP)
    
    return F.softmax(logits_stable, dim=dim)

def stable_softmax_inplace(logits, temperature=1.0, dim=-1):
    """
    Memory-efficient stable softmax
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    # In-place operations where possible
    logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
    logits.sub_(logits_max)  # In-place subtraction
    logits.clamp_(min=-DEFAULT_LOGITS_CLAMP, max=DEFAULT_LOGITS_CLAMP)  # In-place clamping
    
    return F.softmax(logits, dim=dim)

#----------------------------------------------#
#### Loss Functions ####
#----------------------------------------------#
def grad_and_laplacian_mog_density_stable(x, means, precisions, temperature=1.0):
    """
    Optimized numerically stable version of gradient and Laplacian computation
    """
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, dim)
    dim = x.size(-1)
    batch_size, num_components = x.size(0), means.size(0)

    # Compute shared terms
    x_mean = x - means  # Shape: (batch_size, num_components, dim)
    
    # P * (x - μ) - used for both gradient and Laplacian
    x_mean_cov = torch.einsum('kij,bkj->bki', precisions, x_mean)  # (B, K, D)
    
    # (x-μ)ᵀP(x-μ) - used for log probs and Laplacian
    squared_linear = torch.sum(x_mean_cov.square(), dim=-1)  # (B, K) - faster than * multiplication
    
    # Precompute and cache log determinant and trace
    #logdet = stable_logdet(precisions)  # (K,)
    logdet = stable_logdet_hybrid(precisions)
    trace_precision = torch.diagonal(precisions, dim1=-2, dim2=-1).sum(dim=-1)  # (K,)
    
    # Vectorized expansion (faster than .expand())
    logdet_expanded = logdet.unsqueeze(0).expand(batch_size, -1)  # (B, K)
    trace_expanded = trace_precision.unsqueeze(0).expand(batch_size, -1)  # (B, K)


    # Compute log probabilities with numerical stability
    log_probs = 0.5 * (logdet_expanded - squared_linear) - 0.5 * dim * torch.log(
        torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
    
    # Clamp log probabilities to prevent extreme values
    log_probs = torch.clamp(log_probs, min=-DEFAULT_LOGPROB_CLAMP, max=DEFAULT_LOGPROB_CLAMP)
    
    # Compute stable softmax probabilities
    softmax_probs = stable_softmax_jit(log_probs, temperature=temperature)  # (B, K)
    
    # GRADIENT COMPUTATION - vectorized
    gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)  # (B, D)
    
    # LAPLACIAN COMPUTATION - vectorized
    laplacian_component = softmax_probs * (squared_linear - trace_expanded)  # (B, K)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)  # (B,)
    
    # Check for numerical issues
    if torch.any(torch.isnan(gradient)) or torch.any(torch.isnan(laplacian_over_density)):
        print("Warning: NaN detected in gradient or Laplacian computation")
        # Return safe fallback values
        gradient = torch.zeros_like(gradient)
        laplacian_over_density = torch.zeros_like(laplacian_over_density)
    
    return gradient, laplacian_over_density

def grad_and_laplacian_mog_density_chunked_stable(x, means, precisions, 
                                                  batch_chunk_size=DEFAULT_BATCH_CHUNK_SIZE, 
                                                  component_chunk_size=DEFAULT_CENTERS_CHUNK_SIZE,
                                                  temperature=1.0):
    """
    Memory-efficient stable version with optimized chunking
    """
    # Pre-allocate output tensors for better memory efficiency
    total_batches = x.size(0)
    dim = x.size(-1)
    
    gradients = torch.empty(total_batches, dim, dtype=x.dtype, device=x.device)
    laplacians = torch.empty(total_batches, dtype=x.dtype, device=x.device)
    
    # Process in chunks
    for start in range(0, total_batches, batch_chunk_size):
        end = min(start + batch_chunk_size, total_batches)
        x_chunk = x[start:end]
        
        # Use component chunking for large numbers of components
        if means.size(0) > component_chunk_size:
            grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_component_chunked_stable(
                x_chunk, means, precisions, chunk_size=component_chunk_size, temperature=temperature)
        else:
            grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_stable(
                x_chunk, means, precisions, temperature=temperature)
        
        # Direct assignment instead of appending and concatenating
        gradients[start:end] = grad_chunk.detach() if not grad_chunk.requires_grad else grad_chunk
        laplacians[start:end] = laplacian_chunk.detach() if not laplacian_chunk.requires_grad else laplacian_chunk
    
    return gradients, laplacians

def grad_and_laplacian_mog_density_component_chunked_stable(x, means, precisions, 
                                                           chunk_size=DEFAULT_CENTERS_CHUNK_SIZE, 
                                                           temperature=1.0):
    """
    Optimized component chunking with better memory management
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    
    # Pre-allocate tensors for accumulation
    all_log_probs = torch.empty(batch_size, num_components, dtype=x.dtype, device=x.device)
    all_weighted_gradients = torch.empty(batch_size, num_components, dim, dtype=x.dtype, device=x.device)
    all_squared_linear = torch.empty(batch_size, num_components, dtype=x.dtype, device=x.device)
    all_trace_precision = torch.empty(batch_size, num_components, dtype=x.dtype, device=x.device)
    
    # Pre-compute constants
    log_2pi = torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
    
    for start in range(0, num_components, chunk_size):
        end = min(start + chunk_size, num_components)
        chunk_len = end - start
        
        means_chunk = means[start:end]
        precisions_chunk = precisions[start:end]
        
        # Vectorized operations
        x_expanded = x.unsqueeze(1)  # (B, 1, D)
        x_mean = x_expanded - means_chunk  # (B, K_chunk, D)
        
        x_mean_cov = torch.einsum("kij,bkj->bki", precisions_chunk, x_mean)
        squared_linear = torch.sum(x_mean_cov.square(), dim=-1)  # Use .square() instead of *
        
        # Precompute log_det and trace for chunk
        #log_det = stable_logdet(precisions_chunk)  # (K_chunk,)
        log_det = stable_logdet_hybrid(precisions_chunk)  # (K_chunk,)
        trace_chunk = torch.diagonal(precisions_chunk, dim1=-2, dim2=-1).sum(dim=-1)  # (K_chunk,)
        
        # Broadcast efficiently
        log_det_expanded = log_det.unsqueeze(0).expand(batch_size, -1)
        trace_expanded = trace_chunk.unsqueeze(0).expand(batch_size, -1)
        
        log_probs_chunk = 0.5 * (log_det_expanded - squared_linear) - 0.5 * dim * log_2pi
        log_probs_chunk = torch.clamp(log_probs_chunk, min=-DEFAULT_LOGPROB_CLAMP, max=DEFAULT_LOGPROB_CLAMP)
        
        weighted_gradient_chunk = -x_mean_cov
        
        # Direct assignment to pre-allocated tensors
        all_log_probs[:, start:end] = log_probs_chunk
        all_weighted_gradients[:, start:end] = weighted_gradient_chunk
        all_squared_linear[:, start:end] = squared_linear
        all_trace_precision[:, start:end] = trace_expanded
        
        # Clean up chunk-specific tensors
        del x_mean, x_mean_cov, squared_linear, log_det, log_probs_chunk, weighted_gradient_chunk, trace_chunk
        
        # Less frequent cache clearing
        if start % (chunk_size * 5) == 0:  # Reduced frequency
            torch.cuda.empty_cache()
    
    # Final computations
    #softmax_probs = stable_softmax_optimized(all_log_probs, temperature=temperature, dim=1) #optimized version uses less ram and is faster
    softmax_probs = stable_softmax_jit(all_log_probs, temperature=temperature)
    gradient = torch.sum(softmax_probs.unsqueeze(-1) * all_weighted_gradients, dim=1)
    
    laplacian_component = softmax_probs * (all_squared_linear - all_trace_precision)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    return gradient, laplacian_over_density

# Alternative: Ultra-fast version for when you can afford more memory
def grad_and_laplacian_mog_density_vectorized(x, means, precisions, temperature=1.0):
    """
    Fully vectorized version - fastest but uses more memory
    Use when memory is not a constraint
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    
    # Expand all at once
    x_expanded = x.unsqueeze(1)  # (B, 1, D)
    x_mean = x_expanded - means  # (B, K, D)
    
    # Single large einsum - fastest approach
    x_mean_cov = torch.einsum('kij,bkj->bki', precisions, x_mean)
    squared_linear = torch.sum(x_mean_cov.square(), dim=-1)
    
    # Precompute all constants
    #logdet = stable_logdet(precisions)
    logdet = stable_logdet_hybrid(precisions)
    trace_precision = torch.diagonal(precisions, dim1=-2, dim2=-1).sum(dim=-1)
    log_2pi_term = 0.5 * dim * torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
    
    # Vectorized log probability computation
    log_probs = 0.5 * (logdet.unsqueeze(0) - squared_linear) - log_2pi_term
    log_probs = torch.clamp(log_probs, min=-DEFAULT_LOGPROB_CLAMP, max=DEFAULT_LOGPROB_CLAMP)
    
    # Stable softmax
    #softmax_probs = stable_softmax_optimized(log_probs, temperature=temperature, dim=1)
    softmax_probs = stable_softmax_jit(log_probs, temperature=temperature)


    # Vectorized gradient and Laplacian
    gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)
    laplacian_component = softmax_probs * (squared_linear - trace_precision.unsqueeze(0))
    laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    return gradient, laplacian_over_density

def grad_and_laplacian_mog_density_fused(
    x, means, precisions,
    batch_chunk_size=DEFAULT_BATCH_CHUNK_SIZE,
    center_chunk_size=DEFAULT_CENTERS_CHUNK_SIZE,
    temperature=1.0
):
    """
    Fused and double-chunked computation of gradient and laplacian for MoG.
    Reduces intermediate memory allocations and is faster than the separate steps.
    """
    device = x.device
    dtype = x.dtype
    batch_size, dim = x.shape
    num_components = means.shape[0]

    gradient = torch.zeros(batch_size, dim, dtype=dtype, device=device)
    laplacian_over_density = torch.zeros(batch_size, dtype=dtype, device=device)
    log_2pi = torch.log(torch.tensor(2 * torch.pi, device=device, dtype=dtype))

    for b_start in range(0, batch_size, batch_chunk_size):
        b_end = min(b_start + batch_chunk_size, batch_size)
        x_chunk = x[b_start:b_end]  # (B_chunk, D)
        B_chunk = b_end - b_start

        grad_chunk_accum = torch.zeros(B_chunk, dim, dtype=dtype, device=device)
        lap_chunk_accum = torch.zeros(B_chunk, dtype=dtype, device=device)

        for c_start in range(0, num_components, center_chunk_size):
            c_end = min(c_start + center_chunk_size, num_components)
            K_chunk = c_end - c_start

            means_chunk = means[c_start:c_end]
            precisions_chunk = precisions[c_start:c_end]

            # x_mean: (B_chunk, K_chunk, D)
            x_mean = x_chunk.unsqueeze(1) - means_chunk.unsqueeze(0)

            # Compute x_mean_cov and squared_linear in one einsum
            x_mean_cov = torch.einsum("bkd,kde->bke", x_mean, precisions_chunk)
            squared_linear = (x_mean_cov * x_mean_cov).sum(-1)  # (B_chunk, K_chunk)

            # Precompute logdet and trace
            log_det = stable_logdet_hybrid(precisions_chunk)
            trace_chunk = precisions_chunk.diagonal(dim1=-2, dim2=-1).sum(-1)

            # Fused log-probabilities
            log_probs = 0.5 * (log_det.unsqueeze(0) - squared_linear) - 0.5 * dim * log_2pi
            log_probs.clamp_(-DEFAULT_LOGPROB_CLAMP, DEFAULT_LOGPROB_CLAMP)

            softmax_probs = stable_softmax_jit(log_probs, temperature=temperature)

            grad_chunk_accum += torch.sum(softmax_probs.unsqueeze(-1) * (-x_mean_cov), dim=1)
            lap_chunk_accum += torch.sum(softmax_probs * (squared_linear - trace_chunk.unsqueeze(0)), dim=1)

            # Cleanup
            del x_mean, x_mean_cov, squared_linear, log_probs, softmax_probs, log_det, trace_chunk
            torch.cuda.empty_cache()

        gradient[b_start:b_end] = grad_chunk_accum
        laplacian_over_density[b_start:b_end] = lap_chunk_accum

        del grad_chunk_accum, lap_chunk_accum
        torch.cuda.empty_cache()

    return gradient, laplacian_over_density

def score_implicit_matching_stable(factornet, samples, centers, base_epsilon=DEFAULT_EPSILON, 
                                  temperature=DEFAULT_TEMPERATURE, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """
    Enhanced numerically stable version with strategic memory management
    """
    torch.cuda.reset_peak_memory_stats()
    dim = centers.shape[-1]
    centers = centers.clone()

    # Precomputed identity matrix
    #PRECISION_EYE = torch.eye(dim, device=centers.device, dtype=centers.dtype)

    # Precomputed lower-triangular indices
    #TRIL_INDICES = torch.tril_indices(row=dim, col=dim, offset=0, device=centers.device)

    for attempt in range(max_attempts):
        # Initialize variables to None for proper cleanup
        factor_eval = None
        precisions = None
        gradient_eval_log = None
        laplacian_over_density = None
        gradient_eval_log_squared = None
        loss = None
        
        try:
            if attempt != 0:
                print(f"\n⏱️ Attempt {attempt + 1}/{max_attempts}")
            #t0 = time.time()
            current_epsilon = base_epsilon * (2 ** attempt)

            # === FACTORNET FORWARD ===
            factor_eval = factornet(centers)

            if not numerical_health_check(factor_eval, "factor_eval"):
                if attempt < max_attempts - 1:
                    print("❌ Factor eval health check failed, retrying...")
                    raise ValueError("Factor eval health check failed")
                print("⚠️ Proceeding with potentially unstable factor network output")

            # === PRECISION CONSTRUCTION ===
            precisions = vectors_to_precision_chunked_optimized_new(factor_eval, dim, current_epsilon, DEFAULT_CENTERS_CHUNK_SIZE)
            #precisions = vectors_to_precision_precomputed_chunked(factor_eval, dim, current_epsilon, chunk_size=DEFAULT_CENTERS_CHUNK_SIZE, eye=PRECISION_EYE, tril_indices=TRIL_INDICES)
            if not numerical_health_check(precisions, "precisions"):
                if attempt < max_attempts - 1:
                    print("❌ Precisions health check failed, retrying...")
                    raise ValueError("Precisions health check failed")
                print("⚠️ Proceeding with potentially unstable precision matrices")

            # === GRAD & LAPLACIAN ===
            # gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_chunked_stable(
            #     samples, centers, precisions, temperature=temperature)
            gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_fused(
                samples, centers, precisions,
                batch_chunk_size=DEFAULT_BATCH_CHUNK_SIZE,
                center_chunk_size=DEFAULT_CENTERS_CHUNK_SIZE,
                temperature=temperature
            )
            if not numerical_health_check(gradient_eval_log, "gradient_eval_log"):
                if attempt < max_attempts - 1:
                    print("❌ Gradient eval health check failed, retrying...")
                    raise ValueError("Gradient eval health check failed")
                print("⚠️ Proceeding with potentially unstable gradient eval")
                
            if not numerical_health_check(laplacian_over_density, "laplacian_over_density"):
                if attempt < max_attempts - 1:
                    print("❌ Laplacian health check failed, retrying...")
                    raise ValueError("Laplacian health check failed")
                print("⚠️ Proceeding with potentially unstable laplacian")

            # === FINAL LOSS ===
            #t4 = time.time()
            gradient_eval_log_squared = torch.sum(gradient_eval_log.square(), dim=1)
            loss = 2 * laplacian_over_density - gradient_eval_log_squared

            #total_time = time.time() - t0
            if numerical_health_check(loss, "loss"):
                return loss.mean(dim=0)
            elif attempt < max_attempts - 1:
                print(f"❌ Loss health check failed at attempt {attempt + 1}, retrying...")
                raise ValueError("Loss health check failed")
            else:
                print("⚠️ Returning clamped fallback loss")
                # Clamp but retain gradient path
                loss = loss.mean(dim=0)  # preserve reduction over batch
                clamped_loss = torch.clamp(loss, min=-1e10, max=1e10)
                return clamped_loss

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed with error: {e}")
            # Immediate aggressive cleanup on any failure
            cleanup_vars = [factor_eval, precisions, gradient_eval_log, 
                          laplacian_over_density, gradient_eval_log_squared, loss]
            for var in cleanup_vars:
                if var is not None:
                    del var
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if attempt < max_attempts - 1:
                print("Retrying with increased regularization...")
            else:
                # Re-raise on final attempt
                raise e
        
        finally:
            # Proper cleanup - this runs whether we succeed, fail, or continue
            cleanup_vars = [factor_eval, precisions, gradient_eval_log, 
                          laplacian_over_density, gradient_eval_log_squared, loss]
            
            for var in cleanup_vars:
                if var is not None:
                    del var
            
            # Only do expensive cleanup on failures or retries
            if attempt < max_attempts - 1:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for cleanup to complete
    
    # This should never be reached, but just in case
    raise RuntimeError("All attempts failed and no valid result was returned")

def score_implicit_matching_stable_optimized(factornet, samples, centers, base_epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """
    Memory-optimized version with aggressive chunking and minimal allocations
    Specifically designed to handle large numbers of centers (1000+) without OOM
    """
    dim = centers.shape[-1]
    centers = centers.clone()
    batch_size = samples.shape[0]
    num_centers = centers.shape[0]
    
    # Dynamic chunk sizing based on available memory and number of centers
    if num_centers > 1000:
        center_chunk_size = min(50, DEFAULT_CENTERS_CHUNK_SIZE // 2)  # Smaller chunks for many centers
        batch_chunk_size = min(32, DEFAULT_BATCH_CHUNK_SIZE // 2)
    else:
        center_chunk_size = DEFAULT_CENTERS_CHUNK_SIZE
        batch_chunk_size = DEFAULT_BATCH_CHUNK_SIZE

    for attempt in range(max_attempts):
        try:
            if attempt != 0:
                print(f"\n⏱️ Optimized attempt {attempt + 1}/{max_attempts}")
            
            current_epsilon = base_epsilon * (2 ** attempt)
            
            # === FACTORNET FORWARD WITH CHUNKING ===
            # factor_chunks = []
            # for i in range(0, num_centers, center_chunk_size):
            #     chunk_end = min(i + center_chunk_size, num_centers)
            #     centers_chunk = centers[i:chunk_end]

            #     with torch.cuda.device(centers.device):
            #         # Regular float32 precision - stable for all center counts
            #         factor_chunk = factornet(centers_chunk)
            #         factor_chunks.append(factor_chunk.detach())  # Detach to save memory
                
            #     # Aggressive cleanup
            #     del centers_chunk, factor_chunk
            #     if i % (center_chunk_size * 4) == 0:
            #         torch.cuda.empty_cache()
            
            # # Concatenate factor chunks
            # factor_eval = torch.cat(factor_chunks, dim=0)
            # del factor_chunks
            factor_eval = factornet(centers)
            if not numerical_health_check(factor_eval, "factor_eval"):
                if attempt < max_attempts - 1:
                    print("❌ Factor eval health check failed, retrying...")
                    del factor_eval
                    torch.cuda.empty_cache()
                    raise ValueError("Factor eval health check failed")
                print("⚠️ Proceeding with potentially unstable factor network output")
            
            # === PRECISION CONSTRUCTION WITH MICRO-BATCHING ===
            # precision_chunks = []
            # for i in range(0, num_centers, center_chunk_size):
            #     chunk_end = min(i + center_chunk_size, num_centers)
            #     factor_chunk = factor_eval[i:chunk_end]
                
            #     # Use the most memory-efficient precision function
            #     precision_chunk = vectors_to_precision_optimized_new(factor_chunk, dim, current_epsilon)
            #     precision_chunks.append(precision_chunk)
                
            #     del factor_chunk, precision_chunk
            #     if i % (center_chunk_size * 2) == 0:
            #         torch.cuda.empty_cache()
            
            # precisions = torch.cat(precision_chunks, dim=0)
            # del precision_chunks, factor_eval
            precisions = vectors_to_precision_chunked_optimized_new(factor_eval, dim, current_epsilon, center_chunk_size)
            del factor_eval
            if not numerical_health_check(precisions, "precisions"):
                if attempt < max_attempts - 1:
                    print("❌ Precisions health check failed, retrying...")
                    del precisions
                    torch.cuda.empty_cache()
                    raise ValueError("Precisions health check failed")
                print("⚠️ Proceeding with potentially unstable precision matrices")
            
            # === GRAD & LAPLACIAN WITH AGGRESSIVE CHUNKING ===
            gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_fused(
                samples, centers, precisions,
                batch_chunk_size=batch_chunk_size,
                center_chunk_size=center_chunk_size,
                temperature=temperature
            )
            
            del precisions  # Free immediately after use
            
            if not numerical_health_check(gradient_eval_log, "gradient_eval_log"):
                if attempt < max_attempts - 1:
                    print("❌ Gradient eval health check failed, retrying...")
                    del gradient_eval_log, laplacian_over_density
                    torch.cuda.empty_cache()
                    raise ValueError("Gradient eval health check failed")
                print("⚠️ Proceeding with potentially unstable gradient eval")
                
            if not numerical_health_check(laplacian_over_density, "laplacian_over_density"):
                if attempt < max_attempts - 1:
                    print("❌ Laplacian health check failed, retrying...")
                    del gradient_eval_log, laplacian_over_density
                    torch.cuda.empty_cache()
                    raise ValueError("Laplacian health check failed")
                print("⚠️ Proceeding with potentially unstable laplacian")
            
            # === FINAL LOSS ===
            gradient_eval_log_squared = torch.sum(gradient_eval_log.square(), dim=1)
            loss = 2 * laplacian_over_density - gradient_eval_log_squared
            
            # Cleanup intermediate tensors
            del gradient_eval_log, laplacian_over_density, gradient_eval_log_squared
            
            if numerical_health_check(loss, "loss"):
                return loss.mean(dim=0)
            elif attempt < max_attempts - 1:
                print(f"❌ Loss health check failed at attempt {attempt + 1}, retrying...")
                del loss
                torch.cuda.empty_cache()
                raise ValueError("Loss health check failed")
            else:
                print("⚠️ Returning clamped fallback loss")
                loss = loss.mean(dim=0)
                clamped_loss = torch.clamp(loss, min=-1e10, max=1e10)
                return clamped_loss
        
        except Exception as e:
            print(f"❌ Optimized attempt {attempt + 1} failed with error: {e}")
            # Aggressive cleanup on failure
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if attempt < max_attempts - 1:
                print("Retrying with increased regularization...")
            else:
                raise e
    
    raise RuntimeError("All optimized attempts failed and no valid result was returned")
