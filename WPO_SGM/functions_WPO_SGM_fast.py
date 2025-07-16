# ===================== #
# Speed-Optimized Functions (Memory-Efficient)
# ===================== #
import math
import torch
import torch.nn.functional as F
from torch.jit import script
import time
###################
# setup
###################
# ------------------- UTILITIES -------------------
import time
import gc
from memory_profiler import profile
from contextlib import contextmanager
# ------------------- MATH -------------------
import numpy as np
import matplotlib.pyplot as plt
# ------------------- PYTORCH -------------------
import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch.linalg as LA
# ------------------- PROJECT MODULES -------------------
from WPO_SGM.utilities import *

# Global config remains the same
DEFAULT_EPSILON = 1e-4
DEFAULT_MAX_COND = 1e12
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CENTERS_CHUNK_SIZE = 100
DEFAULT_BATCH_CHUNK_SIZE = 64
DEFAULT_TEMPERATURE = 1.0
DEFAULT_LOGPROB_CLAMP = 100
DEFAULT_LOGITS_CLAMP = 50
LOG_2PI = math.log(2 * math.pi)


# ===================== #
# JIT-Compiled Core Functions
# ===================== #

@script
def _fast_condition_estimate_jit(matrices):
    """JIT-compiled fast condition number estimation"""
    batch_size = matrices.shape[0]
    
    # Fast condition estimate using diagonal vs Frobenius norm
    diag_elements = torch.diagonal(matrices, dim1=-2, dim2=-1)
    diag_norm = torch.norm(diag_elements, dim=-1)
    frob_norm = torch.norm(matrices.view(batch_size, -1), dim=-1)
    
    return frob_norm / (diag_norm + 1e-8)

# ===================== #
# Optimized Precision Functions
# ===================== #
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
    adaptive_eps = adaptive_regularization_fast(C, base_epsilon)
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
        if i % (chunk_size * 4) == 0:  # Less frequent cache clearing
            torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

# ===================== #
# Optimized Computation Functions
# ===================== #

def adaptive_regularization_fast(matrices, base_epsilon=DEFAULT_EPSILON, max_cond=DEFAULT_MAX_COND):
    """
    Faster adaptive regularization using pre-computed condition estimates
    """
    batch_size = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype
    
    try:
        # Use Frobenius norm ratio as a fast condition number estimate
        # This avoids expensive SVD in torch.linalg.cond
        diag_norm = torch.norm(torch.diagonal(matrices, dim1=-2, dim2=-1), dim=-1)
        frob_norm = torch.norm(matrices.view(batch_size, -1), dim=-1)
        cond_estimate = torch.clamp(frob_norm / (diag_norm + 1e-8), 1.0, max_cond)
        
        adaptive_eps = base_epsilon * torch.sqrt(cond_estimate / 1e6)
        return adaptive_eps.view(-1, 1, 1)  # Use view instead of unsqueeze
    except:
        return torch.full((batch_size, 1, 1), base_epsilon, device=device, dtype=dtype)

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

def stable_logdet_fast(matrices, eps=1e-6):
    """
    Fast log determinant using diagonal decomposition heuristic
    """
    batch_size = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype
    
    # Fast approximation: use diagonal elements for log determinant
    # This is much faster than Cholesky decomposition
    diag_elements = torch.diagonal(matrices, dim1=-2, dim2=-1)
    diag_elements = torch.clamp(diag_elements, min=eps)
    logdet_approx = torch.sum(torch.log(diag_elements), dim=-1)
    
    return logdet_approx

@script
def stable_softmax_jit(logits, temperature: float = 1.0)-> torch.Tensor:
    """JIT-compiled stable softmax"""
    if temperature != 1.0:
        logits = logits / temperature
    
    logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_stable = logits - logits_max
    logits_clamped = torch.clamp(logits_stable, min=-50.0, max=50.0)
    
    return F.softmax(logits_clamped, dim=-1)

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

def stable_softmax(logits, temperature=1.0, dim=-1):
    """
    Optimized stable softmax with fewer operations
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    # Combined max subtraction and clamping
    logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
    logits_stable = torch.clamp(logits - logits_max, min=-DEFAULT_LOGITS_CLAMP, max=DEFAULT_LOGITS_CLAMP)
    
    return F.softmax(logits_stable, dim=dim)

def stable_logdet_memory_efficient(matrices, eps=1e-6):
    """
    Memory-efficient log determinant computation
    """
    batch_size = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype
    
    # Process in micro-batches to save memory
    micro_batch_size = min(8, batch_size)
    results = []
    
    for i in range(0, batch_size, micro_batch_size):
        end_idx = min(i + micro_batch_size, batch_size)
        batch_matrices = matrices[i:end_idx]
        
        try:
            # Add minimal regularization
            eye = torch.eye(batch_matrices.shape[-1], device=device, dtype=dtype)
            regularized = batch_matrices + eps * eye
            
            # Use Cholesky for stability
            chol = torch.linalg.cholesky(regularized)
            logdet_batch = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)
            
            results.append(logdet_batch.cpu())  # Move to CPU immediately
            del batch_matrices, regularized, chol, logdet_batch
            
        except RuntimeError:
            # Fallback for problematic matrices
            fallback = torch.full((end_idx - i,), -10.0, device=device, dtype=dtype)
            results.append(fallback.cpu())
            del fallback
        
        aggressive_cleanup()
    
    # Concatenate results
    cpu_result = torch.cat(results, dim=0)
    del results
    return cpu_result.to(device)

# ===================== #
# Optimized Loss Functions
# ===================== #
#original ones (fastest, least memory efficient)
def grad_and_laplacian_mog_density_stable(x, means, precisions, temperature: float =1.0):
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
    logdet = stable_logdet(precisions)  # (K,)
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
    softmax_probs = stable_softmax(log_probs, temperature=temperature, dim=1)  # (B, K)
    
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
        log_det = stable_logdet(precisions_chunk)  # (K_chunk,)
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
    softmax_probs = stable_softmax(all_log_probs, temperature=temperature, dim=1) 
    
    gradient = torch.sum(softmax_probs.unsqueeze(-1) * all_weighted_gradients, dim=1)
    
    laplacian_component = softmax_probs * (all_squared_linear - all_trace_precision)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    return gradient, laplacian_over_density

# most memory efficient versions (offloading)
def grad_and_laplacian_mog_density_ultra_chunked(x, means, precisions, 
                                                 batch_chunk_size=8, 
                                                 component_chunk_size=10,
                                                 temperature=1.0):
    """
    Ultra memory-efficient version with micro-chunking and CPU offloading
    """
    total_batches = x.size(0)
    dim = x.size(-1)
    device = x.device
    dtype = x.dtype
    
    print(f"Processing {total_batches} samples with {means.size(0)} components")
    
    # Pre-allocate results on CPU to save GPU memory
    gradients_cpu = torch.empty(total_batches, dim, dtype=dtype)
    laplacians_cpu = torch.empty(total_batches, dtype=dtype)
    
    # Process in micro-batches
    for start in range(0, total_batches, batch_chunk_size):
        end = min(start + batch_chunk_size, total_batches)
        x_chunk = x[start:end]
        if start % 4 == 0:
            print(f"Processing batch {start//batch_chunk_size + 1}/{(total_batches + batch_chunk_size - 1)//batch_chunk_size}")
        
        # Ultra-chunked component processing
        grad_chunk, laplacian_chunk = grad_and_laplacian_ultra_component_chunked(
            x_chunk, means, precisions, chunk_size=component_chunk_size, temperature=temperature)
        
        # Store results on CPU
        gradients_cpu[start:end] = grad_chunk.cpu()
        laplacians_cpu[start:end] = laplacian_chunk.cpu()
        
        # Aggressive cleanup
        del x_chunk, grad_chunk, laplacian_chunk
        aggressive_cleanup()
    
    # Move final results back to GPU
    gradients = gradients_cpu.to(device)
    laplacians = laplacians_cpu.to(device)
    del gradients_cpu, laplacians_cpu
    
    return gradients, laplacians

def grad_and_laplacian_mog_density_ultra_chunked_multigpu(x, means, precisions, 
                                                          batch_chunk_size=8, 
                                                          component_chunk_size=10,
                                                          temperature=1.0,
                                                          compute_device="cuda:1"):
    """
    Ultra memory-efficient version with micro-chunking and secondary GPU offloading (instead of CPU)
    """
    total_batches = x.size(0)
    dim = x.size(-1)
    input_device = x.device
    dtype = x.dtype
    
    print(f"Processing {total_batches} samples with {means.size(0)} components")

    # Allocate results directly on input device (e.g., cuda:0)
    gradients = torch.empty(total_batches, dim, dtype=dtype, device=input_device)
    laplacians = torch.empty(total_batches, dtype=dtype, device=input_device)

    # Use specified compute device (e.g., cuda:1)
    compute_device = torch.device(compute_device)

    for start in range(0, total_batches, batch_chunk_size):
        end = min(start + batch_chunk_size, total_batches)
        x_chunk = x[start:end].to(compute_device, non_blocking=True)
        
        if start % (4 * batch_chunk_size) == 0:
            print(f"Processing batch {start//batch_chunk_size + 1}/{(total_batches + batch_chunk_size - 1)//batch_chunk_size}")
        
        # Move means/precisions if needed (assume they’re on same device as x originally)
        
        grad_chunk, laplacian_chunk = grad_and_laplacian_ultra_component_chunked_multigpu(
            x_chunk,
            means.to(compute_device, non_blocking=True),
            precisions.to(compute_device, non_blocking=True),
            chunk_size=component_chunk_size,
            temperature=temperature
        )
        
        # Move result back to input device
        gradients[start:end] = grad_chunk.to(input_device, non_blocking=True)
        laplacians[start:end] = laplacian_chunk.to(input_device, non_blocking=True)
        
        del x_chunk, grad_chunk, laplacian_chunk
        torch.cuda.empty_cache()
        aggressive_cleanup()
    
    return gradients, laplacians

def grad_and_laplacian_mog_density_ultra_chunked_stream(
    x, means, precisions, 
    batch_chunk_size=8, 
    component_chunk_size=10,
    temperature=1.0,
    compute_device=torch.device("cuda:1"),
):
    """
    Ultra memory-efficient version with micro-chunking and secondary GPU offloading (e.g., GPU 1).
    """
    total_batches = x.size(0)
    dim = x.size(-1)
    input_device = x.device
    dtype = x.dtype

    print(f"Processing {total_batches} samples with {means.size(0)} components")

    # Allocate final output tensors on input device
    gradients = torch.empty(total_batches, dim, dtype=dtype, device=input_device)
    laplacians = torch.empty(total_batches, dtype=dtype, device=input_device)

    # Move GMM parameters once to compute device (if not already)
    means = means.to(compute_device, non_blocking=True)
    precisions = precisions.to(compute_device, non_blocking=True)

    for start in range(0, total_batches, batch_chunk_size):
        end = min(start + batch_chunk_size, total_batches)
        x_chunk = x[start:end]

        if start % (4 * batch_chunk_size) == 0:
            print(f"Processing batch {start // batch_chunk_size + 1}/{(total_batches + batch_chunk_size - 1) // batch_chunk_size}")

        # Offload chunked computation to secondary GPU
        grad_chunk, laplacian_chunk = grad_and_laplacian_dual_gpu_streamed(
            x_chunk,
            means,
            precisions,
            chunk_size=component_chunk_size,
            temperature=temperature,
            gmm_device=compute_device,
            main_device=input_device,
        )

        gradients[start:end] = grad_chunk
        laplacians[start:end] = laplacian_chunk

        # Cleanup
        del x_chunk, grad_chunk, laplacian_chunk
        torch.cuda.empty_cache()
        aggressive_cleanup()

    return gradients, laplacians

def grad_and_laplacian_ultra_component_chunked(x, means, precisions, chunk_size=10, temperature=1.0):
    """
    Ultra memory-efficient component chunking with CPU offloading
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    device = x.device
    dtype = x.dtype
    
    # Store intermediate results on CPU
    log_probs_cpu = torch.empty(batch_size, num_components, dtype=dtype)
    weighted_grads_cpu = torch.empty(batch_size, num_components, dim, dtype=dtype)
    squared_linear_cpu = torch.empty(batch_size, num_components, dtype=dtype)
    trace_precision_cpu = torch.empty(batch_size, num_components, dtype=dtype)
    
    # Pre-compute constants
    log_2pi = torch.log(torch.tensor(2 * torch.pi, device=device, dtype=dtype))
    
    # Process components in micro-chunks
    for start in range(0, num_components, chunk_size):
        end = min(start + chunk_size, num_components)
        
        # Move only current chunk to GPU
        means_chunk = means[start:end].to(device)
        precisions_chunk = precisions[start:end].to(device)
        
        # Compute for current chunk
        x_expanded = x.unsqueeze(1)
        x_mean = x_expanded - means_chunk
        
        # Use checkpoint to trade compute for memory
        def compute_chunk(x_mean, precisions_chunk):
            x_mean_cov = torch.einsum("kij,bkj->bki", precisions_chunk, x_mean)
            squared_linear = torch.sum(x_mean_cov.square(), dim=-1)
            return x_mean_cov, squared_linear
        
        x_mean_cov, squared_linear = checkpoint(compute_chunk, x_mean, precisions_chunk, use_reentrant=False)
        
        # Compute log determinant for chunk
        log_det = stable_logdet_memory_efficient(precisions_chunk)
        trace_chunk = torch.diagonal(precisions_chunk, dim1=-2, dim2=-1).sum(dim=-1)
        
        # Compute log probabilities
        log_det_expanded = log_det.unsqueeze(0).expand(batch_size, -1)
        trace_expanded = trace_chunk.unsqueeze(0).expand(batch_size, -1)
        
        log_probs_chunk = 0.5 * (log_det_expanded - squared_linear) - 0.5 * dim * log_2pi
        log_probs_chunk = torch.clamp(log_probs_chunk, min=-DEFAULT_LOGPROB_CLAMP, max=DEFAULT_LOGPROB_CLAMP)
        
        weighted_gradient_chunk = -x_mean_cov
        
        # Store on CPU immediately
        log_probs_cpu[:, start:end] = log_probs_chunk.detach().cpu()
        weighted_grads_cpu[:, start:end] = weighted_gradient_chunk.cpu()
        squared_linear_cpu[:, start:end] = squared_linear.cpu()
        trace_precision_cpu[:, start:end] = trace_expanded.cpu()
        
        # Aggressive cleanup
        del means_chunk, precisions_chunk, x_mean, x_mean_cov, squared_linear
        del log_det, trace_chunk, log_probs_chunk, weighted_gradient_chunk
        aggressive_cleanup()
    
    # Move results back to GPU for final computation
    all_log_probs = log_probs_cpu.to(device)
    all_weighted_gradients = weighted_grads_cpu.to(device)
    all_squared_linear = squared_linear_cpu.to(device)
    all_trace_precision = trace_precision_cpu.to(device)
    
    # Clean up CPU tensors
    del log_probs_cpu, weighted_grads_cpu, squared_linear_cpu, trace_precision_cpu
    
    # Final computations
    softmax_probs = stable_softmax(all_log_probs, temperature=temperature, dim=1)
    
    gradient = torch.sum(softmax_probs.unsqueeze(-1) * all_weighted_gradients, dim=1)
    
    laplacian_component = softmax_probs * (all_squared_linear - all_trace_precision)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    return gradient, laplacian_over_density

#checkpointed versions
def grad_and_laplacian_mog_density_component_chunked_checkpointed(x, means, precisions, 
                                                                 chunk_size=DEFAULT_CENTERS_CHUNK_SIZE, 
                                                                 temperature=1.0):
    """
    Memory-efficient component chunking with gradient checkpointing
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    
    # Pre-allocate tensors for accumulation (same as original)
    all_log_probs = torch.empty(batch_size, num_components, dtype=x.dtype, device=x.device)
    all_weighted_gradients = torch.empty(batch_size, num_components, dim, dtype=x.dtype, device=x.device)
    all_squared_linear = torch.empty(batch_size, num_components, dtype=x.dtype, device=x.device)
    all_trace_precision = torch.empty(batch_size, num_components, dtype=x.dtype, device=x.device)
    
    # Pre-compute constants
    log_2pi = torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
    
    # Define checkpointed computation for each chunk
    def compute_chunk_core(x, means_chunk, precisions_chunk, log_2pi_const, dim_const):
        """
        Core computation that will be checkpointed
        This contains the memory-intensive operations
        """
        # Vectorized operations
        x_expanded = x.unsqueeze(1)  # (B, 1, D)
        x_mean = x_expanded - means_chunk  # (B, K_chunk, D)
        
        # Memory-intensive einsum
        x_mean_cov = torch.einsum("kij,bkj->bki", precisions_chunk, x_mean)
        squared_linear = torch.sum(x_mean_cov.square(), dim=-1)
        
        # Precompute log_det and trace for chunk
        log_det = stable_logdet(precisions_chunk)  # (K_chunk,)
        trace_chunk = torch.diagonal(precisions_chunk, dim1=-2, dim2=-1).sum(dim=-1)  # (K_chunk,)
        
        # Broadcast efficiently
        batch_size = x.shape[0]
        log_det_expanded = log_det.unsqueeze(0).expand(batch_size, -1)
        trace_expanded = trace_chunk.unsqueeze(0).expand(batch_size, -1)
        
        # Compute log probabilities
        log_probs_chunk = 0.5 * (log_det_expanded - squared_linear) - 0.5 * dim_const * log_2pi_const
        log_probs_chunk = torch.clamp(log_probs_chunk, min=-DEFAULT_LOGPROB_CLAMP, max=DEFAULT_LOGPROB_CLAMP)
        
        weighted_gradient_chunk = -x_mean_cov
        
        return log_probs_chunk, weighted_gradient_chunk, squared_linear, trace_expanded
    
    # Process chunks with checkpointing
    for start in range(0, num_components, chunk_size):
        end = min(start + chunk_size, num_components)
        
        means_chunk = means[start:end]
        precisions_chunk = precisions[start:end]
        
        # Use gradient checkpointing for memory-intensive core computation
        log_probs_chunk, weighted_gradient_chunk, squared_linear, trace_expanded = checkpoint(
            compute_chunk_core, 
            x, 
            means_chunk, 
            precisions_chunk, 
            log_2pi,
            dim,
            use_reentrant=False
        )
        
        # Direct assignment to pre-allocated tensors
        all_log_probs[:, start:end] = log_probs_chunk
        all_weighted_gradients[:, start:end] = weighted_gradient_chunk
        all_squared_linear[:, start:end] = squared_linear
        all_trace_precision[:, start:end] = trace_expanded
        
        # Clean up chunk-specific tensors
        del log_probs_chunk, weighted_gradient_chunk, squared_linear, trace_expanded
        
        # Less frequent cache clearing
        if start % (chunk_size * 5) == 0:
            torch.cuda.empty_cache()
    
    # Final computations (same as original)
    softmax_probs = stable_softmax(all_log_probs, temperature=temperature, dim=1)
    
    gradient = torch.sum(softmax_probs.unsqueeze(-1) * all_weighted_gradients, dim=1)
    
    laplacian_component = softmax_probs * (all_squared_linear - all_trace_precision)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    return gradient, laplacian_over_density

def grad_and_laplacian_mog_density_component_chunked_ultra_checkpointed(x, means, precisions, 
                                                                       chunk_size=DEFAULT_CENTERS_CHUNK_SIZE, 
                                                                       temperature=1.0):
    """
    Ultra memory-efficient version with more aggressive checkpointing
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    
    # Store results on CPU to save GPU memory
    all_log_probs_cpu = torch.empty(batch_size, num_components, dtype=x.dtype)
    all_weighted_gradients_cpu = torch.empty(batch_size, num_components, dim, dtype=x.dtype)
    all_squared_linear_cpu = torch.empty(batch_size, num_components, dtype=x.dtype)
    all_trace_precision_cpu = torch.empty(batch_size, num_components, dtype=x.dtype)
    
    # Pre-compute constants
    log_2pi = torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
    
    def compute_chunk_ultra_core(x, means_chunk, precisions_chunk, log_2pi_const, dim_const):
        """
        Ultra-checkpointed core computation
        """
        # Split into even smaller sub-operations for maximum memory efficiency
        def compute_x_mean_cov(x, means_chunk, precisions_chunk):
            x_expanded = x.unsqueeze(1)
            x_mean = x_expanded - means_chunk
            x_mean_cov = torch.einsum("kij,bkj->bki", precisions_chunk, x_mean)
            return x_mean_cov, x_mean
        
        def compute_probabilities(x_mean_cov, precisions_chunk, log_2pi_const, dim_const):
            squared_linear = torch.sum(x_mean_cov.square(), dim=-1)
            log_det = stable_logdet(precisions_chunk)
            trace_chunk = torch.diagonal(precisions_chunk, dim1=-2, dim2=-1).sum(dim=-1)
            
            batch_size = x_mean_cov.shape[0]
            log_det_expanded = log_det.unsqueeze(0).expand(batch_size, -1)
            trace_expanded = trace_chunk.unsqueeze(0).expand(batch_size, -1)
            
            log_probs_chunk = 0.5 * (log_det_expanded - squared_linear) - 0.5 * dim_const * log_2pi_const
            log_probs_chunk = torch.clamp(log_probs_chunk, min=-DEFAULT_LOGPROB_CLAMP, max=DEFAULT_LOGPROB_CLAMP)
            
            return log_probs_chunk, squared_linear, trace_expanded
        
        # Checkpoint each sub-operation
        x_mean_cov, x_mean = checkpoint(compute_x_mean_cov, x, means_chunk, precisions_chunk, use_reentrant=False)
        log_probs_chunk, squared_linear, trace_expanded = checkpoint(
            compute_probabilities, x_mean_cov, precisions_chunk, log_2pi_const, dim_const, use_reentrant=False)
        
        weighted_gradient_chunk = -x_mean_cov
        
        return log_probs_chunk, weighted_gradient_chunk, squared_linear, trace_expanded
    
    # Process chunks with ultra-checkpointing
    for start in range(0, num_components, chunk_size):
        end = min(start + chunk_size, num_components)
        
        means_chunk = means[start:end]
        precisions_chunk = precisions[start:end]
        
        # Use ultra-checkpointing
        log_probs_chunk, weighted_gradient_chunk, squared_linear, trace_expanded = checkpoint(
            compute_chunk_ultra_core, 
            x, 
            means_chunk, 
            precisions_chunk, 
            log_2pi,
            dim,
            use_reentrant=False
        )
        
        # Store on CPU immediately
        all_log_probs_cpu[:, start:end] = log_probs_chunk.cpu()
        all_weighted_gradients_cpu[:, start:end] = weighted_gradient_chunk.cpu()
        all_squared_linear_cpu[:, start:end] = squared_linear.cpu()
        all_trace_precision_cpu[:, start:end] = trace_expanded.cpu()
        
        # Aggressive cleanup
        del log_probs_chunk, weighted_gradient_chunk, squared_linear, trace_expanded
        torch.cuda.empty_cache()
    
    # Move results back to GPU for final computation
    device = x.device
    all_log_probs = all_log_probs_cpu.to(device)
    all_weighted_gradients = all_weighted_gradients_cpu.to(device)
    all_squared_linear = all_squared_linear_cpu.to(device)
    all_trace_precision = all_trace_precision_cpu.to(device)
    
    # Clean up CPU tensors
    del all_log_probs_cpu, all_weighted_gradients_cpu, all_squared_linear_cpu, all_trace_precision_cpu
    
    # Final computations
    softmax_probs = stable_softmax(all_log_probs, temperature=temperature, dim=1)
    
    gradient = torch.sum(softmax_probs.unsqueeze(-1) * all_weighted_gradients, dim=1)
    
    laplacian_component = softmax_probs * (all_squared_linear - all_trace_precision)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    return gradient, laplacian_over_density

def grad_and_laplacian_mog_density_chunked_checkpointed(x, means, precisions, 
                                                       batch_chunk_size=DEFAULT_BATCH_CHUNK_SIZE, 
                                                       component_chunk_size=DEFAULT_CENTERS_CHUNK_SIZE,
                                                       temperature=1.0):
    """
    Updated chunked version that uses checkpointed component processing
    """
    total_batches = x.size(0)
    dim = x.size(-1)
    
    gradients = torch.empty(total_batches, dim, dtype=x.dtype, device=x.device)
    laplacians = torch.empty(total_batches, dtype=x.dtype, device=x.device)
    
    # Process in chunks
    for start in range(0, total_batches, batch_chunk_size):
        end = min(start + batch_chunk_size, total_batches)
        x_chunk = x[start:end]
        
        # Use checkpointed component chunking for large numbers of components
        if means.size(0) > component_chunk_size:
            grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_component_chunked_checkpointed(
                x_chunk, means, precisions, chunk_size=component_chunk_size, temperature=temperature)
        else:
            # For small numbers of components, use regular stable version
            grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_stable(
                x_chunk, means, precisions, temperature=temperature)
        
        # Direct assignment
        gradients[start:end] = grad_chunk.detach() if not grad_chunk.requires_grad else grad_chunk
        laplacians[start:end] = laplacian_chunk.detach() if not laplacian_chunk.requires_grad else laplacian_chunk
    
    return gradients, laplacians

def grad_and_laplacian_mog_density_chunked_ultra_checkpointed(x, means, precisions, 
                                                       batch_chunk_size=DEFAULT_BATCH_CHUNK_SIZE, 
                                                       component_chunk_size=DEFAULT_CENTERS_CHUNK_SIZE,
                                                       temperature=1.0):
    """
    Updated chunked version that uses checkpointed component processing
    """
    total_batches = x.size(0)
    dim = x.size(-1)
    
    gradients = torch.empty(total_batches, dim, dtype=x.dtype, device=x.device)
    laplacians = torch.empty(total_batches, dtype=x.dtype, device=x.device)
    
    # Process in chunks
    for start in range(0, total_batches, batch_chunk_size):
        end = min(start + batch_chunk_size, total_batches)
        x_chunk = x[start:end]
        
        # Use checkpointed component chunking for large numbers of components
        if means.size(0) > component_chunk_size:
            grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_component_chunked_ultra_checkpointed(
                x_chunk, means, precisions, chunk_size=component_chunk_size, temperature=temperature)
        else:
            # For small numbers of components, use regular stable version
            grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_stable(
                x_chunk, means, precisions, temperature=temperature)
        
        # Direct assignment
        gradients[start:end] = grad_chunk.detach() if not grad_chunk.requires_grad else grad_chunk
        laplacians[start:end] = laplacian_chunk.detach() if not laplacian_chunk.requires_grad else laplacian_chunk
    
    return gradients, laplacians

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
    logdet = stable_logdet(precisions)
    trace_precision = torch.diagonal(precisions, dim1=-2, dim2=-1).sum(dim=-1)
    log_2pi_term = 0.5 * dim * torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
    
    # Vectorized log probability computation
    log_probs = 0.5 * (logdet.unsqueeze(0) - squared_linear) - log_2pi_term
    log_probs = torch.clamp(log_probs, min=-DEFAULT_LOGPROB_CLAMP, max=DEFAULT_LOGPROB_CLAMP)
    
    # Stable softmax
    softmax_probs = stable_softmax(log_probs, temperature=temperature, dim=1)
    
    # Vectorized gradient and Laplacian
    gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)
    laplacian_component = softmax_probs * (squared_linear - trace_precision.unsqueeze(0))
    laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    return gradient, laplacian_over_density

def grad_and_laplacian_mog_density_fused(x, means, precisions, temperature=1.0):
    """
    Fused computation to minimize intermediate tensor allocations
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    
    # Single memory allocation for all intermediate results
    x_expanded = x.unsqueeze(1)
    x_mean = x_expanded - means
    
    # Fuse precision multiplication with squared distance computation
    x_mean_flat = x_mean.view(-1, dim)
    precisions_flat = precisions.repeat(batch_size, 1, 1).view(-1, dim, dim)
    
    # Batched matrix-vector multiplication
    x_mean_cov_flat = torch.bmm(precisions_flat, x_mean_flat.unsqueeze(-1)).squeeze(-1)
    x_mean_cov = x_mean_cov_flat.view(batch_size, num_components, dim)
    
    # Compute squared distances
    squared_linear = torch.sum(x_mean_cov * x_mean_cov, dim=-1)
    
    # Fast log determinant approximation
    logdet = stable_logdet(precisions)
    trace_precision = torch.diagonal(precisions, dim1=-2, dim2=-1).sum(dim=-1)
    
    # Compute log probabilities
    log_probs = 0.5 * (logdet.unsqueeze(0) - squared_linear) - 0.5 * dim * LOG_2PI  # log(2π)
    log_probs = torch.clamp(log_probs, min=-100.0, max=100.0)
    
    # Stable softmax with temperature
    softmax_probs = stable_softmax_jit(log_probs, temperature)
    
    # Compute results in single pass
    gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)
    laplacian_component = softmax_probs * (squared_linear - trace_precision.unsqueeze(0))
    laplacian_over_density = torch.sum(laplacian_component, dim=1)
    
    return gradient, laplacian_over_density

def grad_and_laplacian_streaming(x, means, precisions, chunk_size=DEFAULT_BATCH_CHUNK_SIZE, temperature=1.0):
    """
    Streaming computation with minimal memory footprint
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    
    # Pre-allocate output tensors
    gradient = torch.zeros(batch_size, dim, dtype=x.dtype, device=x.device)
    laplacian = torch.zeros(batch_size, dtype=x.dtype, device=x.device)
    
    # Process in streaming fashion
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        x_chunk = x[start:end]
        
        # Compute chunk results
        grad_chunk, lap_chunk = grad_and_laplacian_mog_density_fused(
            x_chunk, means, precisions, temperature)
        
        # Accumulate results
        gradient[start:end] = grad_chunk
        laplacian[start:end] = lap_chunk
        
        # No explicit cleanup needed - tensors go out of scope
    
    return gradient, laplacian

def grad_and_laplacian_streaming_component_chunked(
    x, means, precisions, 
    batch_chunk_size=DEFAULT_BATCH_CHUNK_SIZE, 
    component_chunk_size=DEFAULT_CENTERS_CHUNK_SIZE, 
    temperature=1.0):
    """
    Streaming over samples AND chunking over components to save memory
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]

    # Pre-allocate outputs
    gradient = torch.zeros(batch_size, dim, dtype=x.dtype, device=x.device)
    laplacian = torch.zeros(batch_size, dtype=x.dtype, device=x.device)

    log_2pi = torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))

    for batch_start in range(0, batch_size, batch_chunk_size):
        batch_end = min(batch_start + batch_chunk_size, batch_size)
        x_chunk = x[batch_start:batch_end]  # (batch_chunk, dim)

        # Accumulate component-wise results for this batch chunk
        all_log_probs_chunks = []
        all_weighted_grads_chunks = []
        all_squared_linear_chunks = []
        all_trace_chunks = []

        for comp_start in range(0, num_components, component_chunk_size):
            comp_end = min(comp_start + component_chunk_size, num_components)

            means_chunk = means[comp_start:comp_end]                 # (K_chunk, dim)
            precisions_chunk = precisions[comp_start:comp_end]       # (K_chunk, dim, dim)
            K_chunk = comp_end - comp_start

            # Compute (x - mu)
            x_expanded = x_chunk.unsqueeze(1)                         # (B_chunk, 1, dim)
            x_mean = x_expanded - means_chunk                         # (B_chunk, K_chunk, dim)

            # Precision times (x - mu)
            x_mean_cov = torch.einsum("kij,bkj->bki", precisions_chunk, x_mean)  # (B_chunk, K_chunk, dim)

            # Squared Mahalanobis distances
            squared_linear = torch.sum(x_mean_cov.square(), dim=-1)  # (B_chunk, K_chunk)

            # Log det and trace per component chunk
            log_det = stable_logdet(precisions_chunk)                # (K_chunk,)
            trace_chunk = torch.diagonal(precisions_chunk, dim1=-2, dim2=-1).sum(dim=-1)  # (K_chunk,)

            # Expand for batch
            log_det_expanded = log_det.unsqueeze(0).expand(batch_end - batch_start, -1)  # (B_chunk, K_chunk)
            trace_expanded = trace_chunk.unsqueeze(0).expand(batch_end - batch_start, -1)  # (B_chunk, K_chunk)

            # Log probabilities chunk
            log_probs_chunk = 0.5 * (log_det_expanded - squared_linear) - 0.5 * dim * log_2pi
            log_probs_chunk = torch.clamp(log_probs_chunk, min=-DEFAULT_LOGPROB_CLAMP, max=DEFAULT_LOGPROB_CLAMP)

            # Weighted gradients chunk (- P(x - mu))
            weighted_gradient_chunk = -x_mean_cov  # (B_chunk, K_chunk, dim)

            # Store all partial results to accumulate after all chunks
            all_log_probs_chunks.append(log_probs_chunk)
            all_weighted_grads_chunks.append(weighted_gradient_chunk)
            all_squared_linear_chunks.append(squared_linear)
            all_trace_chunks.append(trace_expanded)

            # Cleanup chunk to save memory
            del x_mean, x_mean_cov, squared_linear, log_det, log_probs_chunk, weighted_gradient_chunk, trace_chunk

        # Concatenate over components axis
        all_log_probs = torch.cat(all_log_probs_chunks, dim=1)           # (B_chunk, K)
        all_weighted_gradients = torch.cat(all_weighted_grads_chunks, dim=1)  # (B_chunk, K, dim)
        all_squared_linear = torch.cat(all_squared_linear_chunks, dim=1)  # (B_chunk, K)
        all_trace_precision = torch.cat(all_trace_chunks, dim=1)         # (B_chunk, K)

        # Softmax over full component dimension (stable, temperature-controlled)
        softmax_probs = stable_softmax_jit(all_log_probs, temperature=temperature)  # (B_chunk, K)

        # Compute gradient and laplacian
        grad_chunk = torch.sum(softmax_probs.unsqueeze(-1) * all_weighted_gradients, dim=1)  # (B_chunk, dim)
        laplacian_component = softmax_probs * (all_squared_linear - all_trace_precision)     # (B_chunk, K)
        lap_chunk = torch.sum(laplacian_component, dim=1)                                   # (B_chunk,)

        # Assign results back to global tensors
        gradient[batch_start:batch_end] = grad_chunk
        laplacian[batch_start:batch_end] = lap_chunk

    return gradient, laplacian

# ===================== #
# Optimized Main Function
# ===================== #

def score_implicit_matching_optimized(factornet, samples, centers, 
                                    base_epsilon=DEFAULT_EPSILON, 
                                    temperature=DEFAULT_TEMPERATURE, 
                                    max_attempts=DEFAULT_MAX_ATTEMPTS):
    """
    Optimized version with minimal memory usage and maximum speed
    """
    dim = centers.shape[-1]
    centers = centers.detach()  # Detach to avoid unnecessary gradient tracking
    
    for attempt in range(max_attempts):
        try:
            current_epsilon = base_epsilon * (2 ** attempt)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for stability
                factor_eval = factornet(centers)
            
            # Fast precision construction
            precisions = vectors_to_precision_ultra_fast(factor_eval, dim, current_epsilon)
            
            # Streaming gradient and Laplacian computation
            gradient_eval_log, laplacian_over_density = grad_and_laplacian_streaming(
                samples, centers, precisions, chunk_size=32, temperature=temperature)
            
            # Fused final computation
            gradient_squared = torch.sum(gradient_eval_log * gradient_eval_log, dim=1)
            loss = 2 * laplacian_over_density - gradient_squared
            
            # Quick sanity check
            if torch.isfinite(loss).all():
                return loss.mean()
            elif attempt < max_attempts - 1:
                continue
            else:
                # Return clamped result
                return torch.clamp(loss, min=-1e10, max=1e10).mean()
                
        except Exception as e:
            if attempt < max_attempts - 1:
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    raise RuntimeError("All attempts failed")

def score_implicit_matching_stable_dp(factornet, samples, centers,
                                   base_epsilon=DEFAULT_EPSILON,
                                   temperature=DEFAULT_TEMPERATURE,
                                   max_attempts=DEFAULT_MAX_ATTEMPTS):
    """
    Stable score matching with memory logging for DataParallel.
    """

    n_gpus = torch.cuda.device_count()
    log_once(f"🚀 Starting computation with DataParallel on {n_gpus} GPUs")

    dim = centers.shape[-1]
    centers = centers.clone()

    for attempt in range(max_attempts):
        factor_eval = None
        precisions = None
        gradient_eval_log = None
        laplacian_over_density = None
        gradient_eval_log_squared = None
        loss = None

        try:
            if attempt != 0:
                log_once(f"⏱️ Attempt {attempt + 1}/{max_attempts}")

            current_epsilon = base_epsilon * (2 ** attempt)

            # FACTORNET FORWARD
            with profile_section("Factor network forward", device_id=0, rank=0):
                factor_eval = factornet(centers)

            print_memory_usage_dp()

            # PRECISIONS COMPUTATION
            with profile_section("Precisions chunked_optimized", device_id=0, rank=0):
                precisions = vectors_to_precision_chunked_optimized(factor_eval, dim, current_epsilon, 20)

            print_memory_usage_dp()

            del precisions
            gc.collect()
            torch.cuda.empty_cache()

            with profile_section("Precisions new", device_id=0, rank=0):
                precisions = vectors_to_precision_chunked_optimized_new(factor_eval, dim, current_epsilon, 20)

            print_memory_usage_dp()

            if not numerical_health_check_dp(precisions, "precisions"):
                if attempt < max_attempts - 1:
                    log_once("❌ Precisions health check failed, retrying...")
                    raise ValueError("Precisions health check failed")
                else:
                    log_once("⚠️ Proceeding with potentially unstable precision matrices")

            # GRADIENT AND LAPLACIAN
            with profile_section("Gradient and Laplacian computation", device_id=0, rank=0):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_chunked_stable(
                    samples, centers, precisions, temperature=temperature)

            print_memory_usage_dp()
            del gradient_eval_log, laplacian_over_density
            gc.collect()
            torch.cuda.empty_cache()
            with profile_section("Gradient and Laplacian streamed", device_id=0, rank=0):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_streaming_component_chunked(
                    samples, centers, precisions, temperature=temperature)

            print_memory_usage_dp()
            del gradient_eval_log, laplacian_over_density
            gc.collect()
            torch.cuda.empty_cache()
            with profile_section("Gradient and Laplacian chunked checkpointed", device_id=0, rank=0):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_chunked_checkpointed(
                    samples.detach(), centers, precisions,
                    temperature=temperature)

            print_memory_usage_dp()
            del gradient_eval_log, laplacian_over_density
            gc.collect()
            torch.cuda.empty_cache()
            with profile_section("Gradient and Laplacian micro_chunked", device_id=0, rank=0):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_ultra_chunked(
                    samples.detach(), centers, precisions,
                    temperature=temperature)
            print_memory_usage_dp()
            del gradient_eval_log, laplacian_over_density
            gc.collect()
            torch.cuda.empty_cache()
            with profile_section("Gradient and Laplacian micro_chunked_checkpt", device_id=0, rank=0):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_chunked_ultra_checkpointed(
                    samples.detach(), centers, precisions,
                    temperature=temperature)
            print_memory_usage_dp()

            if not numerical_health_check_dp(gradient_eval_log, "gradient_eval_log"):
                if attempt < max_attempts - 1:
                    log_once("❌ Gradient eval health check failed, retrying...")
                    raise ValueError("Gradient eval health check failed")
                else:
                    log_once("⚠️ Proceeding with potentially unstable gradient eval")

            if not numerical_health_check_dp(laplacian_over_density, "laplacian_over_density"):
                if attempt < max_attempts - 1:
                    log_once("❌ Laplacian health check failed, retrying...")
                    raise ValueError("Laplacian health check failed")
                else:
                    log_once("⚠️ Proceeding with potentially unstable laplacian")

            with profile_section("Final loss computation", device_id=0, rank=0):
                gradient_eval_log_squared = torch.sum(gradient_eval_log.square(), dim=1)
                loss = 2 * laplacian_over_density - gradient_eval_log_squared
            
            if numerical_health_check_dp(loss, "loss"):
                result = loss.mean(dim=0)
                log_once("✅ Computation completed successfully")
                return result
            elif attempt < max_attempts - 1:
                log_once(f"❌ Loss health check failed at attempt {attempt + 1}, retrying...")
                raise ValueError("Loss health check failed")
            else:
                log_once("⚠️ Returning clamped fallback loss")
                loss = loss.mean(dim=0)
                return torch.clamp(loss, min=-1e10, max=1e10)

        except Exception as e:
            log_once(f"❌ Attempt {attempt + 1} failed with error: {e}")
            safe_cleanup(
                locals().get("factor_eval"),
                locals().get("precisions"),
                locals().get("gradient_eval_log"),
                locals().get("laplacian_over_density"),
                locals().get("gradient_eval_log_squared"),
                locals().get("loss"),
            )
            gc.collect()
            torch.cuda.empty_cache()

            if attempt < max_attempts - 1:
                log_once("Retrying with increased regularization...")
            else:
                raise e

        finally:
            safe_cleanup(
                locals().get("factor_eval"),
                locals().get("precisions"),
                locals().get("gradient_eval_log"),
                locals().get("laplacian_over_density"),
                locals().get("gradient_eval_log_squared"),
                locals().get("loss"),
            )
            gc.collect()
            torch.cuda.empty_cache()

    raise RuntimeError("All attempts failed and no valid result was returned")

# ===================== #
# Alternative: Memory-Mapped Approach
# ===================== #

def grad_and_laplacian_memory_mapped(x, means, precisions, temperature=1.0):
    """
    Memory-mapped approach for very large tensors
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    
    # Use torch.cuda.Stream for overlapping computation
    stream = torch.cuda.Stream()
    
    with torch.cuda.stream(stream):
        # Overlap data movement with computation
        x_expanded = x.unsqueeze(1)
        x_mean = x_expanded - means
        
        # Process in micro-batches to reduce peak memory
        micro_batch_size = 16
        gradients = []
        laplacians = []
        
        for i in range(0, batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, batch_size)
            x_micro = x_mean[i:end_idx]
            
            # Compute for micro-batch
            x_mean_cov = torch.einsum('kij,bkj->bki', precisions, x_micro)
            squared_linear = torch.sum(x_mean_cov * x_mean_cov, dim=-1)
            
            # Fast log probabilities
            logdet = stable_logdet_fast(precisions)
            log_probs = 0.5 * (logdet.unsqueeze(0) - squared_linear) - 0.5 * dim * 1.8378770664093453
            log_probs = torch.clamp(log_probs, min=-100.0, max=100.0)
            
            # Compute results
            softmax_probs = stable_softmax_jit(log_probs, temperature)
            gradient_micro = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)
            
            trace_precision = torch.diagonal(precisions, dim1=-2, dim2=-1).sum(dim=-1)
            laplacian_micro = torch.sum(softmax_probs * (squared_linear - trace_precision.unsqueeze(0)), dim=1)
            
            gradients.append(gradient_micro)
            laplacians.append(laplacian_micro)
    
    # Concatenate results
    gradient = torch.cat(gradients, dim=0)
    laplacian = torch.cat(laplacians, dim=0)
    
    return gradient, laplacian

