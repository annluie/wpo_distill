#stable version of code
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


# ===================== #
# Global Variables Config
# ===================== #
DEFAULT_EPSILON = 1e-4
DEFAULT_MAX_COND = 1e12
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CENTERS_CHUNK_SIZE = 100
DEFAULT_BATCH_CHUNK_SIZE = 64
DEFAULT_TEMPERATURE = 1.0
DEFAULT_LOGPROB_CLAMP = 100
DEFAULT_LOGITS_CLAMP = 100  # Reduced for stability

'''
# ===================== #
# Global Variables Config - Reduced for memory efficiency
# ===================== #
DEFAULT_EPSILON = 1e-4
DEFAULT_MAX_COND = 1e12
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CENTERS_CHUNK_SIZE = 10  # Reduced from 50
DEFAULT_BATCH_CHUNK_SIZE = 1    # Reduced from 32
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CLAMP = 100
'''

#----------------------------------------------#
#### Precision Matrix Functions ####
#----------------------------------------------#
def vectors_to_precision_stable(vectors, dim, base_epsilon=DEFAULT_EPSILON, max_cond=DEFAULT_MAX_COND):
    """
    Enhanced and corrected version of vectors_to_precision with improved numerical stability
    """
    if torch.isnan(vectors).any():
        print("❌ NaNs in vectors before conversion to precision!")

    batch_size = vectors.shape[0]
    device = vectors.device
    dtype = vectors.dtype

    # Construct lower triangular matrix
    L = torch.zeros(batch_size, dim, dim, dtype=dtype, device=device)
    indices = torch.tril_indices(dim, dim, device=device)
    L[:, indices[0], indices[1]] = vectors  # fixed from squeeze(1)

    # Ensure positive diagonal via softplus
    diag_indices = torch.arange(dim, device=device)
    L[:, diag_indices, diag_indices] = F.softplus(L[:, diag_indices, diag_indices]) + 1e-6

    # Construct precision matrix
    C = torch.matmul(L, L.transpose(1, 2))

    # Adaptive regularization
    adaptive_eps = adaptive_regularization(C, base_epsilon, max_cond)
    identity = torch.eye(dim, device=device, dtype=dtype)
    precision = C + adaptive_eps * identity

    for attempt in range(3):
        try:
            torch.linalg.cholesky(precision)
            break
        except RuntimeError:
            reg_strength = base_epsilon * (10 ** (attempt + 1))
            precision = precision + reg_strength * identity

    return precision

def vectors_to_precision_chunked_stable(vectors, dim, base_epsilon=DEFAULT_EPSILON, chunk_size=DEFAULT_CENTERS_CHUNK_SIZE):
    """
    Chunked version of stable precision matrix computation
    """
    results = []
    for i in range(0, vectors.size(0), chunk_size):
        chunk = vectors[i:i+chunk_size]
        result = vectors_to_precision_stable(chunk, dim, base_epsilon)
        results.append(result)
        
        # Memory cleanup
        del chunk, result
        torch.cuda.empty_cache()
    
    return torch.cat(results, dim=0)

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

def vectors_to_precision_micro_chunked(vectors, dim, base_epsilon=DEFAULT_EPSILON, chunk_size=10):
    """
    Micro-chunked version for extreme memory constraints
    """
    results = []
    total_chunks = (vectors.size(0) + chunk_size - 1) // chunk_size
    
    for i in range(0, vectors.size(0), chunk_size):
        chunk = vectors[i:i+chunk_size]
        result = vectors_to_precision_chunked_optimized(chunk, dim, base_epsilon)
        #result = vectors_to_precision_memory_efficient(chunk, dim, base_epsilon)
        results.append(result.cpu())  # Move to CPU immediately
        
        # Aggressive cleanup after each chunk
        del chunk, result
        aggressive_cleanup()
        
        if i % (chunk_size * 100) == 0:
            print(f"Processed {i//chunk_size + 1}/{total_chunks} chunks")
    
    # Concatenate on CPU first, then move to GPU
    cpu_result = torch.cat(results, dim=0)
    del results
    aggressive_cleanup()
    
    return cpu_result.to(vectors.device)

#----------------------------------------------#
#### Computation Functions ####
#----------------------------------------------#

def adaptive_regularization(matrices, base_epsilon=DEFAULT_EPSILON, max_cond=DEFAULT_MAX_COND):
    """
    Compute adaptive regularization based on condition numbers
    """
    try:
        cond_numbers = torch.linalg.cond(matrices)
        # Clamp condition numbers to prevent extreme values
        cond_clamped = torch.clamp(cond_numbers, 1.0, max_cond)
        # Scale epsilon based on condition number
        adaptive_eps = base_epsilon * torch.sqrt(cond_clamped / 1e6)
        return adaptive_eps.unsqueeze(-1).unsqueeze(-1)  # Shape for broadcasting
    except:
        # Fallback to base epsilon if condition number computation fails
        batch_size = matrices.shape[0]
        return torch.full((batch_size, 1, 1), base_epsilon, device=matrices.device, dtype=matrices.dtype)

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

def stable_logdet_multi_gpu(matrices, eps=1e-6):
    """
    Compute log-determinants in a memory-efficient, multi-GPU manner.
    Uses Cholesky decomposition with offloading to available GPUs.
    """
    batch_size = matrices.shape[0]
    original_device = matrices.device
    dtype = matrices.dtype

    available_gpus = list(range(torch.cuda.device_count()))
    if len(available_gpus) < 2:
        raise RuntimeError("At least 2 GPUs are recommended for multi-GPU offloading.")

    micro_batch_size = min(8, batch_size)
    results = []

    for i in range(0, batch_size, micro_batch_size):
        end_idx = min(i + micro_batch_size, batch_size)
        batch = matrices[i:end_idx]

        # Select a GPU in round-robin fashion
        target_gpu = available_gpus[(i // micro_batch_size) % len(available_gpus)]

        try:
            batch_gpu = batch.to(f'cuda:{target_gpu}', non_blocking=True)
            eye = torch.eye(batch_gpu.shape[-1], device=batch_gpu.device, dtype=dtype)
            reg = batch_gpu + eps * eye
            chol = torch.linalg.cholesky(reg)
            logdet = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)

            # Move back to original device
            results.append(logdet.to(original_device, non_blocking=True))

            del batch, batch_gpu, reg, chol, logdet
        except RuntimeError as e:
            print(f"⚠️ Fallback on batch {i} due to error: {str(e)}")
            fallback = torch.full((end_idx - i,), -10.0, device=original_device, dtype=dtype)
            results.append(fallback)
            del fallback

        aggressive_cleanup()

    return torch.cat(results, dim=0)


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
#original ones (fastest, least memory efficient)
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

def grad_and_laplacian_ultra_component_chunked_multigpu(x, means, precisions, 
                                                        chunk_size=10, 
                                                        temperature=1.0,
                                                        compute_device="cuda:1"):
    """
    Ultra memory-efficient component chunking with secondary GPU offloading (instead of CPU).
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    input_device = x.device
    dtype = x.dtype

    compute_device = torch.device(compute_device)

    # Pre-allocate result tensors on secondary GPU
    log_probs = torch.empty(batch_size, num_components, dtype=dtype, device=compute_device)
    weighted_grads = torch.empty(batch_size, num_components, dim, dtype=dtype, device=compute_device)
    squared_linear_all = torch.empty(batch_size, num_components, dtype=dtype, device=compute_device)
    trace_precision_all = torch.empty(batch_size, num_components, dtype=dtype, device=compute_device)

    # Pre-compute constants
    log_2pi = torch.log(torch.tensor(2 * torch.pi, device=compute_device, dtype=dtype))
    x_on_compute = x.to(compute_device, non_blocking=True)
    x_expanded = x_on_compute.unsqueeze(1)

    for start in range(0, num_components, chunk_size):
        end = min(start + chunk_size, num_components)

        means_chunk = means[start:end].to(compute_device, non_blocking=True)
        precisions_chunk = precisions[start:end].to(compute_device, non_blocking=True)

        x_mean = x_expanded - means_chunk  # [B, C_chunk, D]

        # Use checkpoint to reduce memory during matrix computation
        def compute_chunk(x_mean, precisions_chunk):
            x_mean_cov = torch.einsum("kij,bkj->bki", precisions_chunk, x_mean)
            squared_linear = torch.sum(x_mean_cov.square(), dim=-1)
            return x_mean_cov, squared_linear

        x_mean_cov, squared_linear = checkpoint(compute_chunk, x_mean, precisions_chunk, use_reentrant=False)

        log_det = stable_logdet_memory_efficient(precisions_chunk)
        trace_chunk = torch.diagonal(precisions_chunk, dim1=-2, dim2=-1).sum(dim=-1)

        log_det_expanded = log_det.unsqueeze(0).expand(batch_size, -1)
        trace_expanded = trace_chunk.unsqueeze(0).expand(batch_size, -1)

        log_probs_chunk = 0.5 * (log_det_expanded - squared_linear) - 0.5 * dim * log_2pi
        log_probs_chunk = torch.clamp(log_probs_chunk, min=-DEFAULT_LOGPROB_CLAMP, max=DEFAULT_LOGPROB_CLAMP)

        weighted_gradient_chunk = -x_mean_cov

        log_probs[:, start:end] = log_probs_chunk
        weighted_grads[:, start:end] = weighted_gradient_chunk
        squared_linear_all[:, start:end] = squared_linear
        trace_precision_all[:, start:end] = trace_expanded

        del x_mean_cov, squared_linear, log_det, trace_chunk
        del log_probs_chunk, weighted_gradient_chunk, means_chunk, precisions_chunk
        aggressive_cleanup()

    # Final computations (move to input device for output compatibility)
    all_log_probs = log_probs.to(input_device, non_blocking=True)
    all_weighted_gradients = weighted_grads.to(input_device, non_blocking=True)
    all_squared_linear = squared_linear_all.to(input_device, non_blocking=True)
    all_trace_precision = trace_precision_all.to(input_device, non_blocking=True)

    del log_probs, weighted_grads, squared_linear_all, trace_precision_all

    softmax_probs = stable_softmax(all_log_probs, temperature=temperature, dim=1)
    gradient = torch.sum(softmax_probs.unsqueeze(-1) * all_weighted_gradients, dim=1)

    laplacian_component = softmax_probs * (all_squared_linear - all_trace_precision)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)

    return gradient, laplacian_over_density

def grad_and_laplacian_dual_gpu_streamed(
    x, means, precisions,
    chunk_size=10,
    temperature=1.0,
    gmm_device=torch.device("cuda:1"),
    main_device=torch.device("cuda:0"),
):
    """
    GMM computation on gmm_device (e.g. GPU 1) while model/backward runs on main_device (GPU 0).
    Uses CUDA streams to overlap execution.
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    dtype = x.dtype

    # Allocate outputs on gmm_device
    log_probs_gpu = torch.empty(batch_size, num_components, device=gmm_device, dtype=dtype)
    weighted_grads_gpu = torch.empty(batch_size, num_components, dim, device=gmm_device, dtype=dtype)
    squared_linear_gpu = torch.empty(batch_size, num_components, device=gmm_device, dtype=dtype)
    trace_precision_gpu = torch.empty(batch_size, num_components, device=gmm_device, dtype=dtype)

    # Move input `x` once to GMM device asynchronously
    stream = torch.cuda.Stream(device=gmm_device)
    with torch.cuda.stream(stream):
        x_gpu = x.to(gmm_device, non_blocking=True)
        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=gmm_device, dtype=dtype))

        for start in range(0, num_components, chunk_size):
            end = min(start + chunk_size, num_components)

            means_chunk = means[start:end].to(gmm_device, non_blocking=True)
            precisions_chunk = precisions[start:end].to(gmm_device, non_blocking=True)

            x_expanded = x_gpu.unsqueeze(1)
            x_mean = x_expanded - means_chunk

            def compute_chunk(x_mean, precisions_chunk):
                x_mean_cov = torch.einsum("kij,bkj->bki", precisions_chunk, x_mean)
                squared_linear = torch.sum(x_mean_cov.square(), dim=-1)
                return x_mean_cov, squared_linear

            x_mean_cov, squared_linear = checkpoint(
                compute_chunk, x_mean, precisions_chunk, use_reentrant=False
            )

            log_det = stable_logdet_memory_efficient(precisions_chunk)
            trace_chunk = torch.diagonal(precisions_chunk, dim1=-2, dim2=-1).sum(dim=-1)

            log_det_exp = log_det.unsqueeze(0).expand(batch_size, -1)
            trace_exp = trace_chunk.unsqueeze(0).expand(batch_size, -1)

            log_probs_chunk = 0.5 * (log_det_exp - squared_linear) - 0.5 * dim * log_2pi
            log_probs_chunk = torch.clamp(log_probs_chunk, min=-DEFAULT_LOGPROB_CLAMP, max=DEFAULT_LOGPROB_CLAMP)
            weighted_gradient_chunk = -x_mean_cov

            log_probs_gpu[:, start:end] = log_probs_chunk
            weighted_grads_gpu[:, start:end] = weighted_gradient_chunk
            squared_linear_gpu[:, start:end] = squared_linear
            trace_precision_gpu[:, start:end] = trace_exp

            # Manual aggressive cleanup
            del (
                means_chunk, precisions_chunk, x_mean, x_mean_cov, squared_linear,
                log_det, trace_chunk, log_probs_chunk, weighted_gradient_chunk
            )

    # Wait for stream to finish
    torch.cuda.current_stream(device=gmm_device).wait_stream(stream)

    # Compute final outputs
    softmax_probs = stable_softmax(log_probs_gpu, temperature=temperature, dim=1)

    gradient = torch.sum(softmax_probs.unsqueeze(-1) * weighted_grads_gpu, dim=1)
    laplacian_component = softmax_probs * (squared_linear_gpu - trace_precision_gpu)
    laplacian = torch.sum(laplacian_component, dim=1)

    # Move only final outputs to main device
    return gradient.to(main_device, non_blocking=True), laplacian.to(main_device, non_blocking=True)

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

# score functions
def score_implicit_matching_stable(
    factornet, samples, centers, base_epsilon=DEFAULT_EPSILON, 
    temperature=DEFAULT_TEMPERATURE, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """
    Enhanced numerically stable version with strategic memory management and retry logic.
    """
    torch.cuda.reset_peak_memory_stats()
    dim = centers.shape[-1]
    centers = centers.clone()

    for attempt in range(max_attempts):
        factor_eval = precisions = gradient_eval_log = laplacian_over_density = None
        gradient_eval_log_squared = loss = None

        try:
            if attempt != 0:
                print(f"\n⏱️ Attempt {attempt + 1}/{max_attempts}")
            t0 = time.time()
            current_epsilon = base_epsilon * (2 ** attempt)

            # === FACTORNET FORWARD ===
            with torch.no_grad():  # disables gradient tracking for forward pass retries
                factor_eval = factornet(centers)

            if not numerical_health_check(factor_eval, "factor_eval"):
                if attempt < max_attempts - 1:
                    print("❌ Factor eval health check failed, retrying...")
                    raise ValueError("Factor eval health check failed")
                print("⚠️ Proceeding with potentially unstable factor network output")

            # === PRECISION CONSTRUCTION ===
            precisions = vectors_to_precision_chunked_optimized(
                factor_eval, dim, current_epsilon, DEFAULT_CENTERS_CHUNK_SIZE
            )

            if not numerical_health_check(precisions, "precisions"):
                if attempt < max_attempts - 1:
                    print("❌ Precisions health check failed, retrying...")
                    raise ValueError("Precisions health check failed")
                print("⚠️ Proceeding with potentially unstable precision matrices")

            # === GRAD & LAPLACIAN ===
            gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_chunked_stable(
                samples, centers, precisions, temperature=temperature
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
            gradient_eval_log_squared = torch.sum(gradient_eval_log.square(), dim=1)
            loss = 2 * laplacian_over_density - gradient_eval_log_squared
            total_time = time.time() - t0

            if numerical_health_check(loss, "loss"):
                return loss.mean(dim=0)
            elif attempt < max_attempts - 1:
                print(f"❌ Loss health check failed at attempt {attempt + 1}, retrying...")
                raise ValueError("Loss health check failed")
            else:
                print("⚠️ Returning clamped fallback loss")
                clamped_loss = torch.clamp(loss.mean(dim=0), min=-1e10, max=1e10)
                return clamped_loss

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed with error: {e}")
            safe_cleanup(
                factor_eval, precisions, gradient_eval_log,
                laplacian_over_density, gradient_eval_log_squared, loss
            )
            if attempt < max_attempts - 1:
                print("Retrying with increased regularization...")
            else:
                raise e

        finally:
            # Final cleanup after each attempt
            safe_cleanup(
                factor_eval, precisions, gradient_eval_log,
                laplacian_over_density, gradient_eval_log_squared, loss
            )

    raise RuntimeError("All attempts failed and no valid result was returned")

def score_implicit_matching_stable_ddp(factornet, samples, centers, base_epsilon=DEFAULT_EPSILON, 
                                  temperature=DEFAULT_TEMPERATURE, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """
    Enhanced numerically stable version with strategic memory management and DDP profiling
    """
    # Get rank info
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    if rank == 0:
        print(f"🚀 Starting computation on {world_size} ranks")
    
    dim = centers.shape[-1]
    device_id = centers.device.index if centers.device.index is not None else 0
    centers = centers.clone()  # no checkpointing, so we don't do requires grad to save memory
    #centers = centers.clone().detach().requires_grad_(True)

    for attempt in range(max_attempts):
        # Initialize variables to None for proper cleanup
        factor_eval = None
        precisions = None
        gradient_eval_log = None
        laplacian_over_density = None
        gradient_eval_log_squared = None
        loss = None
        
        try:
            if attempt != 0 and rank == 0:
                print(f"\n⏱️ Attempt {attempt + 1}/{max_attempts}")
            
            current_epsilon = base_epsilon * (2 ** attempt)
            
            # === FACTORNET FORWARD ===
            with profile_section("Factor network forward", device_id, rank):
                
                # Use checkpointing to save memory
                #factor_eval = checkpoint(factornet, centers, use_reentrant=False)
                factor_eval = factornet(centers)

            # === PRECISIONS COMPUTATION ===
            with profile_section("Precisions chunked_optimized", device_id, rank):
                precisions = vectors_to_precision_chunked_optimized(factor_eval, dim, current_epsilon, 20)

            # Clean up and try micro_chunked version
            del precisions
            cleanup_distributed_memory(rank)

            with profile_section("Precisions micro_chunked", device_id, rank):
                precisions = vectors_to_precision_micro_chunked(factor_eval, dim, current_epsilon, 20)

            if not numerical_health_check_ddp(precisions, "precisions", rank):
                if attempt < max_attempts - 1:
                    if rank == 0:
                        print("❌ Precisions health check failed, retrying...")
                    raise ValueError("Precisions health check failed")
                if rank == 0:
                    print("⚠️ Proceeding with potentially unstable precision matrices")

            # === GRAD & LAPLACIAN ===
            with profile_section("Gradient and Laplacian computation", device_id, rank):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_chunked_stable(
                    samples, centers, precisions, temperature=temperature)

            if not numerical_health_check_ddp(gradient_eval_log, "gradient_eval_log", rank):
                if attempt < max_attempts - 1:
                    if rank == 0:
                        print("❌ Gradient eval health check failed, retrying...")
                    raise ValueError("Gradient eval health check failed")
                if rank == 0:
                    print("⚠️ Proceeding with potentially unstable gradient eval")
                
            if not numerical_health_check_ddp(laplacian_over_density, "laplacian_over_density", rank):
                if attempt < max_attempts - 1:
                    if rank == 0:
                        print("❌ Laplacian health check failed, retrying...")
                    raise ValueError("Laplacian health check failed")
                if rank == 0:
                    print("⚠️ Proceeding with potentially unstable laplacian")

            # === FINAL LOSS ===
            with profile_section("Final loss computation", device_id, rank):
                gradient_eval_log_squared = torch.sum(gradient_eval_log.square(), dim=1)
                loss = 2 * laplacian_over_density - gradient_eval_log_squared

            if numerical_health_check_ddp(loss, "loss", rank):
                result = loss.mean(dim=0)
                if rank == 0:
                    print("✅ Computation completed successfully")
                return result
            elif attempt < max_attempts - 1:
                if rank == 0:
                    print(f"❌ Loss health check failed at attempt {attempt + 1}, retrying...")
                raise ValueError("Loss health check failed")
            else:
                if rank == 0:
                    print("⚠️ Returning clamped fallback loss")
                # Clamp but retain gradient path
                loss = loss.mean(dim=0)  # preserve reduction over batch
                clamped_loss = torch.clamp(loss, min=-1e10, max=1e10)
                return clamped_loss

        except Exception as e:
            if rank == 0:
                print(f"❌ Attempt {attempt + 1} failed with error: {e}")
            
            # Immediate aggressive cleanup on any failure
            cleanup_vars = [factor_eval, precisions, gradient_eval_log, 
                          laplacian_over_density, gradient_eval_log_squared, loss]
            for var in cleanup_vars:
                if var is not None:
                    del var
            
            cleanup_distributed_memory(rank)
            
            if attempt < max_attempts - 1:
                if rank == 0:
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
                cleanup_distributed_memory(rank)
    
    # This should never be reached, but just in case
    raise RuntimeError("All attempts failed and no valid result was returned")

def score_implicit_matching_memory_efficient(factornet, samples, centers, 
                                                  base_epsilon=DEFAULT_EPSILON, 
                                                  temperature=DEFAULT_TEMPERATURE, 
                                                  max_attempts=DEFAULT_MAX_ATTEMPTS):
    """
    Ultra memory-efficient version with aggressive memory management
    """
    print(f"Initial GPU memory: {get_memory_usage():.2f} GB")
    
    dim = centers.shape[-1]
    #centers = centers.clone()
    
    for attempt in range(max_attempts):
        try:
            if attempt != 0:
                print(f"\n⏱️ Attempt {attempt + 1}/{max_attempts}")
            
            current_epsilon = base_epsilon * (2 ** attempt)
            
            # === FACTORNET FORWARD WITH CHECKPOINTING ===
            print("Computing factor network...")
            centers = centers.clone().detach().requires_grad_(True)
            factor_eval = checkpoint(factornet, centers, use_reentrant=False)
            #factor_eval = factornet(centers)
            print(f"After factornet: {get_memory_usage():.2f} GB")
            
            # === MICRO-CHUNKED PRECISION CONSTRUCTION ===
            print("Computing precision matrices...")
            precisions = vectors_to_precision_micro_chunked(factor_eval, dim, current_epsilon, chunk_size=5)
            #precisions = vectors_to_precision_chunked_optimized(factor_eval, dim, current_epsilon, chunk_size=5)
            del factor_eval
            aggressive_cleanup()
            print(f"After precisions: {get_memory_usage():.2f} GB")
            
            # === ULTRA-CHUNKED GRAD & LAPLACIAN ===
            print("Computing gradients and Laplacians...")
            gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_ultra_chunked(
                samples, centers, precisions, 
                batch_chunk_size=4,  # Very small chunks
                component_chunk_size=5,
                temperature=temperature
            )
            
            del precisions
            aggressive_cleanup()
            print(f"After grad/laplacian: {get_memory_usage():.2f} GB")
            
            # === FINAL LOSS COMPUTATION ===
            print("Computing final loss...")
            gradient_eval_log_squared = torch.sum(gradient_eval_log.square(), dim=1)
            loss = 2 * laplacian_over_density - gradient_eval_log_squared
            
            del gradient_eval_log, laplacian_over_density, gradient_eval_log_squared
            aggressive_cleanup()
            
            print(f"Final GPU memory: {get_memory_usage():.2f} GB")
            return loss.mean(dim=0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM at attempt {attempt + 1}: {e}")
                aggressive_cleanup()
                
                # Reduce chunk sizes for next attempt
                DEFAULT_CENTERS_CHUNK_SIZE = max(2, DEFAULT_CENTERS_CHUNK_SIZE // 2)
                DEFAULT_BATCH_CHUNK_SIZE = max(2, DEFAULT_BATCH_CHUNK_SIZE // 2)
                print(f"Reducing chunk sizes to {DEFAULT_CENTERS_CHUNK_SIZE}, {DEFAULT_BATCH_CHUNK_SIZE}")
            else:
                print(f"❌ Error at attempt {attempt + 1}: {e}")
            
            if attempt == max_attempts - 1:
                raise e

def score_implicit_matching_ultra_memory_efficient(factornet, samples, centers,
                                                  base_epsilon=DEFAULT_EPSILON,
                                                  temperature=DEFAULT_TEMPERATURE,
                                                  max_attempts=DEFAULT_MAX_ATTEMPTS,
                                                  center_chunk_size=50,
                                                  batch_chunk_size=4,
                                                  component_chunk_size=5):
    """
    Ultra memory-efficient version with aggressive memory management,
    safe backward, and center chunking.
    """
    print(f"Initial GPU memory: {get_memory_usage():.2f} GB")

    dim = centers.shape[-1]

    for attempt in range(max_attempts):
        try:
            if attempt != 0:
                print(f"\n⏱️ Attempt {attempt + 1}/{max_attempts}")

            current_epsilon = base_epsilon * (2 ** attempt)

            # === CHUNKED FORWARD FACTORNET ===
            print("Computing factor network with chunking...")
            center_chunks = []
            centers = centers.detach().requires_grad_(True)  # ensure graph setup
            for i in range(0, centers.shape[0], center_chunk_size):
                chunk = centers[i:i+center_chunk_size]
                chunk_out = checkpoint(factornet, chunk, use_reentrant=False)
                center_chunks.append(chunk_out)
                aggressive_cleanup()
            factor_eval = torch.cat(center_chunks, dim=0)
            print(f"After factornet: {get_memory_usage():.2f} GB")

            # === PRECISION MATRICES ===
            print("Computing precision matrices...")
            precisions = vectors_to_precision_micro_chunked(
                factor_eval, dim, current_epsilon, chunk_size=component_chunk_size
            )
            del factor_eval
            aggressive_cleanup()
            print(f"After precisions: {get_memory_usage():.2f} GB")

            # === GRADIENTS AND LAPLACIANS ===
            print("Computing gradients and Laplacians...")
            gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_ultra_chunked(
                samples.detach(), centers, precisions,
                batch_chunk_size=batch_chunk_size,
                component_chunk_size=component_chunk_size,
                temperature=temperature
            )
            del precisions
            aggressive_cleanup()
            print(f"After grad/laplacian: {get_memory_usage():.2f} GB")

            # === FINAL LOSS ===
            print("Computing final loss...")
            gradient_eval_log_squared = torch.sum(gradient_eval_log.square(), dim=1)
            loss = 2 * laplacian_over_density - gradient_eval_log_squared
            mean_loss = loss.mean(dim=0)

            # Sanity check
            if not torch.isfinite(mean_loss):
                raise RuntimeError("❌ Loss is not finite!")

            del gradient_eval_log, laplacian_over_density, gradient_eval_log_squared, loss
            aggressive_cleanup()

            print(f"Final GPU memory: {get_memory_usage():.2f} GB")
            return mean_loss

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM at attempt {attempt + 1}: {e}")
                aggressive_cleanup()
                torch.cuda.empty_cache()

                center_chunk_size = max(2, center_chunk_size // 2)
                batch_chunk_size = max(2, batch_chunk_size // 2)
                print(f"🔁 Retrying with smaller chunks: center {center_chunk_size}, batch {batch_chunk_size}")
            else:
                print(f"❌ RuntimeError: {e}")
                raise e

    raise RuntimeError("❌ All attempts failed due to memory.")

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

            with profile_section("Precisions micro_chunked", device_id=0, rank=0):
                precisions = vectors_to_precision_micro_chunked(factor_eval, dim, current_epsilon, 20)

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

            with profile_section("Gradient and Laplacian chunked checkpointed", device_id=0, rank=0):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_chunked_checkpointed(
                    samples.detach(), centers, precisions,
                    batch_chunk_size=4,
                    component_chunk_size=5,
                    temperature=temperature)

            print_memory_usage_dp()
            del gradient_eval_log, laplacian_over_density
            gc.collect()
            torch.cuda.empty_cache()
            with profile_section("Gradient and Laplacian micro_chunked", device_id=0, rank=0):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_ultra_chunked(
                    samples.detach(), centers, precisions,
                    batch_chunk_size=4,
                    component_chunk_size=5,
                    temperature=temperature)
            print_memory_usage_dp()
            del gradient_eval_log, laplacian_over_density
            gc.collect()
            torch.cuda.empty_cache()
            with profile_section("Gradient and Laplacian micro_chunked gpu", device_id=0, rank=0):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_ultra_chunked_multigpu(
                    samples.detach(), centers, precisions,
                    batch_chunk_size=4,
                    component_chunk_size=5,
                    temperature=temperature)
            print_memory_usage_dp()
            del gradient_eval_log, laplacian_over_density
            gc.collect()
            torch.cuda.empty_cache()
            with profile_section("Gradient and Laplacian micro_chunked_checkpt", device_id=0, rank=0):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_chunked_ultra_checkpointed(
                    samples.detach(), centers, precisions,
                    batch_chunk_size=4,
                    component_chunk_size=5,
                    temperature=temperature)
            print_memory_usage_dp()

            del gradient_eval_log, laplacian_over_density
            gc.collect()
            torch.cuda.empty_cache()
            with profile_section("Gradient and Laplacian micro_chunked_stream", device_id=0, rank=0):
                gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_ultra_chunked_stream(
                    samples.detach(), centers, precisions,
                    batch_chunk_size=4,
                    component_chunk_size=5,
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

            cleanup_vars = [factor_eval, precisions, gradient_eval_log,
                            laplacian_over_density, gradient_eval_log_squared, loss]
            for var in cleanup_vars:
                if var is not None:
                    del var

            gc.collect()
            torch.cuda.empty_cache()

            if attempt < max_attempts - 1:
                log_once("Retrying with increased regularization...")
            else:
                raise e

        finally:
            cleanup_vars = [factor_eval, precisions, gradient_eval_log,
                            laplacian_over_density, gradient_eval_log_squared, loss]
            for var in cleanup_vars:
                if var is not None:
                    del var

            gc.collect()
            torch.cuda.empty_cache()

    raise RuntimeError("All attempts failed and no valid result was returned")

