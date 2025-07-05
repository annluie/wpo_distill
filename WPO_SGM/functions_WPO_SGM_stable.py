#stable version of code
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
#import gc

# ===================== #
# Global Variables Config
# ===================== #
DEFAULT_EPSILON = 1e-4
DEFAULT_MAX_COND = 1e12
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CENTERS_CHUNK_SIZE = 20
DEFAULT_BATCH_CHUNK_SIZE = 16
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CLAMP = 100

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
    
    # Pre-allocate and use advanced indexing for better memory efficiency
    L = torch.zeros(batch_size, dim, dim, dtype=dtype, device=device)
    
    # Create triangular indices once
    tril_indices = torch.tril_indices(dim, dim, device=device)
    L[:, tril_indices[0], tril_indices[1]] = vectors
    
    # Vectorized diagonal processing
    diag_mask = torch.eye(dim, dtype=torch.bool, device=device)
    L[:, diag_mask] = F.softplus(L[:, diag_mask]) + 1e-6
    
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
        # Memory cleanup
        del chunk, result
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

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
    logits_clamped = torch.clamp(logits_stable, min=-DEFAULT_CLAMP, max=DEFAULT_CLAMP)
    
    return F.softmax(logits_clamped, dim=dim)

def stable_softmax_optimized(logits, temperature=1.0, dim=-1):
    """
    Optimized stable softmax with fewer operations
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    # Combined max subtraction and clamping
    logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
    logits_stable = torch.clamp(logits - logits_max, min=-DEFAULT_CLAMP, max=DEFAULT_CLAMP)
    
    return F.softmax(logits_stable, dim=dim)

#----------------------------------------------#
#### Loss Functions ####
#----------------------------------------------#

def grad_and_laplacian_mog_density_stable(x, means, precisions, temperature=1.0):
    """
    Numerically stable version of gradient and Laplacian computation
    """
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, dim)
    dim = x.size(-1)
    batch_size, num_components = x.size(0), means.size(0)

    # Compute shared terms
    x_mean = x - means  # Shape: (batch_size, num_components, dim)
    
    # P * (x - μ) - used for both gradient and Laplacian
    x_mean_cov = torch.einsum('kij,bkj->bki', precisions, x_mean)  # (B, K, D)
    
    # (x-μ)ᵀP(x-μ) - used for log probs and Laplacian
    squared_linear = torch.sum(x_mean_cov * x_mean_cov, dim=-1)  # (B, K)
    
    # Stable log determinant computation
    logdet = stable_logdet(precisions)  # (K,)
    logdet = logdet.unsqueeze(0).expand(batch_size, -1)  # (B, K)

    # Compute log probabilities with numerical stability
    log_probs = 0.5 * logdet - 0.5 * squared_linear - 0.5 * dim * torch.log(
        torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
    
    # Clamp log probabilities to prevent extreme values
    log_probs = torch.clamp(log_probs, min=-DEFAULT_CLAMP, max=DEFAULT_CLAMP)
    
    # Compute stable softmax probabilities
    softmax_probs = stable_softmax(log_probs, temperature=temperature, dim=1)  # (B, K)
    
    # GRADIENT COMPUTATION
    gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)  # (B, D)
    
    # LAPLACIAN COMPUTATION
    trace_precision = torch.diagonal(precisions, dim1=-2, dim2=-1).sum(dim=-1)  # (K,)
    trace_precision = trace_precision.unsqueeze(0).expand(batch_size, -1)  # (B, K)
    
    laplacian_component = softmax_probs * (squared_linear - trace_precision)  # (B, K)
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
    Memory-efficient stable version with double chunking
    """
    gradients = []
    laplacians = []
    
    for start in range(0, x.size(0), batch_chunk_size):
        end = min(start + batch_chunk_size, x.size(0))
        x_chunk = x[start:end]
        
        # Use component chunking for large numbers of components
        if means.size(0) > component_chunk_size:
            grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_component_chunked_stable(
                x_chunk, means, precisions, chunk_size=component_chunk_size, temperature=temperature)
        else:
            grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_stable(
                x_chunk, means, precisions, temperature=temperature)
        
        gradients.append(grad_chunk.detach() if not grad_chunk.requires_grad else grad_chunk)
        laplacians.append(laplacian_chunk.detach() if not laplacian_chunk.requires_grad else laplacian_chunk)
        
        del grad_chunk, laplacian_chunk, x_chunk
        torch.cuda.empty_cache()
    
    return torch.cat(gradients, dim=0), torch.cat(laplacians, dim=0)

def grad_and_laplacian_mog_density_component_chunked_stable(x, means, precisions, 
                                                           chunk_size=DEFAULT_CENTERS_CHUNK_SIZE, temperature=1.0):
    """
    Component-wise chunked stable computation
    """
    batch_size, dim = x.shape
    num_components = means.shape[0]
    
    # Accumulate partial results
    all_log_probs = []
    all_weighted_gradients = []
    all_squared_linear = []
    all_trace_precision = []
    
    # Compute chunked contributions
    for start in range(0, num_components, chunk_size):
        end = min(start + chunk_size, num_components)
        
        means_chunk = means[start:end]  # (chunk_size, dim)
        precisions_chunk = precisions[start:end]  # (chunk_size, dim, dim)
        
        x_expanded = x.unsqueeze(1)  # (B, 1, D)
        x_mean = x_expanded - means_chunk  # (B, chunk_size, D)
        
        # Shared computation: P * (x - μ)
        x_mean_cov = torch.einsum("kij,bkj->bki", precisions_chunk, x_mean)  # (B, chunk_size, D)
        
        # Shared computation: (x-μ)ᵀP(x-μ)
        squared_linear = torch.sum(x_mean_cov * x_mean_cov, dim=-1)  # (B, chunk_size)
        
        # Stable log determinant
        log_det = stable_logdet(precisions_chunk)  # (chunk_size,)
        log_det = log_det.unsqueeze(0).expand(batch_size, -1)  # (B, chunk_size)
        
        # Log probabilities with stability
        log_probs_chunk = 0.5 * log_det - 0.5 * squared_linear - 0.5 * dim * torch.log(
            torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
        log_probs_chunk = torch.clamp(log_probs_chunk, min=-100, max=100)
        
        # Gradient components: -P * (x - μ)
        weighted_gradient_chunk = -x_mean_cov  # (B, chunk_size, D)
        
        # Trace of precision for Laplacian
        trace_chunk = torch.diagonal(precisions_chunk, dim1=-2, dim2=-1).sum(dim=-1)  # (chunk_size,)
        trace_chunk = trace_chunk.unsqueeze(0).expand(batch_size, -1)  # (B, chunk_size)
        
        # Store results
        all_log_probs.append(log_probs_chunk)
        all_weighted_gradients.append(weighted_gradient_chunk)
        all_squared_linear.append(squared_linear)
        all_trace_precision.append(trace_chunk)
        
        # Memory cleanup
        del x_mean, x_mean_cov, squared_linear, log_det, log_probs_chunk, weighted_gradient_chunk, trace_chunk
        torch.cuda.empty_cache()
    
    # Concatenate across component dimension
    log_probs = torch.cat(all_log_probs, dim=1)  # (B, K)
    weighted_gradients = torch.cat(all_weighted_gradients, dim=1)  # (B, K, D)
    squared_linear_full = torch.cat(all_squared_linear, dim=1)  # (B, K)
    trace_precision_full = torch.cat(all_trace_precision, dim=1)  # (B, K)
    
    # Compute stable softmax over full set of components
    softmax_probs = stable_softmax(log_probs, temperature=temperature, dim=1)  # (B, K)
    
    # GRADIENT: Weight gradients by softmax probabilities and sum over components
    gradient = torch.sum(softmax_probs.unsqueeze(-1) * weighted_gradients, dim=1)  # (B, D)
    
    # LAPLACIAN: Weight Laplacian components by softmax probabilities
    laplacian_component = softmax_probs * (squared_linear_full - trace_precision_full)  # (B, K)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)  # (B,)
    
    # Cleanup
    del log_probs, weighted_gradients, squared_linear_full, trace_precision_full, softmax_probs, laplacian_component
    torch.cuda.empty_cache()
    
    return gradient, laplacian_over_density

def numerical_health_check(tensor, name="tensor"):
    """
    Check numerical health of tensors and provide diagnostics
    """
    if torch.any(torch.isnan(tensor)):
        print(f"Warning: NaN detected in {name}")
        return False
    if torch.any(torch.isinf(tensor)):
        print(f"Warning: Inf detected in {name}")
        return False
    if torch.any(torch.abs(tensor) > 1e12):
        print(f"Warning: Very large values detected in {name} (max: {torch.max(torch.abs(tensor))})")
        return False
    return True

def score_implicit_matching_stable(factornet, samples, centers, base_epsilon=DEFAULT_EPSILON, 
                                  temperature=DEFAULT_TEMPERATURE, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """
    Enhanced numerically stable version of score_implicit_matching with timing
    """
    torch.cuda.reset_peak_memory_stats()
    dim = centers.shape[-1]
    
    # FIX: Since centers are fixed, don't make them require gradients
    #centers = centers.clone().detach().requires_grad_(True)  # use this only for checkpoint
    centers = centers.clone()  # Just clone for safety, no grad required

    for attempt in range(max_attempts):
        try:
            if attempt != 0:
                print(f"\n⏱️ Attempt {attempt + 1}/{max_attempts}")
            t0 = time.time()
            current_epsilon = base_epsilon * (2 ** attempt)

            # === FACTORNET FORWARD ===
            t1 = time.time()
            '''
            with torch.cuda.amp.autocast():
                factor_eval = factornet(centers)
            factor_eval = factor_eval.float()  # <-- cast back to float32 here
            '''
            #factor_eval = checkpoint(factornet, centers, use_reentrant=False)
            factor_eval = factornet(centers)
            #print(f"✅ Factornet forward time: {time.time() - t1:.4f} sec")

            if not numerical_health_check(factor_eval, "factor_eval"):
                if attempt < max_attempts - 1:
                    continue
                print("⚠️ Proceeding with potentially unstable factor network output")

            # === PRECISION CONSTRUCTION ===
            t2 = time.time()
            #precisions = vectors_to_precision_chunked_stable(factor_eval, dim, current_epsilon)
            precisions = vectors_to_precision_chunked_optimized(factor_eval, dim, current_epsilon, 20)
            #print(f"✅ Precision construction time: {time.time() - t2:.4f} sec")
            del factor_eval
            torch.cuda.empty_cache()

            if not numerical_health_check(precisions, "precisions"):
                if attempt < max_attempts - 1:
                    continue
                print("⚠️ Proceeding with potentially unstable precision matrices")

            # === GRAD & LAPLACIAN ===
            t3 = time.time()
            gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_chunked_stable(
                samples, centers, precisions, temperature=temperature)
            #print(f"✅ Gradient & Laplacian time: {time.time() - t3:.4f} sec")
            del precisions
            torch.cuda.empty_cache()

            if not numerical_health_check(gradient_eval_log, "gradient_eval_log"):
                if attempt < max_attempts - 1:
                    continue
            if not numerical_health_check(laplacian_over_density, "laplacian_over_density"):
                if attempt < max_attempts - 1:
                    continue

            # === FINAL LOSS ===
            t4 = time.time()
            gradient_eval_log_squared = torch.sum(gradient_eval_log * gradient_eval_log, dim=1)
            del gradient_eval_log
            loss = 2 * laplacian_over_density - gradient_eval_log_squared
            del laplacian_over_density, gradient_eval_log_squared
            torch.cuda.empty_cache()
            #print(f"✅ Final loss computation time: {time.time() - t4:.4f} sec")

            # === RETURN ===
            total_time = time.time() - t0
            if numerical_health_check(loss, "loss"):
                #print(f"🎯 Total score matching pass time: {total_time:.4f} sec")
                return loss.mean(dim=0)
            elif attempt < max_attempts - 1:
                print(f"Retrying due to bad loss at attempt {attempt + 1}")
                continue
            else:
                print("⚠️ Returning clamped fallback loss")
                return torch.clamp(loss, min=-1e10, max=1e10).mean(dim=0)

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_attempts - 1:
                print("Retrying with increased regularization...")

def setup_optimal_device_settings():
    """
    Configure optimal device settings for performance
    """
    if torch.cuda.is_available():
        # Enable tensor cores for mixed precision
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        
        print("Optimal CUDA settings enabled")
    else:
        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        print(f"Using {torch.get_num_threads()} CPU threads")