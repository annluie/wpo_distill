import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from memory_profiler import profile
#from torch.amp.autocast_mode import autocast
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch.linalg as LA
import torchvision.utils as vutils
import time
#import gc


## vector to precision matrix function
def vectors_to_precision(vectors,dim, epsilon=0.01):
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
    C = torch.matmul(L, L.transpose(1, 2)) + epsilon * torch.eye(dim).to(vectors.device) # (add identity matrix to maintain positive definiteness)
    
    return C

def vectors_to_precision_chunked(vectors, dim, epsilon=0.01,chunk_size=10):
    results = []
    for i in range(0, vectors.size(0), chunk_size):
        chunk = vectors[i:i+chunk_size]
        #results.append(vectors_to_precision(chunk, dim, epsilon))
        results.append(vectors_to_precision_adapt(chunk, dim, epsilon))
    return torch.cat(results, dim=0)
'''
def vectors_to_precision_adapt(vectors, dim, epsilon=0.01):
    batch_size = vectors.shape[0]
    L = torch.zeros(batch_size, dim, dim, dtype=vectors.dtype, device=vectors.device)
    indices = torch.tril_indices(dim, dim)
    L[:, indices[0], indices[1]] = vectors.squeeze(1)
    
    C = torch.matmul(L, L.transpose(1, 2))
    
    # Add regularization proportional to the trace (fast and stable)
    trace_reg = torch.diagonal(C, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1) * epsilon
    identity = torch.eye(dim, device=vectors.device)
    
    return C + trace_reg * identity
'''
def vectors_to_precision_adapt(vectors, dim, epsilon=0.1, min_reg=1e-4):
    """
    Converts Cholesky vectors to precision matrices with adaptive regularization.
    """
    batch_size = vectors.shape[0]
    vectors = vectors.view(batch_size, -1)  # Safe squeeze
    device, dtype = vectors.device, vectors.dtype

    # Create lower-triangular matrix
    L = torch.zeros(batch_size, dim, dim, dtype=dtype, device=device)
    indices = torch.tril_indices(dim, dim, device=device)
    L[:, indices[0], indices[1]] = vectors

    # Ensure positive diagonal
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    L = L + torch.diag_embed(torch.abs(diag) + 1e-6 - diag)

    # Build precision matrix
    C = torch.bmm(L, L.transpose(-1, -2))

    # Adaptive regularization
    diag_C = torch.diagonal(C, dim1=-2, dim2=-1).mean(dim=-1)  # (B,)
    trace_reg = epsilon * diag_C
    reg = torch.maximum(trace_reg, torch.full_like(trace_reg, min_reg))  # (B,)

    # Add per-sample regularization
    identity = torch.eye(dim, device=device, dtype=dtype).expand(batch_size, -1, -1)
    precision = C + reg.view(-1, 1, 1) * identity

    # Final fallback (per sample)
    for i in range(batch_size):
        try:
            _ = torch.linalg.cholesky(precision[i])
        except RuntimeError:
            precision[i] += 1e-3 * identity[i]

    return precision


import torch
import torch.nn.functional as F
'''
def vectors_to_precision(vectors, dim, eps=1e-2):
    """
    Maps an array of 1xdim vectors into Cholesky factors and returns an array of precision matrices,
    ensuring positive definiteness via softplus on the diagonal and regularization.

    Args:
    vectors (torch.Tensor): A tensor of shape (batch_size, 1, dim), where each 1xdim tensor represents the
                            lower triangular part of the Cholesky factor.
    dim (int): The dimension of the square matrix.
    eps (float): Small positive constant to ensure positive definiteness.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, dim, dim), containing the corresponding precision matrices.
    """
    batch_size = vectors.shape[0]
    L = torch.zeros(batch_size, dim, dim, dtype=vectors.dtype, device=vectors.device)
    indices = torch.tril_indices(dim, dim, device=vectors.device)

    L[:, indices[0], indices[1]] = vectors.squeeze(1)

    # Apply softplus to diagonal to ensure strictly positive elements
    diag_indices = torch.arange(dim, device=vectors.device)
    L[:, diag_indices, diag_indices] = F.softplus(L[:, diag_indices, diag_indices])

    # Construct precision matrices
    C = torch.matmul(L, L.transpose(1, 2))

    # Add small ε * I to ensure numerical positive definiteness
    eye = torch.eye(dim, device=vectors.device, dtype=vectors.dtype).expand(batch_size, -1, -1)
    C += eps * eye

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
'''

"""
def vectors_to_precision_chunked(vectors, dim, chunk_size=2, reg=0.01):
    batch_size = vectors.shape[0]
    dtype = torch.float32
    precisions = []
    tril_indices = torch.tril_indices(dim, dim, device=vectors.device)

    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk = vectors[start:end]
        if chunk.ndim == 3 and chunk.shape[1] == 1:
            chunk = chunk.squeeze(1)

        L = torch.zeros((end - start, dim, dim), dtype=dtype, device=vectors.device)
        L[:, tril_indices[0], tril_indices[1]] = chunk.to(dtype=dtype)

        eye = torch.eye(dim, dtype=dtype, device=vectors.device).expand(end - start, -1, -1)
        C_chunk = torch.baddbmm(reg * eye, L, L.transpose(1, 2))  # reg > 0 keeps it PD

        precisions.append(C_chunk)

    return torch.cat(precisions, dim=0)

def vectors_to_precision_chunked(vectors, dim, chunk_size=2, reg=0.01):
    batch_size = vectors.shape[0]
    dtype = torch.float32
    precisions = []
    tril_indices = torch.tril_indices(dim, dim, device=vectors.device)

    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk = vectors[start:end]
        if chunk.ndim == 3 and chunk.shape[1] == 1:
            chunk = chunk.squeeze(1)

        L = torch.zeros((end - start, dim, dim), dtype=dtype, device=vectors.device)
        L[:, tril_indices[0], tril_indices[1]] = chunk.to(dtype=dtype)

        # Apply softplus to diagonal elements for positivity
        diag_indices = torch.arange(dim, device=vectors.device)
        L[:, diag_indices, diag_indices] = F.softplus(L[:, diag_indices, diag_indices])

        eye = torch.eye(dim, dtype=dtype, device=vectors.device).expand(end - start, -1, -1)
        C_chunk = torch.baddbmm(reg * eye, L, L.transpose(1, 2))  # Add jitter for stability

        precisions.append(C_chunk)

    return torch.cat(precisions, dim=0)
"""
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
    #softmax_probs = torch.softmax(log_probs, dim=1)  # Shape: (batch_size, num_components)
    softmax_probs = F.softmax(log_probs, dim=1) #stable softmax

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

#@profile
def laplacian_mog_density_div_density_chunked(x, means, precisions, chunk_size=16):
    results = []
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i:i+chunk_size] 
        result = laplacian_mog_density_div_density(x_chunk, means, precisions)
        results.append(result.detach() if not result.requires_grad else result)
        del result, x_chunk
        #gc.collect()
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

def laplacian_mog_density_div_density_chunked_components(x, means, precisions, chunk_size=32):
    batch_size, dim = x.shape
    num_components = means.shape[0]

    # Accumulate partial results
    all_log_probs = []
    all_squared_linear = []
    all_trace_precision = []

    # Compute chunked contributions
    for start in range(0, num_components, chunk_size):
        end = min(start + chunk_size, num_components)

        means_chunk = means[start:end]  # (chunk_size, dim)
        precisions_chunk = precisions[start:end]  # (chunk_size, dim, dim)

        x_expanded = x.unsqueeze(1)  # (B, 1, D)
        x_mean = x_expanded - means_chunk  # (B, chunk_size, D)

        # Compute log probs using Mahalanobis term: -0.5 * x^T P x + const
        cov_term = torch.einsum("kij,bkj->bki", precisions_chunk, x_mean)  # (B, chunk_size, D)
        squared_linear = torch.sum(cov_term * cov_term, dim=-1)  # (B, chunk_size)

        # log det precision = -log det cov = + log det precision
        log_det = torch.logdet(precisions_chunk)  # (chunk_size,)
        log_det = log_det.unsqueeze(0).expand(batch_size, -1)  # (B, chunk_size)

        log_probs_chunk = 0.5 * log_det - 0.5 * squared_linear - 0.5 * dim * torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))

        # trace of precision
        trace_chunk = torch.diagonal(precisions_chunk, dim1=-2, dim2=-1).sum(dim=-1)  # (chunk_size,)
        trace_chunk = trace_chunk.unsqueeze(0).expand(batch_size, -1)  # (B, chunk_size)

        all_log_probs.append(log_probs_chunk)
        all_squared_linear.append(squared_linear)
        all_trace_precision.append(trace_chunk)

        # Optional memory cleanup
        del x_mean, cov_term, squared_linear, trace_chunk, log_probs_chunk
        torch.cuda.empty_cache()

    # Concatenate across component dimension
    log_probs = torch.cat(all_log_probs, dim=1)  # (B, K)
    squared_linear = torch.cat(all_squared_linear, dim=1)  # (B, K)
    trace_precision = torch.cat(all_trace_precision, dim=1)  # (B, K)

    # Compute softmax over full set of components
    softmax_probs = torch.softmax(log_probs, dim=1)  # (B, K)

    laplacian_component = softmax_probs * (squared_linear - trace_precision)  # (B, K)
    laplacian_over_density = laplacian_component.sum(dim=1)  # (B,)

    return laplacian_over_density


#@profile
def grad_log_mog_density_chunked(x, means, precisions, chunk_size=8):
    outputs = []
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i:i+chunk_size]
        x_chunk.requires_grad_(x.requires_grad)  # No clone + detach
        result = grad_log_mog_density(x_chunk, means, precisions)
        outputs.append(result.detach())
        del result
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)

def grad_and_laplacian_mog_density(x, means, precisions):
    """
    Compute both gradient and Laplacian in a single pass to save memory
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
    
    # cast to double precision to help with stability
    #x = x.double()
    #means = means.double()
    precisions = precisions.double()

    # Log determinant - used for log probs
    #log_det = torch.logdet(precisions)  # (K,)
    #log_det = log_det.unsqueeze(0).expand(batch_size, -1)  # (B, K)

    # 4. Stabilize logdet: use Cholesky + logdet
    EPS = 1e-4  # or slightly higher depending on behavior
    try:
        chol = torch.linalg.cholesky(precisions)  # (K, D, D)
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + EPS), dim=-1)  # (K,)
    except RuntimeError:
        # Fallback to adding jitter if Cholesky fails
        precisions += EPS * torch.eye(precisions.size(-1), device=x.device)
        chol = torch.linalg.cholesky(precisions)
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + EPS), dim=-1)

    logdet = logdet.unsqueeze(0).expand(x.shape[0], -1)  # (B, K)

    # Compute log probabilities once
    log_probs = 0.5 * logdet - 0.5 * squared_linear - 0.5 * dim * torch.log(
        torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
    
    # Compute softmax probabilities once
    softmax_probs = torch.softmax(log_probs, dim=1)  # (B, K)
    
    # GRADIENT COMPUTATION
    gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)  # (B, D)
    
    # LAPLACIAN COMPUTATION
    trace_precision = torch.diagonal(precisions, dim1=-2, dim2=-1).sum(dim=-1)  # (K,)
    trace_precision = trace_precision.unsqueeze(0).expand(batch_size, -1)  # (B, K)
    
    laplacian_component = softmax_probs * (squared_linear - trace_precision)  # (B, K)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)  # (B,)
    
    # Clean up
    del x_mean, x_mean_cov, squared_linear, logdet, log_probs, softmax_probs
    torch.cuda.empty_cache()
    
    return gradient, laplacian_over_density


def grad_and_laplacian_mog_density_chunked_components(x, means, precisions, chunk_size=8):
    """
    Component-chunked version computing both gradient and Laplacian
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
        
        # Log determinant for log probabilities
        #log_det = torch.logdet(precisions_chunk)  # (chunk_size,)
        #log_det = log_det.unsqueeze(0).expand(batch_size, -1)  # (B, chunk_size)
        # cast to double precision to help with stability
        #x = x.double()
        #means = means.double()
        #precisions_chunk = precisions_chunk.double()

    # Log determinant - used for log probs
    #log_det = torch.logdet(precisions)  # (K,)
    #log_det = log_det.unsqueeze(0).expand(batch_size, -1)  # (B, K)

        # 4. Stabilize logdet: use Cholesky + logdet
        
        EPS = 1e-4  # or slightly higher depending on behavior
        try:
            chol = torch.linalg.cholesky(precisions_chunk)  # (K, D, D)
            log_det = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + EPS), dim=-1)  # (K,)
        except RuntimeError:
            # Fallback to adding jitter if Cholesky fails
            precisions += EPS * torch.eye(precisions_chunk.size(-1), device=x.device)
            chol = torch.linalg.cholesky(precisions_chunk)
            log_det = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + EPS), dim=-1)

        #log_det = log_det.unsqueeze(0).expand(x.shape[0], -1)  # (B, K)
        
        #log_det = torch.logdet(precisions_chunk)  # (chunk_size,)
        log_det = log_det.unsqueeze(0).expand(batch_size, -1)  # (B, chunk_size)
        # Log probabilities
        log_probs_chunk = 0.5 * log_det - 0.5 * squared_linear - 0.5 * dim * torch.log(
            torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))
        # Clamp log probabilities to prevent extreme values
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
        del x_mean, x_mean_cov, squared_linear, log_det, log_probs_chunk, weighted_gradient_chunk, trace_chunk, precisions_chunk
        torch.cuda.empty_cache()
    
    # Concatenate across component dimension
    log_probs = torch.cat(all_log_probs, dim=1)  # (B, K)
    weighted_gradients = torch.cat(all_weighted_gradients, dim=1)  # (B, K, D)
    squared_linear_full = torch.cat(all_squared_linear, dim=1)  # (B, K)
    trace_precision_full = torch.cat(all_trace_precision, dim=1)  # (B, K)
    
    # Compute softmax over full set of components (CRITICAL!)
    softmax_probs = torch.softmax(log_probs, dim=1)  # (B, K)
    
    # GRADIENT: Weight gradients by softmax probabilities and sum over components
    gradient = torch.sum(softmax_probs.unsqueeze(-1) * weighted_gradients, dim=1)  # (B, D)
    
    # LAPLACIAN: Weight Laplacian components by softmax probabilities
    laplacian_component = softmax_probs * (squared_linear_full - trace_precision_full)  # (B, K)
    laplacian_over_density = torch.sum(laplacian_component, dim=1)  # (B,)
    del log_probs, weighted_gradients, squared_linear_full, trace_precision_full, softmax_probs, laplacian_component
    torch.cuda.empty_cache()
    return gradient, laplacian_over_density


def grad_and_laplacian_mog_density_double_chunked(x, means, precisions,
                                                 batch_chunk_size=16,
                                                 component_chunk_size=32):
    """
    Double-chunked version: chunks both batch and component dimensions
    """
    gradients = []
    laplacians = []
    
    for start in range(0, x.size(0), batch_chunk_size):
        end = min(start + batch_chunk_size, x.size(0))
        x_chunk = x[start:end]
        
        # Call component-wise chunked function for each batch chunk
        grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density_chunked_components(
            x_chunk, means, precisions, chunk_size=component_chunk_size)
        
        gradients.append(grad_chunk.detach() if not grad_chunk.requires_grad else grad_chunk)
        laplacians.append(laplacian_chunk.detach() if not laplacian_chunk.requires_grad else laplacian_chunk)
        
        del grad_chunk, laplacian_chunk, x_chunk
        torch.cuda.empty_cache()
    
    return torch.cat(gradients, dim=0), torch.cat(laplacians, dim=0)


def grad_and_laplacian_mog_density_batch_chunked(x, means, precisions, chunk_size=16):
    """
    Simple batch chunking version
    """
    gradients = []
    laplacians = []
    
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i:i+chunk_size]
        
        grad_chunk, laplacian_chunk = grad_and_laplacian_mog_density(x_chunk, means, precisions)
        gradients.append(grad_chunk.detach() if not grad_chunk.requires_grad else grad_chunk)
        laplacians.append(laplacian_chunk.detach() if not laplacian_chunk.requires_grad else laplacian_chunk)
        
        del grad_chunk, laplacian_chunk, x_chunk
        torch.cuda.empty_cache()
    
    return torch.cat(gradients, dim=0), torch.cat(laplacians, dim=0)

#grad_eval_comp = torch.compile(grad_log_mog_density_double_chunked)
#----------------------------------------------#
#### STABILITY ENHANCEMENT FUNCTIONS ####
#----------------------------------------------#

def adaptive_regularization(matrices, base_epsilon=1e-4, max_cond=1e12):
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

def stable_logdet(matrices, eps=1e-6, max_attempts=3):
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

def vectors_to_precision_stable(vectors, dim, base_epsilon=1e-4, max_cond=1e12):
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

    # # Spectral clipping
    # try:
    #     precision = torch.nan_to_num(precision, nan=1e-3, posinf=1e3, neginf=-1e3)
    #     eigenvals, eigenvecs = torch.linalg.eigh(precision)
    #     eigenvals_clipped = torch.clamp(eigenvals, min=1e-6, max=1e6)
    #     precision = torch.matmul(eigenvecs, torch.matmul(torch.diag_embed(eigenvals_clipped), eigenvecs.transpose(-2, -1)))
    # except RuntimeError as e:
    #     print("❌ Eigendecomposition failed. Adding extra regularization.")
    #     precision = precision + base_epsilon * 10 * identity

    # Final safety check
    for attempt in range(3):
        try:
            torch.linalg.cholesky(precision)
            break
        except RuntimeError:
            reg_strength = base_epsilon * (10 ** (attempt + 1))
            precision = precision + reg_strength * identity

    return precision

def vectors_to_precision_chunked_stable(vectors, dim, base_epsilon=1e-4, chunk_size=50):
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
    logits_clamped = torch.clamp(logits_stable, min=-100, max=100)
    
    return F.softmax(logits_clamped, dim=dim)

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
    log_probs = torch.clamp(log_probs, min=-100, max=100)
    
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
                                                  batch_chunk_size=16, 
                                                  component_chunk_size=64,
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
                                                           chunk_size=256, temperature=1.0):
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

import time

def score_implicit_matching_stable(factornet, samples, centers, base_epsilon=1e-4, 
                                  temperature=1.0, max_attempts=3):
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


#------------------------------------optimized versions --------------------------
import torch
import torch.nn.functional as F
def vectors_to_precision_optimized(vectors, dim, base_epsilon=1e-4):
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
def vectors_to_precision_chunked_optimized(vectors, dim, epsilon=0.01,chunk_size=10):
    results = []
    for i in range(0, vectors.size(0), chunk_size):
        chunk = vectors[i:i+chunk_size]
        result = vectors_to_precision_optimized(chunk, dim, epsilon)
        results.append(result)
        # Memory cleanup
        del chunk, result
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

def adaptive_regularization_fast(matrices, base_epsilon=1e-4, max_cond=1e12):
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
            #print("factor_eval shape:", factor_eval.shape)  # Should be (K, D_tri)
            #print("Factornet time:", time.time() - t0)
            
            # Ensure consistent dtypes
            if autocast_enabled:
                factor_eval = factor_eval.float()
                centers = centers.float()
                samples = samples.float()
            
            # Optimized precision computation
            #t0 = time.time()
            precisions = vectors_to_precision_optimized(factor_eval, dim, base_epsilon)
            #print("precisions shape:", precisions.shape)
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
        return score_implicit_matching_stable(factornet, samples, centers, 
                                            base_epsilon, temperature)

def memory_efficient_chunked_processing(x, means, precisions, chunk_size_factor=0.8):
    """
    Adaptive chunking based on available memory
    """
    if torch.cuda.is_available():
        # Estimate memory usage and adjust chunk size
        available_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated()
        free_memory = available_memory - used_memory
        
        # Estimate memory per sample and adjust chunk size
        estimated_memory_per_sample = x.numel() * x.element_size() * means.shape[0] * 4
        safe_chunk_size = int(free_memory * chunk_size_factor / estimated_memory_per_sample)
        chunk_size = max(1, min(safe_chunk_size, x.shape[0]))
    else:
        chunk_size = min(32, x.shape[0])  # Conservative CPU default
    
    if chunk_size >= x.shape[0]:
        # No chunking needed
        return grad_and_laplacian_mog_density_optimized(x, means, precisions)
    
    # Process in chunks
    gradients, laplacians = [], []
    precomputed_terms = precompute_precision_terms(precisions)
    
    for i in range(0, x.shape[0], chunk_size):
        chunk = x[i:i+chunk_size]
        grad_chunk, lap_chunk = grad_and_laplacian_mog_density_optimized(
            chunk, means, precisions, precomputed_terms=precomputed_terms
        )
        gradients.append(grad_chunk)
        laplacians.append(lap_chunk)
    
    return torch.cat(gradients), torch.cat(laplacians)

# Additional utility functions for maximum performance
def compile_functions_for_speed():
    """
    Use torch.compile for JIT optimization (PyTorch 2.0+)
    """
    try:
        compiled_functions = {
            'vectors_to_precision': torch.compile(vectors_to_precision_optimized, mode='max-autotune'),
            'grad_and_laplacian': torch.compile(grad_and_laplacian_mog_density_optimized, mode='max-autotune'),
            'score_matching': torch.compile(score_implicit_matching_optimized, mode='max-autotune')
        }
        return compiled_functions
    except AttributeError:
        print("torch.compile not available, using standard functions")
        return None

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

#----------------------------------------------#
def score_implicit_matching(factornet, samples, centers, epsilon=0.01):
    torch.cuda.reset_peak_memory_stats()
    dim = centers.shape[-1]

    #centers.requires_grad_(True)
    centers = centers.clone().detach().requires_grad_(True)
    factor_eval = checkpoint(factornet, centers, use_reentrant=False)
    precisions = vectors_to_precision_chunked(factor_eval, dim, epsilon)
    #v_to_p_comp = torch.compile(vectors_to_precision_chunked)
    #precisions = v_to_p_comp(factor_eval, dim, chunk_size=2)
    #cond_numbers = LA.cond(precisions)  # condition number per matrix
    #print("Condition stats — min:", cond_numbers.min(), "max:", cond_numbers.max(), "mean:", cond_numbers.mean())
    del factor_eval
    torch.cuda.empty_cache()

    #with autocast(device_type='cuda'):  # or 'cpu' if not using GPU with autocast():
    # Compute laplacian/density in chunks
    #laplacian_over_density = laplacian_mog_density_div_density_chunked(samples, centers, precisions)
        #laplacian_over_density_comp = torch.compile(laplacian_mog_density_div_density)
        #laplacian_over_density = laplacian_over_density_comp(samples, centers, precisions)
    # Compute gradient in chunks
    #gradient_eval_log = grad_log_mog_density_chunked(samples, centers, precisions)
        #gradient_eval_log = grad_eval_comp(samples, centers, precisions)

    #compute both at once to save ram
    #torch.cuda.empty_cache()
    #torch.cuda.reset_peak_memory_stats()
    gradient_eval_log, laplacian_over_density = grad_and_laplacian_mog_density_double_chunked(samples, centers, precisions)
    #for i in range(torch.cuda.device_count()):
    #print(f"[non-chunked] GPU {i}: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")
    
    del precisions
    torch.cuda.empty_cache()

    # Compute norm squared of gradient ∇logp
    gradient_eval_log_squared = torch.sum(gradient_eval_log * gradient_eval_log, dim=1)
    del gradient_eval_log
    torch.cuda.empty_cache()

    # Compute final loss
    loss = 2 * laplacian_over_density - gradient_eval_log_squared
    del laplacian_over_density, gradient_eval_log_squared
    torch.cuda.empty_cache()

    #print("Peak memory (bytes):", torch.cuda.max_memory_allocated())
    #print(torch.cuda.memory_summary())

    return loss.mean(dim=0)

    

#----------------------------------------------#
#### plotting functions ####
#----------------------------------------------#

def mog_density_2d_marg(x, means, precisions, n):
    device = x.device
    num_components = means.shape[0]
    num_variables = means.shape[1]
    
    # Ensure n is within a valid range
    if n < 0 or n >= num_variables:
        raise ValueError("Invalid value of n. n should be in the range [0, num_variables - 1].")
    
    # Create indices to select the variables to keep
    keep_indices = [i for i in range(num_variables) if i != n]
    
    # Remove the nth variable from means and adjust the precision matrix
    means = means[:, keep_indices]  # Shape: (num_components, num_variables - 1)
    
    # Create a mask to remove the nth row and column from the precision matrix
    mask = torch.ones(num_variables, dtype=torch.bool, device=device)
    mask[n] = 0
    
    # Apply the mask to create a new precision matrix
    new_precisions = precisions[:, mask, :][:, :, mask]  # Shape: (num_components, num_variables - 1, num_variables - 1)
    
    # Expand the dimensions of x for broadcasting
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, num_variables - 1)
    
    # Create a batch of Multivariate Normal distributions for each component
    mvns = MultivariateNormal(loc=means, precision_matrix=new_precisions)
    
    # Calculate the probabilities for each component
    component_probs = mvns.log_prob(x).exp()  # Shape: (batch_size, num_components)
    
    # Calculate the weighted sum of component probabilities
    densities = component_probs.mean(dim=1)  # Shape: (batch_size,)
    
    return densities,new_precisions

def plot_2d_trained_density_2d_marg(centers,factornet, loss = 0, n = 2, epoch = 0, save_path = None):
    # n: the dimenstion to be marginalized out

    plot_axis = centers.max().item() * 1.1
    device = centers.device
    dim = centers.shape[-1]
    n_x1 = 80
    n_x2 = 80
    x1 = np.linspace(-plot_axis, plot_axis, n_x1)
    x2 = np.linspace(-plot_axis, plot_axis, n_x2)

    # Create a grid of (x1, x2) coordinates using meshgrid
    X1, X2 = np.meshgrid(x1, x2)
    x = torch.tensor(np.stack([X1, X2], axis=2).reshape(-1, 2)).to(device)

    precision_matrix = vectors_to_precision(factornet(centers),dim)
    density, _ =  mog_density_2d_marg(x, centers, precision_matrix, n)
    # print(density.shape)

    density = density.reshape(n_x1, n_x2)
    plot_slice = density.cpu().detach().numpy()
    fig = plt.figure(figsize=(6, 5))
    plt.contourf(x1, x2, plot_slice, cmap='viridis', levels=20)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    loss_str = f'{loss:.3e}'
    plt.title(f'Epoch: {epoch}, Loss: {loss_str}')
    if save_path is not None:
        save_path = save_path + 'epoch'+ str(epoch) +'_marginalized_density_dim_' + str(n) + '.png'
        plt.savefig(save_path)
    
    plt.close(fig)
    return None

def mog_density_marg2d(x, means, precisions, dim1, dim2):
    # keep only dim1 and dim2;
    # other function remove\marginalize the nth dimension
    device = x.device
    num_components = means.shape[0]
    num_variables = means.shape[1]
    
    # Ensure n is within a valid range
    if dim1 >= num_variables or dim2 >= num_variables:
        raise ValueError("Invalid value of dim1 dim2")
    
    # Create indices to select the variables to keep
    # keep_indices = [i for i in range(num_variables) if i != n]
    keep_indices = [dim1, dim2]
    
    # Remove the nth variable from means and adjust the precision matrix
    means = means[:, keep_indices]  # Shape: (num_components, 2)
    
    # Create a mask to remove the nth row and column from the precision matrix
    mask = torch.zeros(num_variables, dtype=torch.bool, device=device)
    mask[dim1] = 1
    mask[dim2] = 1
    
    # Apply the mask to create a new precision matrix
    new_precisions = precisions[:, mask, :][:, :, mask]  # Shape: (num_components, 2, 2)
    
    # Expand the dimensions of x for broadcasting
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, num_variables - 1)
    
    # Create a batch of Multivariate Normal distributions for each component
    mvns = MultivariateNormal(loc=means, precision_matrix=new_precisions)
    
    # Calculate the probabilities for each component
    component_probs = mvns.log_prob(x).exp()  # Shape: (batch_size, num_components)
    
    # Calculate the weighted sum of component probabilities
    densities = component_probs.mean(dim=1)  # Shape: (batch_size,)
    
    return densities,new_precisions

def plot_density_2d_marg(centers,factornet, loss = 0, dim1 = 0, dim2 = 1, epoch = 0, save_path = None):
    # dim1, dim2: the dimenstions to plot
    plot_axis = centers.max().item() * 1.1
    device = centers.device
    dim = centers.shape[-1]
    n_x1 = 80
    n_x2 = 80
    x1 = np.linspace(-plot_axis, plot_axis, n_x1)
    x2 = np.linspace(-plot_axis, plot_axis, n_x2)

    # Create a grid of (x1, x2) coordinates using meshgrid
    X1, X2 = np.meshgrid(x1, x2)
    x = torch.tensor(np.stack([X1, X2], axis=2).reshape(-1, 2)).to(device)

    precision_matrix = vectors_to_precision(factornet(centers),dim)
    density, _ =  mog_density_marg2d(x, centers, precision_matrix, dim1,dim2)

    density = density.reshape(n_x1, n_x2)
    plot_slice = density.cpu().detach().numpy()
    fig = plt.figure(figsize=(6, 5))
    plt.contourf(x1, x2, plot_slice, cmap='viridis', levels=20)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    loss_str = f'{loss:.3e}'
    plt.title(f'Epoch: {epoch}, Loss: {loss_str}')

    if save_path is not None:
        save_path = save_path + 'epoch'+ str(epoch) +'_marginalized_density_dim_' + str(dim1) + '-'  + str(dim2)+ '.png'
        plt.savefig(save_path)
    
    plt.close(fig)
    return None

def scatter_samples_from_model(means, precisions, dim1, dim2, epoch = 0,plot_number = 1000, save_path=None):
    # plot scatter plot of samples from the model
    # keep only dim1 and dim2;
    # other function remove\marginalize the rest dimensions
    
    num_components = means.shape[0]
    num_variables = means.shape[1]
    # device = means.device
    
    # Ensure n is within a valid range
    if dim1 >= num_variables or dim2 >= num_variables:
        raise ValueError("Invalid value of dim1 dim2")
    
    # Create indices to select the variables to keep
    keep_indices = [dim1, dim2]
    
    # Remove the nth variable from means and adjust the precision matrix
    means = means[:, keep_indices]  # Shape: (num_components, 2)
    
    # Create a mask to remove the nth row and column from the precision matrix
    mask = torch.zeros(num_variables, dtype=torch.bool)
    mask[dim1] = 1
    mask[dim2] = 1
    # Apply the mask to create a new precision matrix
    new_precisions = precisions[:, mask, :][:, :, mask]  # Shape: (num_components, 2, 2)

    multivariate_normal = torch.distributions.MultivariateNormal(means, precision_matrix=new_precisions)
    plot_number_factor =  plot_number//num_components +1
    samples = multivariate_normal.sample((plot_number_factor,))
    samples = samples.reshape(-1,2)
    samples = samples[:plot_number,:]
    print(samples.shape)
    samples = samples.cpu().detach().numpy()
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(samples[:,0],samples[:,1],label = 'samples', s = 0.5)
    if save_path is not None:
        save_path = save_path + 'epoch'+ str(epoch) +'_scatter_dim_' + str(dim1) + '-'  + str(dim2)+ '.png'
        plt.savefig(save_path)
    
    plt.close(fig)

    return None

def sample_from_model(factornet, means, sample_number, eps, use_stable=False):
    num_components, dim = means.shape
    comp_num = torch.randint(0, num_components, (sample_number,), device=means.device)
    samples = torch.empty(sample_number, dim, device=means.device)

    unique_indices = comp_num.unique()

    for i in unique_indices:
        idx = (comp_num == i).nonzero(as_tuple=True)[0]
        n_i = idx.shape[0]
        centers_i = means[i].unsqueeze(0).expand(n_i, -1)  # [n_i, dim]

        vectors = factornet(centers_i)  # [n_i, d*(d+1)//2]

        if use_stable:
            precision = vectors_to_precision_stable(vectors, dim, eps)
        else:
            precision = vectors_to_precision(vectors, dim, 0.001)

        mvn = MultivariateNormal(loc=centers_i, precision_matrix=precision)
        samples_i = mvn.rsample()  # [n_i, dim]
        samples[idx] = samples_i

    return samples

def plot_images_with_model(factornet, means, plot_number=10, eps=1e-4, save_path=None):
    dim = means.shape[-1]

    # Get two sets of samples: standard and stable
    samples1 = sample_from_model(factornet, means, plot_number, eps, use_stable=False)
    samples2 = sample_from_model(factornet, means, plot_number, eps, use_stable=True)

    # Denormalize using CIFAR-10 statistics
    def format_samples(samples):
        samples = samples.view(-1, 3, 32, 32)
        samples = denormalize_cifar10(samples)
        samples = torch.clamp(samples, 0, 1)
        return samples

    samples1 = format_samples(samples1)
    samples2 = format_samples(samples2)

    # Plot 2 rows: first row = standard, second row = stable
    fig, axs = plt.subplots(2, plot_number, figsize=(plot_number * 1.5, 4))
    for row, samples in enumerate([samples1, samples2]):
        for i in range(plot_number):
            img = samples[i].permute(1, 2, 0).cpu().numpy()
            axs[row, i].imshow(img)

            # Label the first image of each row
            if i == 0:
                axs[row, i].set_ylabel("Standard" if row == 0 else "Stable", fontsize=12)
            else:
                axs[row, i].set_yticks([])

            axs[row, i].set_xticks([])
            axs[row, i].tick_params(left=False, bottom=False)

    if save_path is not None:
        plt.savefig(save_path + '_sampled_images.png', bbox_inches='tight')
    plt.close(fig)
    return None

def plot_images(means, precisions, plot_number = 10,  epoch = 0, save_path=None):
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

def plot_and_save_centers(centers, save_path, nrow=10, upscale_factor=8):
    centers = centers.view(-1, 3, 32, 32)  # CIFAR-10 shape
    centers = denormalize_cifar10(centers).clamp(0, 1)

    # Create grid image
    grid = vutils.make_grid(centers, nrow=nrow, padding=6)

    # Upscale the grid using bilinear interpolation
    grid = grid.unsqueeze(0)  # add batch dim for interpolation
    height, width = grid.shape[2], grid.shape[3]
    new_height = height * upscale_factor
    new_width = width * upscale_factor
    grid_upscaled = F.interpolate(grid, size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

    # Save upscaled image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vutils.save_image(grid_upscaled, save_path)
    print(f"Saved upscaled centers to {save_path}")

    # Show in matplotlib
    plt.figure(figsize=(new_width / 100, new_height / 100), dpi=100)
    plt.axis("off")
    plt.imshow(grid_upscaled.permute(1, 2, 0).cpu().numpy())
    plt.show()
