import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import lib.toy_data as toy_data
import numpy as np
import argparse
from memory_profiler import profile
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
import gc

## vector to precision matrix function
#@profile
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
    C = torch.matmul(L, L.transpose(1, 2)) + 0.05 * torch.eye(dim).to(vectors.device) # (add identity matrix to maintain positive definiteness)
    
    return C

def vectors_to_precision_chunked(vectors, dim, chunk_size=10):
    batch_size = vectors.size(0)
    output = torch.empty(batch_size, dim, dim, dtype=vectors.dtype, device=vectors.device)
    for i in range(0, batch_size, chunk_size):
        chunk = vectors[i:i+chunk_size]
        output[i:i+chunk_size] = vectors_to_precision(chunk, dim)
    return output

def vectors_to_precision_stab(vectors, dim, eps=1e-4, scale_offdiag=0.1):
    batch_size = vectors.shape[0]
    if vectors.dim() == 3:
        vectors = vectors.squeeze(1)

    tril_indices = torch.tril_indices(dim, dim, device=vectors.device)
    L = torch.zeros(batch_size, dim, dim, dtype=vectors.dtype, device=vectors.device)
    L[:, tril_indices[0], tril_indices[1]] = vectors

    diag = torch.arange(dim, device=vectors.device)
    L[:, diag, diag] = F.softplus(L[:, diag, diag]) + eps

    # Scale off-diagonal terms early in training
    off_diag_mask = torch.ones_like(L).tril(-1)
    L = L * (1 - off_diag_mask + scale_offdiag * off_diag_mask)

    C = L @ L.transpose(-1, -2)
    return C

#----------------------------------------------#
#### Mixture of Gaussians functions ####
#----------------------------------------------#

# Compute MoG density for each data point
#@profile
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
#@profile
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
#@profile
def grad_log_mog_density(x, means, precisions):
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, 2)
    dim = x.size(-1)
    batch_size, num_components = x.size(0), means.size(0)

    # Create a batch of Multivariate Normal distributions for each component
    mvns = MultivariateNormal(loc=means, precision_matrix=precisions)

    # Calculate the log probabilities for each component
    log_probs = mvns.log_prob(x)  # Shape: (batch_size, num_components)
    del mvns
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
    del x_mean_cov, log_probs, softmax_probs
    gc.collect()
    torch.cuda.empty_cache()
    return gradient


# Compute Laplacian
#@profile
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

    mvn = MultivariateNormal(means, precision_matrix=precision_matrix, validate_args=False)
    
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
    mvns = MultivariateNormal(loc=means, precision_matrix=precisions, validate_args=False)

    # Calculate the log probabilities for each component
    log_probs = mvns.log_prob(x)  # Shape: (batch_size, num_components)

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
    gc.collect()
    torch.cuda.empty_cache()
    return laplacian_over_density



#----------------------------------------------#
#@profile
def laplacian_mog_density_div_density_chunked(x, means, precisions, chunk_size=16):
    results = []
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i:i+chunk_size] 
        result = laplacian_mog_density_div_density(x_chunk, means, precisions)
        results.append(result.detach() if not result.requires_grad else result)
        del result, x_chunk
        gc.collect()
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
def laplacian_mog_density_div_density_double_chunked(x, means, precisions, 
                                                    batch_chunk_size=16, 
                                                    component_chunk_size=32):
    results = []
    for start in range(0, x.size(0), batch_chunk_size):
        end = min(start + batch_chunk_size, x.size(0))
        x_chunk = x[start:end]
        # Call component-wise chunked function for each batch chunk
        result_chunk = laplacian_mog_density_div_density_chunked_components(
            x_chunk, means, precisions, chunk_size=component_chunk_size)
        results.append(result_chunk.detach() if not result_chunk.requires_grad else result_chunk)
        del result_chunk, x_chunk
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

def grad_log_mog_density_chunked(x, means, precisions, chunk_size=2):
    outputs = []
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i:i+chunk_size]
        x_chunk.requires_grad_(x.requires_grad)  # No clone + detach
        result = grad_log_mog_density(x_chunk, means, precisions)
        outputs.append(result.detach())
        del result
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)

#@profile
def score_implicit_matching(factornet,samples,centers):
    # detach the samples and centers
    #samples = samples.detach()
    #centers = centers.detach()
    centers.requires_grad_(True)
    dim = centers.shape[-1]
    #factor_eval = checkpoint(factornet, centers, use_reentrant=False)
    factor_eval = factornet(centers)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    precisions = vectors_to_precision_chunked(factor_eval, dim)
    for i in range(torch.cuda.device_count()):
        print(f"[chunked] GPU {i}: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")
    #precisions = vectors_to_precision_chunked(factor_eval, dim)
    #eigs = torch.linalg.eigvalsh(precisions)
    #min_eig = eigs.min(dim=1).values
    #print("Min eigenvalue per batch:", min_eig)
    del factor_eval
    torch.cuda.empty_cache()
    #with autocast("cuda"):
    #with autocast('cpu', dtype=torch.bfloat16):
    #print(f"precisions requires grad: {precisions.requires_grad}")
    #print(f"samples requires grad: {samples.requires_grad}")
    #print(f"centers requires grad: {centers.requires_grad}")
    #laplacian_over_density = laplacian_mog_density_div_density(samples,centers,precisions)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    laplacian_over_density = laplacian_mog_density_div_density_chunked(samples,centers,precisions)
    for i in range(torch.cuda.device_count()):
        print(f"[not double-chunked] GPU {i}: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")
    gradient_eval_log = grad_log_mog_density_chunked(samples,centers,precisions)
    del precisions
    gradient_eval_log_squared = (gradient_eval_log ** 2).sum(dim=1)
    #print(f"laplacian_over_density requires grad: {laplacian_over_density.requires_grad}")
    #print(f"gradient_eval_log requires grad: {gradient_eval_log.requires_grad}")
    """
    # evaluate factor net
    factor_eval = factornet(centers) 
    # create precision matrix from the cholesky factor
    dim = centers.shape[-1]
    precisions = vectors_to_precision(factor_eval,dim)
    print(f"precisions requires grad: {precisions.requires_grad}")
    print(f"samples requires grad: {samples.requires_grad}")
    print(f"centers requires grad: {centers.requires_grad}")
    laplacian_over_density = laplacian_mog_density_div_density(samples,centers,precisions)
    gradient_eval_log = grad_log_mog_density(samples,centers,precisions)
    # square gradient
    
    gradient_eval_log_squared = torch.sum(gradient_eval_log * gradient_eval_log, dim=1)
    print(f"laplacian_over_density requires grad: {laplacian_over_density.requires_grad}")
    print(f"gradient_eval_log requires grad: {gradient_eval_log.requires_grad}")
    """
    del gradient_eval_log
    torch.cuda.empty_cache()
    #loss function
    loss = (2 * laplacian_over_density - gradient_eval_log_squared)

    return loss.mean(dim =0)



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
        save_path = save_path + '_sampled_images.png'
        plt.savefig(save_path)
    plt.close(fig)
    return None

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
        img = samples[i].permute(1, 2, 0).detatch().numpy()  # change from [C, H, W] to [H, W, C]
        axs[i].imshow(img)
        axs[i].axis('off')
    if save_path is not None:
        save_path = save_path + 'epoch'+ str(epoch) + '.png'
        plt.savefig(save_path)
    
    plt.close(fig)

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
