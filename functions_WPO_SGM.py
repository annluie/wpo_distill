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
from torch.cuda.amp import autocast
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
    del x_mean_cov
    gc.collect()
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


def laplacian_mog_density_div_density(x, means, precisions):
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, 2)
    batch_size, num_components = x.size(0), means.size(0)

    # Create a batch of Multivariate Normal distributions for each component
    mvns = MultivariateNormal(loc=means, precision_matrix=precisions)

    # Calculate the log probabilities for each component
    log_probs = mvns.log_prob(x)  # Shape: (batch_size, num_components)

    # Use torch.logsumexp to compute the log of the sum of exponentiated log probabilities
    log_sum_exp = torch.logsumexp(log_probs, dim=1, keepdim=True)  # Shape: (batch_size, 1)

    # Calculate the softmax probabilities along the components dimension
    softmax_probs = torch.softmax(log_probs, dim=1)  # Shape: (batch_size, num_components)

    x_mean = x - means  # Shape: (batch_size, num_components, 2)

    # Calculate the covariance matrix term
    cov_term = torch.matmul(precisions, x_mean.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, num_components, 2)

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

    return laplacian_over_density
#----------------------------------------------#
#BATCHED VERSIONS
@profile
def laplacian_mog_density_div_density_chunked(x, means, precisions, chunk_size=16):
    results = []
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i:i+chunk_size].detach().clone().requires_grad_(x.requires_grad)
        result = laplacian_mog_density_div_density(x_chunk, means, precisions)
        results.append(result.detach() if not result.requires_grad else result)
        del result, x_chunk
        gc.collect()
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)
@profile
def grad_log_mog_density_chunked(x, means, precisions, chunk_size=16):
    results = []
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i:i+chunk_size].detach().clone().requires_grad_(x.requires_grad)
        result = grad_log_mog_density(x_chunk, means, precisions)
        results.append(result.detach() if not result.requires_grad else result)
        del result, x_chunk
        gc.collect()
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)
#----------------------------------------------#
def score_implicit_matching(factornet,samples,centers):
    # detach the samples and centers
    #samples = samples.detach()
    #centers = centers.detach()
    centers.requires_grad_(True)
    dim = centers.shape[-1]
    factor_eval = checkpoint(factornet,centers)
    precisions = vectors_to_precision(factor_eval,dim)
    #with autocast("cuda"):
    with autocast('cpu', dtype=torch.bfloat16):
        #print(f"precisions requires grad: {precisions.requires_grad}")
        #print(f"samples requires grad: {samples.requires_grad}")
        #print(f"centers requires grad: {centers.requires_grad}")
        #laplacian_over_density = laplacian_mog_density_div_density(samples,centers,precisions)
        laplacian_over_density = laplacian_mog_density_div_density_chunked(samples,centers,precisions)
        gradient_eval_log = grad_log_mog_density_chunked(samples,centers,precisions)
        gradient_eval_log_squared = torch.sum(gradient_eval_log * gradient_eval_log, dim=1)
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
        img = samples[i].permute(1, 2, 0).numpy()  # change from [C, H, W] to [H, W, C]
        axs[i].imshow(img)
        axs[i].axis('off')
    if save_path is not None:
        save_path = save_path + 'epoch'+ str(epoch) + '.png'
        plt.savefig(save_path)
    
    plt.close(fig)

    return None