import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import lib.toy_data as toy_data
import numpy as np
import argparse
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
    torch.cuda.empty_cache()
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
    torch.cuda.empty_cache()
    del x_mean_cov
    return gradient / num_components

# Compute grad log pi
def grad_log_mog_density(x, means, precisions):
    #cpu_device = torch.device("cpu") # move to cpu to save memory
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, 2)
    dim = x.size(-1)
    batch_size, num_components = x.size(0), means.size(0)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    #x_cpu = x.to(cpu_device).unsqueeze(1)
    #means_cpu = means.to(cpu_device)
    #precisions_cpu = precisions.to(cpu_device)
    #dim = x_cpu.size(-1)
    #batch_size, num_components = x_cpu.size(0), means_cpu.size(0)

    # Create a batch of Multivariate Normal distributions for each component
    #mvns = MultivariateNormal(loc=means, precision_matrix=precisions)
    mvns_list = safe_multivariate_normal(means,precisions)

    # Calculate the log probabilities for each component
    #log_probs = mvns.log_prob(x)  # Shape: (batch_size, num_components)
    #log_probs = torch.stack([mvn.log_prob(x) for mvn in mvns_list], dim = 1)
    #log_probs = torch.stack([mvn.log_prob(x.cpu()) for mvn in mvns_list], dim=1)
    
    log_probs_list =[]
    for i in range(0, len(mvns_list), 2):
        batch_mvns = mvns_list[i:i+2]
        log_probs_batch = [mvn.log_prob(x.cpu()).squeeze(-1) for mvn in batch_mvns]
        log_probs_list.extend(log_probs_batch)
    for idx, lp in enumerate(log_probs_batch):
        print(f"log_prob_batch[{idx}] shape: {lp.shape}")
    log_probs = torch.stack(log_probs_list, dim=1)
    print(f"log_probs shape after stack: {log_probs.shape}")
    log_probs = log_probs.to(x.device)
    #log_probs = safe_log_probs(x, means, precisions, 1)

    # Use torch.logsumexp to compute the log of the sum of exponentiated log probabilities
    log_sum_exp = torch.logsumexp(log_probs, dim=1, keepdim=True)  # Shape: (batch_size, 1)
    print("log_sum_exp size:", log_sum_exp.shape)
    # Calculate the softmax probabilities along the components dimension
    softmax_probs = torch.softmax(log_probs, dim=1)  # Shape: (batch_size, num_components)
    print("softmax_probs size:", softmax_probs.shape)
    x_mean = x - means  # Shape: (batch_size, num_components, 2)
    x_mean_reshaped = x_mean.view(batch_size, num_components, dim, 1)
    precision_matrix = precisions.unsqueeze(0)  # Shape: (1, num_components, 2, 2)
    precision_matrix = precision_matrix.expand(x.shape[0], -1, -1, -1)  # Shape: (batch_size, num_components, 2, 2)
    
    """
    x_mean_cpu = x_cpu - means_cpu  # Shape: (batch_size, num_components, 2)
    x_mean_reshaped_cpu = x_mean_cpu.view(batch_size, num_components, dim, 1)
    precision_matrix_cpu = precisions_cpu.unsqueeze(0)  # Shape: (1, num_components, 2, 2)
    precision_matrix_cpu = precision_matrix_cpu.expand(x_cpu.shape[0], -1, -1, -1)  # Shape: (batch_size, num_components, 2, 2)
    """
    gc.collect() 
    torch.cuda.empty_cache()
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    #device = torch.device('cuda')
    results = []
    print(torch.cuda.memory_summary())
    for i in range(0, precision_matrix.size(0), 1):
        batch_prec = precision_matrix[i:i+1]
        batch_x = x_mean_reshaped[i:i+1]
        print(f"Batch precision matrix shape: {batch_prec.shape}")
        print(f"Batch x_mean shape: {batch_x.shape}")
        result = torch.matmul(batch_prec, batch_x).squeeze(-1)
        results.append(result)
        torch.cuda.empty_cache()
        print(f"Processing batch {i}-{i+1}")
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    x_mean_cov = torch.cat(results, dim=0).to(x.device)
    print("softmax_probs shape:", softmax_probs.shape)        # Should be [batch_size, K]
    print("x_mean_cov shape:", x_mean_cov.shape)
    print(f"x_mean_cov req grad:{x_mean_cov.requires_grad}")
    print(f"softmax_probs req grad:{softmax_probs.requires_grad}")
    #x_mean_cov_cpu = torch.matmul(precision_matrix, x_mean_reshaped).squeeze(dim = -1)
    #x_mean_cov_cpu = torch.matmul(precision_matrix_cpu, x_mean_reshaped_cpu).squeeze(dim = -1)
    #x_mean_cov = x_mean_cov_cpu.to(x.device)
    # Calculate the gradient of log density with respect to x

    gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov, dim=1)
    #gradient = -torch.sum(softmax_probs.unsqueeze(-1) * x_mean_cov_cpu, dim=1)
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
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

    #mvn = MultivariateNormal(means, precision_matrix=precision_matrix)
    #log_prob = mvn.log_prob(x) # Shape: (batch_size, num_components)
    log_prob =  safe_log_probs(x, means, precisions, batch_size=2)

    prob = log_prob.exp() # Shape: (batch_size, num_components)
    torch.cuda.empty_cache()

    # Calculate the gradient components using matrix multiplication
    x_mean_cov = torch.matmul(precision_matrix, x_mean_reshaped).squeeze(-1)

    squared_linear = torch.sum(x_mean_cov * x_mean_cov, dim=2) # Shape: (batch_size, num_components)

    trace_precision = precision_matrix.view(x.shape[0], num_components, -1)[:, :, ::3].sum(dim=-1)  # Shape: (batch_size, num_components)

    laplacian_component = prob * (squared_linear - trace_precision)  # Shape: (batch_size, num_components)
    laplacian = torch.mean(laplacian_component, dim=1)  # Shape: (batch_size,)
    
    return laplacian


def laplacian_mog_density_div_density(x, means, precisions):
    x = x.unsqueeze(1)  # Shape: (batch_size, 1, 2)
    print("x shape:", x.shape)
    batch_size, num_components = x.size(0), means.size(0)

    # Create a batch of Multivariate Normal distributions for each component
    #mvns = MultivariateNormal(loc=means, precision_matrix=precisions)

    # Calculate the log probabilities for each component
    #log_probs = mvns.log_prob(x)  # Shape: (batch_size, num_components)
    log_probs =safe_log_probs(x,means,precisions, 1)

    # Use torch.logsumexp to compute the log of the sum of exponentiated log probabilities
    log_sum_exp = torch.logsumexp(log_probs, dim=1, keepdim=True)  # Shape: (batch_size, 1)

    # Calculate the softmax probabilities along the components dimension
    softmax_probs = torch.softmax(log_probs, dim=1)  # Shape: (batch_size, num_components)

    x_mean = x - means  # Shape: (batch_size, num_components, 2)

    # Calculate the covariance matrix term
    cov_term = safe_laplacian_cov_term(x,means,precisions,batch_size=4)  # Shape: (batch_size, num_components, 2)

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


def safe_log_probs(x, means, precisions, mbatch_size=16):
    """
    Compute log_probs of x under a mixture of Gaussians with given means and precisions,
    without causing CUDA OOM errors.
    """
    results = []
    num_components = means.size(0)

    for i in range(0, num_components, mbatch_size):
        end = min(i + mbatch_size, num_components)

        mean_batch = means[i:end]                # Shape: (batch, D)
        precision_batch = precisions[i:end]      # Shape: (batch, D, D)

        mvn = MultivariateNormal(loc=mean_batch, precision_matrix=precision_batch)
        # x: (N, 1, D), mvn: (batch, D) → broadcasting: (N, batch)
        # We reshape x to match the batch broadcasting
        log_prob = mvn.log_prob(x.squeeze(1))    # Result: (N,)
        results.append(log_prob)
        
        del mean_batch, precision_batch
        torch.cuda.empty_cache()

    return torch.stack(results, dim=1)  # Final shape: (N, num_components)

def safe_multivariate_normal(means, precisions, batch_size=128):
    """
    Returns a list of MultivariateNormal objects, created in batches.
    Inputs:
        means      : (M, D)
        precisions : (M, D, D)
    Returns:
        mvns_list : List of M MultivariateNormal objects
    """
    mvns_list = []
    M = means.shape[0]
    for i in range(0, M, batch_size):
        end = min(i + batch_size, M)
        means_batch = means[i:end]
        precisions_batch = precisions[i:end]

        # Create list of individual MVNs
        for mean, prec in zip(means_batch, precisions_batch):
            mvn = MultivariateNormal(loc=mean.cpu(), precision_matrix=prec.cpu())
            mvns_list.append(mvn)

        torch.cuda.empty_cache()  # Optional but helps reduce memory spikes    
    return mvns_list


def safe_laplacian_cov_term(x, means, precisions, batch_size=128):
    """
    Computes cov_term = precision @ (x - mean).T in chunks.
    Returns: Tensor of shape (N, M, D)
    """
    x=x.squeeze() # Remove extra singleton dims, e.g., from (N,1,D) → (N,D)
    N, D = x.shape
    M = means.shape[0]
    results = []

    for i in range(0, M, batch_size):
        end = min(i + batch_size, M)
        means_chunk = means[i:end]             # (B, D)
        precisions_chunk = precisions[i:end]   # (B, D, D)

        # Broadcast (N, B, D): (N, 1, D) - (1, B, D)
        x_mean = x.unsqueeze(1) - means_chunk.unsqueeze(0)

        # (N, B, D) → (N, B, D, 1)
        x_mean = x_mean.unsqueeze(-1)

        # (B, D, D) → (1, B, D, D)
        precisions_exp = precisions_chunk.unsqueeze(0)

        # Matrix multiply: (N, B, D, D) x (N, B, D, 1)
        cov_term_chunk = torch.matmul(precisions_exp, x_mean).squeeze(-1)  # (N, B, D)

        results.append(cov_term_chunk)
        torch.cuda.empty_cache()

    return torch.cat(results, dim=1)  # Final shape: (N, M, D)


#----------------------------------------------#
def score_implicit_matching(factornet,samples,centers):
    # evaluate factor net
    factor_eval = factornet(centers) 
    # create precision matrix from the cholesky factor
    dim = centers.shape[-1]
    precisions = vectors_to_precision(factor_eval,dim)
     
    laplacian_over_density = laplacian_mog_density_div_density(samples,centers,precisions)
    torch.cuda.empty_cache()
    gradient_eval_log = grad_log_mog_density(samples,centers,precisions)
    # square gradient
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

def plot_images(means, precisions, epoch = 0, plot_number = 1000, save_path=None):
    # plots plot_number samples from the trained model for image data

    # sample from the multivariate normal distribution
    multivariate_normal = torch.distributions.MultivariateNormal(means, precision_matrix=precisions)
    samples = multivariate_normal.sample((plot_number,))
    # transform images back to original data 
    samples = samples.view(-1, 3, 32, 32)
    samples = samples * 0.5 + 0.5
    fig, axs = plt.subplots(1, plot_number, figsize=(15, 2))
    for i in range(plot_number):
        img = samples[i].permute(1, 2, 0).numpy()  # change from [C, H, W] to [H, W, C]
        axs[i].imshow(img)
        axs[i].axis('off')
    if save_path is not None:
        save_path = save_path + 'epoch'+ str(epoch) +'_scatter_dim_' + '.png'
        plt.savefig(save_path)
    
    plt.close(fig)

    return None
