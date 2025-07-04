#----------------------------------------------#
#### plotting functions ####
#----------------------------------------------#
import matplotlib.pyplot as plt
import os
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
from WPO_SGM.functions_WPO_SGM_stable import vectors_to_precision_stable
from WPO_SGM.functions_WPO_SGM import vectors_to_precision
def sample_from_model(factornet, means, sample_number, eps, use_stable=False):
    num_components, dim = means.shape
    device = means.device
    dtype = means.dtype

    comp_num = torch.randint(0, num_components, (sample_number,), device=device)
    samples = torch.empty(sample_number, dim, device=device, dtype=dtype)

    unique_indices = comp_num.unique()

    for i in unique_indices:
        idx = (comp_num == i).nonzero(as_tuple=True)[0]
        n_i = idx.shape[0]
        centers_i = means[i].unsqueeze(0).expand(n_i, -1)  # [n_i, dim]

        vectors = factornet(centers_i)  # [n_i, d*(d+1)//2]

        # Choose precision method
        try:
            if use_stable:
                precision = vectors_to_precision_stable(vectors, dim, eps)
            else:
                precision = vectors_to_precision(vectors, dim, 0.001)

            # Check positive definiteness
            torch.linalg.cholesky(precision)
        except RuntimeError:
            print("⚠️ Precision matrix not PD, using fallback identity.")
            precision = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0).repeat(n_i, 1, 1)

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


#---------OLD FUNCTIONS ----------------------------------


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

