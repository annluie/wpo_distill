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
from WPO_SGM.functions_WPO_SGM_stable import vectors_to_precision_optimized_new
from WPO_SGM.functions_WPO_SGM import vectors_to_precision
from torchvision.datasets import CIFAR10
from torchvision import transforms

def load_cifar10_training_tensor(device='cpu'):
    transform = transforms.ToTensor()
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    images = torch.stack([img for img, _ in dataset])  # [50000, 3, 32, 32]
    return images.to(device)

@torch.no_grad()
def compute_min_l2_per_sample(generated_batch, training_batch):
    """
    Compute the minimum L2 distance from each generated image to any image in the training set.
    Args:
        generated_batch: [B, 3, 32, 32]
        training_batch: [N, 3, 32, 32]
    Returns:
        Tensor of shape [B] with min L2 distances
    """
    B = generated_batch.shape[0]
    gen_flat = generated_batch.view(B, -1)  # [B, 3072]
    train_flat = training_batch.view(training_batch.shape[0], -1)  # [N, 3072]
    dists = torch.cdist(gen_flat, train_flat, p=2.0)  # [B, N]
    min_l2 = torch.min(dists, dim=1)[0]  # [B]
    return min_l2

import matplotlib.pyplot as plt

@torch.no_grad()
def visualize_nn_matches(generated, training, num_show=10, save_path=None):
    """
    Plots each generated image next to its nearest training image.

    Args:
        generated: [B, 3, 32, 32] tensor (already denormalized and clamped to [0, 1])
        training: [N, 3, 32, 32] tensor (CIFAR-10 training data)
        num_show: how many to visualize
        save_path: optional file path to save the figure
    """
    B = generated.shape[0]
    assert num_show <= B, "num_show exceeds number of generated samples"

    gen_flat = generated.view(B, -1)  # [B, D]
    train_flat = training.view(training.shape[0], -1)  # [N, D]
    dists = torch.cdist(gen_flat, train_flat, p=2.0)  # [B, N]
    nn_indices = dists.argmin(dim=1)  # [B]

    fig, axs = plt.subplots(2, num_show, figsize=(num_show * 2, 4))
    for i in range(num_show):
        # Generated image
        axs[0, i].imshow(generated[i].permute(1, 2, 0).cpu().numpy())
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_ylabel("Generated", fontsize=12)

        # Nearest neighbor in CIFAR-10
        nn_img = training[nn_indices[i]]
        axs[1, i].imshow(nn_img.permute(1, 2, 0).cpu().numpy())
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_ylabel("Nearest Train", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"🔍 Saved NN visualization to {save_path}")
    plt.show()

def plot_images_with_model_parallel(model_parallel, training_samples, plot_number=10, eps=1e-6, save_path=None):
    """
    Plot images using the model parallel setup with all centers
    """
    device = f'cuda:{model_parallel.device_ids[0]}'
    
    # Collect all centers from all devices to the first device
    all_centers = []
    for centers_on_device in model_parallel.centers_per_device:
        if centers_on_device.numel() > 0:  # Skip empty center tensors
            all_centers.append(centers_on_device.to(device))
    
    if not all_centers:
        raise ValueError("No centers found across devices")
    
    # Concatenate all centers
    combined_centers = torch.cat(all_centers, dim=0)
    
    # Use the first device's model for plotting (they should all be identical)
    first_model = model_parallel.factornets[0]
    
    # Plot using all centers
    plot_images_with_model_and_nn(first_model, combined_centers, training_samples, plot_number=plot_number, eps=eps, save_path=save_path)

def plot_images_with_model_and_nn(factornet, means, training_data_tensor, plot_number=10, eps=1e-4, save_path=None):
    dim = means.shape[-1]

    # Get two sets of samples: standard and stable
    samples_standard = sample_from_model(factornet, means, plot_number, eps, use_stable=False)
    samples_stable = sample_from_model(factornet, means, plot_number, eps, use_stable=True)


    # Compute NN matches for stable samples
    with torch.no_grad():
        gen_flat = samples_stable.view(plot_number, -1)              # [B, 3072]
        train_flat = training_data_tensor.view(training_data_tensor.size(0), -1)  # [N, 3072]
        dists = torch.cdist(gen_flat, train_flat, p=2.0)             # [B, N]
        nn_indices = torch.argmin(dists, dim=1)                      # [B]
        nn_matches = training_data_tensor[nn_indices]                # [B, 3, 32, 32]
    '''
    print(f"NN indices: {nn_indices}")
    print(f"Unique NN indices: {nn_indices.unique()}")
    print(f"Number of unique matches: {len(nn_indices.unique())}")
    '''
    # Format and denormalize sampled images
    def format_samples(samples):
        samples = samples.view(-1, 3, 32, 32)
        samples = denormalize_cifar10(samples)
        return torch.clamp(samples, 0, 1)

    samples_standard = format_samples(samples_standard)
    samples_stable = format_samples(samples_stable)
    nn_matches = format_samples(nn_matches)

    # Plot 3 rows: standard, stable, NN matches
    all_samples = [samples_standard, samples_stable, nn_matches]
    labels = ["Standard", "Stable", "NN in Train"]

    fig, axs = plt.subplots(3, plot_number, figsize=(plot_number * 1.5, 6))
    axs = np.atleast_2d(axs)  # Ensures axs[row, i] indexing works

    for row, (samples, label) in enumerate(zip(all_samples, labels)):
        for i in range(plot_number):
            img = samples[i].permute(1, 2, 0).cpu().numpy()
            axs[row, i].imshow(img)
            axs[row, i].set_xticks([])
            axs[row, i].set_yticks([])
            for spine in axs[row, i].spines.values():
                spine.set_visible(False)
            if i == 0:
                axs[row, i].set_ylabel(label, fontsize=12)
    plt.tight_layout(pad=0.5)  # Minimal padding
    if save_path is not None:
        save_file = save_path + '_sampled_images_with_nn.png'
        plt.savefig(save_file, bbox_inches='tight')
        print(f"✅ Saved to {save_file}")
    plt.close(fig)

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
                precision = vectors_to_precision_optimized_new(vectors, dim, eps)
            else:
                precision = vectors_to_precision(vectors, dim, eps)

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

#----------------------- Model Parallel -------------------------------
def sample_from_model_parallel(model_parallel, sample_number=10, eps=1e-6):
    """
    Sample from the model using all centers across all devices
    """
    device = f'cuda:{model_parallel.device_ids[0]}'
    
    # Collect all centers from all devices to the first device
    all_centers = []
    for centers_on_device in model_parallel.centers_per_device:
        if centers_on_device.numel() > 0:  # Skip empty center tensors
            all_centers.append(centers_on_device.to(device))
    
    if not all_centers:
        raise ValueError("No centers found across devices")
    
    # Concatenate all centers
    combined_centers = torch.cat(all_centers, dim=0)
    
    # Use the first device's model for sampling (they should all be identical)
    first_model = model_parallel.factornets[0]
    
    # Sample using all centers
    return sample_from_model(first_model, combined_centers, sample_number=sample_number, eps=eps)

def plot_images_with_model_parallel(model_parallel, training_samples, plot_number=10, eps=1e-6, save_path=None):
    """
    Plot images using the model parallel setup with all centers
    """
    device = f'cuda:{model_parallel.device_ids[0]}'
    
    # Collect all centers from all devices to the first device
    all_centers = []
    for centers_on_device in model_parallel.centers_per_device:
        if centers_on_device.numel() > 0:  # Skip empty center tensors
            all_centers.append(centers_on_device.to(device))
    
    if not all_centers:
        raise ValueError("No centers found across devices")
    
    # Concatenate all centers
    combined_centers = torch.cat(all_centers, dim=0)
    
    # Use the first device's model for plotting (they should all be identical)
    first_model = model_parallel.factornets[0]
    
    # Plot using all centers
    plot_images_with_model_and_nn(first_model, combined_centers, training_samples, plot_number=plot_number, eps=eps, save_path=save_path)

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
