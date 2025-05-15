#!/usr/bin/env python
# coding: utf-8

# # Score-matching informed KDE

# In[ ]:
import subprocess
import sys

def install_dependencies():
    """Run the install_torch.py script to install and import dependencies."""
    try:
        subprocess.check_call([sys.executable, "installer.py"])
        print("Installation script ran successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running installation script: {e}")
import matplotlib.pyplot as plt
import os

import torch
#torch.cuda.empty_cache()
import torch.optim as optim
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
import lib.toy_data as toy_data
import numpy as np
import argparse
import pandas as pd
from pandas.plotting import scatter_matrix as pdsm
#import functions_WPO_SGM as LearnCholesky
import function_cpu as LearnCholesky
import torch.multiprocessing as mp
if __name__ == "__main__":
    install_dependencies()
    mp.set_start_method('spawn')  # Ensure the correct start method is used for Windows
    #import logging
    # git testing


    # In[2]:


    import os
    #os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("GPU is available")
    else:
        device = torch.device('cpu')
        print("GPU is not available")
    print(device)
    """
    """
    try:
        if torch.cuda.is_available():
            # Check how many GPUs are actually detected
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                device = torch.device('cuda')
                # Test the GPU with a small operation
                test_tensor = torch.zeros(1, device=device)
                del test_tensor
                print(f"GPU is available and working. Found {num_gpus} GPU(s)")
            else:
                device = torch.device('cpu')
                print("CUDA is detected but no GPUs are available. Using CPU")
        else:
            device = torch.device('cpu')
            print("GPU is not available. Using CPU")
            
        print(f"Using device: {device}")
        
    except RuntimeError as e:
        print(f"GPU initialization failed: {e}")
        device = torch.device('cpu')
        print(f"Falling back to CPU: {device}")

    device = torch.device("cuda:1")
    #model.to(device)

    #clear memory
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # ### Parsing for scripts
    """
    # In[3]:
    device=torch.device('cpu')

    parser = argparse.ArgumentParser(' ')
    parser.add_argument('--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','swissroll_6D_xy1', 'cifar10'], type = str,default = '2spirals')
    parser.add_argument('--depth',help = 'number of hidden layers of score network',type =int, default = 5)
    parser.add_argument('--hiddenunits',help = 'number of nodes per hidden layer', type = int, default = 64)
    parser.add_argument('--niters',type = int, default = 5000)
    parser.add_argument('--batch_size', type = int,default = 2)
    parser.add_argument('--lr',type = float, default = 2e-3) 
    parser.add_argument('--save',type = str,default = 'experiments/')
    parser.add_argument('--train_kernel_size',type = int, default = 10)
    parser.add_argument('--train_samples_size',type = int, default = 50)
    parser.add_argument('--test_samples_size',type = int, default = 5)
    args = parser.parse_args('')


    # In[4]:


    train_kernel_size = args.train_kernel_size
    train_samples_size = args.train_samples_size
    test_samples_size = args.test_samples_size
    dataset = args.data 
    save_directory = args.save + 'test'+'/'

    print('save_directory',save_directory)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print('Created directory ' + save_directory)

    #print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # ### Precision matrix model

    # In[5]:


    ## Cholesky factor model
    def construct_factor_model(dim:int, depth:int, hidden_units:int):
        '''
        Initializes neural network that models the Cholesky factor of the precision matrix # For nD examples (in theory)
        '''
        chain = []
        chain.append(nn.Linear(dim,int(hidden_units),bias =True)) 
        chain.append(nn.GELU())

        for _ in range(depth-1):
            chain.append(nn.Linear(int(hidden_units),int(hidden_units),bias = True))
            chain.append(nn.GELU())
        chain.append(nn.Linear(int(hidden_units),int(dim*(dim+1)/2),bias = True)) 

        return nn.Sequential(*chain)   


    # ### Helper functions

    # In[6]:


    def evaluate_model(factornet, kernel_centers, num_test_sample):
        '''
        Evaluate the model by computing the average total loss over 10 batch of testing samples
        '''
        total_loss_sum = 0
        device = kernel_centers.device
        for i in range(10):
            p_samples = toy_data.inf_train_gen(dataset,batch_size = num_test_sample)
            testing_samples = torch.tensor(p_samples).to(dtype = torch.float32).to(device)
            total_loss = LearnCholesky.score_implicit_matching(factornet,testing_samples,kernel_centers)
            total_loss_sum += total_loss.item()
            # free memory
            del p_samples,testing_samples,total_loss
            torch.cuda.empty_cache()
        average_total_loss = total_loss_sum / 10
        return average_total_loss

    def save_training_slice_cov(factornet, means, epoch, lr, batch_size, loss_value, save):
        '''
        Save the training slice of the density plot
        '''
        if means.shape[1] != 2:
            return
        plot_axis = means.max().item() * 1.1
        device = means.device
        # Create x as a NumPy array
        x_np = np.meshgrid(np.linspace(-plot_axis, plot_axis, 200), np.linspace(-plot_axis, plot_axis, 200))
        x_np = np.stack(x_np, axis=-1).reshape(-1, 2)

        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        data_dim = x.shape[1]
        precisions = LearnCholesky.vectors_to_precision(factornet(means),data_dim)
        density = LearnCholesky.mog_density(x, means, precisions)
        density = density.reshape(200, 200).T

        # Create a figure
        fig = plt.figure(figsize=(8, 4))
        plt.clf()
        
        plt.subplot(1, 2, 1) 
        plt.contourf(np.linspace(-plot_axis, plot_axis, 200), np.linspace(-plot_axis, plot_axis, 200), density.detach().cpu().numpy(), cmap='viridis')
        plt.axis('square')
        plt.colorbar()     
        
        plt.subplot(1, 2, 2) 
        plt.contourf(np.linspace(-plot_axis, plot_axis, 200), np.linspace(-plot_axis, plot_axis, 200), density.detach().cpu().numpy(), cmap='viridis')
        # Plot the centers
        num_components = torch.min(torch.tensor([means.shape[0], 400]))
        plot_centers = means[:num_components].detach().cpu().numpy()
        plt.scatter(plot_centers[:,1], plot_centers[:,0], s=0.2, c='r')
        plt.axis('square')
        # plt.colorbar()    
        plt.title(f'Epoch: {epoch}, Loss: {loss_value:.3e}')
                
        plt.tight_layout()  # Improve subplot spacing

        # Save the figure
        lr_str = f'{lr:.2e}'
        if save is not None:
            plt.savefig(f'{save}batch_size_{batch_size}lr_{lr_str}_epoch_{epoch}.png')

        plt.close(fig)


    # ### Initialize score network

    # In[7]:


    # check the dataset
    dataset = args.data
    #dataset = 'swissroll'
    #dataset = 'swissroll_6D_xy1'
    #means  = torch.tensor(toy_data.inf_train_gen(dataset, batch_size = 1000)).to(dtype = torch.float32)
    #data_dim = means.shape[1]
    #print('data_dim',data_dim)

    #blah = pd.DataFrame(means)
    #pdsm(blah)


    # ## Initialize Data using CIFAR-10
    # 

    # In[8]:


    # check the dataset
    dataset = args.data
    dataset = 'cifar10'

    means  = torch.tensor(toy_data.inf_train_gen(dataset, batch_size = train_kernel_size)).to(dtype = torch.float32)
    data_dim = means.shape[1]
    # dataset = 'swissroll_6D_xy1'
    """"" # not used anymore since our data is pictures
    means  = torch.tensor(toy_data.inf_train_gen(dataset, batch_size = 1000)).to(dtype = torch.float32)
    data_dim = means.shape[1]
    print('data_dim',data_dim)

    blah = pd.DataFrame(means)
    pdsm(blah)
    """
    #print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # In[ ]:


    depth = args.depth
    hidden_units = args.hiddenunits
    factornet = construct_factor_model(data_dim, depth, hidden_units).to(device).to(dtype = torch.float32)
    factornet = torch.compile(factornet) # help speed up training

    lr = args.lr
    optimizer = optim.Adam(factornet.parameters(), lr=args.lr)

    p_samples = toy_data.inf_train_gen(dataset,batch_size = train_samples_size)
    training_samples = torch.tensor(p_samples).to(dtype = torch.float32).to(device)
    centers  = torch.tensor(toy_data.inf_train_gen(dataset, batch_size = train_kernel_size)).to(dtype = torch.float32).to(device)

    # torch.save(centers, save_directory + 'centers.pt')

    epochs = args.niters
    batch_size = args.batch_size

    # Training the score network
    loss = evaluate_model(factornet, centers, test_samples_size)
    formatted_loss = f'{loss:.3e}'  # Format the average with up to 1e-3 precision
    print(f'Before train, Average total_loss: {formatted_loss}')

    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # In[ ]:


    for step in range(epochs):
        # samples_toydata
        randind = torch.randint(0,train_samples_size,[batch_size,])
        samples = training_samples[randind,:]

        loss = LearnCholesky.score_implicit_matching(factornet,samples,centers)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if not step % 4000:
            loss_value = loss.item()
            print(f'Step: {step}, Loss value: {loss_value:.3e}')

        if not step % 20000:
            loss0 = evaluate_model(factornet, centers, test_samples_size)
            save_training_slice_cov(factornet, centers, step, lr, batch_size, loss0, save_directory)

    formatted_loss = f'{loss:.3e}'  # Format the average with up to 1e-3 precision
    print(f'After train, Average total_loss: {formatted_loss}')

    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # In[ ]:


    # sample from trained model
    # and plot density
    precisions = LearnCholesky.vectors_to_precision(factornet(centers),data_dim)
    LearnCholesky.plot_images(centers, precisions,plot_number=10,save_path=save_directory + 'samples.png')

    """ I think this will not work for general (centers needs to be same as before to plot properly)
    randind = torch.randint(0,1000,[1000,])
    centers = means[randind,:].to(device)
    precisions = LearnCholesky.vectors_to_precision(factornet(centers),data_dim)

    LearnCholesky.scatter_samples_from_model(centers, precisions, dim1 = 0, dim2 = 1,save_path=save_directory + 'samples.png')
    LearnCholesky.plot_density_2d_marg(centers,factornet,dim1 = 0, dim2 = 1, save_path=save_directory + 'density.png')
    """


    # In[ ]:


    # LearnCholesky.scatter_samples_from_model(centers, precisions, dim1 = 2, dim2 = 3, save_path=save_directory + 'samples.png')
    # LearnCholesky.plot_density_2d_marg(centers,factornet, dim1 = 2, dim2 = 3, save_path=save_directory + 'density.png')
    # LearnCholesky.scatter_samples_from_model(centers, precisions, dim1 = 4, dim2 = 5,  save_path=save_directory + 'samples.png')
    # LearnCholesky.plot_density_2d_marg(centers,factornet, dim1 = 4, dim2 = 5,save_path=save_directory + 'density.png')

