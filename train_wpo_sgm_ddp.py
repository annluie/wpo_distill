# Pretrain model (py version of Example_WPO_SGM.ipynb) - DDP Version

###################
# setup
###################
# ------------------- ENV & PATH SETUP -------------------
import os
import sys
import argparse
# ------------------- TIME & LOGGING -------------------
import time
import gc
import logging
from tqdm import trange
from memory_profiler import profile

# ------------------- MATH -------------------
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix as pdsm
import matplotlib.pyplot as plt

# ------------------- PYTORCH -------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# ------------------- PROJECT MODULES -------------------
from WPO_SGM import toy_data
from WPO_SGM import functions_WPO_SGM as LearnCholesky


def setup_ddp(rank, world_size):
    """Initialize the process group for DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the process group"""
    dist.destroy_process_group()


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','swissroll_6D_xy1', 'cifar10'], type = str,default = 'cifar10')
    parser.add_argument('--depth',help = 'number of hidden layers of score network',type =int, default = 5)
    parser.add_argument('--hiddenunits',help = 'number of nodes per hidden layer', type = int, default = 64)
    parser.add_argument('--niters',type = int, default = 20)
    parser.add_argument('--batch_size', type = int,default = 8)
    parser.add_argument('--lr',type = float, default = 2e-3) 
    parser.add_argument('--save',type = str,default = 'cifar10_experiments/')
    parser.add_argument('--train_kernel_size',type = int, default = 50)
    parser.add_argument('--train_samples_size',type = int, default = 500)
    parser.add_argument('--test_samples_size',type = int, default = 5)
    parser.add_argument('--load_model_path', type = str, default = None)
    parser.add_argument('--load_centers_path', type = str, default = None)
    return parser.parse_args()


###################
# functions
###################
#----------------------- NEURAL NETWORK -------------------
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


def load_model(model, centers, load_model_path, load_centers_path, device):   
    """
    Loads model weights from the specified path.
    """
    if load_model_path is not None and os.path.exists(load_model_path):
        state_dict = torch.load(load_model_path, map_location=device)
        
        # Strip "module." prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            state_dict = new_state_dict
        
        # Strip "_orig_mod." prefix if present
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace("_orig_mod.", "")] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        if dist.get_rank() == 0:
            logging.info(f"Loaded model weights from {load_model_path}")
    
    if load_centers_path is not None and os.path.exists(load_centers_path):
        centers = torch.load(load_centers_path, map_location=device)
        if dist.get_rank() == 0:
            logging.info(f"Loaded centers from {load_centers_path}")
    else:
        if dist.get_rank() == 0:
            print(f"No model loaded. Path does not exist: {load_model_path}")
    
    return model, centers

    
#-----------------------  HELPER FUNCTIONS -------------------
# Define a compiled function that takes factornet, samples, centers as inputs
compiled_score = torch.compile(LearnCholesky.score_implicit_matching)

def evaluate_model(factornet, kernel_centers, num_test_sample, dataset, device): 
    '''
    Evaluate the model by computing the average total loss over 10 batch of testing samples
    '''
    with torch.no_grad():
        total_loss_sum = 0
        for i in range(10):
            p_samples = toy_data.inf_train_gen(dataset,batch_size = num_test_sample)
            testing_samples = torch.as_tensor(p_samples, dtype=torch.float32, device=device)
            total_loss = compiled_score(factornet, testing_samples, kernel_centers)
            total_loss_sum += total_loss.item()
             # Free up memory
            del p_samples, testing_samples, total_loss
            gc.collect() #only if using CPU
            torch.cuda.empty_cache()  # Only if using GPU
        average_total_loss = total_loss_sum / 10
    return average_total_loss

def opt_check(factornet, samples, centers, optimizer):
    '''
    Optimization function that computes the loss and performs backpropagation using mixed precision
    '''
    optimizer.zero_grad(set_to_none=True)
    loss = LearnCholesky.score_implicit_matching(factornet, samples, centers)
    #scaler.scale(loss).backward()
    #scaler.step(optimizer)
    #scaler.update()
    loss.backward()
    optimizer.step()
    return loss

#----------------------- SAVE FUNCTIONS -------------------
def save_training_slice_cov(factornet, means, epoch, lr, batch_size, save, train_samples_size, test_samples_size, train_kernel_size):
    '''
    Save the training slice of the NN in parameter-based subfolders,
    with epoch appended to the filename.
    '''
    if save is not None and dist.get_rank() == 0:  # Only save on rank 0
        # Create subfolders for all parameters except epoch
        subfolder = os.path.join(
            save,
            f"sample_size{train_samples_size}",
            f"test_size{test_samples_size}",
            f"batch_size{batch_size}",
            f"centers{train_kernel_size}",
            f"lr{lr}"
        )
        os.makedirs(subfolder, exist_ok=True)

        # Create filename with epoch appended
        filename = os.path.join(subfolder, f"epoch{epoch:04d}_factornet.pth")

        # Save model weights - handle DDP wrapper
        if isinstance(factornet, DDP):
            state_dict = factornet.module.state_dict()
        else:
            state_dict = factornet.state_dict()
        torch.save(state_dict, filename)
        logging.info(f"Saved model checkpoint to {filename}")


def save_training_slice_log(iter_time, loss_time, epoch, max_mem, loss_value, save, epochs):
    '''
    Save the training log for the slice
    '''
    if save is not None and dist.get_rank() == 0:  # Only log on rank 0
        logging.info(f"Training started for epoch {epoch} / {epochs}")
        logging.info(
        f"Step {epoch:04d} | Iter Time: {iter_time:.4f}s | "
        f"Loss Time: {loss_time:.4f}s | Loss: {loss_value:.6f} | "
        f"Max Mem: {max_mem:.2f} MB"
        )   


def main_worker(rank, world_size, args):
    """Main training function for each process"""
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    
    # Extract parameters from args
    train_kernel_size = args.train_kernel_size
    train_samples_size = args.train_samples_size
    test_samples_size = args.test_samples_size
    dataset = args.data 
    epochs = args.niters
    batch_size = args.batch_size
    lr = args.lr
    hidden_units = args.hiddenunits
    depth = args.depth
    save_directory = args.save + 'test'+'/'
    load_model_path = args.load_model_path
    load_centers_path = args.load_centers_path

    #-------------------- Initialize Data -------------------
    # check the dataset
    if dataset not in ['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','swissroll_6D_xy1', 'cifar10']:
        dataset = 'cifar10'
    means  = torch.tensor(toy_data.inf_train_gen(dataset, batch_size = train_kernel_size)).to(dtype = torch.float32)
    data_dim = means.shape[1]
    del means
    torch.cuda.empty_cache()

    #-------------------- Create Save Directory -------------------
    if rank == 0:  # Only create directory on rank 0
        print('save_directory',save_directory)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print('Created directory ' + save_directory)

    # Synchronize all processes
    dist.barrier()

    # Configure the logger (only on rank 0)
    if rank == 0:
        log_filename = os.path.join(save_directory,
                                     f'sample_size{train_samples_size}_test_size{test_samples_size}_batch_size{batch_size}_centers{train_kernel_size}_lr{lr}_training.log'
                                     )
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info(f"---------------------------------------------------------------------------------------------")

    #######################
    # Construct the model
    #######################
    #------------------------ Initialize the model -------------------
    torch.set_float32_matmul_precision('high') # set precision for efficient matrix multiplication
    
    factornet = construct_factor_model(data_dim, depth, hidden_units).to(device).to(dtype = torch.float32)
    centers = torch.tensor(
        toy_data.inf_train_gen(dataset, batch_size=train_kernel_size),
        dtype=torch.float32,
        device=device
    )
    
    # Load model and centers if specified (before wrapping in DDP)
    if load_model_path or load_centers_path:
        if rank == 0:
            print("loading model")
        save_directory = os.path.dirname(load_model_path) if load_model_path else save_directory
        new_dir = os.path.join(save_directory, "loaded")
        if rank == 0 and not os.path.exists(new_dir):
            os.makedirs(new_dir)
        dist.barrier()  # Wait for directory creation
        save_directory = new_dir
        factornet, centers = load_model(factornet, centers, load_model_path, load_centers_path, device)
    
    # Wrap model in DDP
    factornet = DDP(factornet, device_ids=[rank])

    #------------------------ Initialize the optimizer -------------------
    optimizer = optim.Adam(factornet.parameters(), lr=lr)

    # Generate training samples (each process gets the same data for simplicity)
    p_samples = toy_data.inf_train_gen(dataset, batch_size=train_samples_size)
    training_samples = torch.tensor(p_samples, dtype=torch.float32, device=device)

    # Save centers (only on rank 0)
    if rank == 0:
        subfolder = os.path.join(
                    save_directory,
                    f"sample_size{train_samples_size}",
                    f"test_size{test_samples_size}",
                    f"batch_size{batch_size}",
                    f"centers{train_kernel_size}",
                    f"lr{lr}"
                )
        os.makedirs(subfolder, exist_ok=True)
        filename_final = os.path.join(subfolder, 'centers.pt')
        torch.save(centers, filename_final) #save the centers (we fix them in the beginning)
    
    del p_samples

    ###########################
    # Training loop
    ###########################
    gc.collect()
    torch.cuda.empty_cache()
    compiled_opt_check = torch.compile(opt_check) # Compile the optimization function

    # Progress bar only on rank 0
    if rank == 0:
        progress_bar = trange(epochs, desc="Training")
    else:
        progress_bar = range(epochs)

    for step in progress_bar:
        torch.cuda.reset_peak_memory_stats() #reset peak memory stats for the current device
        randind = torch.randint(0, train_samples_size, [batch_size,])
        samples = training_samples[randind, :]
        iter_start = time.time()
        loss = compiled_opt_check(factornet, samples, centers, optimizer)
        loss_value = loss.item()
        iter_end = time.time()
        iter_time = iter_end - iter_start
        max_mem = torch.cuda.max_memory_allocated() / 2**30  # in GiB
        
        if rank == 0:
            if step % 100 == 0:
                print(f"Step {step} started")
                print(f'Step: {step}, Loss value: {loss_value:.3e}')
                print(f"Peak memory usage: {max_mem} GiB")
        
        if step % 5000 == 0 and rank == 0:  # Only evaluate and save on rank 0
            loss_start = time.time()
            loss0 = evaluate_model(factornet, centers, test_samples_size, dataset, device)
            loss_end = time.time()
            loss_time = loss_end - loss_start
            save_training_slice_cov(factornet, centers, step, lr, batch_size, save_directory, 
                                   train_samples_size, test_samples_size, train_kernel_size)
            save_training_slice_log(iter_time, loss_time, step, max_mem, loss0, save_directory, epochs)

        if step < epochs - 1:
            del samples
            gc.collect()
            torch.cuda.empty_cache()
        
        # Synchronize all processes periodically
        if step % 1000 == 0:
            dist.barrier()

    ###############################
    # Evaluate the final model
    ################################
    #---------------------------- Final evaluation -------------------
    gc.collect()
    torch.cuda.empty_cache()

    if rank == 0:  # Only evaluate and save on rank 0
        loss0 = evaluate_model(factornet, centers, test_samples_size, dataset, device)    
        save_training_slice_cov(factornet, centers, step, lr, batch_size, save_directory,
                               train_samples_size, test_samples_size, train_kernel_size)
        formatted_loss = f'{loss0:.3e}'  # Format the average with up to 1e-3 precision
        logging.info(f'After train, Average total_loss: {formatted_loss}')

        #---------------------------- Sample and save -------------------
        with torch.no_grad():
            LearnCholesky.plot_images_with_model(factornet, centers, plot_number=10, save_path=filename_final)
            logging.info(f'Sampled images saved to {filename_final}_sampled_images.png')

    # Clean up DDP
    cleanup_ddp()


def main():
    """Main function to launch distributed training"""
    args = get_args()
    
    # Check if CUDA is available and get number of GPUs
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires CUDA for DDP.")
        return
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for DDP training")
    
    if world_size < 2:
        print("Warning: DDP works best with multiple GPUs. Consider using DataParallel for single GPU.")
    
    # Launch distributed training
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()