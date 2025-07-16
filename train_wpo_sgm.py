## Pretrain model (py version of Example_WPO_SGM.ipynb)

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
import math

# ------------------- PYTORCH -------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
#import torch._dynamo
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
#torch._dynamo.config.suppress_errors = True
# ------------------- PROJECT MODULES -------------------
from plots import *
from WPO_SGM import functions_WPO_SGM as LearnCholesky
from WPO_SGM import toy_data
#from WPO_SGM import function_cpu as LearnCholesky

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
    
    # Final layer - this is crucial for stability
    final_layer = nn.Linear(hidden_units, int(dim*(dim+1)/2), bias=True)
    
    # Initialize final layer to produce identity-like matrices
    with torch.no_grad():
        final_layer.weight.data.fill_(0.0)
        final_layer.bias.data.fill_(0.0)
        
        diagonal_indices = []
        k = 0
        for i in range(dim):
            for j in range(i + 1):  # i >= j for lower triangle
                if i == j:
                    diagonal_indices.append(k)
                k += 1
        final_layer.bias.data[diagonal_indices] = 0.1
    chain.append(final_layer)
    
    #chain.append(nn.Linear(int(hidden_units),int(dim*(dim+1)/2),bias = True)) 
    return nn.Sequential(*chain)

def load_model(model, centers, load_model_path, load_centers_path):   
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
        logging.info(f"Loaded model weights from {load_model_path}")
    
    if load_centers_path is not None and os.path.exists(load_centers_path):
        centers = torch.load(load_centers_path, map_location=device)
        logging.info(f"Loaded centers from {load_centers_path}")
    else:
        print(f"No model loaded. Path does not exist: {load_model_path}")
    
    return model, centers

def load_latest_epoch(save_dir):
    epoch_file = os.path.join(save_dir, "latest_epoch.txt")
    if os.path.exists(epoch_file):
        with open(epoch_file, "r") as f:
            epoch = int(f.read().strip())
        print(f"Resuming from epoch {epoch}")
        return epoch
    return 0

def setup_optimizer_and_scheduler(model, args, total_steps=None):
    """
    Setup optimizer and learning rate scheduler with improved configuration.
    
    Args:
        model: The model to optimize
        args: Arguments containing lr and other hyperparameters
        total_steps: Total training steps (required for OneCycleLR)
    """
    
    # Improved optimizer configuration
    optimizer = optim.AdamW(  # AdamW often works better than Adam
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-4,  # Slightly higher default
        betas=(0.9, 0.999),  # Explicit beta values
        eps=1e-8,
        amsgrad=False  # Can set to True for more stable training
    )

    # Choose scheduler based on training strategy
    scheduler_type = getattr(args, 'scheduler_type', 'reduce_on_plateau')
    
    if scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,           # More standard reduction factor
            patience=5,           # Increased patience for stability
            threshold=1e-4,       # More reasonable threshold
            threshold_mode='rel',
            cooldown=2,           # Increased cooldown
            min_lr=args.lr * 1e-3,  # Min LR as fraction of initial LR
            verbose=True
        )
        
    elif scheduler_type == 'cosine_annealing':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=50,               # Initial restart period
            T_mult=2,             # Multiplication factor for restart period
            eta_min=args.lr * 1e-4,  # Min LR as fraction of initial LR
            last_epoch=-1
        )
        
    elif scheduler_type == 'one_cycle' and total_steps:
        # Often the best choice for modern training
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.3,        # Spend 30% of training ramping up
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,      # initial_lr = max_lr / div_factor
            final_div_factor=1e4  # min_lr = initial_lr / final_div_factor
        )
        
    else:
        # Fallback to simple step scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    
    return optimizer, scheduler
#-----------------------  HELPER FUNCTIONS -------------------
# Define a compiled function that takes factornet, samples, centers as inputs
#compiled_score = torch.compile(LearnCholesky.score_implicit_matching_optimized)
#compiled_score = torch.compile(LearnCholesky.score_implicit_matching)

def evaluate_model(factornet, kernel_centers, num_test_sample): 
    '''
    Evaluate the model by computing the average total loss over 10 batch of testing samples
    '''
    with torch.no_grad():
        total_loss_sum = 0
        device = kernel_centers.device
        for i in range(10):
            p_samples = toy_data.inf_train_gen(dataset,batch_size = num_test_sample)
            testing_samples = torch.as_tensor(p_samples, dtype=torch.float32, device=device)
            #total_loss = compiled_score(factornet, testing_samples, kernel_centers)
            #total_loss = LearnCholesky.score_implicit_matching(factornet, testing_samples, kernel_centers, stab)
            total_loss = LearnCholesky.score_implicit_matching_stable(factornet, testing_samples, kernel_centers, stab)
            #total_loss = LearnCholesky.score_implicit_matching_optimized(factornet, testing_samples, kernel_centers, stab)
            total_loss_sum += total_loss.item()
             # Free up memory
            del p_samples, testing_samples, total_loss
            gc.collect() #only if using CPU
            torch.cuda.empty_cache()  # Only if using GPU
        average_total_loss = total_loss_sum / 10
    return average_total_loss

def opt_check(factornet, samples, centers, optimizer, scheduler=None, scheduler_type='one_cycle', stab=1e-6):
    optimizer.zero_grad(set_to_none=True)
    loss = LearnCholesky.score_implicit_matching_stable(factornet, samples, centers, stab)
    
    # Only print debug info occasionally or when there's an issue
    if torch.isnan(loss) or torch.isinf(loss) or not loss.requires_grad:
        print(f"⚠️ Loss issue detected: {loss}")
        print(f"Loss requires_grad: {loss.requires_grad}")
        print(f"Loss grad_fn: {loss.grad_fn}")
    
    if loss.requires_grad and loss.grad_fn is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(factornet.parameters(), max_norm=100.0)
        optimizer.step()
        # Update scheduler if provided
        if scheduler is not None:
            if scheduler_type == 'reduce_on_plateau':
                # Don't step here - will be called with validation loss
                pass
            elif scheduler_type in ['cosine_annealing', 'one_cycle']:
                scheduler.step()
            else:
                scheduler.step()
    else:
        print("❌ Gradient flow broken!")
        
    return loss

def check_model_gradients(model):
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            print(f"Non-trainable parameter: {name}, shape: {param.shape}")
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return trainable_params > 0

def print_memory_usage(step):
    """Print detailed memory usage for all GPUs"""
    num_gpus = torch.cuda.device_count()
    print(f"Memory usage at step {step}:")
    for device_id in range(num_gpus):
        allocated = torch.cuda.memory_allocated(device_id) / 2**30  # GB
        reserved = torch.cuda.memory_reserved(device_id) / 2**30   # GB
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 2**30  # GB
        print(f"GPU {device_id} Step {step:04d} | "
          f"Allocated: {allocated:.2f}GB | "
          f"Reserved: {reserved:.2f}GB | "
          f"Max: {max_allocated:.2f}GB")

#----------------------- SAVE FUNCTIONS -------------------
def create_save_dir(save):
    '''
    Create a subfolder to save all the outputs
    '''
    if save is not None:
        subfolder = os.path.join(
            save,
            f"sample_size{train_samples_size}",
            f"centers{train_kernel_size}",
            f"batch_size{batch_size}_epochs{epochs}",
            #f"test_size{test_samples_size}",
            f"lr{lr}_hu{hidden_units}_stab{stab}_stabveropt"
            #f"test_size{test_samples_size}_lr{lr}_hu{hidden_units}_stab{stab}_comp"
            #f"lr{lr}_hu{hidden_units}_stab{stab}"
        )
        os.makedirs(subfolder, exist_ok=True)
    else:
        subfolder = os.path.join(
            f"sample_size{train_samples_size}",
            f"centers{train_kernel_size}",
            f"batch_size{batch_size}_epochs{epochs}",
            #f"test_size{test_samples_size}",
            f"lr{lr}_hu{hidden_units}_stab{stab}_stabveropt"
            #f"test_size{test_samples_size}_lr{lr}_hu{hidden_units}_stab{stab}_comp"
            #f"lr{lr}_hu{hidden_units}_stab{stab}"
        )
        os.makedirs(subfolder, exist_ok=True)
    return subfolder

def save_training_slice_cov(factornet, means, epoch, save):
    if save is not None:
        filename = os.path.join(save, f"epoch{epoch:04d}_factornet.pth")
        state_dict = factornet.module.state_dict() if isinstance(factornet, nn.DataParallel) else factornet.state_dict()
        torch.save(state_dict, filename)
        logging.info(f"Saved model checkpoint to {filename}")

        # Latest checkpoint (overwrite each time)
        latest_model_ckpt = os.path.join(save, "latest_factornet.pth")
        torch.save(state_dict, latest_model_ckpt)

        # Save epoch number for resume
        with open(os.path.join(save, "latest_epoch.txt"), "w") as f:
            f.write(str(epoch))
        logging.info(f"Saved latest epoch {epoch} for resuming.")

def save_training_slice_log(iter_time, loss_time, epoch, max_mem, loss_value, save):
    '''
    Save the training log for the slice
    '''
    if save is not None:
        logging.info(f"Training started for epoch {epoch} / {epochs}")
        logging.info(
        f"Step {epoch:04d} | Iter Time: {iter_time:.4f}s | "
        f"Loss Time: {loss_time:.4f}s | Loss: {loss_value:.6f} | "
        f"Max Mem: {max_mem:.2f} GB | "
        )   
###################
# setup
###################
# ------------------- CHECK GPUS -------------------
# check how many GPUs are available
if torch.cuda.is_available():
    devices = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')
    print(f"Using {len(devices)} GPUs with DataParallel: {devices}")
else:
    devices = []
    device = torch.device('cpu')
#device = torch.device('cpu')

# ------------------- SET PARAMETERS -------------------
torch.set_float32_matmul_precision('high') # set precision for efficient matrix multiplication
# Setup optimal device settings once at startup
LearnCholesky.setup_optimal_device_settings()

# setup argument parser
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
parser.add_argument('--stability', type=float, default = 0.01)
parser.add_argument('--weight_decay', type=float, default = 1e-4)
parser.add_argument('--scheduler_type', type=str, default='one_cycle',
                    choices=['reduce_on_plateau', 'cosine_annealing', 'one_cycle', 'step'],
                    help='Type of LR scheduler to use')
args = parser.parse_args()

# set parameters from args
train_kernel_size = args.train_kernel_size
train_samples_size = args.train_samples_size
test_samples_size = args.test_samples_size
dataset = args.data 
epochs = args.niters
batch_size = args.batch_size
lr = args.lr
hidden_units = args.hiddenunits
depth = args.depth
save_directory = args.save
load_model_path = args.load_model_path
load_centers_path = args.load_centers_path
stab = args.stability
weight_decay = args.weight_decay
scheduler_type = args.scheduler_type

#-------------------- Initialize Data -------------------
# check the dataset
if dataset not in ['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','swissroll_6D_xy1', 'cifar10']:
    dataset = 'cifar10'
means  = toy_data.inf_train_gen(dataset, batch_size = train_kernel_size).clone().detach().to(dtype=torch.float32, device=device) # type: ignore
data_dim = means.shape[1]
del means
torch.cuda.empty_cache()

#-------------------- Create Save Directory -------------------
save_directory = create_save_dir(save_directory)
print('save_directory',save_directory)
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print('Created directory ' + save_directory)

# Configure the logger
log_filename = os.path.join(save_directory,'training.log'
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
factornet = construct_factor_model(data_dim, depth, hidden_units).to(device).to(dtype = torch.float32)
p_samples = toy_data.inf_train_gen(dataset,batch_size = train_samples_size)
training_samples = p_samples.clone().detach().to(dtype=torch.float32, device=device) # type: ignore
del p_samples
#centers = toy_data.inf_train_gen(dataset, batch_size=train_kernel_size).clone().detach().to(dtype=torch.float32, device=device) # type: ignore
#factornet = nn.DataParallel(factornet, device_ids=devices)
#Load model and centers if specified
#factornet = torch.compile(factornet, mode="reduce-overhead") # must be compiled before DataParallel

latest_model_ckpt = os.path.join(save_directory, "latest_factornet.pth")
latest_centers_ckpt = os.path.join(save_directory, "latest_centers.pth")

if load_model_path is None and os.path.exists(latest_model_ckpt):
    load_model_path = latest_model_ckpt

if load_centers_path is None and os.path.exists(latest_centers_ckpt):
    load_centers_path = latest_centers_ckpt

start_epoch = 0
if load_model_path or load_centers_path:
    print(f"Loading checkpoint from {load_model_path}, {load_centers_path}")
    # Load model and centers, centers can be None and replaced if needed below
    factornet, centers = load_model(factornet, None, load_model_path, load_centers_path)
    start_epoch = load_latest_epoch(save_directory)

# If no centers loaded (e.g. fresh start), generate and save them
if start_epoch == 0 or centers is None:
    print("Generating new centers...")
    centers = training_samples[:train_kernel_size]
    
    centers_path = os.path.join(save_directory, 'centers.pt')
    torch.save(centers, centers_path)
    
    centers_img_path = os.path.join(save_directory, 'centers.png')
    LearnCholesky.plot_and_save_centers(centers, centers_img_path)
else:
    print(f"Using loaded centers from checkpoint.")

print(f"Starting training from epoch {start_epoch} / {epochs}")

if devices:
    factornet = nn.DataParallel(factornet, device_ids=devices) # Wrap model in DataParallel, must be done after loading the model

#------------------------ Initialize the optimizer -------------------
lr = args.lr
# Set total steps for OneCycleLR if needed
steps_per_epoch = max(1, train_samples_size // batch_size)
total_steps = epochs * steps_per_epoch
start_step = start_epoch * steps_per_epoch
remaining_steps = total_steps - start_step

if remaining_steps <= 0:
    print("Training already completed based on loaded checkpoint. Exiting.")
    sys.exit(0)

optimizer, scheduler = setup_optimizer_and_scheduler(factornet, args, total_steps)

# Print scheduler info for debugging
print(f"Scheduler type: {scheduler_type}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total steps: {total_steps}")
print(f"Epochs: {epochs}")
if scheduler_type == 'one_cycle':
    print(f"OneCycleLR configured with max_lr={lr}, total_steps={total_steps}")

'''
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    threshold=10.0,           # only triggers if loss doesn't improve by at least 10
    threshold_mode='abs',
    cooldown=2,
    min_lr=1e-7,
    #verbose=True
)
'''
# Call this before training
if not check_model_gradients(factornet):
    print("ERROR: No trainable parameters found!")
###########################
# Training loop
###########################
gc.collect()
torch.cuda.empty_cache()
#scaler = torch.amp.GradScaler(enabled=False)  #mixed precision gradient scaler
compiled_opt_check = opt_check # compile the optimization function

for step in trange(start_step,total_steps, desc="Training"):
    torch.cuda.reset_peak_memory_stats() #reset peak memory stats for the current device
    randind = torch.randint(0, train_samples_size, [batch_size,])
    samples = training_samples[randind, :]
    iter_start = time.time()
    # with torch.autograd.detect_anomaly():
    loss = compiled_opt_check(factornet, samples, centers, optimizer, scheduler=scheduler, scheduler_type=scheduler_type, stab=stab)
    loss_value = loss.item()
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Invalid loss (NaN or Inf) at step {step}. Exiting.")
        sys.exit(1)
    iter_end = time.time()
    iter_time = iter_end - iter_start
    max_mem = torch.cuda.max_memory_allocated() / 2**30  # in GiB
    #print(f"Peak memory usage: {max_mem} GiB")
    #print_memory_usage(step)
    '''
    for device_id in [0, 1]:  # assuming 2 GPUs: cuda:0 and cuda:1
        print(f"Memory summary for cuda:{device_id}")
        print(torch.cuda.memory_summary(device=f'cuda:{device_id}', abbreviated=False))
    '''

    if step % 100 == 0:
        print(f"Step {step} started")
        print(f'Step: {step}, Loss value: {loss_value:.3e}')
        # FIXED: Add current learning rate logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current LR: {current_lr:.2e}')
        with open(os.path.join(save_directory, "loss_log.csv"), "a") as f:
            f.write(f"{step},{loss_value},{current_lr}\n")
    if step % 200 == 0:
        loss_start = time.time()
        loss0 = evaluate_model(factornet, centers, test_samples_size)
        loss_end = time.time()
        loss_time = loss_end - loss_start
        save_training_slice_log(iter_time, loss_time, step, max_mem, loss0, save_directory)

        # FIXED: Only call scheduler.step() for reduce_on_plateau here
        # OneCycleLR is already stepped in opt_check after each optimizer.step()
        if not math.isnan(loss0) and not math.isinf(loss0):
            # Get old LR before stepping
            old_lrs = [group['lr'] for group in optimizer.param_groups]

            # Step scheduler appropriately - ONLY for reduce_on_plateau
            if scheduler_type == 'reduce_on_plateau':
                scheduler.step(loss0) # type: ignore
                # Get new LR after stepping
                new_lrs = [group['lr'] for group in optimizer.param_groups]
                # Log LR changes
                for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                    if old_lr != new_lr:
                        logging.info(f"LR changed for param group {i} from {old_lr:.4e} to {new_lr:.4e} at step {step}")
            # Note: OneCycleLR and others are stepped after each optimizer.step() in opt_check
        else:
            print("⚠️ Warning: Skipping LR scheduler step due to invalid val_loss:", loss0)
        # Sample and save generated images at intermediate steps
        with torch.no_grad():
            generated = LearnCholesky.sample_from_model(factornet, training_samples, sample_number=10, eps=stab)
            l2 = torch.mean((generated - training_samples[:10])**2).item()
            print(f"[Step {step}] L2 to training data: {l2:.2f}")
            logging.info(f"L2 {l2:.2f} | ")
    if step % 500 ==0:
        with torch.no_grad():
            filename_step_sample = os.path.join(save_directory, f"step{step:05d}")
            LearnCholesky.plot_images_with_model(factornet, centers, plot_number=10, eps=stab, save_path=filename_step_sample)
            logging.info(f"Saved samples at step {step} to {filename_step_sample}")
    if step % 1000 == 0:
        save_training_slice_cov(factornet, centers, step, save_directory)
    if step < total_steps - 1:  # FIXED: Use total_steps instead of epochs
        del samples
        gc.collect()
        torch.cuda.empty_cache()

###############################
# Evaluate the final model
################################
#---------------------------- Final evaluation -------------------
gc.collect()
torch.cuda.empty_cache()

loss0 = evaluate_model(factornet, centers, test_samples_size)    
save_training_slice_cov(factornet, centers, step, save_directory)
formatted_loss = f'{loss0:.3e}'  # Format the average with up to 1e-3 precision
logging.info(f'After train, Average total_loss: {formatted_loss}')

#---------------------------- Sample and save -------------------
with torch.no_grad():
    LearnCholesky.plot_images_with_model(factornet, centers, plot_number=10, eps=stab, save_path=save_directory)
    logging.info(f'Sampled images saved to {save_directory}_sampled_images.png')
