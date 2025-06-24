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
#from WPO_SGM import functions_WPO_SGM as LearnCholesky
from WPO_SGM import function_cpu as LearnCholesky


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
    devices = ['cpu']
    device = torch.device('cpu')
#device = torch.device('cpu')

# ------------------- SET PARAMETERS -------------------
torch.set_float32_matmul_precision('high') # set precision for efficient matrix multiplication

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
print('save_directory',save_directory)
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print('Created directory ' + save_directory)


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

    
#-----------------------  HELPER FUNCTIONS -------------------
# Define a compiled function that takes factornet, samples, centers as inputs
compiled_score = torch.compile(LearnCholesky.score_implicit_matching)

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
            total_loss = LearnCholesky.score_implicit_matching(factornet, testing_samples, kernel_centers)
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
    torch.nn.utils.clip_grad_norm_(factornet.parameters(), max_norm=1.0) #gradient clipping
    optimizer.step()
    return loss

#----------------------- SAVE FUNCTIONS -------------------
import os

def save_training_slice_cov(factornet, means, epoch, lr, batch_size, save):
    '''
    Save the training slice of the NN in parameter-based subfolders,
    with epoch appended to the filename.
    '''
    if save is not None:
        # Create subfolders for all parameters except epoch
        subfolder = os.path.join(
            save,
            f"sample_size{train_samples_size}",
            f"test_size{test_samples_size}",
            f"batch_size{batch_size}",
            f"centers{train_kernel_size}",
            f"lr{lr}_hu{hidden_units}"
        )
        os.makedirs(subfolder, exist_ok=True)

        # Create filename with epoch appended
        filename = os.path.join(subfolder, f"epoch{epoch:04d}_factornet.pth")

        # Save model weights
        state_dict = factornet.module.state_dict() if isinstance(factornet, nn.DataParallel) else factornet.state_dict()
        torch.save(state_dict, filename)
        logging.info(f"Saved model checkpoint to {filename}")

        # Save centers if needed
        # centers_filename = os.path.join(subfolder, f"epoch{epoch:04d}_centers.pth")
        # torch.save(means, centers_filename)
        # logging.info(f"Saved centers to {centers_filename}")


def save_training_slice_log(iter_time, loss_time, epoch, max_mem, loss_value, save):
    '''
    Save the training log for the slice
    '''
    if save is not None:
        logging.info(f"Training started for epoch {epoch} / {epochs}")
        logging.info(
        f"Step {epoch:04d} | Iter Time: {iter_time:.4f}s | "
        f"Loss Time: {loss_time:.4f}s | Loss: {loss_value:.6f} | "
        f"Max Mem: {max_mem:.2f} MB"
        )   

# Configure the logger
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
factornet = construct_factor_model(data_dim, depth, hidden_units).to(device).to(dtype = torch.float32)
centers = torch.tensor(
    toy_data.inf_train_gen(dataset, batch_size=train_kernel_size),
    dtype=torch.float32,
    device=device
)
#factornet = nn.DataParallel(factornet, device_ids=devices)
# Load model and centers if specified
if load_model_path or load_centers_path:
    print("loading model")
    save_directory = os.path.dirname(load_model_path) if load_model_path else save_directory
    new_dir = os.path.join(save_directory, "loaded")
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    save_directory = new_dir  # ✅ Set after creation
    factornet, centers = load_model(factornet, centers, load_model_path, load_centers_path)
factornet = nn.DataParallel(factornet, device_ids=devices) # Wrap model in DataParallel, must be done after loading the model
#factornet = torch.compile(factornet, mode="max-autotune") # must be compiled after DataParallel

#------------------------ Initialize the optimizer -------------------
lr = args.lr
optimizer = optim.Adam(factornet.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( #learning rate scheduler
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    threshold=1e-4,
    cooldown=2,
    min_lr=1e-7,
    verbose=True
)

p_samples = toy_data.inf_train_gen(dataset,batch_size = train_samples_size)
training_samples = torch.tensor(p_samples, dtype=torch.float32, device=device)

subfolder = os.path.join(
            save_directory,
            f"sample_size{train_samples_size}",
            f"test_size{test_samples_size}",
            f"batch_size{batch_size}",
            f"centers{train_kernel_size}",
            f"lr{lr}_hu{hidden_units}"
        )
os.makedirs(subfolder, exist_ok=True)
#centers = centers.to('cpu')
filename_final = os.path.join(subfolder, 'centers.pt')
#generate centers as subset of training samples
centers = training_samples[torch.randperm(training_samples.size(0))[:train_kernel_size]]
torch.save(centers, filename_final) #save the centers (we fix them in the beginning)
filename_final = os.path.join(subfolder, 'centers.png')
LearnCholesky.plot_and_save_centers(centers, filename_final)
del p_samples

###########################
# Training loop
###########################
gc.collect()
torch.cuda.empty_cache()
#scaler = torch.amp.GradScaler(enabled=False)  #mixed precision gradient scaler
compiled_opt_check = opt_check # compile the optimization function

for step in trange(epochs, desc="Training"):
    torch.cuda.reset_peak_memory_stats() #reset peak memory stats for the current device
    randind = torch.randint(0, train_samples_size, [batch_size,])
    samples = training_samples[randind, :]
    iter_start = time.time()
    # with torch.autograd.detect_anomaly():
    loss = compiled_opt_check(factornet, samples, centers, optimizer)
    loss_value = loss.item()
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Invalid loss (NaN or Inf) at step {step}. Exiting.")
        sys.exit(1)
    iter_end = time.time()
    iter_time = iter_end - iter_start
    max_mem = torch.cuda.max_memory_allocated() / 2**30  # in GiB
    print(f"Peak memory usage: {max_mem} GiB")
    if step % 100 == 0:
        print(f"Step {step} started")
        print(f'Step: {step}, Loss value: {loss_value:.3e}')
        with open(os.path.join(subfolder, "loss_log.csv"), "a") as f:
            f.write(f"{step},{loss_value}\n")
    if step % 200 == 0:
        loss_start = time.time()
        loss0 = evaluate_model(factornet, centers, test_samples_size)
        loss_end = time.time()
        loss_time = loss_end - loss_start
        save_training_slice_cov(factornet, centers, step, lr, batch_size, save_directory)
        save_training_slice_log(iter_time, loss_time, step, max_mem, loss0, save_directory)
        if not torch.isnan(loss0) and not torch.isinf(loss0):
                # Get old LR before stepping scheduler
                old_lrs = [group['lr'] for group in optimizer.param_groups]
                scheduler.step(loss0)

                # Get new LR after stepping scheduler
                new_lrs = [group['lr'] for group in optimizer.param_groups]

                # Check if any LR changed
        for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
            if old_lr != new_lr:
                logging.info(f"LR changed for param group {i} from {old_lr:.2e} to {new_lr:.2e} at step {step}")
            else:
                print("⚠️ Warning: Skipping LR scheduler step due to invalid val_loss:", loss0)
        # Sample and save generated images at intermediate steps
        with torch.no_grad():
            filename_step_sample = os.path.join(subfolder, f"step{step:05d}")
            LearnCholesky.plot_images_with_model(factornet, centers, plot_number=10, save_path=filename_step_sample)
            logging.info(f"Saved samples at step {step} to {filename_step_sample}")
            generated = LearnCholesky.sample_from_model(factornet, training_samples, sample_number=10)
            l2 = torch.mean((generated - training_samples[:10])**2).item()
            print(f"[Step {step}] L2 to training data: {l2:.2f}")
    if step < epochs - 1:
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
save_training_slice_cov(factornet, centers, step, lr, batch_size, save_directory)
formatted_loss = f'{loss0:.3e}'  # Format the average with up to 1e-3 precision
logging.info(f'After train, Average total_loss: {formatted_loss}')

#---------------------------- Sample and save -------------------
with torch.no_grad():
    LearnCholesky.plot_images_with_model(factornet, centers, plot_number=10, save_path=subfolder)
    logging.info(f'Sampled images saved to {filename_final}_sampled_images.png')
