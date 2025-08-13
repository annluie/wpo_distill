## Pretrain model (py version of Example_WPO_SGM.ipynb) - Model Parallel Version for Google Cloud

###################
# setup
###################
# ------------------- ENV & PATH SETUP -------------------
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:32'
import sys
import argparse
# ------------------- TIME & LOGGING -------------------
import time
import gc
import logging
from tqdm import trange
from memory_profiler import profile
import tempfile
import shutil
from google.cloud import storage
# ------------------- MATH -------------------
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix as pdsm
import matplotlib.pyplot as plt
import math
# ------------------- PYTORCH -------------------
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint_sequential
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
import torch.utils.checkpoint as cp
import torch.nn as nn
# ------------------- PROJECT MODULES -------------------
import utilities.plots as plotting, utilities.save as saving, utilities.diagnostics as check
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import toy_data
import config.load as load, config.logging as log, config.setup_model as setup

from train_wpo_sgm_model_parallel import ModelParallelWPO

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
    
    return nn.Sequential(*chain)

def evaluate_model(model_parallel, num_test_sample, num_batches=2):
    """
    Evaluate the model by computing the average total loss over test samples.
    Modified for model parallel approach - uses all centers across all devices
    """
    device = f'cuda:{model_parallel.device_ids[0]}'  # Use first device
    total_loss_sum = 0.0

    model_parallel.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate test samples directly on the correct device
            p_samples = toy_data.inf_train_gen(dataset, batch_size=num_test_sample)
            if not torch.is_tensor(p_samples):
                p_samples = torch.tensor(p_samples, device=device, dtype=torch.float32)
            else:
                p_samples = p_samples.to(device=device, dtype=torch.float32, non_blocking=True)

            # Use model parallel forward pass (this uses all centers across all devices)
            loss = model_parallel(p_samples, stab)
            total_loss_sum += loss.item()

    return total_loss_sum / num_batches

#----------------------- GOOGLE CLOUD FUNCTIONS -------------------
def upload_dir_to_gcs(local_dir, gcs_path):
    """Function to upload a directory to Google Cloud Storage"""
    client = storage.Client()
    path = gcs_path[len("gs://"):]  # remove gs://
    bucket_name, prefix = path.split("/", 1)
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_file = os.path.join(root, filename)
            # Compute relative path from local_dir root, then join with gcs prefix
            rel_path = os.path.relpath(local_file, local_dir).replace("\\", "/")
            gcs_file_path = os.path.join(prefix, rel_path).replace("\\", "/")
            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(local_file)
            logging.info(f"Uploaded {local_file} to gs://{bucket_name}/{gcs_file_path}")

def create_save_dir_gcloud(args, local_save_directory):
    '''
    Create a subfolder for Google Cloud compatible saving
    '''
    subfolder = os.path.join(
        local_save_directory,
        f"sample_size{args.train_samples_size}",
        f"centers{args.train_kernel_size}",
        f"batch_size{args.batch_size}_epochs{args.niters}",
        f"lr{args.lr}_hu{args.hiddenunits}_stab{args.stability}_modelparallel_gcloud"
    )
    os.makedirs(subfolder, exist_ok=True)
    return subfolder

def save_training_slice_cov_gcloud(model_parallel, step, save_dir, optimizer=None, scheduler=None):
    """
    Save model, optimizer, and scheduler state for Google Cloud
    Modified for model parallel approach
    """
    if save_dir is not None:
        # Save all device models
        for i, factornet in enumerate(model_parallel.factornets):
            filename = os.path.join(save_dir, f"step{step:05d}_factornet_device{i}.pth")
            torch.save(factornet.state_dict(), filename)
            logging.info(f"Saved model checkpoint for device {i} to {filename}")

        # Save latest model checkpoints
        for i, factornet in enumerate(model_parallel.factornets):
            latest_model_ckpt = os.path.join(save_dir, f"latest_factornet_device{i}.pth")
            torch.save(factornet.state_dict(), latest_model_ckpt)

        # Save optimizer and scheduler state
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "latest_optimizer.pth"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_dir, "latest_scheduler.pth"))

        # Save step number
        with open(os.path.join(save_dir, "latest_step.txt"), "w") as f:
            f.write(str(step))
        logging.info(f"Saved latest step {step} for resuming.")

def save_checkpoint_gcloud(model_parallel, optimizer, scheduler, step, save_directory):
    """Save training checkpoint for Google Cloud."""
    checkpoint = {
        'model_state_dicts': [net.state_dict() for net in model_parallel.factornets],
        'device_ids': model_parallel.device_ids,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
    }
    checkpoint_path = os.path.join(save_directory, "latest_checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"💾 Saved checkpoint at step {step}")
    print(f"💾 Saved checkpoint at step {step} to {checkpoint_path}")

def save_training_slice_log_gcloud(iter_time, loss_time, epoch, epochs, max_mem, loss_value, save):
    '''
    Save the training log for the slice
    '''
    if save is not None:
        logging.info(f"Training started for epoch {epoch} / {epochs}")
        logging.info(
        f"Step {epoch:04d} | Iter Time: {iter_time:.4f}s | "
        f"Loss Time: {loss_time:.4f}s | Loss: {loss_value:.6f} | "
        f"Max Mem: {max_mem:.2f} GB"
        )   

###################
# setup
###################
# ------------------- CHECK GPUS -------------------
def setup_devices():
    """Setup and validate GPU devices for model parallel training."""
    if not torch.cuda.is_available():
        print("❌ ERROR: Model parallel training requires CUDA")
        sys.exit(1)
        
    devices = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')
    print(f"Using {len(devices)} GPUs with Model Parallel: {devices}")
    if len(devices) < 2:
        print("⚠️ Warning: Model parallel training works best with multiple GPUs")
    
    print(f"Using {len(devices)} GPUs with Model Parallel: {devices}")
    return device, devices

device, devices = setup_devices()

# ------------------- SET PARAMETERS -------------------
torch.set_float32_matmul_precision('high') # set precision for efficient matrix multiplication

# Setup optimal device settings once at startup
LearnCholesky.setup_optimal_device_settings()

# setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings','swissroll_6D_xy1', 'cifar10'], type = str,default = 'cifar10')
parser.add_argument('--depth',help = 'number of hidden layers of score network',type =int, default = 5)
parser.add_argument('--hiddenunits',help = 'number of nodes per hidden layer', type = int, default = 64)
parser.add_argument('--niters',type = int, default = 1000)
parser.add_argument('--batch_size', type = int,default = 8)
parser.add_argument('--lr',type = float, default = 1e-4) 
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--stability', type=float, default=0.01, help='Stability parameter')
parser.add_argument('--scheduler_type', type=str, default='one_cycle',
                   choices=['reduce_on_plateau', 'cosine_annealing', 'one_cycle', 'step'],
                   help='Type of LR scheduler to use')
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
stab = args.stability
weight_decay = args.weight_decay
scheduler_type = args.scheduler_type
load_model_path = args.load_model_path
load_centers_path = args.load_centers_path

#-------------------- Initialize Data -------------------
means_data = toy_data.inf_train_gen(dataset, batch_size = train_kernel_size)
means = torch.tensor(means_data, dtype=torch.float32, device=device) if not torch.is_tensor(means_data) else means_data.clone().detach().to(dtype=torch.float32, device=device)
data_dim = means.shape[1]
del means
torch.cuda.empty_cache()

#-------------------- Create Save Directory for Google Cloud -------------------
save_directory = os.environ.get("AIP_MODEL_DIR", args.save)
print("Saving outputs to:", save_directory)
# Create a temp directory for saving during training
local_save_directory = tempfile.mkdtemp(prefix="training_temp_modelparallel_")
print(f"Saving training data locally in {local_save_directory}")

# Optional: if local fallback
if not save_directory.startswith("gs://"):
    save_directory = create_save_dir_gcloud(args, save_directory)
    local_save_directory = save_directory  # Use the main save directory if it exists
else:
    # For cloud training, create proper subdirectory structure in temp location
    local_save_directory = create_save_dir_gcloud(args, local_save_directory)

log_filename = os.path.join(local_save_directory,
                             f'sample_size{train_samples_size}_test_size{test_samples_size}_batch_size{batch_size}_centers{train_kernel_size}_lr{lr}_hu{hidden_units}_modelparallel_training.log'
                            )
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logging.info(f"---------------------------------------------------------------------------------------------")

#######################
# Construct the model
#######################
# ------------------------ Initialize the model ------------------------
factornet = construct_factor_model(data_dim, depth, hidden_units).to(device).to(dtype=torch.float32)
p_samples = toy_data.inf_train_gen(dataset, batch_size=train_samples_size)
if torch.is_tensor(p_samples):
    training_samples = p_samples.clone().detach().to(dtype=torch.float32, device=device)
else:
    training_samples = torch.tensor(p_samples, dtype=torch.float32, device=device)
del p_samples

# ------------------------ Load or generate centers ------------------------
# Determine centers path (argument or fallback)
if load_centers_path is None:
    centers_path = os.path.join(local_save_directory, "latest_centers.pth")
else:
    centers_path = load_centers_path

centers = load.load_centers(centers_path, device=str(device))

# If fresh training or failed to load, regenerate centers
if centers is None:
    print("Generating new centers...")
    centers = training_samples[:train_kernel_size]
    torch.save(centers, centers_path)
    print(f"✅ Saved centers to {centers_path}")

    centers_img_path = os.path.join(local_save_directory, "centers.png")
    plotting.plot_and_save_centers(centers, centers_img_path)
else:
    print("✅ Using loaded centers from checkpoint.")

# ------------------------ Setup Model Parallel Training ------------------------
print(f"Setting up model parallel training with {len(devices)} GPUs")
if len(devices) < 2:
    print("Warning: Model parallel training works best with multiple GPUs")

# Create model parallel wrapper
model_parallel = ModelParallelWPO(factornet, centers, devices)

# Setup optimizer for all device models
all_params = []
for factornet_device in model_parallel.factornets:
    all_params.extend(list(factornet_device.parameters()))

# ------------------------ Setup optimizer & scheduler ------------------------
steps_per_epoch = max(1, train_samples_size // batch_size)
total_steps = epochs * steps_per_epoch

# Add total_steps to args for scheduler setup
args.total_steps = total_steps

optimizer, scheduler = setup.setup_optimizer_scheduler(args, all_params)

# ------------------------ Debug Info ------------------------
print(f"Scheduler type: {scheduler_type}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total steps: {total_steps}")
print(f"Epochs: {epochs}")
if scheduler_type == 'one_cycle':
    print(f"OneCycleLR configured with max_lr={lr}, total_steps={total_steps}")

# ------------------------ Load model checkpoint ------------------------
latest_checkpoint_path = os.path.join(local_save_directory, "latest_checkpoint.pth")
if load_model_path is None and os.path.exists(latest_checkpoint_path):
    load_model_path = latest_checkpoint_path

start_step = 0

if load_model_path is not None and os.path.exists(load_model_path):
    model_parallel, optimizer, scheduler, start_step = load.load_model_parallel_checkpoint(
        load_model_path, model_parallel, optimizer, scheduler
    )
    if model_parallel is not None:
        print(f"✅ Loaded checkpoint from {load_model_path}, resuming from step {start_step}")
    else:
        print(f"❌ Failed to load checkpoint from {load_model_path}, starting fresh")
        start_step = 0
else:
    print("🆕 No checkpoint found; starting fresh.")

# ------------------------ Gradient Check ------------------------
if not check.check_model_gradients(model_parallel):
    print("❌ ERROR: No trainable parameters found!")

###########################
# Training loop
###########################
gc.collect()
torch.cuda.empty_cache()

print(f"Starting model parallel training from step {start_step}...")
model_parallel.train()

try:
    for step in trange(start_step, total_steps, desc="Training"):
        torch.cuda.reset_peak_memory_stats()
        randind = torch.randint(0, train_samples_size, [batch_size,])
        samples = training_samples[randind, :]
        iter_start = time.time()
        
        # Model parallel training step
        optimizer.zero_grad(set_to_none=True)
        loss = model_parallel(samples, stab)
        
        if torch.isnan(loss) or torch.isinf(loss) or not loss.requires_grad:
            print(f"⚠️ Loss issue detected: {loss}")
            continue
        
        if loss.requires_grad and loss.grad_fn is not None:
            loss.backward()
            # Clip gradients for all device models
            for factornet_device in model_parallel.factornets:
                torch.nn.utils.clip_grad_norm_(factornet_device.parameters(), max_norm=100.0)
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
            
        loss_value = loss.item()
        iter_end = time.time()
        iter_time = iter_end - iter_start
        max_mem = torch.cuda.max_memory_allocated() / 2**30  # in GiB
        
        if step % 100 == 0:
            print(f"Step {step} started")
            print(f'Step: {step}, Loss value: {loss_value:.3e}')
            # Add current learning rate logging
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current LR: {current_lr:.2e}')
            with open(os.path.join(local_save_directory, "loss_log.csv"), "a") as f:
                f.write(f"{step},{loss_value},{current_lr}\n")
        
        if step % 100 == 0:
            loss_start = time.time()
            loss0 = evaluate_model(model_parallel, test_samples_size)
            loss_end = time.time()
            loss_time = loss_end - loss_start
            save_training_slice_log_gcloud(iter_time, loss_time, step, total_steps, max_mem, loss0, local_save_directory)

            # Only call scheduler.step() for reduce_on_plateau here
            if not math.isnan(loss0) and not math.isinf(loss0):
                # Get old LR before stepping
                old_lrs = [group['lr'] for group in optimizer.param_groups]

                # Step scheduler appropriately - ONLY for reduce_on_plateau
                if scheduler_type == 'reduce_on_plateau':
                    scheduler.step(loss0)
                    # Get new LR after stepping
                    new_lrs = [group['lr'] for group in optimizer.param_groups]
                    # Log LR changes
                    for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                        if old_lr != new_lr:
                            logging.info(f"LR changed for param group {i} from {old_lr:.4e} to {new_lr:.4e} at step {step}")
            else:
                print("⚠️ Warning: Skipping LR scheduler step due to invalid val_loss:", loss0)
            
            # Sample and save generated images at intermediate steps
            with torch.no_grad():
                # Use model parallel sampling with all centers
                generated = plotting.sample_from_model_parallel(model_parallel, sample_number=10, eps=stab)
                l2 = torch.mean((generated - training_samples[:10])**2).item()
                print(f"[Step {step}] L2 to training data: {l2:.2f}")
                logging.info(f"L2 {l2:.2f} | ")
        
        if step % 200 == 0:
            with torch.no_grad():
                filename_step_sample = os.path.join(local_save_directory, f"step{step:05d}")
                # Use model parallel plotting with all centers
                plotting.plot_images_with_model_parallel(model_parallel, training_samples, plot_number=10, eps=stab, save_path=filename_step_sample)
                logging.info(f"Saved samples at step {step} to {filename_step_sample}")
            
            save_checkpoint_gcloud(model_parallel, optimizer, scheduler, step, local_save_directory)
        
        if step % 500 == 0:
            save_training_slice_cov_gcloud(model_parallel, step, local_save_directory, optimizer, scheduler)
            
        if step < total_steps - 1:
            del samples
            gc.collect()
            torch.cuda.empty_cache()

except Exception as e:
    logging.error(f"Training interrupted with error: {e}")
    # Optionally re-raise if you want to stop after logging
    for handler in logging.root.handlers:
        handler.flush()
finally:
    ###############################
    # Evaluate the final model
    ################################
    #---------------------------- Final evaluation -------------------
    gc.collect()
    torch.cuda.empty_cache()

    loss0 = evaluate_model(model_parallel, test_samples_size)    
    save_training_slice_cov_gcloud(model_parallel, step, local_save_directory, optimizer, scheduler)
    formatted_loss = f'{loss0:.3e}'  # Format the average with up to 1e-3 precision
    logging.info(f'After train, Average total_loss: {formatted_loss}')
    #---------------------------- Sample and save -------------------
    with torch.no_grad():
        # Use model parallel plotting with all centers for final sampling
        plotting.plot_images_with_model_parallel(model_parallel, training_samples, plot_number=10, eps=stab, save_path=local_save_directory)
        logging.info(f'Sampled images saved to {local_save_directory}_sampled_images.png')

    print("✅ Model parallel training completed!")

    ###########################
    # Upload to Google Cloud Storage
    ############################
    if save_directory.startswith("gs://"):
        logging.info("Uploading all saved data to Google Cloud Storage...")
        upload_dir_to_gcs(local_save_directory, save_directory)
        logging.info("Upload complete.")
        # Cleanup local temp dir
        shutil.rmtree(local_save_directory)
        logging.info("Cleaned up local temp directory.")
    else:
        logging.info(f"Training completed. Results saved locally to {local_save_directory}")
    
    sys.stdout.flush()
    sys.stderr.flush()