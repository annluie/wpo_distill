## Pretrain model (py version of Example_WPO_SGM.ipynb) - Model Parallel Version


###################
# setup
###################
# ------------------- ENV & PATH SETUP -------------------
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:32'
import sys
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
import torch.multiprocessing as mp
# Removed mixed precision imports since not used for large center counts
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint_sequential
import torch.distributed as dist
#import torch._dynamo
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
import torch.utils.checkpoint as cp
import torch.nn as nn
#torch._dynamo.config.suppress_errors = True
# ------------------- PROJECT MODULES -------------------
import utilities.plots as plotting, utilities.save as saving, utilities.diagnostics as check
#from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import functions_WPO_SGM_fast as LearnCholesky
from WPO_SGM import toy_data
#from WPO_SGM import function_cpu as LearnCholesky
import config.load as load, config.logging as log, config.setup_model as setup

from train_wpo_sgm_model_parallel import ModelParallelWPO

# Optimized forward pass to reduce function call overhead
class OptimizedModelParallelWPO(ModelParallelWPO):
    """Optimized version of ModelParallelWPO with reduced overhead"""
    
    def __init__(self, factornet, centers, device_ids):
        super().__init__(factornet, centers, device_ids)
        # Pre-compile the loss function if possible
        self._compiled_loss = None
        # Pre-allocate gradient accumulation buffers
        self.gradient_accumulation_steps = 1
        self._accumulated_loss = None
        
    def parameters(self):
        """Efficiently return all parameters from all devices"""
        for factornet in self.factornets:
            yield from factornet.parameters()
    
    def set_gradient_accumulation_steps(self, steps):
        """Set gradient accumulation steps for memory efficiency"""
        self.gradient_accumulation_steps = steps
        
    def forward(self, samples, stab=1e-6):
        """Highly optimized forward pass"""
        # Pre-allocate samples on devices (avoid repeated transfers)
        if not hasattr(self, '_samples_cache') or self._samples_cache[0].shape[0] != samples.shape[0]:
            self._samples_cache = [torch.empty_like(samples, device=f'cuda:{device_id}')
                                 for device_id in self.device_ids]
        
        # Copy samples to all devices efficiently
        for i, device_id in enumerate(self.device_ids):
            self._samples_cache[i].copy_(samples, non_blocking=True)
        
        # Compute loss components on each device
        loss_components = []
        
        for i, device_id in enumerate(self.device_ids):
            device_samples = self._samples_cache[i]
            device_centers = self.centers_per_device[i]
            device_factornet = self.factornets[i]
            
            # Skip devices with no centers
            if device_centers.shape[0] == 0:
                continue
            
            # Compute loss component on this device
            with torch.cuda.device(device_id):
                # if hasattr(LearnCholesky, 'score_implicit_matching_stable_optimized'):
                #     loss_component = LearnCholesky.score_implicit_matching_stable_optimized(
                #         device_factornet, device_samples, device_centers, stab)
                # else:
                #     loss_component = LearnCholesky.score_implicit_matching_stable(
                #         device_factornet, device_samples, device_centers, stab)
                loss_component = LearnCholesky.score_implicit_matching_stable_optimized(
                        device_factornet, device_samples, device_centers, stab)
                loss_components.append(loss_component)
        
        # Efficiently combine losses
        if len(loss_components) == 1:
            return loss_components[0]
        
        main_device = f'cuda:{self.device_ids[0]}'
        total_loss = loss_components[0]
        
        for loss_component in loss_components[1:]:
            total_loss = total_loss + loss_component.to(main_device, non_blocking=True)
        
        return total_loss / len(loss_components)

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

def evaluate_model(model_parallel, num_test_sample, num_batches=2):
    """
    Optimized evaluation function with reduced overhead
    """
    device = f'cuda:{model_parallel.device_ids[0]}'  # Use first device
    total_loss_sum = 0.0

    model_parallel.eval()
    with torch.no_grad():
        # Pre-allocate tensor to avoid repeated allocations
        p_samples_tensor = torch.empty(num_test_sample, training_samples.shape[1],
                                     device=device, dtype=torch.float32)
        
        for _ in range(num_batches):
            # Generate test samples directly on the correct device
            p_samples = toy_data.inf_train_gen(dataset, batch_size=num_test_sample)
            if not torch.is_tensor(p_samples):
                p_samples_tensor.copy_(torch.from_numpy(p_samples))
            else:
                p_samples_tensor.copy_(p_samples)

            # Use model parallel forward pass (this uses all centers across all devices)
            loss = model_parallel(p_samples_tensor, stab)
            total_loss_sum += loss.item()

    model_parallel.train()  # Reset to training mode
    return total_loss_sum / num_batches


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

# Additional optimizations for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

# Enable compilation optimizations
torch._C._set_graph_executor_optimize(True)
try:
    torch._C._jit_set_texpr_fuser_enabled(True)
except AttributeError:
    pass  # Not available in all PyTorch versions
args = load.parse_arguments()

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
means_data = toy_data.inf_train_gen(dataset, batch_size = train_kernel_size)
means = torch.tensor(means_data, dtype=torch.float32, device=device) if not torch.is_tensor(means_data) else means_data.clone().detach().to(dtype=torch.float32, device=device)
data_dim = means.shape[1]
del means, means_data

#-------------------- Create Save Directory -------------------
save_directory = saving.create_save_dir_from_args(args)
print('save_directory',save_directory)
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print('Created directory ' + save_directory)

# Configure the logger
log.setup_logging(save_directory)

#######################
# Construct the model
#######################
# ------------------------ Initialize the model ------------------------
factornet = construct_factor_model(data_dim, depth, hidden_units).to(device, dtype=torch.float32)
p_samples = toy_data.inf_train_gen(dataset, batch_size=train_samples_size)
if torch.is_tensor(p_samples):
    training_samples = p_samples.to(dtype=torch.float32, device=device)
    training_samples = training_samples.contiguous()
else:
    training_samples = torch.tensor(p_samples, dtype=torch.float32, device=device).contiguous()
del p_samples

# Pin memory for faster transfers if not already on GPU
if training_samples.device.type == 'cpu':
    training_samples = training_samples.pin_memory()

# ------------------------ Load or generate centers ------------------------
# Determine centers path (argument or fallback)
if load_centers_path is None:
    centers_path = os.path.join(save_directory, "latest_centers.pth")
else:
    centers_path = load_centers_path

centers = load.load_centers(centers_path, device=str(device))

# If fresh training or failed to load, regenerate centers
if centers is None:
    print("Generating new centers...")
    centers = training_samples[:train_kernel_size]
    torch.save(centers, centers_path)
    print(f"✅ Saved centers to {centers_path}")

    centers_img_path = os.path.join(save_directory, "centers.png")
    plotting.plot_and_save_centers(centers, centers_img_path)
else:
    print("✅ Using loaded centers from checkpoint.")

# ------------------------ Setup Model Parallel Training ------------------------
print(f"Setting up model parallel training with {len(devices)} GPUs")
if len(devices) < 2:
    print("Warning: Model parallel training works best with multiple GPUs")

# Create optimized model parallel wrapper
model_parallel = OptimizedModelParallelWPO(factornet, centers, devices)

# Set gradient accumulation for memory efficiency with large center counts
if train_kernel_size > 500:
    model_parallel.set_gradient_accumulation_steps(2)
    print(f"Using gradient accumulation with {model_parallel.gradient_accumulation_steps} steps")

# Setup optimizer for all device models
all_params = []
for factornet_device in model_parallel.factornets:
    all_params.extend(list(factornet_device.parameters()))

# Mixed precision removed for simplicity and compatibility with large center counts
print("Mixed precision training: Disabled (removed for large center compatibility)")

# ------------------------ Setup optimizer & scheduler ------------------------
steps_per_epoch = max(1, train_samples_size // batch_size)
total_steps = epochs * steps_per_epoch

optimizer, scheduler = setup.setup_optimizer_scheduler(args, all_params)

# ------------------------ Debug Info ------------------------
print(f"Scheduler type: {scheduler_type}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total steps: {total_steps}")
print(f"Epochs: {epochs}")
if scheduler_type == 'one_cycle':
    print(f"OneCycleLR configured with max_lr={lr}, total_steps={total_steps}")

# ------------------------ Load model checkpoint ------------------------
latest_checkpoint_path = os.path.join(save_directory, "latest_checkpoint.pth")
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

# Pre-allocate tensors to reduce allocation overhead
batch_indices = torch.empty(batch_size, dtype=torch.long, device=device)

# Pre-generate random indices for multiple steps to reduce overhead
index_buffer_size = min(1000, total_steps - start_step)
random_indices_buffer = torch.randint(0, train_samples_size,
                                     [index_buffer_size, batch_size],
                                     device=device, dtype=torch.long)
buffer_idx = 0

# Pre-allocate sample tensor to avoid repeated allocations
samples_tensor = torch.empty(batch_size, training_samples.shape[1],
                           device=device, dtype=torch.float32)

for step in trange(start_step, total_steps, desc="Training"):
    iter_start = time.time()
    
    # Use pre-generated indices to reduce randint overhead
    if buffer_idx >= index_buffer_size:
        random_indices_buffer = torch.randint(0, train_samples_size,
                                             [index_buffer_size, batch_size],
                                             device=device, dtype=torch.long)
        buffer_idx = 0
    
    randind = random_indices_buffer[buffer_idx]
    buffer_idx += 1
    
    # Use pre-allocated tensor for better memory efficiency
    samples_tensor = training_samples[randind]
    
    # Model parallel training step
    optimizer.zero_grad(set_to_none=True)
    loss = model_parallel(samples_tensor, stab)
    
    if torch.isnan(loss) or torch.isinf(loss) or not loss.requires_grad:
        print(f"⚠️ Loss issue detected: {loss}")
        continue
    
    if loss.requires_grad and loss.grad_fn is not None:
        loss.backward()
        # Clip gradients for all device models - use unscaled version for efficiency
        torch.nn.utils.clip_grad_norm_(model_parallel.parameters(), max_norm=100.0)
        optimizer.step()
        
        # Update scheduler if provided
        if scheduler is not None:
            try:
                if scheduler_type in ['cosine_annealing', 'one_cycle']:
                    scheduler.step()
                elif scheduler_type == 'step':
                    scheduler.step()
                # Step and reduce_on_plateau schedulers are handled elsewhere
            except (TypeError, AttributeError, RuntimeError):
                pass  # Skip scheduler update if it fails
    else:
        print("❌ Gradient flow broken!")
        
    loss_value = loss.item()
    iter_end = time.time()
    iter_time = iter_end - iter_start
    
    # Reduce expensive operations frequency
    if step % 100 == 0:
        max_mem = torch.cuda.max_memory_allocated() / 2**30  # in GiB
        print(f"Step {step} started")
        print(f'Step: {step}, Loss value: {loss_value:.3e}')
        # Add current learning rate logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current LR: {current_lr:.2e}')
        with open(os.path.join(save_directory, "loss_log.csv"), "a") as f:
            f.write(f"{step},{loss_value},{current_lr}\n")
    
    # Reduce evaluation frequency to every 200 steps instead of 100
    if step % 200 == 0:
        loss_start = time.time()
        loss0 = evaluate_model(model_parallel, test_samples_size, num_batches=1)  # Reduce batches
        loss_end = time.time()
        loss_time = loss_end - loss_start
        max_mem = torch.cuda.max_memory_allocated() / 2**30  # in GiB
        saving.save_training_slice_log(iter_time, loss_time, step, total_steps, max_mem, loss0, save_directory)

        # Only call scheduler.step() for reduce_on_plateau here
        if not math.isnan(loss0) and not math.isinf(loss0):
            # Get old LR before stepping
            old_lrs = [group['lr'] for group in optimizer.param_groups]

            # Step scheduler appropriately - ONLY for reduce_on_plateau
            try:
                if scheduler_type == 'reduce_on_plateau':
                    # Use the correct method for ReduceLROnPlateau
                    if hasattr(scheduler, 'step') and hasattr(scheduler, 'mode'):
                        scheduler.step(loss0)  # ReduceLROnPlateau needs metrics
                elif scheduler_type == 'step':
                    scheduler.step()
                # Other schedulers are handled in the main training loop
            except TypeError:
                # Some schedulers might need different arguments
                pass
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
            generated = plotting.sample_from_model_parallel(model_parallel, sample_number=5, eps=stab)  # Reduce samples
            l2 = torch.mean((generated - training_samples[:5])**2).item()
            print(f"[Step {step}] L2 to training data: {l2:.2f}")
            logging.info(f"L2 {l2:.2f} | ")
    
    if step % 500 == 0:  # Increased frequency for plotting
        with torch.no_grad():
            filename_step_sample = os.path.join(save_directory, f"step{step:05d}")
            # Use model parallel plotting with all centers
            plotting.plot_images_with_model_parallel(model_parallel, training_samples, plot_number=10, eps=stab, save_path=filename_step_sample)
            logging.info(f"Saved samples at step {step} to {filename_step_sample}")
        
        saving.save_checkpoint(model_parallel, optimizer, scheduler, step, save_directory)
    
    if step % 1000 == 0:  # Reduced frequency
        saving.save_training_slice_cov(model_parallel, step, save_directory, optimizer, scheduler)
        
    # Remove expensive cleanup operations from the hot loop
    # Only cleanup periodically and more strategically
    if step % 100 == 0:
        torch.cuda.empty_cache()
    elif step % 25 == 0 and train_kernel_size > 1000:  # More frequent cleanup for large center counts
        torch.cuda.empty_cache()

###############################
# Evaluate the final model
################################
#---------------------------- Final evaluation -------------------
gc.collect()
torch.cuda.empty_cache()

loss0 = evaluate_model(model_parallel, test_samples_size)    
saving.save_training_slice_cov(model_parallel, step, save_directory, optimizer, scheduler)
formatted_loss = f'{loss0:.3e}'  # Format the average with up to 1e-3 precision
logging.info(f'After train, Average total_loss: {formatted_loss}')
#---------------------------- Sample and save -------------------
with torch.no_grad():
    # Use model parallel plotting with all centers for final sampling
    plotting.plot_images_with_model_parallel(model_parallel, training_samples, plot_number=10, eps=stab, save_path=save_directory)
    logging.info(f'Sampled images saved to {save_directory}_sampled_images.png')

print("✅ Model parallel training completed!")