import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import os
import sys
from typing import List, Tuple
import gc
from tqdm import trange
import concurrent.futures
import threading

# Import your existing modules
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import toy_data
import config.load as load
from utilities.plots import *

def construct_factor_model(dim: int, depth: int, hidden_units: int):
        '''
        Initializes neural network that models the Cholesky factor of the precision matrix
        '''
        chain = []
        chain.append(nn.Linear(dim, int(hidden_units), bias=True)) 
        chain.append(nn.GELU())

        for _ in range(depth-1):
            chain.append(nn.Linear(int(hidden_units), int(hidden_units), bias=True))
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

class ModelParallelWPO(nn.Module):
    """
    Model parallel implementation that splits centers across GPUs
    """
    def __init__(self, factornet, centers, device_ids: List[int]):
        super().__init__()
        self.device_ids = device_ids
        self.num_devices = len(device_ids)
        
        # Replicate the factornet on all devices - just copy the entire model
        self.factornets = nn.ModuleList()
        for device_id in device_ids:
            # Create a deep copy of the factornet and move to device
            import copy
            factornet_copy = copy.deepcopy(factornet).to(f'cuda:{device_id}')
            self.factornets.append(factornet_copy)
        
        self.centers_per_device = self._split_centers(centers, device_ids)

    def _split_centers(self, centers, device_ids):
        """Split centers across devices as evenly as possible"""
        num_centers = centers.shape[0]
        centers_per_device = []
        
        # Calculate split sizes
        base_size = num_centers // len(device_ids)
        remainder = num_centers % len(device_ids)
        
        start_idx = 0
        for i, device_id in enumerate(device_ids):
            # Give remainder centers to first few devices
            size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + size
            
            # FIXED: Ensure we don't go out of bounds
            if start_idx >= num_centers:
                # No more centers for this device
                centers_per_device.append(torch.empty(0, centers.shape[1], device=f'cuda:{device_id}'))
            else:
                end_idx = min(end_idx, num_centers)
                device_centers = centers[start_idx:end_idx].to(f'cuda:{device_id}')
                centers_per_device.append(device_centers)
            
            start_idx = end_idx
        
        return centers_per_device
    
    def forward(self, samples, stab=1e-6):
        """
        Forward pass with model parallelism
        """
        # Move samples to all devices
        samples_per_device = [samples.to(f'cuda:{device_id}') 
                             for device_id in self.device_ids]
        
        # Compute loss components on each device
        loss_components = []
        
        for i, device_id in enumerate(self.device_ids):
            device_samples = samples_per_device[i]
            device_centers = self.centers_per_device[i]
            device_factornet = self.factornets[i]
            
            # Compute loss component on this device
            with torch.cuda.device(device_id):
                loss_component = LearnCholesky.score_implicit_matching_stable(
                    device_factornet, device_samples, device_centers, stab
                )
                loss_components.append(loss_component)
        
        # Gather all loss components to the first device and average
        main_device = f'cuda:{self.device_ids[0]}'
        total_loss = torch.zeros(1, device=main_device)
        
        for loss_component in loss_components:
            total_loss += loss_component.to(main_device)
        
        return total_loss / len(loss_components)

class CenterChunkedTrainer:
    """
    Alternative approach: Process centers in chunks sequentially
    """
    def __init__(self, factornet, centers, chunk_size=50):
        self.factornet = factornet
        self.centers = centers
        self.chunk_size = chunk_size
        self.num_chunks = (centers.shape[0] + chunk_size - 1) // chunk_size
        
    def compute_loss(self, samples, stab=1e-6):
        """
        Compute loss by processing centers in chunks
        """
        total_loss = torch.tensor(0.0, device=samples.device, requires_grad=True)
        num_centers = self.centers.shape[0]
        
        # FIXED: Accumulate gradients properly
        for i in range(0, num_centers, self.chunk_size):
            end_idx = min(i + self.chunk_size, num_centers)
            centers_chunk = self.centers[i:end_idx]
            
            # Compute loss for this chunk
            chunk_loss = LearnCholesky.score_implicit_matching_stable(
                self.factornet, samples, centers_chunk, stab
            )
            
            # FIXED: Proper weighted accumulation
            chunk_weight = (end_idx - i) / num_centers
            if i == 0:
                total_loss = chunk_loss * chunk_weight
            else:
                total_loss = total_loss + (chunk_loss * chunk_weight)
            
            # Clean up intermediate tensors but keep total_loss
            del centers_chunk, chunk_loss
        
        return total_loss

class GradientAccumulationTrainer:
    """
    Use gradient accumulation to handle large center sets
    """
    def __init__(self, factornet, centers, accumulation_steps=4):
        self.factornet = factornet
        self.centers = centers
        self.accumulation_steps = accumulation_steps
        self.chunk_size = centers.shape[0] // accumulation_steps
        
    def training_step(self, samples, optimizer, stab=1e-6):
        """
        Training step with gradient accumulation
        """
        optimizer.zero_grad()
        total_loss = 0.0
        
        for i in range(0, self.centers.shape[0], self.chunk_size):
            end_idx = min(i + self.chunk_size, self.centers.shape[0])
            centers_chunk = self.centers[i:end_idx]
            
            # Compute loss for this chunk
            chunk_loss = LearnCholesky.score_implicit_matching_stable(
                self.factornet, samples, centers_chunk, stab
            )
            
            # Scale loss by number of accumulation steps
            scaled_loss = chunk_loss / self.accumulation_steps
            scaled_loss.backward()
            
            total_loss += chunk_loss.item()
            
            # Clean up
            del centers_chunk, chunk_loss, scaled_loss
            torch.cuda.empty_cache()
        
        # Update parameters after accumulating gradients
        torch.nn.utils.clip_grad_norm_(self.factornet.parameters(), max_norm=1.0)
        optimizer.step()
        
        return total_loss / self.accumulation_steps

def setup_model_parallel_training(factornet, centers, device_ids, args):
    """
    Setup model parallel training
    """
    print(f"Setting up model parallel training on devices: {device_ids}")
    print(f"Total centers: {centers.shape[0]}")
    print(f"Centers per device: {[centers.shape[0] // len(device_ids) + (1 if i < centers.shape[0] % len(device_ids) else 0) for i in range(len(device_ids))]}")
    
    # Create model parallel wrapper
    model_parallel = ModelParallelWPO(factornet, centers, device_ids)
    
    # Setup optimizer for all device models
    all_params = []
    for factornet_device in model_parallel.factornets:
        all_params.extend(list(factornet_device.parameters()))
    
    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 1e-4)
    )
    
    return model_parallel, optimizer

def setup_chunked_training(factornet, centers, chunk_size=None):
    """
    Setup chunked training approach
    """
    if chunk_size is None:
        # Estimate chunk size based on available memory
        available_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated(0)
        free_memory = available_memory - used_memory
        
        # Rough estimate: each center uses ~dim^2 * 4 bytes for precision matrix
        dim = centers.shape[1]
        bytes_per_center = dim * dim * 4 * 2  # Factor of 2 for safety
        max_centers = int(free_memory * 0.5 / bytes_per_center)  # Use 50% of free memory
        chunk_size = min(max_centers, 500)  # Cap at 500
        
    print(f"Using chunked training with chunk size: {chunk_size}")
    return CenterChunkedTrainer(factornet, centers, chunk_size)

def setup_gradient_accumulation_training(factornet, centers, accumulation_steps=None):
    """
    Setup gradient accumulation training
    """
    if accumulation_steps is None:
        # Auto-determine based on memory constraints
        accumulation_steps = max(2, centers.shape[0] // 50)  # Aim for ~50 centers per step
        
    print(f"Using gradient accumulation with {accumulation_steps} steps")
    return GradientAccumulationTrainer(factornet, centers, accumulation_steps)

# Example usage functions
def train_with_model_parallel(model_parallel, training_samples, optimizer, scheduler, args):
    """
    Training loop for model parallel approach
    """
    model_parallel.train()
    
    for step in trange(args.niters, desc="Training"):
        # Sample batch
        randind = torch.randint(0, training_samples.shape[0], [args.batch_size])
        samples = training_samples[randind, :]
        
        # Forward pass
        optimizer.zero_grad()
        loss = model_parallel(samples, args.stability)
        loss.backward()
        
        # Gradient clipping and optimization
        for factornet_device in model_parallel.factornets:
            torch.nn.utils.clip_grad_norm_(factornet_device.parameters(), max_norm=100.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
            
        # Memory cleanup
        del samples, loss
        torch.cuda.empty_cache()

def train_with_chunked_approach(chunked_trainer, training_samples, optimizer, scheduler, args):
    """
    Training loop for chunked approach
    """
    chunked_trainer.factornet.train()
    
    for step in range(args.niters):
        # Sample batch
        randind = torch.randint(0, training_samples.shape[0], [args.batch_size])
        samples = training_samples[randind, :]
        
        # Forward pass with chunking
        optimizer.zero_grad()
        loss = chunked_trainer.compute_loss(samples, args.stability)
        loss.backward()
        
        # Optimization
        torch.nn.utils.clip_grad_norm_(chunked_trainer.factornet.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
            
        # Memory cleanup
        del samples, loss
        torch.cuda.empty_cache()

def train_with_gradient_accumulation(grad_accum_trainer, training_samples, optimizer, scheduler, args):
    """
    Training loop for gradient accumulation approach
    """
    grad_accum_trainer.factornet.train()
    
    for step in range(args.niters):
        # Sample batch
        randind = torch.randint(0, training_samples.shape[0], [args.batch_size])
        samples = training_samples[randind, :]
        
        # Training step with gradient accumulation
        loss = grad_accum_trainer.training_step(samples, optimizer, args.stability)
        
        if scheduler:
            scheduler.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
            
        # Memory cleanup
        del samples
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Model Parallel WPO Training Module")
    print("Import this module and use the setup functions in your main training script")
