#----------------------------------------------#
#### Import dependencies ####
#----------------------------------------------#
# ------------------- UTILITIES -------------------
import time
import gc
from contextlib import contextmanager
# ------------------- MATH -------------------

# ------------------- PYTORCH -------------------
import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
# ------------------- PROJECT MODULES -------------------

#----------------------------------------------#
#### Utility functions ####
#----------------------------------------------#
# setup functions
def setup_optimal_device_settings():
    """
    Configure optimal device settings for performance
    """
    if torch.cuda.is_available():
        # Enable tensor cores for mixed precision
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        
        print("Optimal CUDA settings enabled")
    else:
        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        print(f"Using {torch.get_num_threads()} CPU threads")

def setup_memory_efficient_settings():
    """
    Configure settings for memory efficiency
    """
    if torch.cuda.is_available():
        # Conservative settings for memory efficiency
        torch.backends.cudnn.benchmark = False  # Disable for memory consistency
        torch.backends.cudnn.deterministic = True
        # Also add this for better memory management
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set memory fraction if needed
        # torch.cuda.set_per_process_memory_fraction(0.8)
        
        print("Memory-efficient CUDA settings enabled")
        print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Using CPU - no GPU memory concerns")

# numerical health checks
def numerical_health_check(tensor, name="tensor"):
    """
    Check numerical health of tensors and provide diagnostics
    """
    if torch.any(torch.isnan(tensor)):
        print(f"Warning: NaN detected in {name}")
        return False
    if torch.any(torch.isinf(tensor)):
        print(f"Warning: Inf detected in {name}")
        return False
    if torch.any(torch.abs(tensor) > 1e12):
        print(f"Warning: Very large values detected in {name} (max: {torch.max(torch.abs(tensor))})")
        return False
    return True

def numerical_health_check_ddp(tensor, name, rank=None):
    """
    Check tensor health across all ranks
    """
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    
    if tensor is None:
        print(f"[Rank {rank}] {name} is None")
        return False
    
    # Local checks
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if dist.is_initialized():
        # Convert to float tensors for all_reduce (SUM operation)
        has_nan_tensor = torch.tensor(float(has_nan), device=tensor.device, dtype=torch.float32)
        has_inf_tensor = torch.tensor(float(has_inf), device=tensor.device, dtype=torch.float32)
        
        # All-reduce with SUM - if any rank has issues, sum will be > 0
        dist.all_reduce(has_nan_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(has_inf_tensor, op=dist.ReduceOp.SUM)
        
        # Convert back to boolean
        has_nan = has_nan_tensor.item() > 0
        has_inf = has_inf_tensor.item() > 0
    
    is_healthy = not (has_nan or has_inf)
    
    if not is_healthy:
        print(f"[Rank {rank}] ❌ {name} health check failed - NaN: {has_nan}, Inf: {has_inf}")
    
    return is_healthy

def numerical_health_check_dp(tensor, name): # Helper numerical health check for DP (not using dist)
    if tensor is None:
        print(f"⚠️ {name} tensor is None")
        return False
    if torch.isnan(tensor).any():
        print(f"⚠️ {name} contains NaNs")
        return False
    if torch.isinf(tensor).any():
        print(f"⚠️ {name} contains Infs")
        return False
    return True

# cleanup functions
def aggressive_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def safe_cleanup(*vars_):
    for v in vars_:
        try:
            if isinstance(v, torch.Tensor):
                _ = v.detach()  # ✅ safe version
            del v
        except:
            pass
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

def cleanup_distributed_memory(rank=None):
    """
    Coordinated memory cleanup across all ranks
    """
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Synchronize before cleanup
    if dist.is_initialized():
        dist.barrier()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Synchronize after cleanup
    if dist.is_initialized():
        dist.barrier()
    
    if rank == 0:  # Only print from rank 0 to avoid spam
        print("🧹 Distributed memory cleanup completed")

# profiling functions
def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

def get_memory_usage_for_device(device_id, rank=None):
    """
    Get memory usage for a specific device with DDP awareness
    """
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    
    if torch.cuda.is_available() and device_id is not None:
        # Ensure we're checking the right device
        current_device = torch.cuda.current_device()
        if current_device != device_id:
            torch.cuda.set_device(device_id)
        
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3  # GB
        
        print(f"[Rank {rank}] Device {device_id} - Allocated: {allocated:.2f}GB, "
              f"Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
        
        return allocated, reserved, max_allocated
    else:
        print(f"[Rank {rank}] CUDA not available or device_id is None")
        return 0, 0, 0
    
@contextmanager
def profile_section(section_name, device_id, rank=None):
    """
    Context manager for profiling memory and time with DDP support
    """
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Reset peak memory stats at the start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)
    
    # Synchronize all processes before timing
    if dist.is_initialized():
        dist.barrier()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_id)
    
    start_time = time.time()
    
    try:
        yield
    finally:
        # Synchronize before measuring end time
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_id)
        
        if dist.is_initialized():
            dist.barrier()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print timing info
        print(f"[Rank {rank}] {section_name} time: {duration:.2f}s")
        
        # Print memory info
        get_memory_usage_for_device(device_id, rank)

def log_once(msg):
    """Print only once (e.g., on main device 0)."""
    # For DataParallel, just print once per process
    print(msg)

def print_memory_usage_dp():
    """Print memory usage stats on all GPUs."""
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        print(f"GPU {i}: Allocated {allocated:.3f} GB | Reserved {reserved:.3f} GB | Max Allocated {max_allocated:.3f} GB")


