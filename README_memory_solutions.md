# Memory-Efficient WPO-SGM Training Solutions

This document provides solutions for handling large numbers of centers in WPO-SGM training when you can't fit everything on a single GPU.

## Problem

The original `train_wpo_sgm_stable.py` has memory usage dominated by the number of centers. With DataParallel, the centers tensor gets replicated across all GPUs rather than being split, which multiplies memory usage instead of reducing it.

## Solutions Provided

### 1. Model Parallel Training (`train_wpo_sgm_model_parallel.py`)

**Best for:** When you have multiple GPUs and want to split centers across them.

**How it works:**
- Splits centers across multiple GPUs
- Replicates the model on each GPU
- Computes loss components in parallel
- Combines results on the main GPU

**Usage:**
```python
from train_wpo_sgm_model_parallel import setup_model_parallel_training

# Setup model parallel with specific GPUs
device_ids = [0, 1, 2, 3]  # Use GPUs 0-3
model_parallel, optimizer = setup_model_parallel_training(
    factornet, centers, device_ids, args
)
```

### 2. Chunked Training

**Best for:** When you have limited GPU memory but can process centers sequentially.

**How it works:**
- Processes centers in small chunks
- Computes loss for each chunk separately
- Combines results with proper weighting

**Usage:**
```python
from train_wpo_sgm_model_parallel import setup_chunked_training

# Setup chunked training
chunk_size = 50  # Process 50 centers at a time
chunked_trainer = setup_chunked_training(factornet, centers, chunk_size)

# Training step
loss = chunked_trainer.compute_loss(samples, stab=1e-6)
```

### 3. Gradient Accumulation Training

**Best for:** When you want to maintain gradient flow while processing centers in batches.

**How it works:**
- Splits centers into chunks
- Accumulates gradients across chunks
- Updates parameters after processing all chunks

**Usage:**
```python
from train_wpo_sgm_model_parallel import setup_gradient_accumulation_training

# Setup gradient accumulation
accumulation_steps = 4  # Accumulate over 4 chunks
grad_accum_trainer = setup_gradient_accumulation_training(
    factornet, centers, accumulation_steps
)

# Training step (handles gradient accumulation internally)
loss = grad_accum_trainer.training_step(samples, optimizer, stab=1e-6)
```

### 4. Automatic Strategy Selection (`train_wpo_sgm_memory_efficient.py`)

**Best for:** Automatic selection of the best strategy based on your hardware.

**Features:**
- Estimates memory requirements
- Automatically chooses the best strategy
- Provides manual override options
- Comprehensive error handling

## Quick Start

### Option 1: Use the automatic script (Recommended)

```bash
# Automatic strategy selection
python train_wpo_sgm_memory_efficient.py --train_kernel_size 1000 --data cifar10

# Force a specific strategy
python train_wpo_sgm_memory_efficient.py --strategy chunked --chunk_size 50

# Model parallel with specific settings
python train_wpo_sgm_memory_efficient.py --strategy model_parallel --train_kernel_size 2000
```

### Option 2: Modify your existing script

Add these imports to your existing `train_wpo_sgm_stable.py`:

```python
from train_wpo_sgm_model_parallel import (
    setup_chunked_training, 
    setup_gradient_accumulation_training
)

# Replace your training loop with chunked approach
chunked_trainer = setup_chunked_training(factornet, centers, chunk_size=50)
optimizer = optim.AdamW(chunked_trainer.factornet.parameters(), lr=args.lr)

# Training loop
for step in range(args.niters):
    randind = torch.randint(0, training_samples.shape[0], [args.batch_size])
    samples = training_samples[randind, :]
    
    optimizer.zero_grad()
    loss = chunked_trainer.compute_loss(samples, args.stability)
    loss.backward()
    optimizer.step()
```

## Memory Estimates

The script automatically estimates memory usage:

```
Memory estimates:
  Centers: 0.12 GB
  Precision matrices: 36.00 GB  # This is the bottleneck!
  Gradients: 0.01 GB
  Total estimate: 36.13 GB
```

## Strategy Selection Logic

1. **Single GPU**: If total memory ≤ 80% of GPU memory
2. **Model Parallel**: If multiple GPUs and total memory ≤ 60% of total GPU memory
3. **Chunked**: If total memory ≤ 150% of GPU memory
4. **Gradient Accumulation**: For very large memory requirements

## Command Line Options

### Basic Options
- `--train_kernel_size`: Number of centers (main memory driver)
- `--batch_size`: Batch size for training samples
- `--data`: Dataset to use (cifar10, 8gaussians, etc.)

### Memory Strategy Options
- `--strategy`: Choose strategy (auto, single_gpu, model_parallel, chunked, gradient_accumulation)
- `--chunk_size`: Chunk size for chunked training
- `--accumulation_steps`: Steps for gradient accumulation

### Example Commands

```bash
# Small experiment (fits on single GPU)
python train_wpo_sgm_memory_efficient.py --train_kernel_size 100

# Medium experiment (use chunking)
python train_wpo_sgm_memory_efficient.py --train_kernel_size 500 --strategy chunked --chunk_size 50

# Large experiment (use model parallel)
python train_wpo_sgm_memory_efficient.py --train_kernel_size 2000 --strategy model_parallel

# Very large experiment (use gradient accumulation)
python train_wpo_sgm_memory_efficient.py --train_kernel_size 5000 --strategy gradient_accumulation --accumulation_steps 8
```

## Performance Considerations

### Model Parallel
- **Pros**: True parallelism, can handle very large center sets
- **Cons**: Communication overhead between GPUs, requires multiple GPUs

### Chunked Training
- **Pros**: Works on single GPU, simple implementation
- **Cons**: Sequential processing, may be slower

### Gradient Accumulation
- **Pros**: Maintains proper gradient flow, memory efficient
- **Cons**: Slower convergence, more complex bookkeeping

## Troubleshooting

### Out of Memory Errors
1. Reduce `--train_kernel_size`
2. Reduce `--batch_size`
3. Use `--strategy chunked` with smaller `--chunk_size`
4. Use `--strategy gradient_accumulation` with more `--accumulation_steps`

### Slow Training
1. Try `--strategy model_parallel` if you have multiple GPUs
2. Increase `--chunk_size` if using chunked training
3. Reduce `--accumulation_steps` if using gradient accumulation

### Memory Monitoring
The script automatically prints memory usage:
```
Memory: 8.45GB allocated, 9.12GB reserved
```

## Integration with Existing Code

To integrate these solutions into your existing training pipeline:

1. **Import the modules:**
```python
from train_wpo_sgm_model_parallel import setup_chunked_training
```

2. **Replace your loss computation:**
```python
# Old way
loss = LearnCholesky.score_implicit_matching_stable(factornet, samples, centers, stab)

# New way (chunked)
chunked_trainer = setup_chunked_training(factornet, centers, chunk_size=50)
loss = chunked_trainer.compute_loss(samples, stab)
```

3. **Update your optimizer:**
```python
# Point optimizer to the wrapped model
optimizer = optim.AdamW(chunked_trainer.factornet.parameters(), lr=args.lr)
```

## Advanced Usage

### Custom Chunk Sizes
```python
# Estimate optimal chunk size based on available memory
available_memory = torch.cuda.get_device_properties(0).total_memory
used_memory = torch.cuda.memory_allocated(0)
free_memory = available_memory - used_memory

dim = centers.shape[1]
bytes_per_center = dim * dim * 4 * 2  # Rough estimate
optimal_chunk_size = int(free_memory * 0.5 / bytes_per_center)
```

### Mixed Strategies
You can combine strategies for maximum efficiency:
```python
# Use model parallel with chunking on each device
for device_id in device_ids:
    device_centers = centers_per_device[device_id]
    chunked_trainer = setup_chunked_training(factornet, device_centers, chunk_size=20)
```

This comprehensive solution should handle your memory constraints while maintaining training effectiveness.
