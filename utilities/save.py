# saving functions

###################
# import dependencies
###################
# ------------------- ENV & PATH SETUP -------------------
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:32'

# ------------------- TIME & LOGGING -------------------
import logging
# ------------------- PYTORCH -------------------
import torch
# ------------------- PROJECT MODULES -------------------
from utilities.plots import *
from WPO_SGM import functions_WPO_SGM_stable as LearnCholesky
from WPO_SGM import toy_data
#from WPO_SGM import function_cpu as LearnCholesky

###################
# saving functions
###################
def create_save_dir_from_args(args):
    '''
    Create a subfolder to save all the outputs
    '''
    if args.save is not None:
        subfolder = os.path.join(
                args.save,
                f"sample_size{args.train_samples_size}",
                #f"test_size{args.test_samples_size}",
                f"centers{args.train_kernel_size}",
                f"batch_size{args.batch_size}_epochs{args.niters}",
                f"lr{args.lr}_hu{args.hiddenunits}_stab{args.stability}_modelparallel"
                #f"lr{lr}_hu{hidden_units}_stab{stab}"
            )
        os.makedirs(subfolder, exist_ok=True)
    else:
        subfolder = os.path.join(
                f"sample_size{args.train_samples_size}",
                #f"test_size{args.test_samples_size}",
                f"centers{args.train_kernel_size}",
                f"batch_size{args.batch_size}_epochs{args.niters}",
                f"lr{args.lr}_hu{args.hiddenunits}_stab{args.stability}_modelparallel"
            )
    return subfolder

def save_training_slice_cov(model_parallel, step, save_dir, optimizer=None, scheduler=None):
    """
    Save model, optimizer, and scheduler state (centers saved separately only once)
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

def save_checkpoint(model_parallel, optimizer, scheduler, step, save_directory):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dicts': [net.state_dict() for net in model_parallel.factornets],
            'device_ids': model_parallel.device_ids,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': step,
        }
        checkpoint_path = os.path.join(save_directory, "latest_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"💾 Saved checkpoint at step {step}")
        print(f"💾 Saved checkpoint at step {step} to {checkpoint_path}")

def save_training_slice_log(iter_time, loss_time, epoch, epochs, max_mem, loss_value, save):
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
