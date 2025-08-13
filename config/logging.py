## logging functions
import os
import logging
import math

def setup_logging(save_directory):
        """Configure logging for training."""
        log_file = os.path.join(save_directory, 'training.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info("=" * 100)
        logging.info("Starting new training session")

def save_training_slice_log(iter_time, loss_time, epoch, total_steps, max_mem, loss_value, save):
    '''
    Save the training log for the slice
    '''
    if save is not None:
        logging.info(f"Training started for epoch {epoch} / {total_steps}")
        logging.info(
        f"Step {epoch:04d} | Iter Time: {iter_time:.4f}s | "
        f"Loss Time: {loss_time:.4f}s | Loss: {loss_value:.6f} | "
        f"Max Mem: {max_mem:.2f} GB | "
        )  

def save_log(factornet, step, loss_value, optimizer, save_directory):
        """Log training progress."""
        print(f"Step {step}, Loss: {loss_value:.3e}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.2e}")
        
        # Save to CSV log
        with open(os.path.join(save_directory, "loss_log.csv"), "a") as f:
            f.write(f"{step},{loss_value},{current_lr}\n")
        
        # Evaluate model
        val_loss = self.evaluate_model(factornet, self.args.test_samples_size)
        logging.info(f"Step {step:04d} | Loss: {loss_value:.6f} | Val Loss: {val_loss:.6f}")
        
        # Update scheduler for reduce_on_plateau
        if self.args.scheduler_type == 'reduce_on_plateau' and not (math.isnan(val_loss) or math.isinf(val_loss)):
            # Get scheduler from optimizer (assuming single scheduler)
            # This is a simplified approach - you may need to pass scheduler here
            pass
