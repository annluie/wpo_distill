# utils.py
#
# some of the original utilities used by FFJORD
import os
import math
from numbers import Number
import logging, argparse
import torch

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WPO-SGM Model Parallel Training")
    
    # Data and model parameters
    parser.add_argument('--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 
                                          'moons', '2spirals', 'checkerboard', 'rings',
                                          'swissroll_6D_xy1', 'cifar10'], 
                       default='cifar10', help='Dataset to use')
    parser.add_argument('--depth', type=int, default=5, 
                       help='Number of hidden layers in score network')
    parser.add_argument('--hiddenunits', type=int, default=64, 
                       help='Number of nodes per hidden layer')
    
    # Training parameters
    parser.add_argument('--niters', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--stability', type=float, default=0.01, help='Stability parameter')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_type', type=str, default='one_cycle',
                       choices=['reduce_on_plateau', 'cosine_annealing', 'one_cycle', 'step'],
                       help='Type of LR scheduler to use')
    
    # Data sizes
    parser.add_argument('--train_kernel_size', type=int, default=50, help='Number of centers')
    parser.add_argument('--train_samples_size', type=int, default=500, help='Training sample size')
    parser.add_argument('--test_samples_size', type=int, default=5, help='Test sample size')
    
    # I/O parameters
    parser.add_argument('--save', type=str, default='cifar10_experiments/', 
                       help='Save directory')
    parser.add_argument('--load_model_path', type=str, default=None, 
                       help='Path to load model checkpoint')
    parser.add_argument('--load_centers_path', type=str, default=None, 
                       help='Path to load centers')
    
    return parser.parse_args()

#----------------------- SAVE FUNCTIONS -------------------
def create_save_dir_from_args(args):
    '''
    Create a subfolder to save all the outputs
    '''
    if args.save is not None:
        subfolder = os.path.join(
                args.save,
                f"sample_size{args.train_samples_size}",
                f"test_size{args.test_samples_size}",
                f"batch_size{args.batch_size}",
                f"centers{args.train_kernel_size}",
                f"lr{args.lr}_hu{args.hiddenunits}_stab{args.stability}_stabveropt"
                #f"lr{lr}_hu{hidden_units}_stab{stab}"
            )
        os.makedirs(subfolder, exist_ok=True)
    else:
        subfolder = os.path.join(
                f"sample_size{args.train_samples_size}",
                f"test_size{args.test_samples_size}",
                f"batch_size{args.batch_size}",
                f"centers{args.train_kernel_size}",
                f"lr{args.lr}_hu{args.hiddenunits}_stab{args.stability}_stabveropt"
                #f"lr{lr}_hu{hidden_units}_stab{stab}"
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

def save_checkpoint(self, model_parallel, optimizer, scheduler, step):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dicts': [net.state_dict() for net in model_parallel.factornets],
            'device_ids': model_parallel.device_ids,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': step,
        }
        
        checkpoint_path = os.path.join(self.save_directory, "latest_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"💾 Saved checkpoint at step {step}")

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

#----------OLD STUFF--------------------------------
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0
        self.sum = 0  #

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.sum += val
        self.val = val

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#
# def inf_generator(iterable):
#     """Allows training with DataLoaders in a single infinite loop:
#         for i, (x, y) in enumerate(inf_generator(train_loader)):
#     """
#     iterator = iterable.__iter__()
#     while True:
#         try:
#             yield iterator.__next__()
#         except StopIteration:
#             iterator = iterable.__iter__()
#
#
# def save_checkpoint(state, save, epoch):
#     if not os.path.exists(save):
#         os.makedirs(save)
#     filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
#     torch.save(state, filename)
#
#
# def isnan(tensor):
#     return (tensor != tensor)
#
#
# def logsumexp(value, dim=None, keepdim=False):
#     """Numerically stable implementation of the operation
#     value.exp().sum(dim, keepdim).log()
#     """
#     if dim is not None:
#         m, _ = torch.max(value, dim=dim, keepdim=True)
#         value0 = value - m
#         if keepdim is False:
#             m = m.squeeze(dim)
#         return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
#     else:
#         m = torch.max(value)
#         sum_exp = torch.sum(torch.exp(value - m))
#         if isinstance(sum_exp, Number):
#             return m + math.log(sum_exp)
#         else:
#             return m + torch.log(sum_exp)
#
#
#
