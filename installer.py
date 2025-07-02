import subprocess
import sys


print("Installing Matplotlib, Scikit-learn, Pandas, PyTorch, and TorchVision...")

def install_dependencies():
    try:
        # Install both torch and torchvision
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib","scikit-learn","pandas", "torch", "torchvision",
                               "tqdm", "gc" "logging","memory_profiler", "pandas-profiling"])
        print("All required dependencies have been installed successfully.")
    except subprocess.CalledProcessError:
        print("An error occurred while installing PyTorch and TorchVision.")
