# Use official PyTorch base image (CPU or CUDA)
# GPU version
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
#CPU version
# FROM pytorch/pytorch:2.1.2-cpu

# Optional: Use CUDA image if you're using a GPU VM
# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set working directory inside the container
WORKDIR /app

# Set environment variables for non-interactive apt
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/PDT
# Set PyTorch CUDA memory allocation configuration
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,expandable_segments:True


# Install required system packages for torch.compile
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy local requirements and code
COPY requirements.txt .
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e . 

# Optional: expose port if it's a web app
# EXPOSE 8080

ENTRYPOINT ["torchrun"]

CMD ["--nproc_per_node=2", "train_wpo_sgm.py"]