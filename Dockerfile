# Use official PyTorch base image (CPU or CUDA)
# GPU version
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime 
#CPU version
# FROM pytorch/pytorch:2.1.2-cpu

# Optional: Use CUDA image if you're using a GPU VM
# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set working directory inside the container
WORKDIR /app

# Install required system packages for torch.compile
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy local requirements and code
COPY requirements.txt .
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: expose port if it's a web app
# EXPOSE 8080

# Default command
CMD ["python", "Example_WPO_SGM.py"]