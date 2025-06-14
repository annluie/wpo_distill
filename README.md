This project uses code from "Wasserstein proximal operators describe score-based generative models and resolve memorization".
https://arxiv.org/abs/2402.06162

Running the code:
First, build the docker image: docker build -t your-image-name .
Then, run using the number of gpus you have: docker run --gpus all --rm your-image-name \
  torchrun --nproc_per_node=<NUM_GPUS> --standalone train_wpo_sgm.py --arg1 val1 --arg2 val2


