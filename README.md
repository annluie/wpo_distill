This project uses code from "Wasserstein proximal operators describe score-based generative models and resolve memorization".
https://arxiv.org/abs/2402.06162

Running the code:
First, build the docker image: docker build -t your-image-name .
Then, run using the number of gpus you have: docker run --gpus all \
  -v $(pwd):/app \
  your_image_name \
  python train_wpo_sgm.py \
  --data cifar10 \
  --depth 5 \
  --hiddenunits 64 \
  --niters 50000 \
  --batch_size 2 \
  --lr 0.002 \
  --save cifar10_experiments/ \
  --train_kernel_size 1000 \
  --train_samples_size 50000 \
  --test_samples_size 500 \
  --load_model_path None \
  --load_centers_path None


   
