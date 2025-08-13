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
  --niters 10000 \
  --batch_size 2 \
  --lr 0.002 \
  --save cifar10_experiments/ \
  --train_kernel_size 100 \
  --train_samples_size 500 \
  --test_samples_size 5 \
  --load_model_path None \
  --load_centers_path None

--data cifar10 
  --depth 5 
  --hiddenunits 64 \
  --niters 10000 \
  --batch_size 2 \
  --lr 0.002 \
  --save cifar10_experiments/ \
  --train_kernel_size 100 \
  --train_samples_size 500 \
  --test_samples_size 5 \
  --load_model_path None \
  --load_centers_path None
   
--data cifar10
--depth 5
--hiddenunits 64
--niters 10000
--batch_size 4
--lr 0.002
--save cifar10_experiments/
--train_kernel_size 100
--train_samples_size 500
--test_samples_size 5
--load_model_path None
--load_centers_path None

--data
cifar10
--depth
5
--hiddenunits
64
--niters
10000
--batch_size
4
--lr
0.002
--save
cifar10_experiments/
--train_kernel_size
100
--train_samples_size
500
--test_samples_size
5
--load_model_path
None
--load_centers_path
None
Running the code:
First, build the docker image: docker build -t your-image-name .
Then, run using the number of gpus you have: docker run --gpus all \
  -v $(pwd):/app \
  your_image_name \
  python train_wpo_sgm.py \
  --data cifar10 \
  --depth 5 \
  --hiddenunits 64 \
  --niters 10000 \
  --batch_size 2 \
  --lr 0.002 \
  --save cifar10_experiments/ \
  --train_kernel_size 100 \
  --train_samples_size 500 \
  --test_samples_size 5 \
  --load_model_path None \
  --load_centers_path None

--data cifar10 
  --depth 5 
  --hiddenunits 64 \
  --niters 10000 \
  --batch_size 2 \
  --lr 0.002 \
  --save cifar10_experiments/ \
  --train_kernel_size 100 \
  --train_samples_size 500 \
  --test_samples_size 5 \
  --load_model_path None \
  --load_centers_path None
   
--data cifar10
--depth 5
--hiddenunits 64
--niters 10000
--batch_size 4
--lr 0.002
--save cifar10_experiments/
--train_kernel_size 100
--train_samples_size 500
--test_samples_size 5
--load_model_path None
--load_centers_path None

--data
cifar10
--depth
5
--hiddenunits
64
--niters
10000
--batch_size
4
--lr
0.002
--save
cifar10_experiments/
--train_kernel_size
100
--train_samples_size
500
--test_samples_size
5
--load_model_path
None
--load_centers_path
None

python train_wpo_sgm_stable.py --niters 50 --batch_size 64 --lr 0.0001 --stability 0.001 --train_kernel_size 200 --train_samples_size 50000 --test_samples_size 50
conda create -n wpo_env python=3.11
cd wpo_distill
conda activate wpo_env
pip install -r requirements.txt 
python3 train_wpo_sgm_stable_parallel.py --niters 50 --batch_size 32 --train_kernel_size 1000 --train_samples_size 50000 --test_samples_size 5 --lr 0.0001 --stability 0.001 --hiddenunits 64 --scheduler_type 'step'