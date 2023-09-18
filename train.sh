CUDA_DEVICE=3


#CUDA_VISIBLE_DEVICES=3 python train.py --dataset DD --pool_ratio 0.10 --num_pool 2 --epochs 1000 &
#CUDA_VISIBLE_DEVICES=4 python train.py --dataset PTC_MR --pool_ratio 0.10 --num_pool 2 --epochs 1000 &
#CUDA_VISIBLE_DEVICES=6 python train.py --dataset PROTEINS --pool_ratio 0.10 --num_pool 2 --epochs 1000 &
#CUDA_VISIBLE_DEVICES=7 python train.py --dataset FRANKENSTEIN --pool_ratio 0.10 --num_pool 2 --epochs 1000 &


CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset DD --pool_ratio 0.10 --num_pool 2 --epochs 1000
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset PTC_MR --pool_ratio 0.10 --num_pool 2 --epochs 1000
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset PROTEINS --pool_ratio 0.10 --num_pool 2 --epochs 1000
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset IMDB-BINARY --pool_ratio 0.10 --num_pool 2 --epochs 1000
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset FRANKENSTEIN --pool_ratio 0.10 --num_pool 2 --epochs 1000
