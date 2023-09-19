CUDA_DEVICE=7

K_START_FACTOR=0.10
K_INC_FACTOR=0.05
K_END_FACTOR=0.90



for k in $(seq $K_START_FACTOR $K_INC_FACTOR $K_END_FACTOR);
do
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset DD --pool_ratio "$k" --num_pool 2 --epochs 100
      CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset PTC_MR --pool_ratio "$k" --num_pool 2 --epochs 100
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset PROTEINS --pool_ratio "$k" --num_pool 2 --epochs 100
          #CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset IMDB-BINARY --pool_ratio "$k" --num_pool 2 --epochs 100
            #CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset FRANKENSTEIN --pool_ratio "$k" --num_pool 2 --epochs 100
          done
