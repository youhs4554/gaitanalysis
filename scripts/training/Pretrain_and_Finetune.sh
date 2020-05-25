#!/bin/bash
PYTHON=/opt/conda/envs/torch/bin/python;

# pre-training first
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 $PYTHON main.py --backbone r2plus1d_18 --model_depth 18 --pretrained_path "" --data_root /data/GaitData/RawFrames --multi_gpu --with_segmentation --device_ids 0,1,2,3,4,5,6,7 --batch_size 32 --learning_rate 1e-4 --n_threads 8 --mode train --model_arch AGNet-pretrain --n_iter 201 --CV 5 &&

# fine-tunning ...
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 $PYTHON main.py --backbone r2plus1d_18 --model_depth 18 --pretrained_path "" --data_root /data/GaitData/RawFrames --multi_gpu --with_segmentation --device_ids 0,1,2,3,4,5,6,7 --pretrain_epoch 200 --batch_size 32 --learning_rate 1e-4 --n_threads 8 --mode train --model_arch AGNet --n_iter 201 --CV 5