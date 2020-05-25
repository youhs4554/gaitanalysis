#!/bin/bash
PYTHON=/home/hossay/anaconda3/envs/torch/bin/python;

# backbone : 3D-ResNet50, model : naive, dropout : 0.3
$PYTHON main.py --backbone 3D-resnet --model_depth 50 --pretrained_path pretrained/resnet-50-kinetics.pth --multi_gpu --device_ids 0,1,2,3,4,5,6,7 --batch_size 32 --learning_rate 1e-2 --n_threads 8 --mode train --model_arch naive --n_iter 31 --CV 5 --drop_rate 0.3 &&

# backbone : 3D-ResNet50, model : SPP, merge : 1x1_C, dropout : 0.3
$PYTHON main.py --backbone 3D-resnet --model_depth 50 --pretrained_path pretrained/resnet-50-kinetics.pth --multi_gpu --device_ids 0,1,2,3,4,5,6,7 --batch_size 32 --learning_rate 1e-2 --n_threads 8 --mode train --model_arch SPP --merge_type 1x1_C --n_iter 31 --CV 5 --drop_rate 0.3