#!/bin/bash
PYTHON=/home/hossay/anaconda3/envs/torch/bin/python;

# backbone : 3D-ResNet50, model : HPP, merge : None, n_groups : 1, dropout : 0.0, attention : True
$PYTHON main.py --backbone 3D-resnet --model_depth 50 --pretrained_path pretrained/3D-ResNets/resnet-50-kinetics.pth --multi_gpu --device_ids 0,1,2,3,4,5,6,7 --batch_size 32 --learning_rate 1e-1 --n_threads 8 --mode train --model_arch HPP --merge_type 1x1_C --n_iter 201 --CV 5 --n_groups 1 --drop_rate 0.0 --attention &&

# backbone : 3D-ResNet50, model : HPP, merge : None, n_groups : 2, dropout : 0.0, attention : True
$PYTHON main.py --backbone 3D-resnet --model_depth 50 --pretrained_path pretrained/3D-ResNets/resnet-50-kinetics.pth --multi_gpu --device_ids 0,1,2,3,4,5,6,7 --batch_size 32 --learning_rate 1e-1 --n_threads 8 --mode train --model_arch HPP --merge_type 1x1_C --n_iter 201 --CV 5 --n_groups 2 --drop_rate 0.0 --attention &&

# backbone : 3D-ResNet50, model : HPP, merge : None, n_groups : 3, dropout : 0.0, attention : True
$PYTHON main.py --backbone 3D-resnet --model_depth 50 --pretrained_path pretrained/3D-ResNets/resnet-50-kinetics.pth --multi_gpu --device_ids 0,1,2,3,4,5,6,7 --batch_size 32 --learning_rate 1e-1 --n_threads 8 --mode train --model_arch HPP --merge_type 1x1_C --n_iter 201 --CV 5 --n_groups 3 --drop_rate 0.0 --attention &&

# backbone : 3D-ResNet50, model : HPP, merge : None, n_groups : 4, dropout : 0.0, attention : True
$PYTHON main.py --backbone 3D-resnet --model_depth 50 --pretrained_path pretrained/3D-ResNets/resnet-50-kinetics.pth --multi_gpu --device_ids 0,1,2,3,4,5,6,7 --batch_size 32 --learning_rate 1e-1 --n_threads 8 --mode train --model_arch HPP --merge_type 1x1_C --n_iter 201 --CV 5 --n_groups 4 --drop_rate 0.0 --attention