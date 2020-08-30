#!/bin/bash
PYTHON=/opt/conda/envs/torch/bin/python



#**************************************#
# backbone network variation experiments
#**************************************#

# backbone = r2plus1d_18 (sample duration = 16)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_18 --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16
    )
done

# backbone = r2plus1d_34 (sample_duration = 16); already included in other exp. -> skip!

# backbone = inflated_resnet50 (sample duration = 16)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone inflated_resnet50 --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16
    )
done

# backbone = inflated_resnet101 (sample duration = 16)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone inflated_resnet101 --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16
    )
done

#**************************************#
# sample duration variation experiments
#**************************************#

# sample duration = 16
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16
    )
done

# sample duration = 64
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 12 --sample_duration 64
    )
done



