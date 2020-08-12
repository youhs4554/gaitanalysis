#!/bin/bash
PYTHON=/opt/conda/envs/torch/bin/python

#**************************************#
# sample duration variation experiments
#**************************************#

# sample duration = 64 (w/ mixup)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 12 --sample_duration 64 --mixup
    )
done

# sample duration = 16 (w/ mixup)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --sample_duration 16 --mixup
    )
done

#**************************************#
# backbone network variation experiments
#**************************************#

# backbone = r2plus1d_18 (w/ mixup, sample duration = 16)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_18 --dataset UCF101 --fold $fold --sample_duration 16 --mixup
    )
done

# backbone = inflated_resnet101 (w/ mixup, sample duration = 16)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone inflated_resnet101 --dataset UCF101 --fold $fold --sample_duration 16 --mixup
    )
done


#*****************************#
# mixup abiliation experiments
#*****************************#

# w/o mixup (sample duration = 64)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 12 --sample_duration 64
    )
done

# w/o mixup (sample duration = 16)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --sample_duration 16
    )
done
