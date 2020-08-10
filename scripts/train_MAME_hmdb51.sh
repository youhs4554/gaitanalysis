#!/bin/bash
PYTHON=/opt/conda/envs/torch/bin/python

#**************************************#
# sample duration variation experiments
#**************************************#

# sample duration = 32 (w/ mixup)
for fold in {1..3}; do
    (
        echo "[HMDB51] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch DefaultMAMENet --task classification --backbone r2plus1d_34_32_kinetics --dataset HMDB51 --fold $fold --batch_size 24 --sample_duration 32 --mixup
    )
done

# sample duration = 16 (w/ mixup)
for fold in {1..3}; do
    (
        echo "[HMDB51] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch DefaultMAMENet --task classification --backbone r2plus1d_34_32_kinetics --dataset HMDB51 --fold $fold --sample_duration 16 --mixup
    )
done

# sample duration = 8 (w/ mixup)
for fold in {1..3}; do
    (
        echo "[HMDB51] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch DefaultMAMENet --task classification --backbone r2plus1d_34_32_kinetics --dataset HMDB51 --fold $fold --sample_duration 8 --mixup
    )
done

#**************************************#
# backbone network variation experiments
#**************************************#

# backbone = r2plus1d_18 (w/ mixup, sample duration = 16)
for fold in {1..3}; do
    (
        echo "[HMDB51] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch DefaultMAMENet --task classification --backbone r2plus1d_18 --dataset HMDB51 --fold $fold --sample_duration 16 --mixup
    )
done

# backbone = r2plus1d_34_8_kinetics (w/ mixup, sample duration = 8)
for fold in {1..3}; do
    (
        echo "[HMDB51] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch DefaultMAMENet --task classification --backbone r2plus1d_34_8_kinetics --dataset HMDB51 --fold $fold --sample_duration 8 --mixup
    )
done

# backbone = inflated_resnet101 (w/ mixup, sample duration = 16)
for fold in {1..3}; do
    (
        echo "[HMDB51] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch DefaultMAMENet --task classification --backbone inflated_resnet101 --dataset HMDB51 --fold $fold --sample_duration 16 --mixup
    )
done


#*****************************#
# mixup abiliation experiments
#*****************************#

# w/o mixup
for fold in {1..3}; do
    (
        echo "[HMDB51] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch DefaultMAMENet --task classification --backbone r2plus1d_34_32_kinetics --dataset HMDB51 --fold $fold --sample_duration 16
    )
done
