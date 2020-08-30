#!/bin/bash
PYTHON=/opt/conda/envs/torch/bin/python

#**************************************#
# backbone network variation experiments
#**************************************#

# backbone = r2plus1d_18 (sample duration = 16)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_18 --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16
    )
done

# backbone = r2plus1d_34 (sample_duration = 16); already included in other exp. -> skip!

# backbone = inflated_resnet50 (sample duration = 16)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone inflated_resnet50 --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16
    )
done

# backbone = inflated_resnet101 (sample duration = 16)
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone inflated_resnet101 --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16
    )
done


#**************************************#
# STCB(x1) location experiments
#**************************************#

# squad = [1,0,0]
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16 --squad 1,0,0
    )
done

# squad = [0,1,0]
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16 --squad 0,1,0
    )
done

# squad = [0,0,1]
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16 --squad 0,0,1
    )
done


#**************************************#
# STCB squad experiments
#**************************************#

# =============        
#    2-STCB
# =============        

# squad = [0,1,1] -> 2-STCB
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16 --squad 0,1,1
    )
done

# squad = [1,0,1] -> 2-STCB
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16 --squad 1,0,1
    )
done

# squad = [1,1,0] -> 2-STCB
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16 --squad 1,1,0
    )
done

# =============        
#    4-STCB
# =============        

# squad = [2,1,1] -> 4-STCB
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16 --squad 2,1,1
    )
done

# squad = [1,2,1] -> 4-STCB
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16 --squad 1,2,1
    )
done

# squad = [1,1,2] -> 4-STCB
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16 --squad 1,1,2
    )
done

#**************************************#
# sample duration variation experiments
#**************************************#

# sample duration = 16
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 60 --sample_duration 16
    )
done

# sample duration = 64
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        CUDA_VISIBLE_DEVICES=3,4,5 $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_size 12 --sample_duration 64
    )
done



