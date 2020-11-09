#!/bin/bash


#**************************************#
# STCB(x1) location experiments
#**************************************#

# squad = [1,0,0]
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_per_gpu 8 --sample_duration 16 --squad 1,0,0
    )
done

# squad = [0,1,0]
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_per_gpu 8 --sample_duration 16 --squad 0,1,0
    )
done

# squad = [0,0,1]
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_per_gpu 8 --sample_duration 16 --squad 0,0,1
    )
done

#**************************************#
# STCB counts experiments
#**************************************#

# =============        
#    2-STCB
# =============        
# squad = [1,1,0] -> 2-STCB
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_per_gpu 8 --sample_duration 16 --squad 1,1,0
    )
done

# =============        
#    4-STCB
# =============        

# squad = [2,2,0] -> 4-STCB
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_per_gpu 8 --sample_duration 16 --squad 2,2,0
    )
done

# =============        
#    8-STCB
# =============        
# squad = [2,3,3] -> 8-STCB
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_per_gpu 8 --sample_duration 16 --squad 2,3,3
    )
done