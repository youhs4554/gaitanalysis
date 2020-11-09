#!/bin/bash

# sample duration = 64
for fold in {1..3}; do
    (
        echo "[UCF101] fold-$fold..." &&
        $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_34_32_kinetics --dataset UCF101 --fold $fold --batch_per_gpu 4 --sample_duration 64
    )
done