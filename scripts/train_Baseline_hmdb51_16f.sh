#!/bin/bash

# sample duration = 16
for fold in {1..3}; do
    (
        echo "[HMDB51] fold-$fold..." &&
        $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_34_32_kinetics --dataset HMDB51 --fold $fold --batch_per_gpu 8 --sample_duration 16
    )
done