#!/bin/bash
PYTHON=/opt/conda/envs/torch/bin/python

# sample duration = 32
for fold in {1..3}; do
    (
        echo "[HMDB51] fold-$fold..." &&
        $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_34_32_ig65m --dataset HMDB51 --fold $fold --batch_per_gpu 4 --sample_duration 32 
    )
done