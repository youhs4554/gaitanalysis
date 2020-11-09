#!/bin/bash
PYTHON=/opt/conda/envs/torch/bin/python

# sample duration = 8
for fold in {1..3}; do
    (
        echo "[CesleaFDD6] fold-$fold..." &&
        $PYTHON main.py --model_arch STCNet --task classification --backbone r2plus1d_34_8_ig65m --dataset CesleaFDD6 --fold $fold --batch_per_gpu 4 --sample_duration 8 --squad 2,3,3
    )
done