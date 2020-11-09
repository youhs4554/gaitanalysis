#!/bin/bash

# sample duration = 8
for fold in {1..3}; do
    (
        echo "[CesleaFDD6] fold-$fold..." &&
        $PYTHON main.py --model_arch FineTunedConvNet --task classification --backbone r2plus1d_34_8_ig65m --dataset CesleaFDD6 --fold $fold --batch_per_gpu 4 --sample_duration 8 
    )
done