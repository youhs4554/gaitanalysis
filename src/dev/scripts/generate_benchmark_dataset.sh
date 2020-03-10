#!/bin/bash
PYTHON=/home/hossay/anaconda3/envs/torch/bin/python

URFD_DETECTION_ROOT="../../preprocess/data/URFD"
MULTICAM_DETECTION_ROOT="../../preprocess/data/MulticamFD"

function merge_data() {
    $PYTHON -c "
import os
import pandas as pd

print('Merge {} dataset...'.format(os.path.basename('$1')))
data_frames = []
for dirpath, dirnames, filenames in os.walk('$1'):
    for i in filenames:
        p = os.path.join(dirpath, i)
        data_frames.append(pd.read_pickle(p))
data_tag = os.path.basename('$1')
save_path = os.path.join(
    os.path.dirname('$1'), f'person_detection_and_tracking_results_drop-{data_tag}.pkl')
pd.concat(data_frames).to_pickle(save_path)
#os.system('rm -rf {}'.format('$1'))
"

}

# URFD datasets
for i in {0..7}; do
    (
        echo "Starts on GPU-$i..."
        $PYTHON generate_dataset.py --data_gen --dataset URFD --mode preprocess__frame --video_home /data/FallDownData/URFD/video --input_file $URFD_DETECTION_ROOT/person_detection_and_tracking_results_drop-$i.pkl --darknet_api_home ../../preprocess/darknet --interval_sel "" --disable_tracking --device_yolo $i
    ) &
done
wait

merge_data $URFD_DETECTION_ROOT

# MulticamFD datasets
for i in {0..7}; do
    (
        echo "Starts on GPU-$i..."
        $PYTHON generate_dataset.py --data_gen --dataset MulticamFD --mode preprocess__frame --video_home /data/FallDownData/MulticamFD/video --input_file $MULTICAM_DETECTION_ROOT/person_detection_and_tracking_results_drop-$i.pkl --darknet_api_home ../../preprocess/darknet --interval_sel "" --disable_tracking --device_yolo $i
    ) &
done
wait

merge_data $MULTICAM_DETECTION_ROOT
