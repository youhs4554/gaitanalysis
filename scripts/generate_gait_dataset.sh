#!/bin/bash

GAIT_DETECTION_ROOT="../../preprocess/data/Gaitparams_PD"

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
os.system('rm -rf {}'.format('$1'))
"

}

# GAIT datasets
for i in {0..7}; do
    (
        echo "Starts on GPU-$i..."
        $PYTHON generate_dataset.py --data_gen --dataset Gaitparams_PD --mode preprocess__frame --video_home /data/GaitData/Video --input_file $GAIT_DETECTION_ROOT/person_detection_and_tracking_results_drop-$i.pkl --target_file ../../preprocess/data/targets_dataframe.pkl --darknet_api_home ../../preprocess/darknet --meta_home /data/GaitData/MetaData_converted --fps 24 --device_yolo $i
    ) &
done
wait

merge_data $GAIT_DETECTION_ROOT
