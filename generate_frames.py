import pandas as pd
import os


def check_dir(dir):
    if not os.path.exists(dir):
        os.system(f'mkdir -p {dir}')
    return dir


video_home = check_dir('/data/GaitData/Video')
save_dir = check_dir('/data/GaitData/RawFrames')

dataset = pd.read_pickle(
    '/home/hossay/gaitanalysis/preprocess/data/person_detection_and_tracking_results_drop-merged.pkl')

video_names = list(set(dataset.vids))

for video_name in video_names:
    save_subdir = check_dir(os.path.join(
        save_dir, os.path.splitext(video_name)[0]))
    vpath = os.path.join(video_home, video_name+'.avi')
    os.system(
        f"ffmpeg -i {vpath} {save_subdir}/thumb%04d.jpg -hide_banner")
