import pandas as pd
import numpy as np
import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm

if __name__ == '__main__':
    ucf_home = "/data/torch_data/UCF-101/frames"
    hmdb_home = "/data/torch_data/HMDB51/frames"

    # listup all detection results in .txt file
    ucf_files = natsorted(glob(ucf_home+'/*/*/*.txt'))
    hmdb_files = natsorted(glob(hmdb_home+'/*/*/*.txt'))

    ucf_anno_file = '/data/torch_data/UCF-101/detection_yolov4.txt'
    hmdb_anno_file = '/data/torch_data/HMDB51/detection_yolov4.txt'

    def merge_results(filenames, anno_file):
        frame_home = '/'.join(filenames[0].split('/')[:-3])
        # guarantee format that ends with '/'
        frame_home = frame_home.rstrip('/') + '/'

        with open(anno_file, 'w') as outfile:
            for fname in tqdm(filenames):
                with open(fname) as infile:
                    lines = infile.readlines()
                    if not lines:
                        outfile.write(
                            fname[len(frame_home):] + ' ' + " ".join(["0", ] + ["0.0"]*4) + '\n')
                    else:
                        for line in lines:
                            outfile.write(fname[len(frame_home):] + ' ' + line)

    merge_results(ucf_files, ucf_anno_file)    # ucf
    merge_results(hmdb_files, hmdb_anno_file)  # hmdb
