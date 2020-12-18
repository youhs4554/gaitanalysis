from glob import glob
import sys
import os
from natsort import natsorted
import numpy as np

if __name__ == '__main__':

    n_split = 8

    meta = {
        "URFD": ["/data/FallDownData/URFD/frames", "URFD_evalList"],
        "MulticamFD": ["/data/FallDownData/MulticamFD/frames", "MulticamFD_evalList"],
    }

    for x in ["URFD", "MulticamFD"]:
        frame_root, save_dir = meta.get(x)
        os.system("mkdir -p {}".format(save_dir))

        # listup all .jpg files
        all_files = natsorted(glob(frame_root+'/*/*/*.jpg'))

        for i, split in enumerate(np.array_split(all_files, n_split)):
            with open(os.path.join(save_dir, f"chunk{i:02d}.txt"), 'w') as outfile:
                outfile.write('\n'.join(split))
