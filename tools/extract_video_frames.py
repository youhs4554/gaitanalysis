import os, sys
from glob import glob
from natsort import natsorted
from tqdm import tqdm

def save_frames(root, ext='.avi'): 
    root = root.rstrip('/') 
    files = natsorted(glob(root + '/*/*')) 
    for vpath in tqdm(files):
        lab, vname = vpath.split('/')[-2:]
        vname = vname.replace(ext, '')
        frame_dir = os.path.join(os.path.dirname(root), 'frames', lab, vname) 
        os.system('mkdir -p {}'.format(frame_dir)) 
        os.system(f"ffmpeg -i {vpath} {frame_dir}/thumb%04d.jpg -hide_banner")

if __name__ == '__main__':
    root = sys.argv[1]
    save_frames(root)
