import os, sys
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import pathlib
import subprocess

def save_frames(root, ext='.avi', sample_rate=5): 
    root = root.rstrip('/') 
    files = natsorted(glob(root + '/*/*')) 
    for vpath in tqdm(files):
        lab, vname = vpath.split('/')[-2:]
        vname = vname.replace(ext, '')
        frame_dir = os.path.join(os.path.dirname(root), 'frames', lab, vname)#.replace("(", "\(").replace(")", "\)")
        #pathlib.Path(frame_dir).mkdir(exist_ok=True)
        subprocess.call(["mkdir", "-p", frame_dir])
        subprocess.call(["ffmpeg", "-i", vpath, "-vf", f"select=not(mod(n\,{sample_rate}))", "-vsync", "vfr", "-q:v", "2", f"{frame_dir}/thumb%04d.jpg", "-hide_banner"])
        #os.system(f"ffmpeg -i {vpath} {frame_dir}/thumb%04d.jpg -hide_banner")

if __name__ == '__main__':
    root = sys.argv[1]
    sample_rate = sys.argv[2]
    save_frames(root, sample_rate=sample_rate)
