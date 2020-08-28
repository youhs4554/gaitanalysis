from PIL import Image
import numpy as np
import glob, os
from natsort import natsorted
import tqdm
import cv2
import cv2


def main(args):
    frame_home = os.path.join(args.data_root, "tvl1_flow")
    video_home = os.path.join(args.data_root, "video")

    assert os.path.exists(frame_home) and os.path.exists(
        video_home
    ), "opticla flow frame dir must have suffix of 'tvl1_flow' and video dir must have suffix of 'video'"

    frame_sub_dirs = glob.glob(os.path.join(frame_home, "*", "*"))  # u,v pairs
    frame_sub_dirs = list(
        filter(lambda x: not os.path.splitext(x)[-1], frame_sub_dirs)
    )  # exclude junks(*.bin)
    u_frame_dirs = natsorted(
        list(
            filter(
                lambda x: os.path.dirname(x[len(frame_home) :]) == "/u", frame_sub_dirs
            )
        )
    )
    v_frame_dirs = natsorted(list(set(frame_sub_dirs) - set(u_frame_dirs)))

    org_video_names = natsorted(glob.glob(os.path.join(video_home, "*", "*")))
    org_video_names = [x[len(video_home) + 1 :] for x in org_video_names]
    labs, names = zip(*[os.path.split(x) for x in org_video_names])

    save_root = os.path.join(os.path.dirname(frame_home), "video" + "_flow")
    os.system(f"mkdir -p {save_root}")

    for ud, vd in tqdm.tqdm(list(zip(u_frame_dirs, v_frame_dirs))):
        uf = [os.path.join(ud, p) for p in os.listdir(ud)]
        vf = [os.path.join(ud, p) for p in os.listdir(vd)]

        vname = os.path.basename(ud) + ".avi"
        if vname.startswith("v_HandstandPushups"):
            # exception case
            vname = vname.replace("v_HandstandPushups", "v_HandStandPushups")
        cls = labs[names.index(vname)]

        org_video = os.path.join(video_home, cls, vname)
        org_vcap = cv2.VideoCapture(org_video)
        fps = org_vcap.get(cv2.CAP_PROP_FPS)

        save_dir = os.path.join(save_root, cls)
        os.system(f"mkdir -p {save_dir}")
        vname = os.path.join(save_dir, vname)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        h, w = cv2.imread(uf[0], 0).shape
        out = cv2.VideoWriter(vname, fourcc, fps, (w, h))

        for u, v in zip(uf, vf):
            u = cv2.imread(u, 0)
            v = cv2.imread(v, 0)
            z = np.zeros_like(u)
            compound = cv2.merge((u, v, z))
            out.write(compound)

        out.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data/torch/UCF-101")
    args = parser.parse_args()
    main(args)
