from utils.preprocessing import PatientLocalizer, COPAnalyizer, Worker
import opts
from preprocess.darknet.python.extract_bbox import set_gpu
import numpy as np
from tqdm import tqdm
import os

if __name__ == '__main__':
    opt = opts.parse_opts()

    opt.benchmark = opt.dataset in ["URFD", "MulticamFD"]

    if opt.mode == 'preprocess__frame':
        # patient localizer & interval selector
        set_gpu(opt.device_yolo)

        localizer = PatientLocalizer(darknet_api_home=opt.darknet_api_home)

        if opt.interval_sel not in ['COP', 'DAPs', '']:
            raise ValueError(
                'unrecognizable interveal selection method. got {}'.format(opt.interval_sel))

        interval_selector = None
        if opt.interval_sel == 'COP':
            interval_selector = COPAnalyizer(opt.meta_home, opt.fps)

        worker = Worker(localizer, interval_selector, opt)
        worker.run()
