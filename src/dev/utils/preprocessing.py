import re
from tqdm import tqdm
import csv
from collections import namedtuple
import cv2
from preprocess.preprocess_metadata import create_target_df
from preprocess.tracking_patient import run_tracker
from preprocess.darknet.python.extract_bbox import *
from itertools import islice
from PIL import Image
import torch
import pandas as pd
import os
import numpy as np
import torch.nn.functional as F

# # Add COP Files containing start/end timing info


class COPAnalyizer(object):
    def __init__(self, meta_home='/data/GaitData/MetaData_converted', fps=24):
        self.meta_home = meta_home
        self.fps = fps

    def get_interval(self, vid, return_index=True):
        regex = re.compile(r'(.*)_test_(.*)_trial_(.*)')
        pid, test_ix, trial_ix = regex.search(vid).groups()

        cop_file_path = os.path.join(
            self.meta_home, f'{pid}_cop_test_{test_ix}_trial_{trial_ix}.txt')

        fp = open(cop_file_path, 'r', encoding='utf-8')
        reader = csv.reader(fp, delimiter='\t')

        tmp = []

        for i, line in enumerate(reader):
            if i == 0:
                continue  # skip first line!
            # parse each line
            _, time, _, pos, _ = line

            if not time:
                break  # if we encounter empty str, break this loop!
            time, pos = [eval(x) for x in [time, pos]]

            tmp.append([time, pos])

        tmp = [x[0] for x in tmp if x[0] > 0 and x[1] <= 350]

        start, end = tmp[0], tmp[-1]
        if return_index:
            start, end = [int(self.fps * t) for t in [start, end]]

        return start, end


class HumanScaleAnalyizer(object):
    def __init__(self, opt):
        self.opt = opt

    def check_interval(self, pos):

        # determine given pos is in interval

        cond = pos == "hhi"

        if cond:
            return True
        else:
            return False


# In[10]:


def check_direction(positions):
    if positions[0] < positions[-1]:
        return 'approach'
    else:
        return 'leave'


def convert_path(p, darknet_home):
    # convert path to use darknet api
    try:
        p = os.path.join(darknet_home, p)
    except:
        print('catch')
    # convert python 3 string -> python2 bytes
    p = p.encode()

    return p


class PatientLocalizer(object):
    path_dict = {"cfg_path": "cfg/yolov3.cfg",
                 "weight_path": "yolov3.weights",
                 "anno_path": "cfg/coco.data"}

    def __init__(self, darknet_api_home=darknet_home):
        for p_key, p_val in self.path_dict.items():
            self.path_dict[p_key] = convert_path(p_val, darknet_api_home)

        self.im = IMAGE()
        self.net = load_net(
            self.path_dict["cfg_path"], self.path_dict["weight_path"], 0)
        self.meta = load_meta(self.path_dict["anno_path"])

    def __len__(self):
        return self.c

    def detect_from_video(self, video):
        video_name = os.path.basename(os.path.splitext(video)[0])

        cap = cv2.VideoCapture(video)

        self.c = 0  # counter
        idx = 0     # real index

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            idx += 1  # real idx

            res = detect(self.net, self.meta, self.im, frame)
            person_BB = []
            for i in range(len(res)):
                if res[i]:
                    lab, conf, (bx, by, bw, bh) = res[i]
                    if lab.decode() == 'person':
                        p_BB = [bx, by, bw, bh]
                        person_BB.append(p_BB)

            if not person_BB:
                continue

            self.c += 1  # counter

            res = dict(person_pos=person_BB,
                       idx=idx, vid=video_name)
            nt = namedtuple('res', res.keys())(
                *res.values())  # convert dict -> named tuple

            yield nt

        cap.release()


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


class Worker(object):
    def __init__(self, localizer, interval_selector, opt):

        self.opt = opt

        if opt.data_gen:
            # input data part
            input_lines = ['']

            if not os.path.exists(opt.chunk_vid_home):
                os.system(f'mkdir -p {opt.chunk_vid_home}')
                all_video_files = [os.path.join(opt.video_home, v) + '\n' for v in os.listdir(opt.video_home) if
                                   not v.startswith('vid') and os.path.exists(os.path.join(opt.meta_home, '{0}_cop_{1}_{2}_{3}_{4}.txt'.format(*os.path.splitext(v)[0].split('_'))))]
                for ix, partial in enumerate(chunk(all_video_files, math.ceil(len(all_video_files)/opt.chunk_parts))):
                    with open(f'{opt.chunk_vid_home}/vids-part{ix}.txt', 'w') as f:
                        f.writelines(partial)

            with open(f'{opt.chunk_vid_home}/vids-part{opt.device_yolo}.txt') as f:
                video_files = list(x.strip() for x in f.readlines())

            pbar = tqdm(video_files)
            for video in pbar:
                pbar.set_description(
                    f"Processing ***** {os.path.basename(video)}")
                self._run(video, localizer, interval_selector, input_lines)

            prefix, ext = os.path.splitext(opt.input_file)
            input_file_path = prefix + '-' + str(opt.device_yolo) + ext

            pd.DataFrame([x.split('\t') for x in input_lines[0].strip().split('\n')],
                         columns=['vids', 'idx', 'pos']).to_pickle(input_file_path)

            if opt.device_yolo == 0:
                # for prevent safe file saving
                # todo. parameterize column selection func with opt
                # target data part
                create_target_df(meta_home=opt.meta_home, save_path=opt.target_file,
                                 single_cols=['Velocity', 'Cadence'],
                                 pair_cols=['Cycle Time(sec)', 'Stride Length(cm)', 'Stride Velocity', 'HH Base Support(cm)',
                                            'Swing Time(sec)', 'Stance Time(sec)', 'Double Supp. Time(sec)',
                                            'Toe In / Out',
                                            "Stride Length Std Dev", "Stride Time Std Dev"]
                                 )

                target_df = pd.read_pickle(opt.target_file)

                old_columns = [('Stride Length(cm)', 'Stride Velocity')]
                new_columns = ['Stride Time(sec)']

                for new_col, old_cols in zip(new_columns, old_columns):
                    for tail in ['/L', '/R']:
                        target_df[new_col + tail] = target_df[old_cols[0]+tail] / \
                            target_df[old_cols[1]+tail]

                old_columns = [('Stride Length Std Dev', 'Stride Length(cm)'),
                               ('Stride Time Std Dev', 'Stride Time(sec)')]

                new_columns = ['Stride Length(cm)^2',
                               'Stride Time(sec)^2']

                for new_col, old_cols in zip(new_columns, old_columns):
                    for tail in ['/L', '/R']:
                        target_df[new_col + tail] = target_df[old_cols[0] +
                                                              tail]**2 + target_df[old_cols[1]+tail]**2

                old_columns = [('Stride Length Std Dev', 'Stride Length(cm)'),
                               ('Stride Time Std Dev', 'Cycle Time(sec)'),
                               ('Swing Time(sec)', 'Cycle Time(sec)'),
                               ('Stance Time(sec)', 'Cycle Time(sec)'),
                               ('Double Supp. Time(sec)', 'Cycle Time(sec)')]

                new_columns = ['CV Stride Length', 'CV Stride Time',
                               "Swing % of Cycle", "Stance % of Cycle", "Double Supp % Cycle"]

                for new_col, old_cols in zip(new_columns, old_columns):
                    for tail in ['/L', '/R']:
                        target_df[new_col + tail] = 100 * target_df[old_cols[0] +
                                                                    tail] / target_df[old_cols[1]+tail]

                target_df['Stride Length Var/L'] = target_df['Stride Length Std Dev/L']**2
                target_df['Stride Length Var/R'] = target_df['Stride Length Std Dev/R']**2

                # save data frame as pickle file
                target_df.to_pickle(opt.target_file)

    def _run(self, video, localizer, interval_selector, input_lines=None):
        vid = os.path.splitext(os.path.basename(video))[0]
        tracking_log = {}

        start_end = None
        if interval_selector:
            start_end = interval_selector.get_interval(vid=vid)

        return run_tracker(localizer, video, tracking_log, maxlen=self.opt.maxlen,
                           save_dir=self.opt.data_root, input_lines=input_lines, start_end=start_end,
                           analize=False, plot_dist=False)

    def _preprocess_inputdata(self, input_data, spatial_transform):
        inputs = []

        for cropped in input_data[::self.opt.delta]:
            img = cv2.resize(
                cropped, (self.opt.sample_size, self.opt.sample_size))
            inputs.append(img)

        # zero padding
        inputs = np.pad(inputs, ((0, self.opt.sample_duration - len(inputs)), (0, 0), (0, 0), (0, 0)),
                        'constant', constant_values=0)

        inputs = [spatial_transform(Image.fromarray(img)) for img in inputs]
        inputs = torch.stack(inputs, 0).permute(1, 0, 2, 3)

        return inputs

    def _run_demo(self, net, video, localizer,
                  startTime, endTime,
                  spatial_transform, target_transform):

        net.eval()

        self.opt.data_root = None  # do not save arr (.npy)

        # todo. develop interval selection methods....\
        # depth calculation / same ratio of human bboxes

        #input_data = self._run(video, localizer, interval_selector)
        #input_data = self._preprocess_inputdata(input_data, spatial_transform)

        start_ix, end_ix = [int(self.opt.fps * t)
                            for t in [startTime, endTime]]
        cap = cv2.VideoCapture(video)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize(
                (self.opt.img_size, self.opt.img_size))
            img_tensor = spatial_transform(img)
            frames.append(img_tensor)

        # time intervals of interest
        frames = frames[start_ix:end_ix+1]

        frame_indices = torch.LongTensor(
            np.linspace(
                0, len(frames), self.opt.sample_duration, endpoint=False).astype(np.int)
        )
        input_data = torch.stack(frames)[frame_indices].permute(1, 0, 2, 3)

        padding = (0, 0,
                   0, 0,
                   0, self.opt.sample_duration - len(frame_indices))

        # zero padding
        input_data = F.pad(input_data, padding)

        with torch.no_grad():
            y_pred, _ = net(input_data[None, :])
            y_pred = target_transform.inverse_transform(
                y_pred.detach().cpu().numpy())

        return y_pred
