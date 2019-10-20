import numpy as np
import os
import json

from utils.target_columns import get_target_columns
from utils.train_utils import AverageMeter

import torch


def test(data_loader, model, opt, plotter,
         logpath, score_func, target_transform, target_columns):

    mode = 'uniform_average' if opt.score_avg else 'raw_values'

    model.eval()

    scores = AverageMeter()

    y_true, y_pred = [], []

    for _ in range(len(data_loader)):
        inputs, masks, targets, vids = next(data_loader)

        res = model(inputs)
        seg_outputs, reg_outputs, x4s = tuple(zip(*res))

        targets = targets.numpy()
        reg_outputs = torch.cat(reg_outputs).detach().cpu().numpy()

        score_val = score_func(
            targets,
            reg_outputs,
            multioutput=mode
        )

        scores.update(score_val, inputs.size(0))

        # y_true, y_pred for later usage
        y_true.append(targets)
        y_pred.append(reg_outputs)

    # end of epoch, avged score for entire testset
    score_map = {
        k: v for k, v in zip(target_columns, scores.avg)
    }

    if not os.path.exists(logpath):
        os.system(f'mkdir -p {logpath}')

    # log it!
    with open(os.path.join(logpath, 'results.json'), 'w') as fp:
        json.dump(score_map, fp)

    print('='*6 + 'Scores summary' + '='*6, score_map, sep='\n')

    return np.vstack(y_true), np.vstack(y_pred)


class Tester(object):
    def __init__(self,
                 model,
                 opt,
                 score_func=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None):

        self.model = model
        self.opt = opt
        self.score_func = score_func

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.target_columns = get_target_columns(opt)

    def fit(self, ds, plotter):
        print('Start testing...')

        from torch.utils.data import DataLoader

        setattr(ds, 'spatial_transform', self.spatial_transform)
        setattr(ds, 'temporal_transform', self.temporal_transform)

        # define dataloader
        test_loader = DataLoader(ds,
                                 batch_size=self.opt.batch_size,
                                 shuffle=False,
                                 num_workers=self.opt.n_threads)

        # convert to iterable
        test_loader = iter(test_loader)

        with torch.no_grad():
            # test model!
            y_true, y_pred = test(test_loader, self.model, self.opt, plotter,
                                  self.opt.logpath, self.score_func,
                                  self.target_transform, self.target_columns)

        return y_true, y_pred
