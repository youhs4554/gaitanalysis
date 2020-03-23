import numpy as np
import os
import json

from utils.target_columns import get_target_columns
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test(data_loader, model, opt, plotter,
         logpath, score_func, target_transform, target_columns, criterion):

    mode = 'uniform_average' if opt.score_avg else 'raw_values'

    model.eval()

    scores = AverageMeter()

    y_true, y_pred = [], []

    data_loader = iter(data_loader)

    for _ in range(len(data_loader)):
        inputs, masks, targets, vids, valid_lengths = next(data_loader)

        res = model(inputs)
        if not opt.enable_guide:
            reg_outputs, = tuple(zip(*res))

        else:
            if opt.model_arch == 'AGNet-pretrain':
                reg_outputs, _, seg_outputs = tuple(zip(*res))
            else:
                reg_outputs, seg_outputs = tuple(zip(*res))

        targets = target_transform.inverse_transform(targets.cpu().numpy())
        reg_outputs = target_transform.inverse_transform(
            torch.cat(reg_outputs).detach().cpu().numpy())

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

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    merged = np.concatenate([y_true, y_pred], 1)

    import pandas as pd
    df = pd.DataFrame(merged, columns=[
        ['y_true']*len(target_columns) + ['y_pred']*len(target_columns),
        target_columns * 2
    ])

    df.to_pickle(os.path.join(logpath, 'full_testing_results.pkl'))

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

    def fit(self, test_loader, plotter, criterion):
        print('Start testing...')

        with torch.no_grad():
            # test model!
            y_true, y_pred = test(test_loader, self.model, self.opt, plotter,
                                  self.opt.logpath, self.score_func,
                                  self.target_transform, self.target_columns, criterion)

        return y_true, y_pred
