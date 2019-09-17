import numpy as np

from utils.target_columns import get_target_columns
from utils.train_utils import AverageMeter

import torch


def test(data_loader, model, opt, score_logger, score_func, target_transform):

    mode = 'uniform_average' if opt.score_avg else 'raw_values'

    model.eval()

    scores = AverageMeter()

    y_true, y_pred = [], []

    for _ in range(len(data_loader)):
        inputs, targets, vids = next(data_loader)

        outputs = model(inputs)

        targets = targets.numpy()
        outputs = target_transform.inverse_transform(
            torch.cat(outputs).detach().cpu().numpy())

        score_val = score_func(
            targets,
            outputs,
            multioutput=mode
        )

        scores.update(score_val, inputs.size(0))

        # y_true, y_pred for later usage
        y_true.append(targets)
        y_pred.append(outputs)

    # end of epoch, log avged score for entire testset
    score_logger.log({c: v for c, v in zip(score_logger.header, scores.avg)})

    print('='*6 + 'Scores summary' + '='*6)
    for c, v in zip(score_logger.header, scores.avg):
        print(f'{c}: {v:.3f}')

    return np.vstack(y_true), np.vstack(y_pred)


class Tester(object):
    def __init__(self,
                 model,
                 opt,
                 test_logger,
                 score_func=None,
                 input_transform=None, target_transform=None):

        self.model = model
        self.opt = opt
        self.test_logger = test_logger
        self.score_func = score_func

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.target_columns = get_target_columns(opt)

    def fit(self, ds,):
        print('Start testing...')

        from torch.utils.data import DataLoader

        setattr(ds, 'input_transform', self.input_transform)
        setattr(ds, 'target_transform', self.target_transform)

        # define dataloader
        test_loader = DataLoader(ds,
                                 batch_size=self.opt.batch_size,
                                 shuffle=False,
                                 num_workers=self.opt.n_threads, pin_memory=True)

        # convert to iterable
        test_loader = iter(test_loader)

        with torch.no_grad():
            # test model!
            y_true, y_pred = test(test_loader, self.model, self.opt,
                                  self.test_logger, self.score_func, self.target_transform)

        return y_true, y_pred
