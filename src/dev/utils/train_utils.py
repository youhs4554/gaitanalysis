from sklearn.metrics import (r2_score, mean_absolute_error,
                             mean_squared_log_error, confusion_matrix, classification_report, roc_curve, auc)
from utils.generate_model import init_state
from sklearn.model_selection import KFold
import torch
import time
import os
import numpy as np
import csv
import json
from utils.target_columns import get_target_columns
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import cv2
import torchvision.transforms.functional as F
from itertools import islice
import collections
from multiprocessing import Pool
import glob
from tqdm import tqdm
import seaborn as sns
import h5py


def computeLoss(net_outputs, targets, loss_funcs, model_arch='AGNet-pretrain', masks=None,
                contain_outputs=True):

    if len(loss_funcs) > 1:
        main_lossfunc, aux_lossfunc = loss_funcs
    elif len(loss_funcs) == 1:
        main_lossfunc, = loss_funcs

    out_dict = {}
    aux_outputs = None
    if model_arch == 'GuidelessNet':
        main_outputs, = tuple(zip(*net_outputs))
    elif model_arch == 'AGNet-pretrain':
        main_outputs, _, aux_outputs = tuple(zip(*net_outputs))
    elif model_arch == 'AGNet':
        main_outputs, aux_outputs = tuple(zip(*net_outputs))
        main_outputs = [main_outputs[i][:, -4:]
                        for i in range(len(main_outputs))]
        targets = targets[:, -4:]

    main_loss = main_lossfunc(main_outputs, targets)
    out_dict['main_loss'] = main_loss
    if contain_outputs:
        out_dict['main_outputs'] = main_outputs

    if aux_outputs is not None:
        aux_loss = aux_lossfunc(aux_outputs, masks)
        out_dict['aux_loss'] = aux_loss
        if contain_outputs:
            out_dict['aux_outputs'] = aux_outputs

    out_dict['joint_loss'] = torch.sum(
        torch.stack([out_dict[k]
                     for k in out_dict.keys() if k.endswith('_loss')])
    )

    return out_dict


def train_model(step, epoch, train_loader, test_loader, model, criterion1, criterion2, optimizer, opt,
                plotter, target_transform, target_columns, fold):

    metrices_functions = {
        'regression': [
            r2_score, mean_absolute_error, mean_squared_log_error
        ],
        'classification': classification_report
    }

    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
            data_loader = train_loader
        else:
            model.eval()
            torch.manual_seed(0)  # for same result
            data_loader = test_loader

        running_joint_loss = 0.0
        running_main_loss = 0.0
        if opt.enable_guide:
            running_aux_loss = 0.0

        # update plotter at every epoch
        update_cycle = 10 if phase == 'train' else len(data_loader)

        y_pred = []
        y_true = []
        y_prob = []

        for i, (inputs, masks, targets, vids, valid_lengths) in enumerate(data_loader):

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                res = model(inputs)
                out_dict = computeLoss(res, targets, loss_funcs=[criterion1, criterion2],
                                       model_arch=opt.model_arch, masks=masks, contain_outputs=True)

            if phase == 'train':
                out_dict['joint_loss'].backward()
                optimizer.step()
                step[0] += 1

            running_joint_loss += out_dict['joint_loss'].item()
            running_main_loss += out_dict['main_loss'].item()
            if opt.enable_guide:
                running_aux_loss += out_dict['aux_loss'].item()

            if opt.benchmark:
                y_true.append(targets.cpu().numpy())
                y_pred.append(
                    torch.cat(out_dict['main_outputs']).detach().argmax(1).cpu().numpy())
                y_prob.append(
                    torch.cat(out_dict['main_outputs']).detach().softmax(1).cpu().numpy()[:, 1])   # probability of falling ('1')

            else:
                y_true.append(target_transform.inverse_transform(
                    targets.cpu().numpy()))
                y_pred.append(target_transform.inverse_transform(
                    torch.cat(out_dict['main_outputs']).detach().cpu().numpy()))

            if (i + 1) % update_cycle == 0:
                _loss = running_joint_loss / update_cycle
                _main_loss = running_main_loss / update_cycle
                if opt.enable_guide:
                    _aux_loss = running_aux_loss / update_cycle

                log_str = '[fold-{0}] {phase} Epoch: [{1}][{2}/{3}]\t Main. Loss {main_loss:.4f}\t'.format(
                    fold,
                    epoch,
                    i + 1,
                    len(data_loader),
                    phase=phase,
                    main_loss=_main_loss)

                if opt.enable_guide:
                    log_str += 'Aux. Loss {aux_loss:.4f}'.format(
                        aux_loss=_aux_loss)

                print(log_str)

                # update visdom window
                plotter.plot('loss', phase, 'avg_loss__trace',
                             step[0], _loss)
                plotter.plot('main_loss', phase, 'avg_main_loss__trace',
                             step[0], _main_loss)
                if opt.enable_guide:
                    plotter.plot('aux_loss', phase, 'avg_aux_loss__trace',
                                 step[0], _aux_loss)
                plotter.plot('lr', phase, 'lr', step[0],
                             optimizer.param_groups[0]['lr'])

                if phase == 'train':
                    # re-init running_losses
                    running_joint_loss = 0.0
                    running_main_loss = 0.0
                    if opt.enable_guide:
                        running_aux_loss = 0.0

        if opt.benchmark:
            true_vals = np.array(
                y_true[:-1]).reshape(-1).tolist() + y_true[-1].reshape(-1).tolist()
            pred_vals = np.array(
                y_pred[:-1]).reshape(-1).tolist() + y_pred[-1].reshape(-1).tolist()
            prob_vals = np.array(
                y_prob[:-1]).reshape(-1).tolist() + y_prob[-1].reshape(-1).tolist()

            target_names = ['class-0', 'class-1']

            classification_res = classification_report(
                true_vals, pred_vals, target_names=target_names, output_dict=True)

            scores = {}
            for name, report in classification_res.items():
                if name == 'accuracy':
                    scores[name] = report
                elif name == 'class-0':
                    scores['specificity'] = report['recall']
                elif name == 'class-1':
                    scores['sensitivity'] = report['recall']
                    scores['f1-score'] = report['f1-score']

        else:
            # regression
            scores = {}
            for score_func in metrices_functions['regression']:
                score = score_func(
                    np.vstack(y_true),
                    np.vstack(y_pred),
                    multioutput='raw_values',
                )

                avg_score = score.mean()
                scores[score_func.__name__] = avg_score

                # write socres per each gait-parameter
                for n in range(len(target_columns)):
                    plotter.plot(target_columns[n] + '_score', phase,
                                 target_columns[n] + '_' +
                                 score_func.__name__ + '__' + 'trace',
                                 step[0], score[n])

        # write averaged scores
        for key in scores.keys():
            plotter.plot(key+'_avg', phase, key + '__' + 'avg' + '__' + 'trace',
                         step[0], scores[key])

    return running_joint_loss, scores, true_vals, pred_vals, prob_vals


class Trainer(object):
    def __init__(self,
                 model,
                 criterion1, criterion2,
                 optimizer,
                 scheduler,
                 opt,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 plotter=None, fold=1):

        self.model = model
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.opt = opt
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.plotter = plotter
        self.fold = fold

        self.target_columns = get_target_columns(opt)

    def fit(self, train_loader, test_loader, metrice='f1-score'):
        print('Start training...')

        step = [0]

        best_score = -1.0
        best_score_dict = None

        for epoch in range(self.opt.n_iter):
            test_loss, test_scores, true_vals, pred_vals, prob_vals = train_model(step, epoch, train_loader, test_loader, self.model,
                                                                                  self.criterion1, self.criterion2, self.optimizer,
                                                                                  self.opt, self.plotter, self.target_transform,
                                                                                  self.target_columns, fold=self.fold)
            if test_scores[metrice] > best_score:
                # save model...
                ckpt_dir = os.path.join(self.opt.ckpt_dir,
                                        '_'.join(filter(lambda x: x != '',
                                                        [self.opt.attention_str,
                                                         self.opt.model_arch,
                                                         self.opt.merge_type,
                                                         self.opt.arch,
                                                         self.opt.dataset])))

                if not os.path.exists(ckpt_dir):
                    os.system(f'mkdir -p {ckpt_dir}')
                save_file_path = os.path.join(ckpt_dir,
                                              'model_fold-{}.pth'.format(self.fold))
                states = {
                    'epoch': epoch + 1,
                    'arch': self.opt.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(states, save_file_path)

                # update best_score
                best_score = test_scores[metrice]
                best_score_dict = test_scores

                best_true_vals = true_vals
                best_pred_vals = pred_vals
                best_prob_vals = prob_vals

                print(
                    f"@@ EPOCH : {epoch} Best `{metrice}` has been updated (value: {best_score:.5f}). Save best model at {save_file_path}")

            if self.scheduler is not None:
                self.scheduler.step(test_loss)

        with h5py.File("results.hdf5", "a") as f:
            name = self.opt.dataset + '_' + self.opt.model_indicator
            grp = f.require_group(name)
            grp = grp.require_group("fold-{}".format(self.fold))
            grp.require_dataset("true_vals", (len(best_true_vals),),
                                dtype='i')[...] = best_true_vals
            grp.require_dataset("pred_vals", (len(best_pred_vals),),
                                dtype='i')[...] = best_pred_vals
            grp.require_dataset("prob_vals", (len(best_prob_vals),),
                                dtype='f')[...] = best_prob_vals

        return best_score_dict
