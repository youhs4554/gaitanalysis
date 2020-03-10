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


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def normalize_img(img):
    img = (img - np.min(img))/np.max(img)
    return np.uint8(255 * img)


def normalize_video(video):
    video = (video-video.min())/(video.max()-video.min())*255
    video = torch.from_numpy(video.cpu().numpy().astype(np.uint8))
    return video


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


class Logger(object):

    def __init__(self, path, header):
        if not os.path.exists(path):
            os.system(f'mkdir -p {os.path.dirname(path)}')
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def get_mtl_loss(named_params, layers_to_share=['layer3', 'layer4'], alpha=0.1):

    params_of_task = collections.defaultdict(list)

    for name, p in named_params:
        task_name, layer_name = name.split('.')[1:3]
        if p.requires_grad and \
                layer_name in layers_to_share:
            params_of_task[task_name].append(p.view(-1))

    for k, v in params_of_task.items():
        params_of_task[k] = torch.cat(v)

    centroid_of_params = torch.stack(
        list(params_of_task.values())).mean(dim=0)

    # # for multi-task (soft weight sharing)
    distanceNorms_from_centroids = torch.stack([
        (v-centroid_of_params).norm(2) for v in params_of_task.values()]).mean(dim=0)

    mtl_penalty = centroid_of_params.norm(2) + \
        alpha * distanceNorms_from_centroids

    return mtl_penalty


def train_epoch(step, epoch, data_loader, model, criterion1, criterion2, optimizer, opt,
                plotter, target_transform, target_columns):
    print('train at epoch {}'.format(epoch))

    model.train()

    running_loss = 0.0
    running_reg_loss = 0.0
    if opt.enable_guide:
        running_seg_loss = 0.0

    # update plotter at every epoch
    update_cycle = len(data_loader)
    update_cycle = 10

    y_pred = []
    y_true = []
    y_prob = []

    metrices_functions = {
        'regression': [
            r2_score, mean_absolute_error, mean_squared_log_error
        ],
        'classification': classification_report
    }

    for i, (inputs, masks, targets, vids, valid_lengths) in enumerate(data_loader):
        res = model(inputs)

        loss = 0.0
        if not opt.enable_guide:
            reg_outputs, = tuple(zip(*res))
            reg_loss = criterion1(reg_outputs, targets)
            loss += reg_loss
        else:
            if opt.model_arch == 'AGNet-pretrain':
                reg_outputs, _, seg_outputs = tuple(zip(*res))
                reg_loss = criterion1(reg_outputs, targets)
            else:
                reg_outputs, seg_outputs = tuple(zip(*res))
                reg_loss = criterion1([reg_outputs[i][:, -4:]
                                       for i in range(len(reg_outputs))], targets[:, -4:])

            # for i in range(8):
            #     for j in range(4):
            #         print(seg_outputs[i][j].min().item(),
            #               seg_outputs[i][j].max().item())

            seg_loss = criterion2(seg_outputs, masks)

            loss += (reg_loss + seg_loss)

        if opt.benchmark:
            y_true.append(targets.cpu().numpy())
            y_pred.append(
                torch.cat(reg_outputs).detach().argmax(1).cpu().numpy())
            y_prob.append(
                torch.cat(reg_outputs).detach().softmax(1).cpu().numpy()[:, 1])   # probability of falling ('1')

        else:
            y_true.append(target_transform.inverse_transform(
                targets.cpu().numpy()))
            y_pred.append(target_transform.inverse_transform(
                torch.cat(reg_outputs).detach().cpu().numpy()))

        running_loss += loss.item()
        running_reg_loss += reg_loss.item()
        if opt.enable_guide:
            running_seg_loss += seg_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step[0] += 1

        print(i)

        if (i + 1) % update_cycle == 0:
            _loss = running_loss / update_cycle
            _reg_loss = running_reg_loss / update_cycle
            if opt.enable_guide:
                _seg_loss = running_seg_loss / update_cycle

            log_str = 'Epoch: [{0}][{1}/{2}]@{3}\t Reg. Loss {reg_loss:.4f}\t'.format(
                epoch,
                i + 1,
                len(data_loader),
                1,  # opt.fold
                reg_loss=_reg_loss)

            if opt.enable_guide:
                log_str += 'Seg. Loss {seg_loss:.4f}'.format(
                    seg_loss=_seg_loss)

            print(log_str)

            # import ipdb
            # ipdb.set_trace()

            # update visdom window
            plotter.plot('loss', 'train', 'avg_loss__trace',
                         step[0], _loss)
            plotter.plot('reg_loss', 'train', 'avg_reg_loss__trace',
                         step[0], _reg_loss)
            if opt.enable_guide:
                plotter.plot('seg_loss', 'train', 'avg_seg_loss__trace',
                             step[0], _seg_loss)
            plotter.plot('lr', 'train', 'lr', step[0],
                         optimizer.param_groups[0]['lr'])

            # re-init running_loss
            running_loss = 0.0
            running_reg_loss = 0.0
            running_seg_loss = 0.0

    if opt.benchmark:
        true_vals = np.array(
            y_true[:-1]).reshape(-1).tolist() + y_true[-1].reshape(-1).tolist()
        pred_vals = np.array(
            y_pred[:-1]).reshape(-1).tolist() + y_pred[-1].reshape(-1).tolist()
        prob_vals = np.array(
            y_prob[:-1]).reshape(-1).tolist() + y_prob[-1].reshape(-1).tolist()

        # Compute ROC curve and ROC area for each class
        fpr, tpr, thresh = roc_curve(true_vals, prob_vals)

        roc_auc = auc(fpr, tpr)

        plt.ioff()

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC_AUC_at_train_EPOCH_{epoch}')
        plt.legend(loc="lower right")

        # wirte ROC-AUC curve on visdom
        plotter.matplot(f'ROC_AUC_at_train_EPOCH_{epoch}', plt)

        # metrices to evaluate performance
        target_names = ['0', '1']

        classification_res = classification_report(
            true_vals, pred_vals, target_names=target_names, output_dict=True)

        scores = {}
        for name, report in classification_res.items():
            if name == 'accuracy':
                scores[name] = report
            elif name == 'weighted avg':
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
                plotter.plot(target_columns[n] + '_score', 'train',
                             target_columns[n] + '_' +
                             score_func.__name__ + '__' + 'trace',
                             step[0], score[n])

    # write averaged scores
    for key in scores.keys():
        plotter.plot(key+'_avg', 'train', key + '__' + 'avg' + '__' + 'trace',
                     step[0], scores[key])

    if epoch % opt.checkpoint == 0:
        ckpt_dir = os.path.join(opt.ckpt_dir,
                                '_'.join(filter(lambda x: x != '',
                                                [opt.attention_str,
                                                 opt.model_arch,
                                                 opt.merge_type,
                                                 opt.arch,
                                                 opt.group_str])))

        if not os.path.exists(ckpt_dir):
            os.system(f'mkdir -p {ckpt_dir}')
        save_file_path = os.path.join(ckpt_dir,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)


def evaluate(step, epoch, data_loader,
             model, criterion1, criterion2, opt, plotter,
             target_transform, target_columns):
    print('evaluation at epoch {}'.format(epoch))

    model.eval()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    if opt.enable_guide:
        seg_losses = AverageMeter()

    y_pred = []
    y_true = []
    y_prob = []

    for i, (inputs, masks, targets, vids, valid_lengths) in enumerate(data_loader):
        res = model(inputs)

        loss = 0.0
        if not opt.enable_guide:
            reg_outputs, = tuple(zip(*res))
            reg_loss = criterion1(reg_outputs, targets)
            loss += reg_loss
        else:
            if opt.model_arch == 'AGNet-pretrain':
                reg_outputs, _, seg_outputs = tuple(zip(*res))
                reg_loss = criterion1(reg_outputs, targets)
            else:
                reg_outputs, seg_outputs = tuple(zip(*res))
                reg_loss = criterion1([reg_outputs[i][:, -4:]
                                       for i in range(len(reg_outputs))], targets[:, -4:])

            seg_loss = criterion2(seg_outputs, masks)

            loss += (reg_loss+seg_loss)

        if opt.benchmark:
            y_true.append(targets.cpu().numpy())
            y_pred.append(
                torch.cat(reg_outputs).detach().argmax(1).cpu().numpy())
            y_prob.append(
                torch.cat(reg_outputs).detach().softmax(1).cpu().numpy()[:, 1])   # probability of falling ('1')
        else:
            y_true.append(target_transform.inverse_transform(
                targets.cpu().numpy()))
            y_pred.append(target_transform.inverse_transform(
                torch.cat(reg_outputs).detach().cpu().numpy()))

        losses.update(loss.item(), 1)
        reg_losses.update(reg_loss.item(), 1)
        if opt.enable_guide:
            seg_losses.update(seg_loss.item(), 1)

    avg_loss = losses.avg
    avg_reg_loss = reg_losses.avg
    if opt.enable_guide:
        avg_seg_loss = seg_losses.avg

    if opt.benchmark:
        true_vals = np.array(
            y_true[:-1]).reshape(-1).tolist() + y_true[-1].reshape(-1).tolist()
        pred_vals = np.array(
            y_pred[:-1]).reshape(-1).tolist() + y_pred[-1].reshape(-1).tolist()
        prob_vals = np.array(
            y_prob[:-1]).reshape(-1).tolist() + y_prob[-1].reshape(-1).tolist()

        # Compute ROC curve and ROC area for each class
        fpr, tpr, thresh = roc_curve(true_vals, prob_vals)

        roc_auc = auc(fpr, tpr)

        plt.ioff()

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC_AUC_at_train_EPOCH_{epoch}')
        plt.legend(loc="lower right")

        # wirte ROC-AUC curve on visdom
        plotter.matplot(f'ROC_AUC_at_train_EPOCH_{epoch}', plt)

        # metrices to evaluate performance
        target_names = ['0', '1']

        classification_res = classification_report(
            true_vals, pred_vals, target_names=target_names, output_dict=True)

        scores = {}
        for name, report in classification_res.items():
            if name == 'accuracy':
                scores[name] = report
            elif name == 'weighted avg':
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
                plotter.plot(target_columns[n] + '_score', 'test',
                             target_columns[n] + '_' +
                             score_func.__name__ + '__' + 'trace',
                             step[0], score[n])

    # write averaged scores
    for key in scores.keys():
        plotter.plot(key+'_avg', 'test', key + '__' + 'avg' + '__' + 'trace',
                     step[0], scores[key])

    if opt.enable_guide:
        # import ipdb
        # ipdb.set_trace()
        log_str = 'Epoch: [{0}][{1}/{2}]@{3}\tReg. Loss {reg_loss:.4f}\tSeg. Loss {seg_loss:.4f}\tScore {avg_score:.3f}'.format(
            epoch,
            i + 1,
            len(data_loader),
            1,  # opt.fold,
            reg_loss=avg_reg_loss,
            seg_loss=avg_seg_loss,
            avg_score=avg_score)
    else:
        log_str = 'Epoch: [{0}][{1}/{2}]@{3}\tReg. Loss {reg_loss:.4f}\tScore {avg_score:.3f}'.format(
            epoch,
            i + 1,
            len(data_loader),
            reg_loss=avg_reg_loss,
            avg_score=avg_score)

    print(log_str)

    # update visdom window
    plotter.plot('loss', 'val', 'avg_loss__trace',
                 step[0], avg_loss)
    plotter.plot('reg_loss', 'val', 'avg_reg_loss__trace',
                 step[0], avg_reg_loss)
    if opt.enable_guide:
        plotter.plot('seg_loss', 'val', 'avg_seg_loss__trace',
                     step[0], avg_seg_loss)
    plotter.plot('score', 'val', 'avg_score__trace',
                 step[0], avg_score)

    return avg_loss, avg_score


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
                 plotter=None):

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

        self.target_columns = get_target_columns(opt)

    def fit(self, train_loader, test_loader):
        print('Start training...')

        step = [0]

        for epoch in range(self.opt.n_iter):
            # train loop
            train_epoch(step, epoch, train_loader, self.model,
                        self.criterion1, self.criterion2, self.optimizer,
                        self.opt, self.plotter, self.target_transform,
                        self.target_columns)
            with torch.no_grad():
                # at every train epoch, evaluate model!
                test_loss, test_score = evaluate(step, epoch,
                                                 test_loader,
                                                 self.model,
                                                 self.criterion1, self.criterion2,
                                                 self.opt,
                                                 self.plotter,
                                                 self.target_transform,
                                                 self.target_columns)
            # self.scheduler.step(test_loss)
