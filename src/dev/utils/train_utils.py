from sklearn.metrics.regression import r2_score
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


def train_epoch(step, epoch, split, data_loader, model, criterion1, criterion2, optimizer, opt,
                plotter, score_func, target_transform, target_columns):

    print('train at epoch {} @ split-{}'.format(epoch, split))

    model.train()

    running_loss = 0.0
    running_scores = [0.0 for _ in range(len(target_columns))]

    # update plotter at every 1/10 of epoch
    update_cycle = len(data_loader) // 10

    for i, (inputs, masks, targets, vids, valid_lengths) in enumerate(data_loader):
        res = model(inputs)
        reg_outputs, seg_outputs = tuple(zip(*res))

        reg_loss = criterion1(reg_outputs, targets)
        seg_loss = criterion2(seg_outputs, masks)

        loss = reg_loss + seg_loss

        score = score_func(
            target_transform.inverse_transform(targets.cpu().numpy()),
            target_transform.inverse_transform(
                torch.cat(reg_outputs).detach().cpu().numpy()),
            multioutput='raw_values',
        )

        running_loss += loss.item()
        for n in range(len(target_columns)):
            running_scores[n] += score[n]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step[0] += 1

        if (i + 1) % update_cycle == 0:
            _loss = running_loss / update_cycle
            _scores = [running_scores[n] /
                       update_cycle for n in range(len(target_columns))]
            print('Epoch@Split: [{0}][{1}/{2}]@{3}\t'
                  'Loss {loss:.4f}\t'
                  'Score {avg_score:.3f}'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      split,
                      loss=_loss,
                      avg_score=np.mean(_scores)))

            # update visdom window
            plotter.plot('loss', 'train', 'avg_loss__trace', step[0], _loss)
            plotter.plot('score', 'train', 'avg_score__trace',
                         step[0], np.mean(_scores))
            plotter.plot('lr', 'train', 'lr', step[0],
                         optimizer.param_groups[0]['lr'])

            for n in range(len(target_columns)):
                plotter.plot(target_columns[n] + '_score', 'train',
                             target_columns[n] + '_' +
                             'score' + '__' + 'trace',
                             step[0], _scores[n])

            # re-init running_loss & running_scores
            running_loss = 0.0
            running_scores = [0.0 for _ in range(len(target_columns))]

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


def validate(step, epoch, split, data_loader,
             model, criterion1, criterion2, opt, plotter, score_func,
             target_transform, target_columns):
    print('validation at epoch {} @ split-{}'.format(epoch, split))

    model.eval()
    losses = AverageMeter()
    multi_scores = [AverageMeter() for _ in range(len(target_columns))]
    for i, (inputs, masks, targets, vids, valid_lengths) in enumerate(data_loader):
        res = model(inputs)

        reg_outputs, seg_outputs = tuple(zip(*res))

        reg_loss = criterion1(reg_outputs, targets)
        seg_loss = criterion2(seg_outputs, masks)

        loss = reg_loss + seg_loss

        score = score_func(
            target_transform.inverse_transform(targets.cpu().numpy()),
            target_transform.inverse_transform(
                torch.cat(reg_outputs).detach().cpu().numpy()),
            multioutput='raw_values',
        )

        losses.update(loss.item(), inputs.size(0))
        for n in range(len(target_columns)):
            multi_scores[n].update(score[n], inputs.size(0))

    avg_loss = losses.avg
    avg_score = sum([s.avg for s in multi_scores])/len(multi_scores)

    print('Epoch@Split: [{0}][{1}/{2}]@{3}\t'
          'Loss {loss:.4f}\t'
          'Score {avg_score:.3f}'.format(
              epoch,
              i + 1,
              len(data_loader),
              split,
              loss=avg_loss,
              avg_score=avg_score))

    # update visdom window
    plotter.plot('loss', 'val', 'avg_loss__trace', step[0], avg_loss)
    plotter.plot('score', 'val', 'avg_score__trace',
                 step[0], np.mean(avg_score))

    for n in range(len(target_columns)):
        plotter.plot(target_columns[n] + '_score', 'val', target_columns[n] +
                     '_' + 'score' + '__' + 'trace', step[0], score[n])

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
                 score_func=r2_score):

        self.model = model
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.opt = opt
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.score_func = score_func

        self.target_columns = get_target_columns(opt)

    def fit(self, ds, dataloader_generator, ds_class, plotter):
        entire_vids = np.array(ds.vids)

        # K-fold CV
        kf = KFold(n_splits=self.opt.CV)
        from collections import defaultdict

        CV_results = defaultdict(list)

        # logdir for CV(K-fold Cross Validation)
        CV_logdir = os.path.join(os.path.dirname(
            self.opt.logpath), 'CV')

        if not os.path.exists(CV_logdir):
            os.system(f"mkdir -p {CV_logdir}")

        CV_logpath = os.path.join(CV_logdir, 'results.json')

        print('Start training...')

        cv_loss = 0.
        cv_score = 0.

        step = [0]

        for split, (train, valid) in enumerate(kf.split(entire_vids)):

            train_vids, valid_vids = entire_vids[train], entire_vids[valid]

            train_loader = dataloader_generator(self.opt, ds, train_vids, ds_class,
                                                phase='train',
                                                spatial_transform=self.spatial_transform['train'],
                                                temporal_transform=self.temporal_transform['train'],
                                                shuffle=True)
            valid_loader = dataloader_generator(self.opt, ds, valid_vids, ds_class,
                                                phase='valid',
                                                spatial_transform=self.spatial_transform['test'],
                                                temporal_transform=self.temporal_transform['test'],
                                                shuffle=False)

            epoch_status = defaultdict(list)
            for epoch in range(self.opt.n_iter):
                # train loop
                train_epoch(step, epoch, split, train_loader, self.model,
                            self.criterion1, self.criterion2, self.optimizer,
                            self.opt, plotter,
                            self.score_func, self.target_transform,
                            self.target_columns)
                with torch.no_grad():
                    # at every train epoch, validate model!
                    valid_loss, valid_score = validate(step, epoch, split,
                                                       valid_loader,
                                                       self.model,
                                                       self.criterion1, self.criterion2,
                                                       self.opt,
                                                       plotter,
                                                       self.score_func,
                                                       self.target_transform,
                                                       self.target_columns)
                # self.scheduler.step(valid_loss)

                epoch_status['loss'].append(valid_loss)
                epoch_status['score'].append(valid_score)

            last_loss = epoch_status['loss'][-1]
            last_score = epoch_status['score'][-1]

            CV_results[f'split-{split}'].append(
                dict(loss=last_loss,
                     score=last_score
                     ))

            # todo. add only last epoch into cv_loss
            cv_loss += last_loss
            cv_score += last_score

            if not self.opt.warm_start:
                # if warm-starting is False, re-init the state
                print('Re-initializing states...')
                self.model, self.criterion1, self.criterion2, self.optimizer, self.scheduler = init_state(
                    self.opt)

            # at every end of split save cv_results ( as it takes too much time...)
            with open(CV_logpath, 'w') as fp:
                json.dump(CV_results, fp)

        CV_results['avg_loss'] = cv_loss / self.opt.CV
        CV_results['avg_score'] = cv_score / self.opt.CV

        # save CV_results as file.
        with open(CV_logpath, 'w') as fp:
            json.dump(CV_results, fp)
