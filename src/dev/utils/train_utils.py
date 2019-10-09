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


def normalize_img(img):
    img = (img - np.min(img))/np.max(img)
    return np.uint8(255 * img)


def draw_actmap(reg_actmap, inputs, weight_output, plotter, opt):
    nc, L, h, w = reg_actmap.shape
    step = inputs.shape[1] // L
    mean = np.array(opt.mean)
    std = np.array(opt.std)

    for t in range(L):
        actmap = reg_actmap[:, t]  # (nc,h,w)
        # (128, 128, 3)
        input_img = (inputs.transpose(1, 2, 3, 0)[
                     step*t:step*(t+1)]*std + mean).mean(0)
        heatmaps = []
        for n in range(nc):
            # dot-product for all logits
            actmap_img = weight_output[n].dot(
                actmap.reshape((nc, h*w))).reshape((h, w))
            actmap_img = normalize_img(actmap_img)
            input_img = normalize_img(input_img)
            actmap_img = cv2.resize(actmap_img,
                                    (opt.sample_size, opt.sample_size))
            actmap_img = cv2.applyColorMap(
                actmap_img, cv2.COLORMAP_JET)

            # bgr -> rgb
            actmap_img = cv2.cvtColor(actmap_img, cv2.COLOR_BGR2RGB)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            res = torch.clamp(F.to_tensor(actmap_img) * 0.3 +
                              F.to_tensor(input_img) * 0.5, 0., 1.)

            heatmaps.append(res)

        actmap_grid = make_grid(heatmaps)
        plotter.viz.image(actmap_grid, win='actmap_hist', env=plotter.env,
                          opts=dict(caption=f't={t}',
                                    store_history=True,
                                    title=f'Activation_Maps at {t}-th'),)


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


def train_epoch(step, epoch, split, data_loader, model, criterion1, criterion2, optimizer, opt,
                plotter, score_func, target_transform, target_columns):

    print('train at epoch {} @ split-{}'.format(epoch, split))

    model.train()

    running_loss = 0.0
    running_scores = [0.0 for _ in range(len(target_columns))]

    # update plotter at every 1/10 of epoch
    update_cycle = len(data_loader) // 10

    for i, (inputs, masks, targets, vids) in enumerate(data_loader):
        res = model(inputs)
        seg_outputs, reg_outputs, reg_actmap = tuple(zip(*res))

        loss = 0.2 * criterion1(seg_outputs, masks) + \
            0.8 * criterion2(reg_outputs, targets)

        score = score_func(
            target_transform.inverse_transform(targets.numpy()),
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

            input_video = inputs[0, :].permute(1, 2, 3, 0)
            plotter.draw_video('train__input_video',
                               f'Input (epoch : {epoch}, step : {step[0]})',
                               torch.tensor((input_video-input_video.min())/(input_video.max()-input_video.min())*255, dtype=torch.uint8))

            cmap = plt.get_cmap('jet')

            output_video = cmap(
                seg_outputs[0][0, 1].detach().cpu().numpy())[..., :3]
            plotter.draw_video('train__output_video',
                               f'Output (epoch : {epoch}, step : {step[0]})',
                               torch.tensor((output_video-output_video.min())/(output_video.max()-output_video.min())*255, dtype=torch.uint8))

            target_video = cmap(masks[0, 1].numpy())[..., :3]
            plotter.draw_video('train__target_video',
                               f'Target (epoch : {epoch}, step : {step[0]})',
                               torch.tensor((target_video-target_video.min())/(target_video.max()-target_video.min())*255, dtype=torch.uint8))

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
    for i, (inputs, masks, targets, vids) in enumerate(data_loader):
        res = model(inputs)

        seg_outputs, reg_outputs, reg_actmap = tuple(zip(*res))

        loss = 0.2 * criterion1(seg_outputs, masks) + \
            0.8 * criterion2(reg_outputs, targets)
        score = score_func(
            target_transform.inverse_transform(targets.numpy()),
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

    # reg_actmap = reg_actmap[0][0].detach().cpu().numpy()
    # input_img = inputs[0].cpu().numpy()

    # weight_output = list(model.parameters())[-2].cpu().data.numpy()

    # # draw activation maps
    # draw_actmap(reg_actmap, input_img, weight_output, plotter, opt)

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
                                                shuffle=True)

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
