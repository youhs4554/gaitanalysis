from sklearn.metrics.regression import r2_score
from utils.generate_model import init_state
from sklearn.model_selection import KFold
import torch
import time
import os
import numpy as np
import csv
from utils.target_columns import get_target_columns


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


def train_epoch(epoch, split, data_loader, model, criterion, optimizer, opt,
                epoch_logger, score_func, target_transform):

    print('train at epoch {} @ split-{}'.format(epoch, split))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        score = score_func(
            target_transform.inverse_transform(targets.numpy()),
            target_transform.inverse_transform(
                torch.cat(outputs).detach().cpu().numpy())
        )

        losses.update(loss.item(), inputs.size(0))
        scores.update(score, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch@Split: [{0}][{1}/{2}]@{3}\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Score {score.val:.3f} ({score.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  split,
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  score=scores))

    epoch_logger.log({
        'epoch@split': f'{epoch}@{split}',
        'loss': losses.avg,
        'score': scores.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        if opt.model_arch == 'HPP':
            ckpt_dir = os.path.join(opt.ckpt_dir, opt.model_arch + '_' +
                                    opt.merge_type + '_' +
                                    'finetuned_with' + '_' + opt.arch)
        else:
            ckpt_dir = os.path.join(opt.ckpt_dir,
                                    opt.model_arch + '_' +
                                    'finetuned_with' + '_' + opt.arch)

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


def validate(epoch, split, data_loader,
             model, criterion, logger, score_func, target_transform):
    print('validation at epoch {} @ split-{}'.format(epoch, split))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        score = score_func(
            target_transform.inverse_transform(targets.numpy()),
            target_transform.inverse_transform(
                torch.cat(outputs).detach().cpu().numpy())
        )

        losses.update(loss.item(), inputs.size(0))
        scores.update(score, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch@Split: [{0}][{1}/{2}]@{3}\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Scores {score.val:.3f} ({score.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  split,
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  score=scores))

    logger.log({'epoch@split': f'{epoch}@{split}',
                'loss': losses.avg, 'score': scores.avg})

    return losses.avg, scores.avg


class Trainer(object):
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 opt,
                 train_logger,
                 val_logger,
                 input_transform=None, target_transform=None,
                 score_func=r2_score):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.opt = opt
        self.train_logger = train_logger
        self.valid_logger = val_logger
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.score_func = score_func

        self.target_columns = get_target_columns(opt)

    def fit(self, ds, dataloader_generator):
        entire_vids = np.array(ds.vids)

        # K-fold CV
        kf = KFold(n_splits=self.opt.CV)
        from collections import defaultdict

        cv_result = defaultdict(list)
        print('Start training...')

        cv_loss = 0.
        cv_score = 0.

        for split, (train, valid) in enumerate(kf.split(entire_vids)):

            train_vids, valid_vids = entire_vids[train], entire_vids[valid]

            train_loader = dataloader_generator(self.opt, ds, train_vids,
                                                self.input_transform,
                                                self.target_transform,
                                                shuffle=True)
            valid_loader = dataloader_generator(self.opt, ds, valid_vids,
                                                self.input_transform,
                                                self.target_transform,
                                                shuffle=False)

            epoch_status = defaultdict(list)

            for epoch in range(self.opt.n_iter):
                # train loop
                train_epoch(epoch, split, train_loader, self.model,
                            self.criterion, self.optimizer,
                            self.opt, self.train_logger,
                            self.score_func, self.target_transform)

                with torch.no_grad():
                    # at every train epoch, validate model!
                    valid_loss, valid_score = validate(epoch, split,
                                                       valid_loader,
                                                       self.model,
                                                       self.criterion,
                                                       self.valid_logger,
                                                       self.score_func,
                                                       self.target_transform)

                self.scheduler.step(valid_loss,)

                epoch_status['loss'].append(valid_loss)
                epoch_status['score'].append(valid_score)

            avg_loss = np.mean(epoch_status['loss'])
            avg_score = np.mean(epoch_status['score'])

            # todo. early stopping based on avg_loss or avg_score,
            # and user might select criterion

            # todo.\
            #  best model selection functionality can be added,
            # if we need the hyperparameter searching algorithm
            # (e.g. Gridsearch | Randomsearch | etc.) \

            # todo. \
            #  best model selection can be implented by running .fit function
            # parallel manner... after running all iterations, we can select
            # best model based on
            #  avg_loss or avg_score

            cv_result[f'split-{split}'].append(
                dict(loss=avg_loss,
                     score=avg_score)
            )

            # todo. add only last epoch into cv_loss
            cv_loss += avg_loss
            cv_score += avg_score

            if not self.opt.warm_start:
                # if warm-starting is False, re-init the state
                print('Re-initializing states...')
                self.model, self.optimizer, self.scheduler = init_state(
                    self.opt)

        cv_result['avg_loss'] = cv_loss / self.opt.CV
        cv_result['avg_score'] = cv_score / self.opt.CV

        print(cv_result)
