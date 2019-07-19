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

from sklearn.metrics.regression import r2_score

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, score_func, target_transform):

    print('train at epoch {}'.format(epoch))

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
            target_transform.inverse_transform(torch.cat(outputs).detach().cpu().numpy())
        )

        losses.update(loss.item(), inputs.size(0))
        scores.update(score, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Score {score.val:.3f} ({score.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  score=scores))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'score': scores.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        ckpt_dir = os.path.join(opt.ckpt_dir, 'finetuned_with_'+ opt.arch)
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


def validate(epoch, data_loader, model, criterion, logger, score_func, target_transform):
    print('validation at epoch {}'.format(epoch))

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
            target_transform.inverse_transform(torch.cat(outputs).detach().cpu().numpy())
        )

        losses.update(loss.item(), inputs.size(0))
        scores.update(score, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Scores {score.val:.3f} ({score.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  score=scores))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'score': scores.avg})

    return losses.avg, scores.avg


from sklearn.model_selection import KFold

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
        data_locations = np.array(ds.vids)

        # K-fold CV
        kf = KFold(n_splits=self.opt.CV)
        print('Start training...')

        for train, valid in kf.split(data_locations):

            train_vids, valid_vids = data_locations[train], data_locations[valid]

            train_loader = dataloader_generator(self.opt, ds, train_vids,
                                                self.input_transform, self.target_transform, shuffle=True)
            valid_loader = dataloader_generator(self.opt, ds, valid_vids,
                                                self.input_transform, self.target_transform, shuffle=False)

            for i in range(self.opt.n_iter):
                # train loop
                train_epoch(i, train_loader, self.model, self.criterion, self.optimizer,
                            self.opt, self.train_logger, self.score_func, self.target_transform)

                with torch.no_grad():
                    # at every train epoch, validate model!
                    valid_loss, valid_score = validate(i, valid_loader, self.model, self.criterion,
                                                       self.valid_logger, self.score_func, self.target_transform)

                self.scheduler.step(valid_loss,)

        # TODO. Reinitialize model and monitor valid_score for best model selection along with all CV-splits