import torch
import torch.nn as nn
import collections


class EarlyStopping(object):
    # implementation from
    # https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    #
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)


class Logger:
    def __init__(self, plotter, phase):
        self.plotter = plotter
        self.__global_step = 0
        self.phase = phase

    @property
    def global_step(self):
        return self.__global_step

    @global_step.setter
    def global_step(self, gstep):
        self.__global_step = gstep

    def write(self, **kwargs):
        for k, v in kwargs.items():
            self.plotter.plot(k, self.phase, k, self.global_step, v)


class AverageMeter():
    def __init__(self, window_size=20):
        self.deq = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.deq.append(val)
        self.total += val*n
        self.count += n

    @property
    def avg(self):
        v = torch.tensor(list(self.deq), dtype=torch.float32)
        return v.mean().item()

    # @property
    # def avg(self):
    #     return self.total / self.count


class ScoreMeter(AverageMeter):
    def __init__(self, metric_name,
                 metric_func, need_score, task, window_size=20):
        super(ScoreMeter, self).__init__(window_size=window_size)

        self.metric_name = metric_name
        self.metric_func = metric_func
        self.need_score = need_score
        self.task = task

    def update(self, out, targets):
        if self.need_score:
            try:
                s = self.metric_func(targets.detach().cpu().numpy(
                ), out.softmax(1).detach().cpu().numpy()[:, 1])
            except ValueError:
                return
        else:
            if self.task == 'classification':
                s = self.metric_func(targets.detach().cpu().numpy(),
                                     out.argmax(1).detach().cpu().numpy())
            else:
                s = self.metric_func(targets.detach().cpu().numpy(),
                                     out.detach().cpu().numpy())
        
        super().update(val=s)


def predict_single_clip(model, batch, task):
    images, masks, targets, vids, valid_lengths = batch
    out, loss_dict = model(images, masks, targets=targets)

    # average for all GPUs
    loss_dict = {k: loss_dict[k].mean() for k in loss_dict}

    d = {}
    loss = loss_dict[task]

    d['clip'] = out, loss, targets

    return d


def predict_multiple_clip(model, batch, task):
    # considers clip/video-level targets at once(at collate_fn)
    images, masks, clip_level_targets, video_level_targets, *_ = batch

    out, clip_level_loss_dict = model(
        images, masks, targets=clip_level_targets)

    # average for all GPUs
    clip_level_loss_dict = {k: clip_level_loss_dict[k].mean()
                            for k in clip_level_loss_dict}

    if isinstance(model, nn.DataParallel):
        model = model.module

    d = {}
    video_level_out = out.view(video_level_targets.size(0), -1, 2).mean(1)
    video_level_loss = model.predictor.criterion(
        video_level_out, video_level_targets)

    clip_level_out = out
    clip_level_loss = clip_level_loss_dict[task]

    d['video'] = video_level_out, video_level_loss, video_level_targets
    d['clip'] = clip_level_out, clip_level_loss, clip_level_targets

    return d
