from datetime import datetime
from functools import partial
from matplotlib.pyplot import winter
import numpy as np
import pandas as pd
import torch
import os
import collections
import sklearn.metrics
from torch.nn.modules import loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from .visualization import VisdomPlotter
from models.utils import load_pretrained_ckpt
from ._utils import (Logger, AverageMeter, ScoreMeter,
                     predict_single_clip, predict_multiple_clip, EarlyStopping, update_losses, get_running_losses)
import copy


@torch.no_grad()
def evaluate(model, data_loader, metrics, task, multiple_clip=False):
    model.eval()

    print()
    print("Start Evaluation...")

    levels = ['video', 'clip'] if multiple_clip else ['clip']

    loss_meter_group = {
        level: collections.defaultdict(
            partial(AverageMeter, window_size=len(data_loader))) for level in levels
    }

    score_meters_group = {
        level: [ScoreMeter(level+'_'+metric_name, metric_func, need_score, task=task, window_size=len(data_loader))
                for metric_name, (metric_func, need_score) in metrics.items()] for level in levels
    }

    out_history = collections.defaultdict(list)
    target_history = collections.defaultdict(list)

    for batch in data_loader:
        if torch.cuda.is_available():
            batch = [item.cuda()
                     if isinstance(item, torch.Tensor) else item
                     for item in batch]

        if multiple_clip:
            out_dict = predict_multiple_clip(
                model, batch, task)
        else:
            out_dict = predict_single_clip(
                model, batch)

        for level in out_dict:
            loss_meters = loss_meter_group[level]
            out, loss_dict, targets = out_dict[level]

            # update loss_meters
            update_losses(loss_dict, loss_meters)

            out_history[level].append(out.detach().cpu())
            target_history[level].append(targets.detach().cpu())

    for level in levels:
        out_history_cat = torch.cat(out_history[level])
        target_history_cat = torch.cat(target_history[level])

        score_meters = score_meters_group[level]
        for i in range(len(score_meters)):
            sm = score_meters[i]
            sm.update(out_history_cat, target_history_cat)
            score_meters[i] = sm
        score_meters_group[level] = score_meters

    loss_val_group = {level: get_running_losses(
        loss_meter_group[level]) for level in levels}
    score_val_group = {level: {
        sm.metric_name: sm.avg for sm in score_meters_group[level]} for level in levels}

    return loss_val_group, score_val_group, levels


def train_one_epoch(model, optimizer, train_loader, valid_loader,
                    fold, n_folds, epoch, n_epochs, validation_freq, metrics,
                    train_logger=None, valid_logger=None,
                    task='classification', multiple_clip=True,
                    lr_scheduler=None, warmup_scheduler=None):
    """
    Usage:

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, train_loader,
                            valid_loader, epoch, validation_freq=10)
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)

    """
    # consider first positioned metric as main metric
    main_metric = list(metrics)[0]

    epoch_step = 0

    # initialize running_vars
    loss_meters = collections.defaultdict(
        partial(AverageMeter, window_size=validation_freq))
    score_meters = [ScoreMeter('clip'+'_'+metric_name, metric_func, need_score, task=task, window_size=validation_freq)
                    for metric_name, (metric_func, need_score) in metrics.items()]

    print("Start Training...")
    print()

    best_val_score = -1.0
    best_model_wts = None

    out_history = []
    target_history = []

    model.train()
    enable_tsn = train_loader.dataset.dataset.num_samples > 1
    for batch in train_loader:
        if torch.cuda.is_available():
            batch = [item.cuda()
                     if isinstance(item, torch.Tensor) else item
                     for item in batch]

        images, masks, targets = batch

        # if len(targets.cpu().unique()) == 1:
        #     print('Continue highly imbalance batch...')
        #     continue
        out, loss_dict = model(
            images, masks, targets=targets, enable_tsn=enable_tsn)

        out_history.append(out.detach().cpu())
        target_history.append(targets.detach().cpu())

        loss_dict = {k: loss_dict[k].mean() for k in loss_dict}
        losses = sum(loss for loss in loss_dict.values())

        # backprop on multi-task loss
        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        if warmup_scheduler is not None:
            if epoch <= int(n_epochs * 0.75):
                warmup_scheduler.step()
            else:
                if lr_scheduler is not None:
                    lr_scheduler.step()

        epoch_step += 1
        # set global_step for logging
        train_logger.global_step = epoch * len(train_loader) + epoch_step
        valid_logger.global_step = epoch * len(train_loader) + epoch_step

        # update losses
        update_losses(loss_dict, loss_meters)
        if epoch_step % validation_freq == 0:
            running_loss = get_running_losses(loss_meters)
            running_loss_log = "*"*5 + " Multi-task Losses " + "*"*5 + "\n" + \
                "\n".join([f"- {k}: {v:.4f}" for k, v in running_loss.items()])

            print(
                '[Train] FOLD:[{fold}/{n_folds}] EP:[{epoch}/{n_epochs}]\tSTEPS:[{epoch_step}/{n_steps}]\n{running_loss_log}'.format(
                    fold=fold, n_folds=n_folds,
                    epoch=epoch, n_epochs=n_epochs,
                    epoch_step=epoch_step,
                    n_steps=len(train_loader),
                    running_loss_log=running_loss_log
                ))

            # evaluate with validation dataset
            valid_loss_val_group, valid_score_val_group, levels = evaluate(
                model, valid_loader, metrics=metrics, task=task, multiple_clip=multiple_clip)

            main_valid_scores = {
                level: valid_score_val_group[level][level+'_'+main_metric] for level in levels}
            # level_main = 'video' if multiple_clip else 'clip'
            cur_val_score = main_valid_scores["clip"]
            if cur_val_score >= best_val_score:
                best_val_score = cur_val_score
                best_model_wts = copy.deepcopy(model.state_dict())

            valid_clip_loss = valid_loss_val_group['clip']
            valid_clip_scores = valid_score_val_group['clip']

            valid_clip_loss_log = "*"*5 + " Multi-task Losses " + "*"*5 + "\n" + \
                "\n".join([f"- {k}: {v:.4f}" for k,
                           v in valid_clip_loss.items()])
            valid_clip_score_log = "*"*5 + " Scores " + "*"*5 + "\n" + "{main_metric} : {valid_clip_score}\n".format(
                valid_clip_loss_log=valid_clip_loss_log, valid_clip_score=valid_clip_scores[
                    'clip_'+main_metric],
                main_metric=main_metric
            )

            log_str = "[Valid] FOLD:[{fold}/{n_folds}] EP:[{epoch}/{n_epochs}]\tSTEPS:[{epoch_step}/{n_steps}]\n{valid_clip_loss_log}\n{valid_clip_score_log}\n".format(
                fold=fold, n_folds=n_folds,
                epoch=epoch, n_epochs=n_epochs,
                epoch_step=epoch_step,
                n_steps=len(train_loader),
                valid_clip_loss_log=valid_clip_loss_log,
                valid_clip_score_log=valid_clip_score_log)

            if multiple_clip:
                valid_video_loss = valid_loss_val_group['video']
                valid_video_scores = valid_score_val_group['video']
                valid_video_loss_log = "*"*5 + " Multi-task Losses " + "*"*5 + "\n" + \
                    "\n".join([f"- {k}: {v:.4f}" for k,
                               v in valid_video_loss.items()])

                valid_video_score_log = "*"*5 + " Scores " + "*"*5 + "\n" + "{main_metric} : {valid_video_score}\n".format(
                    valid_video_loss_log=valid_video_loss_log, valid_video_score=valid_video_scores[
                        'video_'+main_metric],
                    main_metric=main_metric
                )

                log_str += "[video] {valid_video_loss_log}\n {valid_video_score_log}\t ".format(
                    valid_video_loss_log=valid_video_loss_log, valid_video_score_log=valid_video_score_log
                )

            print(log_str)

            out_history_cat = torch.cat(out_history)
            target_history_cat = torch.cat(target_history)
            for i in range(len(score_meters)):
                sm = score_meters[i]
                sm.update(out_history_cat, target_history_cat)
                score_meters[i] = sm

            running_scores = {sm.metric_name: sm.avg for sm in score_meters}
            # update visdom window
            train_logger.write(**{
                'lr': optimizer.param_groups[0]['lr'],
                **{'clip_' + k: v for k,
                   v in running_loss.items()},
                **running_scores
            })

            eval_log = {}
            for level in levels:
                eval_log.update({
                    **{level+'_' + k: v for k, v in valid_loss_val_group[level].items()},
                    **valid_score_val_group[level]
                })

            valid_logger.write(**eval_log)

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, CosineAnnealingLR):
            pass
        else:
            lr_scheduler.step()

    # load best_wts
    model.load_state_dict(best_model_wts)

    return best_val_score, model


def create_dir(cb):
    def wrapper_func(*args, **kwargs):
        save_dir = kwargs.get("save_dir")
        if save_dir is not None:
            print("creating a new directory at {} .....".format(save_dir))
            os.system("mkdir -p {}".format(save_dir))

        return cb(*args, **kwargs)

    return wrapper_func


class NeuralNetworks(object):
    def __init__(self, model, optimizer, n_folds=1, fold=1,
                 lr_scheduler=None, warmup_scheduler=None, default_metrics_callbacks=None, task='classification'):

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warmup_scheduler = warmup_scheduler
        self.n_folds = n_folds  # default : 1, if bigger than 1, cross-validate
        self.fold = fold
        self.task = task

    @create_dir
    def train(self, train_loader, valid_loader, n_epochs, validation_freq=10, multiple_clip=False, metrics=None, save_dir=''):

        fold = self.fold

        model_path = os.path.join(
            save_dir, 'model_fold-{}.pth'.format(fold))
        env_name = os.path.basename(save_dir) + "-fold-{}".format(fold)
        env_name = env_name + "_" + \
            datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")

        # visdom plot interface
        plotter = VisdomPlotter(env_name=env_name)

        # train/validation logging objs
        train_logger = Logger(plotter, phase='train')
        valid_logger = Logger(plotter, phase='valid')

        # early-stopping
        # es = EarlyStopping(mode='max', patience=20,
        #                    percentage=True, min_delta=0.01)
        es = None

        best_score = -1.0
        for epoch in range(n_epochs):
            # validation_score is  `evaluate(...)[list(metrics)[0]]`
            validation_score, new_model = train_one_epoch(self.model, self.optimizer, train_loader, valid_loader,
                                                          fold, self.n_folds, epoch, n_epochs, validation_freq,
                                                          lr_scheduler=self.lr_scheduler, warmup_scheduler=self.warmup_scheduler,
                                                          metrics=self.get_metrics(metrics), train_logger=train_logger, valid_logger=valid_logger,
                                                          task=self.task, multiple_clip=multiple_clip)
            if es is not None:
                # early stop criterion is met, we can stop now
                if es.step(torch.tensor(validation_score)):
                    print("\nEarly stop the training loop...\n")
                    break

            if validation_score >= best_score:
                # save-best model
                torch.save(new_model, model_path)

                # update best_score
                best_score = validation_score
                print(
                    f"@@ EPOCH : {epoch} Best `{list(metrics)[0]}` has been updated (value: {best_score:.5f}). Save best model at {model_path}")
        plotter.viz.save([plotter.env])

        # read json file
        import json
        env_file = os.path.join(
            os.environ["HOME"], ".visdom", plotter.env + ".json")
        logs = json.load(open(env_file))["jsons"]

        lr_history = []
        train_history = []
        valid_history = []
        for win_title, win_id in plotter.plots.items():
            log_data = logs[win_id]["content"]["data"]

            if len(log_data) > 1:
                train_logs, valid_logs = log_data
                train_history.append(
                    [win_title, train_logs["x"], train_logs["y"]])
                valid_history.append(
                    [win_title, valid_logs["x"], valid_logs["y"]])
            else:
                if win_title == "lr":
                    lr_history.append(
                        [win_title, log_data[0]["x"], log_data[0]["y"]])
                else:
                    valid_history.append(
                        [win_title, log_data[0]["x"], log_data[0]["y"]])

        def history_to_dataframe(history):
            history = np.array(history)
            titles = history[:, 0].repeat(2).tolist()
            history = history[:, 1:].tolist()
            history = pd.DataFrame(
                np.array(list(zip(*history))).transpose((2, 1, 0)).reshape(-1, len(history)*2), columns=[titles, ["x", "y"]*len(history)])

            return history

        train_history_df = history_to_dataframe(train_history)
        valid_history_df = history_to_dataframe(valid_history)
        lr_history_df = history_to_dataframe(lr_history)

        with pd.ExcelWriter(env_file.replace(".json", ".xlsx")) as writer:
            train_history_df.to_excel(writer, sheet_name='train_history')
            valid_history_df.to_excel(writer, sheet_name='valid_history')
            lr_history_df.to_excel(writer, sheet_name='lr_history')

    def test(self, test_loader, metrics=None, multiple_clip=False, pretrained_path=''):
        trained_model = load_pretrained_ckpt(
            self.model, pretrained_path=pretrained_path)

        return evaluate(trained_model, test_loader,
                        metrics=self.get_metrics(metrics),
                        task=self.task, multiple_clip=multiple_clip)

    def get_metrics(self, metrics):
        if metrics is None:
            metrics = list(self.default_metrics_callbacks.keys())

        if isinstance(metrics, list):
            items = []
            for m in metrics:
                if self.default_metrics_callbacks.get(m) is not None:
                    items.append((m, self.default_metrics_callbacks.get(m)))
                else:
                    print('### Metric `{}` is ignored...'.format(m))
            metrics = collections.OrderedDict(items)
        elif isinstance(metrics, dict):
            metrics = collections.OrderedDict(metrics)
        else:
            raise ValueError("invalid metrics")
        return metrics

    def get_features(self, layer_name=''):
        # TODO. return feature-map by layername
        """
        Module wrapper that returns intermediate layers from a model
        It has a strong assumption that the modules have been registered
        into the model in the same order as they are used.
        This means that one should **not** reuse the same nn.Module
        twice in the forward if you want this to work.
        Additionally, it is only able to query submodules that are directly
        assigned to the model. So if `model` is passed, `model.feature1` can
        be returned, but not `model.feature1.layer2`.
        Arguments:
            model (nn.Module): model on which we will extract the features
            return_layers (Dict[name, new_name]): a dict containing the names
                of the modules for which the activations will be returned as
                the key of the dict, and the value of the dict is the name
                of the returned activation (which the user can specify).
        Examples::
            >>> m = torchvision.models.resnet18(pretrained=True)
            >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
            >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
            >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
            >>> out = new_m(torch.rand(1, 3, 224, 224))
            >>> print([(k, v.shape) for k, v in out.items()])
            >>>     [('feat1', torch.Size([1, 64, 56, 56])),
            >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
        """
        raise NotImplementedError('yet..')


class VideoClassifier(NeuralNetworks):

    """
    """

    __metrics = {
        'accuracy': (lambda y_true, y_pred: sklearn.metrics.accuracy_score(y_true, y_pred), False),
        'roc_auc': (lambda y_true, y_score: sklearn.metrics.roc_auc_score(y_true, y_score), True),
        'ap': (lambda y_true, y_score: sklearn.metrics.average_precision_score(y_true, y_score), True),
        'f1-score': (lambda y_true, y_pred: sklearn.metrics.f1_score(y_true, y_pred), False),
    }

    def __init__(self, model, optimizer, n_folds=1, fold=1,
                 lr_scheduler=None, warmup_scheduler=None):
        super(VideoClassifier, self).__init__(
            model, optimizer, n_folds, fold, lr_scheduler, warmup_scheduler, task='classification')

    @property
    def default_metrics_callbacks(self):
        return self.__metrics


class VideoRegressor(NeuralNetworks):
    __metrics = {
        'mae': (lambda y_true, y_pred: sklearn.metrics.mean_absolute_error(y_true, y_pred), False),
        'r2': (lambda y_true, y_pred: sklearn.metrics.r2_score(y_pred, y_true), False),
        'msle': (lambda y_true, y_pred: sklearn.metrics.mean_squared_log_error(y_true, y_pred), False),
    }

    def __init__(self, model, optimizer, n_folds=1, fold=1,
                 lr_scheduler=None, warmup_scheduler=None):

        super(VideoRegressor, self).__init__(
            model, optimizer, n_folds, fold, lr_scheduler, warmup_scheduler, task='regression')
