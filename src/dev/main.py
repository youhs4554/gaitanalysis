from utils.preprocessing import (
    PatientLocalizer,
    COPAnalyizer,
    HumanScaleAnalyizer,
    Worker,
)
from utils.mean import get_mean, get_std
import utils.visualization as viz
from utils.target_columns import get_target_columns
from utils.testing_utils import Tester
from utils.train_utils import Trainer
from utils.transforms import (
    Compose, ToTensor,
    MultiScaleRandomCrop,
    MultiScaleCornerCrop,
    Normalize,
    TemporalRandomCrop,
    TemporalCenterCrop,
    LoopPadding,
    RandomRotation3D,
    RandomResizedCrop3D,
    RandomResizedCrop3D,
    ToTensor3D,
    Normalize3D,
    CenterCrop3D,
    Resize3D,
    RandomHorizontalFlip3D
)
from utils.generate_model import init_state, load_trained_ckpt
import torch
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    FunctionTransformer,
    StandardScaler,
    QuantileTransformer,
    OneHotEncoder,
)
from preprocess.darknet.python.extract_bbox import set_gpu
import torchvision.transforms as TF
import torch.nn.functional as F
from opts import parse_opts
from visualdl import LogWriter
import visdom
from utils.parallel import DataParallelCriterion
import datasets.benchmark
import datasets.gaitregression
import sklearn
import time
import os
from torch import nn
import warnings

warnings.filterwarnings("ignore")


def prepare_data(opt, fold=1):
    opt.data_root = opt.data_root.rstrip("/")
    opt.benchmark = opt.dataset in ["URFD", "MulticamFD"]

    # attention indicator
    opt.attention_str = "Attentive" if opt.attention else "NonAttentive"
    opt.group_str = f"G{opt.n_groups}" if opt.n_groups > 0 else ""
    opt.arch = "{}-{}".format(opt.backbone, opt.model_depth)

    target_columns = get_target_columns(opt)

    # define regression model
    net, criterion1, criterion2, optimizer, scheduler = init_state(opt)

    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value, dataset=opt.mean_dataset)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if opt.train_crop == "random":
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == "corner":
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == "center":
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=["c"]
        )

    model_indicator = "_".join(
        filter(
            lambda x: x != "",
            [
                opt.attention_str,
                opt.model_arch,
                opt.merge_type,
                opt.arch,
                opt.group_str,
            ],
        )
    )
    opt.model_indicator = model_indicator

    # result log path
    logpath = os.path.join(opt.log_dir, model_indicator, opt.mode)
    opt.logpath = logpath

    plotter = viz.VisdomPlotter(
        env_name=opt.dataset + '_' +
        opt.model_indicator + "-fold-{}".format(fold)
    )

    if opt.dataset == "Gaitparams_PD":
        spatial_transform = {
            "train": Compose(
                [
                    TF.RandomRotation(degrees=(0, 0)),
                    TF.RandomResizedCrop(
                        size=opt.sample_size,
                        scale=(opt.sample_size / opt.img_size, 1.0),
                    ),
                    ToTensor(opt.norm_value),
                    norm_method,
                ]
            ),
            "test": Compose(
                [TF.CenterCrop(opt.sample_size), ToTensor(
                    opt.norm_value), norm_method]
            ),
        }

        temporal_transform = {
            "train": None,  # TemporalRandomCrop(opt.sample_duration),
            "test": None,  # TemporalCenterCrop(opt.sample_duration),
        }

        target_transform = StandardScaler()

        # prepare dataset  (train/test split)
        data = datasets.gaitregression.prepare_dataset(
            input_file=opt.input_file,
            target_file=opt.target_file,
            target_columns=target_columns,
            chunk_parts=opt.chunk_parts,
            target_transform=target_transform,
        )

        if opt.with_segmentation:
            ds_class = datasets.gaitregression.GAITSegRegDataset
        else:
            ds_class = datasets.gaitregression.GAITDataset

        train_ds = ds_class(
            X=data["train_X"],
            y=data["train_y"],
            opt=opt,
            phase="train",
            spatial_transform=spatial_transform["train"],
            temporal_transform=temporal_transform["train"],
        )

        test_ds = ds_class(
            X=data["test_X"],
            y=data["test_y"],
            opt=opt,
            phase="test",
            spatial_transform=spatial_transform["test"],
            temporal_transform=temporal_transform["test"],
        )

    elif opt.benchmark:
        spatial_transform = {
            "train": Compose(
                [
                    Resize3D(size=opt.img_size),
                    RandomResizedCrop3D(
                        transform2D=TF.RandomResizedCrop(
                            size=(opt.sample_size, opt.sample_size))),
                    # scale=(opt.sample_size / opt.img_size, 1.0))),
                    RandomHorizontalFlip3D(),
                    ToTensor3D(),
                    Normalize3D(
                        mean=opt.mean,
                        std=opt.std
                    ),
                ]
            ),
            "test": Compose(
                [
                    Resize3D(size=opt.img_size),
                    CenterCrop3D((opt.sample_size, opt.sample_size)),
                    ToTensor3D(),
                    Normalize3D(
                        mean=opt.mean,
                        std=opt.std
                    ),
                ]
            ),
        }

        temporal_transform = {
            "train": None,  # TemporalRandomCrop(opt.sample_duration),
            "test": None,  # TemporalCenterCrop(opt.sample_duration),
        }

        target_transform = None

        train_ds = datasets.benchmark.FallDataset(
            root=opt.data_root,
            train=True,
            annotation_path=os.path.join(
                os.path.dirname(opt.data_root), "TrainTestlist"
            ),
            detection_file_path=opt.input_file,
            frames_per_clip=opt.sample_duration,
            # step_between_clips=10,
            fold=fold,
            transform=spatial_transform["train"],
            preCrop=opt.precrop,
        )

        train_ds[10]

        test_ds = datasets.benchmark.FallDataset(
            root=opt.data_root,
            train=False,
            annotation_path=os.path.join(
                os.path.dirname(opt.data_root), "TrainTestlist"
            ),
            detection_file_path=opt.input_file,
            frames_per_clip=opt.sample_duration,
            # step_between_clips=10,
            fold=fold,
            transform=spatial_transform["test"],
            preCrop=opt.precrop,
        )

    else:
        NotImplementedError("Does not support other datasets until now..")

    # Construct train/test dataloader for selected fold
    train_loader = torch.utils.data.DataLoader(train_ds, pin_memory=True,
                                               batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.n_threads)
    test_loader = torch.utils.data.DataLoader(test_ds, pin_memory=True,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.n_threads)

    return (
        opt,
        net,
        criterion1,
        criterion2,
        optimizer,
        scheduler,
        spatial_transform,
        temporal_transform,
        target_transform,
        plotter,
        train_loader,
        test_loader,
        target_columns,
    )


def train(opt, fold, metrice='f1-score'):
    opt, net, criterion1, criterion2, optimizer, scheduler, spatial_transform, temporal_transform, target_transform, plotter, train_loader, test_loader, target_columns = \
        prepare_data(opt, fold)

    trainer = Trainer(
        model=net,
        criterion1=criterion1,
        criterion2=criterion2,
        optimizer=optimizer,
        scheduler=None,
        opt=opt,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        plotter=plotter, fold=fold
    )

    score_dict = trainer.fit(train_loader, test_loader,
                             metrice=metrice)

    return score_dict


def cross_validation(opt, metrice='f1-score'):
    cv_scores = []
    for fold in range(1, opt.n_folds+1):
        score_dict = train(opt, fold, metrice=metrice)
        cv_scores.append(score_dict[metrice])

    print()
    print('-'*64)
    print('{0}-fold CV result with {1} : {2:.4f}'.format(opt.n_folds,
                                                         metrice, np.mean(cv_scores)))
    print()
    print('-'*64)

    return cv_scores


def test(opt, fold):

    opt, net, criterion1, criterion2, optimizer, scheduler, spatial_transform, temporal_transform, target_transform, plotter, train_loader, test_loader, target_columns = prepare_data(
        opt, fold)

    net = load_trained_ckpt(opt, net)

    tester = Tester(
        model=net,
        opt=opt,
        score_func=sklearn.metrics.r2_score,
        spatial_transform=spatial_transform[opt.mode],
        temporal_transform=temporal_transform[opt.mode],
        target_transform=target_transform,
    )

    y_true, y_pred = tester.fit(
        test_loader=test_loader, criterion=criterion2
    )

    # visualize
    viz.scatterplots(target_columns, y_true, y_pred, save_dir="./tmp")


def demo(opt):
    opt, net, criterion1, criterion2, optimizer, scheduler, spatial_transform, temporal_transform, target_transform, plotter, train_loader, test_loader, target_columns = prepare_data(
        opt)

    from demo import app as flask_app

    # patient localizer & interval selector
    if opt.segm_method == "yolo":
        set_gpu(opt.device_yolo)

    interval_selector, localizer = None, None
    if opt.interval_sel == "COP":
        interval_selector = COPAnalyizer(opt.meta_home, opt.fps)
        localizer = PatientLocalizer(darknet_api_home=opt.darknet_api_home)
    elif opt.interval_sel == "Scale":
        interval_selector = HumanScaleAnalyizer(opt)
    elif opt.interval_sel == "DAPs":
        raise NotImplementedError(
            "DAPs interval selection is not implemented yet.")

    worker = Worker(localizer, interval_selector, opt)
    worker.run()

    # set runner
    flask_app.set_runner(
        opt,
        net,
        localizer,
        interval_selector,
        worker,
        spatial_transform,
        target_transform,
        target_columns,
    )

    # run flask server
    print("Demo server is waiting for you...")
    flask_app.app.run(host="0.0.0.0", port=opt.port)


def main():
    opt = parse_opts()

    if opt.mode == "cv":
        cross_validation(opt)  # K-fold cross-validation (cv)
    elif opt.mode == 'train':
        train(opt, fold=1)   # train-for single split
    elif opt.mode == "test":
        test(opt, fold=1)    # test-for single split
    elif opt.mode == "demo":
        demo(opt)            # demo, running RESTful API server.


if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', True)

    main()
