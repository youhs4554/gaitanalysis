from torch import nn
import os
import time
import sklearn
import datasets.gaitregression
from utils.parallel import DataParallelCriterion
import visdom
from visualdl import LogWriter
from opts import parse_opts
from utils.generate_model import init_state, load_trained_ckpt
from utils.transforms import (
    Compose, ToTensor, MultiScaleRandomCrop, MultiScaleCornerCrop, Normalize,
    TemporalRandomCrop, TemporalCenterCrop, LoopPadding)
from utils.train_utils import Trainer, Logger
from utils.testing_utils import Tester
from utils.target_columns import get_target_columns
import utils.visualization as viz
from utils.mean import get_mean, get_std
from utils.preprocessing import (
    PatientLocalizer,
    COPAnalyizer,
    HumanScaleAnalyizer,
    Worker,
)
import torch.nn.functional as F
import torchvision.transforms as TF
from preprocess.darknet.python.extract_bbox import set_gpu

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    opt = parse_opts()

    # attention indicator
    opt.attention_str = 'Attentive' if opt.attention else 'NonAttentive'
    opt.group_str = f"G{opt.n_groups}" if opt.n_groups > 0 else ''
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

    spatial_transform = {
        "train": Compose(
            [
                TF.RandomRotation(degrees=(0, 0)),
                TF.RandomResizedCrop(size=opt.sample_size,
                                     scale=(opt.sample_size/opt.img_size, 1.0)),
                ToTensor(opt.norm_value),
                norm_method,
            ]
        ),
        "test": Compose(
            [
                TF.CenterCrop(opt.sample_size),
                ToTensor(opt.norm_value),
                norm_method,
            ]
        ),
    }

    temporal_transform = {
        "train": None,  # TemporalRandomCrop(opt.sample_duration),
        "test": None,  # TemporalCenterCrop(opt.sample_duration),
    }

    from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, QuantileTransformer
    from sklearn.pipeline import Pipeline

    # target transform
    # target_transform = QuantileTransformer(
    #     random_state=0, output_distribution="normal"
    # )

    target_transform = StandardScaler()

    model_indicator = '_'.join(filter(lambda x: x != '', [opt.attention_str,
                                                          opt.model_arch,
                                                          opt.merge_type,
                                                          opt.arch,
                                                          opt.group_str]))

    # result log path
    logpath = os.path.join(opt.log_dir, model_indicator,
                           opt.mode)

    opt.logpath = logpath

    plotter = viz.VisdomPlotter(env_name=model_indicator)

    if opt.dataset == "Gaitparams_PD":
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
            X=data["train_X"], y=data["train_y"], opt=opt, phase='train',
        )

        test_ds = ds_class(
            X=data["test_X"], y=data["test_y"], opt=opt, phase='test',
        )

        dataloader_generator = (
            datasets.gaitregression.generate_dataloader_for_crossvalidation
        )

    else:
        NotImplementedError("Does not support other datasets until now..")

    if opt.mode == "train":
        trainer = Trainer(
            model=net,
            criterion1=criterion1, criterion2=criterion2,
            optimizer=optimizer,
            scheduler=scheduler,
            opt=opt,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
        )

        trainer.fit(
            ds=train_ds, dataloader_generator=dataloader_generator,
            ds_class=ds_class,
            plotter=plotter)

    elif opt.mode == "test":
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
            ds=test_ds, plotter=plotter, criterion=criterion2)

        # visualize
        viz.scatterplots(target_columns, y_true, y_pred, save_dir="./tmp")

    elif opt.mode == "demo":
        from demo import app as flask_app
        # patient localizer & interval selector
        set_gpu(opt.device_yolo)

        localizer = PatientLocalizer(darknet_api_home=opt.darknet_api_home)

        interval_selector = None
        if opt.interval_sel == "COP":
            interval_selector = COPAnalyizer(opt.meta_home, opt.fps)
        elif opt.interval_sel == "Scale":
            interval_selector = HumanScaleAnalyizer(opt)

        worker = Worker(localizer, interval_selector, opt)

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
