from torch import nn
import os
import sklearn
import datasets.gaitregression
from utils.parallel import DataParallelCriterion
from opts import parse_opts
from utils.generate_model import init_state, load_trained_ckpt
from utils.transforms import (
    Compose, ToTensor, MultiScaleRandomCrop, MultiScaleCornerCrop, Normalize)
from utils.train_utils import Trainer, Logger
from utils.testing_utils import Tester
from utils.target_columns import (
    get_target_columns, get_target_columns_by_group)
import utils.visualization as viz
from utils.mean import get_mean, get_std
from utils.preprocessing import (
    PatientLocalizer,
    COPAnalyizer,
    HumanScaleAnalyizer,
    Worker,
)
from preprocess.darknet.python.extract_bbox import set_gpu

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    opt = parse_opts()

    target_columns = get_target_columns(opt)

    # define regression model
    net, optimizer, scheduler = init_state(opt)

    criterion = nn.MSELoss()
    criterion = DataParallelCriterion(
        criterion, device_ids=eval(opt.device_ids + ","))

    opt.arch = "{}-{}".format(opt.backbone, opt.model_depth)
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
                # crop_method, #### disable crop method
                # RandomHorizontalFlip(), ### disable flip
                ToTensor(opt.norm_value),
                norm_method,
            ]
        ),
        "test": Compose(
            [
                # crop_method, #### disable crop method
                ToTensor(opt.norm_value),
                norm_method,
            ]
        ),
    }

    # temporal_transform = TemporalRandomCrop(
    #     opt.sample_duration)  # disable temporal crop method

    target_transform_func = sklearn.preprocessing.QuantileTransformer

    if opt.dataset == "Gaitparams_PD":
        # prepare dataset  (train/test split)
        data = datasets.gaitregression.prepare_dataset(
            input_file=opt.input_file,
            target_file=opt.target_file,
            target_columns=target_columns,
            chunk_parts=opt.chunk_parts,
        )

        train_ds = datasets.gaitregression.GAITDataset(
            X=data["train_X"], y=data["train_y"], opt=opt
        )
        test_ds = datasets.gaitregression.GAITDataset(
            X=data["test_X"], y=data["test_y"], opt=opt
        )

        dataloader_generator = (
            datasets.gaitregression.generate_dataloader_for_crossvalidation
        )

        # target transform
        target_transform = target_transform_func(
            random_state=0, output_distribution="normal"
        ).fit(data["target_df"].values)

    else:
        NotImplementedError("Does not support other datasets until now..")

    if opt.mode == "train":
        train_logger = Logger(
            os.path.join(
                opt.log_dir,
                opt.model_arch
                + "_"
                + opt.merge_type
                + "_"
                + "finetuned_with"
                + "_"
                + opt.arch,
                "train.tsv",
            ),
            ["epoch@split", "loss", "score", "lr"],
        )
        valid_logger = Logger(
            os.path.join(
                opt.log_dir,
                opt.model_arch
                + "_"
                + opt.merge_type
                + "_"
                + "finetuned_with"
                + "_"
                + opt.arch,
                "valid.tsv",
            ),
            ["epoch@split", "loss", "score"],
        )

        trainer = Trainer(
            model=net,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            opt=opt,
            train_logger=train_logger,
            val_logger=valid_logger,
            input_transform=spatial_transform[opt.mode],
            target_transform=target_transform,
        )

        trainer.fit(ds=train_ds, dataloader_generator=dataloader_generator)

    elif opt.mode == "test":
        net = load_trained_ckpt(opt, net)

        test_logger = Logger(os.path.join(
            opt.log_dir, "test.log"), target_columns)

        tester = Tester(
            model=net,
            opt=opt,
            test_logger=test_logger,
            score_func=sklearn.metrics.r2_score,
            input_transform=spatial_transform[opt.mode],
            target_transform=target_transform,
        )

        y_true, y_pred = tester.fit(ds=test_ds)

        # visualize
        viz.scatterplots(target_columns, y_true, y_pred, save_dir="./tmp")

        for group, grid_size, fig_size in zip(
            ["temporal", "spatial", "etc"],
            [(4, 2), (2, 2), (2, 2)],
            [(20, 20), (20, 11), (20, 11)],
        ):
            group_cols = get_target_columns_by_group(group)
            viz.dist_plots(
                target_columns,
                group_cols,
                y_true,
                y_pred,
                save_dir="./tmp",
                grid_size=grid_size,
                figsize=fig_size,
                group=group,
            )
            viz.margin_plots(
                target_columns,
                group_cols,
                y_true,
                y_pred,
                save_dir="./tmp",
                grid_size=grid_size,
                figsize=fig_size,
                group=group,
            )

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
        flask_app.run(host="0.0.0.0", port=opt.port)
