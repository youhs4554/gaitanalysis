from opts import parse_opts
from utils.generate_model import init_state
from utils.transforms import *
from utils.train_utils import Trainer, Logger
from utils.testing_utils import Tester
from utils.target_columns import get_target_columns, get_target_columns_by_group
import utils.visualization as viz

import datasets.gaitregression
from utils.mean import get_mean, get_std
from utils.parallel import DataParallelModel, DataParallelCriterion

import sklearn

import os
from torch import nn


if __name__ == '__main__':
    opt = parse_opts()

    target_columns = get_target_columns(opt)

    # define regression model
    net, optimizer, scheduler = init_state(opt)

    criterion = nn.MSELoss()
    criterion = DataParallelCriterion(criterion, device_ids=eval(opt.device_ids+','))

    opt.arch = '{}-{}'.format(opt.backbone, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])

    spatial_transform = Compose([
        ### crop_method, #### disable crop method
        RandomHorizontalFlip(),
        ToTensor(opt.norm_value), norm_method
    ])

    ### temporal_transform = TemporalRandomCrop(opt.sample_duration) ### disable temporal crop method

    target_transform_func = sklearn.preprocessing.QuantileTransformer

    if opt.dataset=='Gaitparams_PD':
        # prepare dataset  (train/test split)
        data = datasets.gaitregression.prepare_dataset(input_file=opt.input_file, target_file=opt.target_file,
                                              target_columns=target_columns)

        train_ds = datasets.gaitregression.GAITDataset(X=data['train_X'], y=data['train_y'], opt=opt)
        test_ds = datasets.gaitregression.GAITDataset(X=data['test_X'], y=data['test_y'], opt=opt)

        dataloader_generator = datasets.gaitregression.generate_dataloader_for_crossvalidation

        # target transform
        target_transform = target_transform_func(random_state=0, output_distribution='normal').fit(
            data['target_df'].values)

    else:
        NotImplementedError("Does not support other datasets until now..")

    if opt.mode == 'train':
        train_logger = Logger(
            os.path.join(opt.log_dir,
                         opt.model_arch + '_' + opt.merge_type + '_' + 'finetuned_with' + '_' + opt.arch,
                         'train.tsv'),
            ['epoch@split', 'loss', 'score', 'lr'])
        valid_logger = Logger(
            os.path.join(opt.log_dir,
                         opt.model_arch + '_' + opt.merge_type + '_' + 'finetuned_with' + '_' + opt.arch,
                         'valid.tsv'),
            ['epoch@split', 'loss', 'score'])

        trainer = Trainer(model=net, criterion=criterion,
                          optimizer=optimizer, scheduler=scheduler,
                          opt=opt,
                          train_logger=train_logger, val_logger=valid_logger,
                          input_transform=spatial_transform, target_transform=target_transform)

        trainer.fit(ds=train_ds, dataloader_generator=dataloader_generator)

    elif opt.mode == 'test':
        if opt.model_arch == 'HPP':
            model_path = os.path.join(opt.ckpt_dir,
                         opt.model_arch + '_' + opt.merge_type + '_' + 'finetuned_with' + '_' + opt.arch, 'save_' + opt.test_epoch + '.pth')
        elif opt.model_arch == 'naive':
            model_path = os.path.join(opt.ckpt_dir,
                                      opt.model_arch + '_' + 'finetuned_with' + '_' + opt.arch,
                                      'save_' + opt.test_epoch + '.pth')

        print(f"Load trained model from {model_path}...")

        # laod pre-trained model
        pretrain = torch.load(model_path)
        net.load_state_dict(pretrain['state_dict'])

        test_logger = Logger(
            os.path.join(opt.log_dir, 'test.log'),
            target_columns)

        tester = Tester(model=net,
                        opt=opt,
                        test_logger=test_logger, score_func=sklearn.metrics.r2_score,
                        input_transform=spatial_transform, target_transform=target_transform)

        y_true, y_pred = tester.fit(ds=test_ds)

        # visualize
        viz.scatterplots(target_columns, y_true, y_pred, save_dir='./tmp')

        for group, grid_size, fig_size in zip(['temporal', 'spatial', 'etc'], [(4,2),(2,2),(2,2)], [(20,20),(20,11),(20,11)]):
            group_cols = get_target_columns_by_group(group)
            viz.dist_plots(target_columns, group_cols, y_true, y_pred, save_dir='./tmp', grid_size=grid_size, figsize=fig_size, group=group)
            viz.margin_plots(target_columns, group_cols, y_true, y_pred, save_dir='./tmp', grid_size=grid_size, figsize=fig_size, group=group)


    else:
        ValueError("Invalid mode. Select ( train | test )")
