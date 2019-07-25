import argparse

def parse_opts():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--dataset',
        default='Gaitparams_PD',
        type=str,
        help='Dataset to use ( Gaitparams_PD | others... )',
    )
    parser.add_argument(
        '--input_file',
        default="../../preprocess/data/person_detection_and_tracking_results_drop.pkl",
        type=str,
        help='File path of input dataframe file (.pkl)',
    )
    parser.add_argument(
        '--target_file',
        default="../../preprocess/data/targets_dataframe.pkl",
        type=str,
        help='File path of target dataframe file (.pkl)',
    )
    parser.add_argument(
        '--log_dir',
        default="./logs",
        type=str,
        help='Directory path of log',
    )

    parser.add_argument(
        '--frame_home',
        default="/data/GaitData/CroppedFrameArrays",
        type=str,
        help='Directory path of frames cropped by YOLO',
    )
    parser.add_argument(
        '--target_columns',
        default='all',
        type=str,
        help='Target columns to use ( all | spatial | temporal )',
    )
    parser.add_argument(
        '--pretrained_path',
        default='./pretrained/resnet-50-kinetics.pth',
        type=str,
        help='Path to pretrained model file',
    )
    parser.add_argument(
        '--backbone',
        default=None,
        type=str,
        help='Which networks to use ( resnet | others )',
    )
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of backbone networks',
    )
    parser.add_argument(
        '--device_ids',
        default="0,1,2,3",
        type=str,
        help='GPU ids to use',
    )

    parser.add_argument(
        '--sample_size',
        default=(384,128),
        type=tuple,
        help='Input image size (h,w) fed into backbone',
    )
    parser.add_argument(
        '--sample_duration',
        default=50,
        type=int,
        help='Input image lenght fed into backbone',
    )
    parser.add_argument(
        '--delta',
        default=6,
        type=int,
        help='Sampling frequency of video frame ( 1 frame / \delta )',
    )
    parser.add_argument(
        '--multi_gpu',
        action='store_true',
        help='If true, enable multi GPU system.')
    parser.set_defaults(multi_gpu=True)
    parser.add_argument(
        '--mode',
        default='train',
        type=str,
        help='Specify mode ( training | testing )',
    )
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='Batch size',
    )
    parser.add_argument(
        '--n_iter',
        default=100,
        type=int,
        help='Number of iterations for training',
    )
    parser.add_argument(
        '--n_threads',
        default=16,
        type=int,
        help='Number of thread to use for dataloader',
    )
    parser.add_argument(
        '--CV',
        default=5,
        type=int,
        help='Number of splits for K-fold cross validation',
    )
    parser.add_argument(
        '--train_crop',
        default='',
        type=str,
        help=
        'Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)'
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument(
        '--mean_dataset',
        default='activitynet',
        type=str,
        help=
        'dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument(
        '--no_mean_norm',
        action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=True)
    parser.add_argument(
        '--std_norm',
        action='store_true',
        help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--ckpt_dir',
        default='./ckpt_repos',
        type=str,
        help='Directory where trained model is saved.')
    parser.add_argument(
        '--test_epoch',
        default=90,
        type=str,
        help='Epoch to test')
    parser.add_argument(
        '--score_avg',
        action='store_true',
        help='If true, averaged score is calculated. If false, scores per each gait-params are calculated')
    parser.set_defaults(score_avg=False)
    parser.add_argument(
        '--model_arch',
        default='HPP',
        type=str,
        help='Specify mode for regression model (naive | HPP)')
    parser.add_argument(
        '--merge_type',
        default='addition',
        type=str,
        help='Merge type of multiple scale HPP features (addition | 1x1_C)')

    # and so on...

    args = parser.parse_args()

    return args





