import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        default='Gaitparams_PD',
        type=str,
        help='Dataset to use ( Gaitparams_PD | FallDown | and others; will be added later... )',
    )
    parser.add_argument(
        '--detection_file',
        default="../../preprocess/data/person_detection_and_tracking_results_drop.pkl",
        type=str,
        help='File path of detection dataframe file (.pkl)',
    )
    parser.add_argument(
        '--target_file',
        default="../../preprocess/data/targets_dataframe-Gaitparams_PD.pkl",
        type=str,
        help='File path of target dataframe file (.pkl)',
    )
    parser.add_argument(
        '--data_root',
        default=None,
        type=str,
        help='Directory path of data',
    )
    parser.add_argument(
        '--model_arch',
        default='HPP',
        type=str,
        help='Specify mode for regression model ( DefaultAGNet | ConcatenatedAGNet | FineTunedConvNet )')
    parser.add_argument(
        '--task',
        default='classification',
        type=str,
        help='Select task ( classification | regression ).'
    )
    parser.add_argument(
        '--chunk_vid_home',
        default="./video_chunks",
        type=str,
        help='Directory path of chunked vids list.',
    )
    parser.add_argument(
        '--chunk_parts',
        default=8,
        type=int,
        help='Number of chunk parts.',
    )
    parser.add_argument(
        '--meta_home',
        default="/data/GaitData/MetaData_converted",
        type=str,
        help='Directory path of meta data.',
    )
    parser.add_argument(
        '--darknet_api_home',
        default="../../preprocess/darknet",
        type=str,
        help='Main directory of darknet API.',
    )
    parser.add_argument(
        '--target_columns',
        default='basic',
        type=str,
        help='Target columns to use ( basic | advanced )',
    )
    parser.add_argument(
        '--pretrained_path',
        default='',
        type=str,
        help='Path to pretrained model file for concatenatedAGNet',
    )
    parser.add_argument(
        '--backbone',
        default='r2plus1d',
        type=str,
        help='Which networks to use as a backbone ( r2plus1d | r3d_18 | mc3_18 )',
    )
    parser.add_argument(
        '--device_yolo',
        default=9,
        type=int,
        help='GPU id for segmentation method (yolo | ..).',
    )
    parser.add_argument(
        '--img_size',
        default=144,
        type=int,
        help='Resized input image size',
    )
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Sampled input image size fed into backbone',
    )
    parser.add_argument(
        '--raw_h',
        default=480,
        type=int,
        help='Input raw frame height.',
    )
    parser.add_argument(
        '--raw_w',
        default=640,
        type=int,
        help='Input raw frame width.',
    )
    parser.add_argument(
        '--sample_duration',
        default=64,
        type=int,
        help='Input image lenght fed into backbone',
    )
    parser.add_argument(
        '--maxlen',
        default=300,
        type=int,
        help='Maximum sampling length of input image.',
    )
    parser.add_argument(
        '--fps',
        default=24,
        type=int,
        help='FPS of raw video frame',
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
    parser.set_defaults(multi_gpu=False)
    parser.add_argument(
        '--multiple_clip',
        action='store_true',
        help='If true, enable multiple_clip.')
    parser.set_defaults(multiple_clip=False)
    parser.add_argument(
        '--with_segmentation',
        action='store_true',
        help='If true, add segmentation dataset.'
    )
    parser.set_defaults(with_segmentation=False)
    parser.add_argument(
        '--mode',
        default='train',
        type=str,
        help='Specify mode ( cv | train | test | demo | preprocess__frame | preprocess__feature )',
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
        '--n_folds',
        default=5,
        type=int,
        help='Number of folds to cross validated. Should be between 1 to 5',
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument(
        '--max_gradnorm',
        default=5.0,
        type=float,
        help='Maximum value of gradients used for gradient-clipping.'
    )
    parser.add_argument(
        '--ckpt_dir',
        default="/data/GaitData/ckpt_dir",
        type=str,
        help='Directory where trained model is saved.')
    parser.add_argument(
        '--data_gen',
        action='store_true',
        help='If true, you can generate dataset.')
    parser.set_defaults(data_gen=False)
    parser.add_argument(
        '--balanced_batch',
        action='store_true',
        help='If true, you can balanced-class batch')
    parser.set_defaults(balanced_batch=False)
    parser.add_argument(
        '--interval_sel',
        default='COP',
        type=str,
        help='Inteval selection methods to use ( COP | Scale | DAPs ).'
    )
    parser.add_argument(
        '--port',
        default=40000,
        type=int,
        help='Service port for gaitanalysis.'
    )
    parser.add_argument(
        '--load_pretrained',
        action='store_true',
        help='If true, you can use pretrained feature arrray as input.')
    parser.set_defaults(load_pretrained=False)
    parser.set_defaults(precrop=False)
    parser.add_argument(
        '--disable_tracking',
        action='store_true',
        help='If true, disable tracking patients, select only a first-detected person instead.')
    parser.set_defaults(disable_tracking=False)
    # and so on...

    args = parser.parse_args()

    return args
