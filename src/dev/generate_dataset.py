from utils.preprocessing import PatientLocalizer, COPAnalyizer, Worker
import opts
from preprocess.darknet.python.extract_bbox import set_gpu
import numpy as np
from tqdm import tqdm
import os

if __name__ == '__main__':
    opt = opts.parse_opts()

    if opt.mode == 'preprocess__frame':

        # patient localizer & interval selector
        set_gpu(opt.device_yolo)

        localizer = PatientLocalizer(darknet_api_home=opt.darknet_api_home)

        interval_selector = COPAnalyizer(opt.meta_home, opt.fps)
        worker = Worker(localizer, interval_selector, opt)

    elif opt.mode == 'preprocess__feature':
        from datasets.gaitregression import GAITDataset
        from utils.mean import get_mean, get_std
        from preprocess.preprocess_metadata import create_target_df
        import pandas as pd
        import sklearn
        from utils.transforms import (
            Compose, ToTensor, MultiScaleRandomCrop, MultiScaleCornerCrop, Normalize)
        from utils.generate_model import generate_backbone
        from torch.utils.data import DataLoader
        from utils.parallel import DataParallelModel
        from torch import nn

        X = pd.read_pickle(opt.input_file)
        y = create_target_df(meta_home=opt.meta_home,
                             save_path=opt.target_file,
                             single_cols=['Velocity', 'Cadence',
                                          'Functional Amb. Profile'],
                             pair_cols=['Cycle Time(sec)',
                                        'Stride Length(cm)',
                                        'HH Base Support(cm)',
                                        'Swing Time(sec)',
                                        'Stance Time(sec)',
                                        'Double Supp. Time(sec)',
                                        'Toe In / Out']
                             )

        opt.arch = "{}-{}".format(opt.backbone, opt.model_depth)
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value, dataset=opt.mean_dataset)

        if opt.no_mean_norm and not opt.std_norm:
            norm_method = Normalize([0, 0, 0], [1, 1, 1])
        else:
            norm_method = Normalize(opt.mean, opt.std)

        # input transform
        input_transform = Compose(
            [
                # crop_method, #### disable crop method
                # RandomHorizontalFlip(), ### disable flip
                ToTensor(opt.norm_value),
                norm_method,
            ]
        )

        # target transform
        target_transform_func = sklearn.preprocessing.QuantileTransformer

        target_transform = target_transform_func(
            random_state=0, output_distribution="normal"
        ).fit(y.values)

        ds = GAITDataset(X=X, y=y, opt=opt,
                         input_transform=input_transform,
                         target_transform=target_transform,
                         load_pretrained=False)
        ds_loader = DataLoader(ds, batch_size=opt.batch_size)
        ds_loader = iter(ds_loader)

        net = generate_backbone(opt)
        num_feats = net.fc.in_features

        net = nn.Sequential(*list(net.children())[:-1])

        # Enable GPU model & data parallelism
        if opt.multi_gpu:
            net = DataParallelModel(
                net, device_ids=eval(opt.device_ids + ',', ))

        net.cuda()

        save_dir = os.path.join(os.path.dirname(opt.data_root),
                                'FeatsArrays', opt.arch)
        if not os.path.exists(save_dir):
            os.system(f'mkdir -p {save_dir}')

        for img, _, vids in tqdm(ds_loader, desc='[FeX] Status'):
            # img : (N,C,L,H,W)
            img = img.permute(0, 2, 1, 3, 4)  # (N, L, C, H, W)
            img = img.contiguous().view(-1, *img.size()
                                        [2:])  # (N*L, C, H, W)
            feats_from_multiGPU = net(img)

            for feats, vid in zip(feats_from_multiGPU, vids):
                np_feats = feats.detach().cpu().numpy()
                np_feats = np.squeeze(np_feats, (2, 3))
                save_path = os.path.join(save_dir, vid+'.npy')
                np.save(save_path, np_feats)
