from utils.preprocessing import (
    PatientLocalizer,
    COPAnalyizer,
    HumanScaleAnalyizer,
    Worker,
)
from utils.validate_activations import ActivationMapProvider
from utils.train_utils import Trainer
import utils.data
from preprocess.darknet.python.extract_bbox import set_gpu
from opts import parse_opts
import cv2
import torch
from utils.generate_model import load_trained_ckpt
import numpy as np
import warnings
import json
warnings.filterwarnings("ignore")


def train(opt, fold, metrice='f1-score'):
    opt, net, criterion1, criterion2, optimizer, lr_scheduler, warmup_scheduler, spatial_transform, temporal_transform, target_transform, plotter, train_loader, test_loader, target_columns = \
        utils.data.prepare_data(opt, fold)

    trainer = Trainer(
        model=net,
        criterion1=criterion1,
        criterion2=criterion2,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        warmup_scheduler=warmup_scheduler,
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
        cv_scores.append(score_dict)

    test_score_averaged = np.mean([x[metrice] for x in cv_scores])

    print()
    print('-'*64)
    print('{0}-fold CV result with {1} : {2:.4f}'.format(opt.n_folds,
                                                         metrice, test_score_averaged))

    score_histories = {'fold-' + str(i): d for i, d in enumerate(cv_scores)}

    prefix = opt.dataset + '_' + opt.model_indicator
    json.dump(score_histories, open(
        'results/{}_CV_score_histories.json'.format(prefix), 'w'))

    print()
    print('-'*64)

    return cv_scores


def test(opt, fold):

    opt, net, criterion1, criterion2, optimizer, lr_scheduler, warmup_scheduler, spatial_transform, temporal_transform, target_transform, plotter, train_loader, test_loader, target_columns = \
        utils.data.prepare_data(opt, fold)
    net = load_trained_ckpt(opt, net)

    # init ActivationMapProvider
    actmap_provider = ActivationMapProvider(net, opt)

    def denormalization(img, mean, std):
        return img.permute(1, 2, 0).numpy()*std + mean

    test_ds = test_loader.dataset

    layer_sel = f"{opt.model_arch.lower()}.conv_1x1"

    for i in range(10):
        i = 12750  # falling-sample index
        img_tensor, mask_tensor, target, vid, _ = test_ds[i]
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        activation_result = actmap_provider.compute(
            img_tensor=img_tensor, activation_layer=layer_sel)

        # Move CUDA tensor to CPU
        img_tensor = img_tensor.cpu()

        overlay_seq = []

        for t in range(img_tensor.size(1)):
            img_ = np.uint8(
                255*denormalization(img_tensor[:, t], opt.mean, opt.std))
            heatmap_ = np.uint8(255*activation_result[t])
            heatmap_ = cv2.applyColorMap(heatmap_, cv2.COLORMAP_JET)
            frame = np.uint8(heatmap_*0.3 + img_*0.5)
            overlay_seq.append(frame)

        out = cv2.VideoWriter(f'falling-sample-{opt.model_indicator}.avi',
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              24.0, (opt.sample_size, opt.sample_size))
        for frame in overlay_seq:
            out.write(frame)

        out.release()

        print(vid)


def demo(opt):
    opt, net, criterion1, criterion2, optimizer, lr_scheduler, warmup_scheduler, spatial_transform, temporal_transform, target_transform, plotter, train_loader, test_loader, target_columns = \
        utils.data.prepare_data(opt)

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
        train(opt, fold=opt.test_fold)   # train-for single split
    elif opt.mode == "test":
        test(opt, fold=opt.test_fold)    # test-for single split
    elif opt.mode == "demo":
        demo(opt)            # demo, running RESTful API server.


if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', True)

    main()
