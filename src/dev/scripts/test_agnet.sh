# test model
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 python main.py --input_file ../../preprocess/data/person_detection_and_tracking_results_drop.pkl --target_file ../../preprocess/data/targets_dataframe.pkl --backbone r2plus1d_18 --model_depth 18 --pretrained_path "" --data_root /data/GaitData/RawFrames --multi_gpu --with_segmentation --device_ids 0,1,2,3,4,5,6,7 --test_epoch 40 --pretrain_epoch 160 --batch_size 32 --n_threads 8 --mode test --model_arch AGNet