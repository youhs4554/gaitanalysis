for x in {1..8}
do
    python train.py --cfg_file manifest/DefaultAGNet_URFD\@r2plus1d_18_cv.cfg --fold $x
done