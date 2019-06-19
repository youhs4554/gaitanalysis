import tensorflow as tf
import c3d_wrapper
import os
from IPython.core.debugger import set_trace

os.environ["CUDA_VISIBLE_DEVICES"]="2"
MODEL_PATH = '/data/GaitData/pretrained/C3D/conv3d_deepnetA_sport1m_iter_1900000_TF.model'
BATCH_SIZE = 1


# define graph
net = c3d_wrapper.C3DNet(
    pretrained_model_path=MODEL_PATH, trainable=False,
    batch_size=BATCH_SIZE)

tf_video_clip = tf.placeholder(tf.float32,
                               [None, None, 112, 112, 3],
                               name='tf_video_clip')  # (batch,num_frames,112,112,3)
tf_output = net(inputs=tf_video_clip)

# create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print('Total labels: {}'.format(len(labels)))

import cv2
import numpy as np

mean_val = np.load('train01_16_128_171_mean.npy').transpose(1,2,3,0)

stacked_arr = np.load('/data/GaitData/CroppedFrameArrays/am022_test_2_trial_1.npy')


import time
t0 = time.time()
vid = []
for img in stacked_arr:
    vid.append(cv2.resize(img, (171,128)))

vid = np.array(vid)

print(time.time()-t0)

X = vid[0:16]-mean_val
X = X[:, 8:120, 30:142, :]

set_trace()

output = sess.run(tf_output, feed_dict={tf_video_clip:[X]})

set_trace()

#plt.plot(output[0]); plt.show()

print('Position of maximum probability: {}'.format(output[0].argmax()))
print('Maximum probability: {:.5f}'.format(max(output[0])))
print('Corresponding label: {}'.format(labels[output[0].argmax()]))

# sort top five predictions from softmax output
top_inds = output[0].argsort()[::-1][:5]  # reverse sort and take five largest items
print('\nTop 5 probabilities and labels:')

for i in top_inds:
    print('{:.5f} {}'.format(output[0][i], labels[i]))

