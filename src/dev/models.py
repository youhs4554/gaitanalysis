from __future__ import print_function, division
import os

from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from IPython.core.debugger import set_trace
import itertools
import seaborn as sns
from tqdm import tqdm
import random
import cv2
from natsort import natsorted
import collections
from IPython import display
import pylab as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.regression import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from skorch import NeuralNetRegressor
from skorch.helper import predefined_split
from skorch import callbacks
from sklearn.model_selection import GridSearchCV

import tensorflow as tf

import c3d_wrapper
from data_utils import *
from models import *
from params import *


class TF_Model:
    def __init__(self, batch_size=BATCH_SIZE_OF_TFMODEL, model_path=MODEL_PATH, mean_file=MEAN_FILE):
        # define graph
        self.net = c3d_wrapper.C3DNet(
            pretrained_model_path=model_path, trainable=False,
            batch_size=batch_size)

        self.tf_video_clip = tf.placeholder(tf.float32,
                                       [batch_size, None, 112, 112, 3],
                                       name='tf_video_clip')  # (batch,num_frames,112,112,3)
        
        self.net(inputs=self.tf_video_clip)
        
        self.mean_val = np.load(mean_file).transpose(1,2,3,0)

            
        # create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
    
    def run(self, X, layer):
        return self.sess.run(self.net.feats_dict[layer], feed_dict={self.tf_video_clip: [X]})[0]    
    
class Decoder(nn.Module):
    def __init__(self, num_filters = [512,256,128,64,1]):
        
        super(Decoder, self).__init__()
        
        # input_shape = (b,512,20,4,4)
        
        self.decode = nn.Sequential(
            # b, 512, 20, 4, 4
            nn.Conv3d(num_filters[0], num_filters[0], 
                      kernel_size=1, stride=1),
            nn.BatchNorm3d(num_filters[0]), 
            nn.ReLU(True),
            
            # b, 256, 39, 8, 8
            nn.ConvTranspose3d(num_filters[0], num_filters[1], 
                               kernel_size=(3,4,4), stride=2, 
                               padding=1),
            nn.BatchNorm3d(num_filters[1]), 
            nn.ReLU(True),
                        
            # b, 128, 75, 16, 16
            nn.ConvTranspose3d(num_filters[1], num_filters[2], 
                               kernel_size=(3,4,4), stride=2,
                               padding=(2,1,1)),
            nn.BatchNorm3d(num_filters[2]), 
            nn.ReLU(True),
            
            # b, 64, 150, 32, 32
            nn.ConvTranspose3d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_filters[3]), 
            nn.ReLU(True),
            
            # b, 3, 300, 64, 64
            nn.ConvTranspose3d(num_filters[3], num_filters[4], kernel_size=4, stride=2, padding=1),
            nn.Tanh()            
        )

    def forward(self, x):
        '''
            x : size = (B, C, D, H, W)
        '''
        return self.decode(x)
    
    
class Reconstruction_pretrained(nn.Module):
    def __init__(self):
        super(Reconstruction_pretrained, self).__init__()
        self.decoder = Decoder()
    
            
    def forward(self, encoded):
        decoded = self.decoder(encoded)
        return decoded

    
class Concat(nn.Module):
    def __init__(self, val, dim):
        super(Concat, self).__init__()
        
        self.val = val
        self.dim = dim
        
    def forward(self, x):
        return torch.cat([self.val, x], self.dim)

    
class Encoder(nn.Module):
    def __init__(self, num_filters = [1,32,64,128,256]):
        
        super(Encoder, self).__init__()
        
        self.encode = nn.Sequential(
            # b, 32, 150, 32, 32
            nn.Conv3d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1), # ()
            nn.BatchNorm3d(num_filters[1]), 
            nn.ReLU(True),
            
            # b, 64, 75, 16, 16
            nn.Conv3d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_filters[2]), 
            nn.ReLU(True),
            
            # b, 128, 25, 8, 8
            nn.Conv3d(num_filters[2], num_filters[3], kernel_size=(5,4,4), stride=(3,2,2), padding=1),
            nn.BatchNorm3d(num_filters[3]), 
            nn.ReLU(True),
            
            # b, 256, 8, 4, 4
            nn.Conv3d(num_filters[3], num_filters[4], kernel_size=(6,4,4), stride=(3,2,2), padding=1),
            nn.BatchNorm3d(num_filters[4]), 
            nn.ReLU(True),            
            
         )

    def forward(self, x):
        '''
            x : size = (B, C, D, H, W)
        '''
        return self.encode(x)

class Decoder_Unet_style(nn.Module):
    def __init__(self, enc_feats, num_filters = [256,128,64,32,1]):
        
        super(Decoder_Unet_style, self).__init__()
        
        self.decode = nn.Sequential(
            Concat(enc_feats[-1], 1),
            
            # b, 128, 25, 8, 8
            nn.ConvTranspose3d(2*num_filters[0], num_filters[1], kernel_size=(6,4,4), stride=(3,2,2), padding=1),
            nn.BatchNorm3d(num_filters[1]), 
            nn.ReLU(True),
            
            Concat(enc_feats[-2], 1),
            
            # b, 64, 75, 16, 16
            nn.ConvTranspose3d(2*num_filters[1], num_filters[2], kernel_size=(5,4,4), stride=(3,2,2), padding=1),
            nn.BatchNorm3d(num_filters[2]), 
            nn.ReLU(True),
            
            Concat(enc_feats[-3], 1),

            # b, 32, 150, 32, 32
            nn.ConvTranspose3d(2*num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_filters[3]), 
            nn.ReLU(True),
            
            
            Concat(enc_feats[-4], 1),
            
            # b, 1, 300, 64, 64
            nn.ConvTranspose3d(2*num_filters[3], num_filters[4], kernel_size=4, stride=2, padding=1),
            nn.Tanh()            
            
        )

    def forward(self, x):
        '''
            x : size = (B, C, D, H, W)
        '''
        return self.decode(x)
    

    
    
class Reconstruction_fromscratch(nn.Module):
    def __init__(self):
        super(Reconstruction_fromscratch, self).__init__()
        
        self.encoder = Encoder()
    
    def encode(self, X):
        enc_feats = []
        for layer in self.encoder.encode:
            if type(layer).__name__ == 'ReLU':
                enc_feats.append(X)
            X = layer(X)
        
        encoded = enc_feats[-1]
        
        return Decoder_Unet_style(enc_feats).to(torch.device("cuda:0")), encoded
    
            
    def forward(self, X):
        decoder, encoded = self.encode(X)
        decoded = decoder(encoded)
        return decoded, encoded  # <- return a tuple of two values
    
    
class AutoEncoderNet(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, encoded = y_pred  # <- unpack the tuple that was returned by `forward`
        loss_reconstruction = super(AutoEncoderNet, self).get_loss(decoded, y_true, *args, **kwargs)
        loss_l1 = 1e-3 * torch.abs(encoded).sum()
        
        return loss_reconstruction + loss_l1        
    
    
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()
    def forward(self, x):
        '''
        
            x : size = (N,C,D,H,W)
        '''
        return torch.mean(x, (2,3,4))

    
class Residual(nn.Module):
    def __init__(self, C_in, C_out):
        super(Residual, self).__init__()
        
        self.conv_1x1 = nn.Sequential(nn.Conv3d(C_in, C_out, kernel_size=1, stride=1, bias=False),
                                      nn.BatchNorm3d(C_out),
                                      nn.ReLU(True)
                                     )
        
        self.model = nn.Sequential(
            nn.Conv3d(C_out, C_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(C_out),
            nn.ReLU(True),
            
            nn.Conv3d(C_out, C_out//4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(C_out//4),
            nn.ReLU(True),
            
            nn.Conv3d(C_out//4, C_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(C_out),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        res = self.conv_1x1(x)
        x = self.model(res)
        
        return res + x
    
    
class Regression_pretrained(nn.Module):
    def __init__(self):
        super(Regression_pretrained, self).__init__()
        
        # input_shape = (b,512,20,4,4)
        
        self.model = nn.Sequential(
            Residual(C_in=512, C_out=256),
            
            # (b,512,10,4,4)
            nn.MaxPool3d(kernel_size=(4,1,1), stride=(2,1,1), padding=(1,0,0)),
            
            Residual(C_in=256, C_out=256),
            
            # (b,512,5,4,4)
            nn.MaxPool3d(kernel_size=(4,1,1), stride=(2,1,1), padding=(1,0,0)),

            Residual(C_in=256, C_out=256),
            
            # (b,512,2,4,4)
            nn.MaxPool3d(kernel_size=(3,1,1), stride=(2,1,1), padding=(0,0,0)),
            
            Residual(C_in=256, C_out=256),
            
            # (b,512,1,4,4)
            nn.MaxPool3d(kernel_size=(4,1,1), stride=(2,1,1), padding=(1,0,0)),
            
            View(-1,256*1*4*4),
            
            nn.Dropout(0.5),
            
            nn.Linear(256*1*4*4, 15)
        )
        
        
    
    def forward(self, x):
        x = self.model(x)
        
        return x