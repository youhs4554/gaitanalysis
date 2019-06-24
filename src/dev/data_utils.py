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
from torchvision.utils import save_image

from skorch import NeuralNetRegressor
from skorch.helper import predefined_split
from skorch import callbacks
from sklearn.model_selection import GridSearchCV

import tensorflow as tf

import c3d_wrapper
from models import *
from params import *


def pid2vid(pid):
    num, test_id, trial_id = pid.split('_')
    return '_'.join([num, 'test', test_id, 'trial', trial_id])
    

def vid2pid(vid):
    split = vid.split('_')
    return '_'.join([split[0], split[2], split[4]])

def visualize_conv_featsmap(vid, feature_extraction_model,
                  frame_home=FRAME_HOME,
                  frames_per_clip=FRAMES_PER_CLIP, 
                  feats_maxlen=FEATS_MAXLEN,
                  layer='conv1',
                  save_dir = './feats_viz'):
    
    if os.path.exists(save_dir):
        os.system(f'rm -rf {save_dir}')
    os.makedirs(save_dir)

    stacked_arr = np.load(os.path.join(frame_home, vid) + '.npy')
    feats = extract_features(stacked_arr, feature_extraction_model,
                             frame_home,
                             frames_per_clip, 
                             feats_maxlen, layer=layer)

    # normalize (0-1)
    feats /= feats.max() # --> because mininum is 0 (relued)
    for t in range(feats.shape[1]):
        fms = feats[:,t,:,:]   # (C, H, W)
        to_save = []
        for fm in fms:
            # fm : (H,W,3)
            fm = cv2.applyColorMap((fm*255).astype(np.uint8), cv2.COLORMAP_JET) # change color map
            fm = cv2.resize(cv2.cvtColor(fm, cv2.COLOR_BGR2RGB), (64,64))
            to_save.append(TF.to_tensor(fm))  # (3,H,W)
        to_save = torch.stack(to_save) # (C,3,H,W)
        
        save_image(to_save, os.path.join(save_dir, f'frame-{t}.png'))

def extract_features(stacked_arr, feature_extraction_model,
                     frame_home,
                     frames_per_clip, 
                     feats_maxlen, layer):

    def preprocess_clip(clip):
        vid = []
        for img in clip:
            vid.append(cv2.resize(img, (171,128)))

        vid = np.array(vid)

        leng = len(vid)

        vid = vid - feature_extraction_model.mean_val[:leng]
        vid = vid[:, 8:120, 30:142, :]
        
        # (16, 112, 112, 3)
        vid = np.pad(vid, ((frames_per_clip-leng,0),(0,0),(0,0),(0,0)), 'constant')

        return vid

    res = []
    feats_len_per_clip = None
    
    while True:
        clip = stacked_arr[:frames_per_clip]
        if len(clip) == 0: break

        clip = preprocess_clip(clip)
        
        feature = feature_extraction_model.run(clip, layer)
        if 'conv' in layer:
            feature = feature.transpose(3,0,1,2)   # (C,D,H,W)
            feats_len_per_clip = feature.shape[1]
        
        res.append(feature)

        # move to next slice !
        stacked_arr = stacked_arr[frames_per_clip:]

    if 'conv' in layer:
        return np.concatenate(res, 1) # concatenate through time dimension
    else:
        return res


def save_features(vids, feature_extraction_model,
                  frame_home=FRAME_HOME,
                  frames_per_clip=FRAMES_PER_CLIP, 
                  feats_maxlen=FEATS_MAXLEN,
                  save_dir=FEATS_SAVE_DIR, layer='conv1'):
    
    save_dir = os.path.join(save_dir, layer)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ix in tqdm(range(len(vids))):
        vid = vids[ix]
        stacked_arr = np.load(os.path.join(frame_home, vid) + '.npy')
        feats = extract_features(stacked_arr, feature_extraction_model,
                                 frame_home,
                                 frames_per_clip, 
                                 feats_maxlen, layer)
        np.save(os.path.join(save_dir, vid), feats)
        
from sklearn.model_selection import train_test_split

def filter_input_df_with_vids(df, vids):
    return df[df['vids'].isin(vids)]

def filter_target_df_with_vids(df, vids):
    target_ids = [ vid2pid(vid) for vid in vids ]
    return df.loc[target_ids]

def split_dataset_with_vids(input_df, target_df, vids, test_size=0.3, random_state=42):
    train_vids, test_vids = train_test_split(vids, test_size=test_size, random_state=random_state)

    train_X, train_y = filter_input_df_with_vids(input_df,train_vids), filter_target_df_with_vids(target_df,train_vids)
    test_X, test_y = filter_input_df_with_vids(input_df,test_vids), filter_target_df_with_vids(target_df, test_vids)
        
    return train_X, train_y, train_vids, test_X, test_y, test_vids



def report_lerning_process(columns, epoch, phase, y_pred, y_true, loss_history):
    
    pred_and_gt = { k:[] for k in columns }

    for i,col in enumerate(columns):
        pred_and_gt[col].append([y_pred[:,i], y_true[:,i]])
    
    data = collections.defaultdict(list)

    pp = []
    gg = []
    for i,col in enumerate(pred_and_gt.keys()):
        transposed_data = list(zip(*pred_and_gt[col]))
        preds = np.concatenate(transposed_data[0])
        gts = np.concatenate(transposed_data[1])

        pp.append(preds)
        gg.append(gts)

        for p,g in zip(preds, gts):
            data["name"].append(col)
            data["pred"].append(p)
            data["gt"].append(g)

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20,20))
    fig.suptitle("Epoch={} / Phase={}\nLoss={:.8f}".format(epoch,
                                                           phase,
                                                           loss_history[phase][-1]),
                                                           fontsize=30)
    
    axes = axes.flatten()
    
    for i,col in enumerate(columns):
        part_of_df = df.loc[df.name==col]
        ax = axes[i]
        part_of_df.plot.scatter(x="pred", y="gt", c='green', ax=ax, label='data')
        ax.set_title(f'name={col}')
    
    for i,(preds,gts) in enumerate(zip(pp,gg)):
        ax = axes[i]
        ax.plot([min(preds), max(preds)], [min(gts), max(gts)], 'r--', label='GT=PRED')
        ax.legend()
                    
    ax1, ax2 = axes[len(columns):][:2] # last two axes : plot learning curve (train/test)

    for ax,name,color in zip([ax1, ax2], ['train','valid'],['blue','orange']):
        ax.plot(loss_history[name], color=color)
        ax.set_title(f'Learning Curve ({name})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    plt.savefig(f'status_{phase}.png')
    plt.cla()    
    
def prepare_dataset(input_file, target_file, feature_extraction_model=None, layer='conv1'):
    # data prepare first !!
    input_df = pd.read_pickle(input_file)
    target_df = pd.read_pickle(target_file)[target_columns]
    
    # save conv features from pretrained c3d net
    possible_vids = list(set(input_df.vids))
    if feature_extraction_model:
        save_features(vids=possible_vids, feature_extraction_model=feature_extraction_model,
                      feats_maxlen=FEATS_MAXLEN, layer=layer)
    
    # split dataset (train/test)
    train_X, train_y, train_vids, test_X, test_y, test_vids = split_dataset_with_vids(input_df, target_df, possible_vids, test_size=0.3, random_state=42)
    
    # # target scaler
    scaler = StandardScaler()
    train_y.loc[:,:] = scaler.fit_transform(train_y.values)

    # scaler = None
    
    return dict(train_X=train_X, train_y=train_y, train_vids=train_vids,
                test_X=test_X, test_y=test_y, test_vids=test_vids, 
                input_df=input_df, target_df=target_df, scaler=scaler)

def dataset_init(task, X, y, scaler=None, name=None):
    main_task, policy = task.split('@')

    if main_task == 'regression':
        if policy == 'pretrained':
            dataset = GAITDataset_Regression(X, y, scaler=scaler, name=name)
        else:
            # TODO. haha
            raise NotImplementedError('dataset for regression with from scratch policy is not implemented')
            
    elif main_task == 'reconstruction':
        if policy=='pretrained':
            dataset = GAITDataset_Reconstruction_Pretrained(X, name=name)
        else:
            dataset = GAITDataset_Reconstruction_FromScratch(X, name=name)
            
    return dataset

def normalize_img(pic, shape=(64,64), color_map='gray'):
    if color_map=='gray':
        code = cv2.COLOR_BGR2GRAY
        mean = (0.5,)
        std = (0.5,)
    elif color_map=='rgb':
        code = cv2.COLOR_BGR2RGB
        mean = (0.5,0.5,0.5,)
        std = (0.5,0.5,0.5,)

    pic = cv2.resize(cv2.cvtColor(pic, code), shape)

    pic = TF.to_tensor(pic) # scale to [0.0, 1.0]
    pic = TF.normalize(pic, mean, std).permute(1,2,0).numpy()   # scale to [-1.0, 1.0]

    return pic


class GAITDataset_Regression(Dataset):
    def __init__(self,
                 X, y,
                 feats_save_dir=FEATS_SAVE_DIR, scaler=None, name=None):

        self.X = X
        self.y = y
        self.vids = list(set(X.vids))
        
        self.feats_save_dir = os.path.join(feats_save_dir, 'fc1')
        self.name = name
        
        if scaler:
            scaled_values = scaler.transform(y)
            self.y.loc[:,:] = scaled_values
            
    def __len__(self):
        return len(self.vids)
    
    def __getitem__(self, idx):
        
        vid = self.vids[idx]
        
        feats = np.load(os.path.join(self.feats_save_dir, vid) + '.npy')
        #feats = np.pad(feats, ((0,0),(0,320-feats.shape[1]),(0,0),(0,0)), 'constant')
        #feats = np.pad(feats, ((0,20-feats.shape[0]),(0,0)), 'constant')
        feats = np.mean(feats, axis=0)
        
        targets = self.y.loc[vid2pid(vid)].values
        
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    
    
    
class GAITDataset_Reconstruction_Pretrained(Dataset):
    def __init__(self,
                 X, 
                 reconstruction_shape=(64,64),
                 frame_home=FRAME_HOME, frame_maxlen=FRAME_MAXLEN,
                 feats_save_dir=FEATS_SAVE_DIR, name=None):
        
        self.X = X
        self.vids = list(set(X.vids))
        
        self.reconstruction_shape = reconstruction_shape        
        self.feats_save_dir = feats_save_dir
        self.frame_home = frame_home
        self.frame_maxlen = frame_maxlen
        self.name = name
            
    def __len__(self):
        return len(self.vids)
    
    def __getitem__(self, idx):
        
        vid = self.vids[idx]
        
        feats = np.load(os.path.join(self.feats_save_dir, vid) + '.npy')
        
        stacked_arr = np.load(os.path.join(self.frame_home, vid) + '.npy')
        
        frames = []
        
        for cropped in stacked_arr:  
            pic = normalize_img(cropped, shape=self.reconstruction_shape, color_map='gray')
            frames.append(pic)
            
        # zero padding
        frames = np.pad(frames, ((0,self.frame_maxlen-len(frames)),(0,0),(0,0),(0,0)),
                                               'constant', constant_values=0).transpose(3,0,1,2)
        
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(frames, dtype=torch.float32)
    
    
    
    
class GAITDataset_Reconstruction_FromScratch(Dataset):
    def __init__(self,
                 X, 
                 reconstruction_shape=(64,64),
                 frame_home=FRAME_HOME, frame_maxlen=FRAME_MAXLEN,
                 name=None):
        
        self.X = X
        self.vids = list(set(X.vids))
        
        self.reconstruction_shape = reconstruction_shape
        self.frame_home = frame_home
        self.frame_maxlen = frame_maxlen
        self.name = name
            
    def __len__(self):
        return len(self.vids)
    
    def __getitem__(self, idx):
        
        vid = self.vids[idx]
        
        stacked_arr = np.load(os.path.join(self.frame_home, vid) + '.npy')
        
        frames = []
        
        for cropped in stacked_arr:  
            pic = normalize_img(cropped, shape=self.reconstruction_shape, color_map='gray')
            frames.append(pic)

        # zero padding
        frames = np.pad(frames, ((0,self.frame_maxlen-len(frames)),(0,0),(0,0),(0,0)),
                                               'constant', constant_values=0).transpose(3,0,1,2)
        
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(frames, dtype=torch.float32)
            
         
            
