#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

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
import json

from natsort import natsorted
import collections
from IPython import display
import pylab as pl

from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from skorch import NeuralNetRegressor
from skorch.helper import predefined_split
from skorch import callbacks
from collections import defaultdict         

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.regression import median_absolute_error, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import KFold
import c3d_wrapper
from data_utils import *
import models
from params import *
from statsmodels.graphics.gofplots import qqplot


# # TF model for feature extraction

# In[ ]:


# pretrained c3d net( tensorflow )
# tf_model = TF_Model()

#visualize_conv_featsmap('7157030_test_0_trial_1', tf_model, layer='conv1')


# In[ ]:


from skorch.callbacks import Callback
from torchvision.utils import save_image

def to_tensor_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.permute(0,3,1,2)
    x = x.view(x.size(0), 1, 64, 64)
    return x

class Save_Reconstruction_Results(Callback):
    def __init__(self, path):
        self.path = path
        
    def on_epoch_end(self, net, **kwargs):
        for name in ['train', 'valid']:
            dataset = kwargs['dataset_'+name]
            rand_ix = np.random.randint(len(dataset))
            X,y = dataset[rand_ix]
            
            save_dir = os.path.join(self.path, name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # target img
            y = y.numpy().transpose(1,2,3,0)  # (maxlen,h,w,3)
            
            # predicted img
            pred = net.predict(X[None,:])[0].transpose(1,2,3,0) # (maxlen,h,w,3)
            
            for sub_name,pic in zip(['target', 'pred'], [y,pred]):
                pic = to_tensor_img(torch.from_numpy(pic))
                save_image(pic, os.path.join(save_dir,sub_name+'.png'))

                
def fetch_samples_from_dataset(dataset):
    X = []
    Y = []
    for item in dataset:
        X.append(item[0].numpy())
        Y.append(item[1].numpy())
        
    return np.array(X), np.array(Y)

def check_array(arr):
    dim = arr.shape[1]
    
    # select non-nan rows
    arr = arr[(~np.isnan(arr)).all(1)].reshape(-1, dim)    
    
    # select non-inf rows
    arr = arr[(arr!=np.inf).all(1)]
    
    return arr

def mape(y_true, y_pred, scaler=None, reduce_all=True):
    if scaler:
        # when skorch callbacks is executed for epoch_scoring
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)
        
    with np.errstate(divide='ignore'):
        mape = np.abs((y_true-y_pred)/y_true)

    if reduce_all:
        return check_array(mape).mean()
    else:
        return check_array(mape).mean(0)

def root_mean_squared_error(y_true, y_pred, reduce_all=True):
    if reduce_all:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        return np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))

def record_score(y_pred, y_true, phase, save_path="./scores"):
    MAE = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    MAPE = mape(y_true, y_pred, reduce_all=False)
    RMSE = root_mean_squared_error(y_true, y_pred, reduce_all=False)
    R2 = r2_score(y_true, y_pred, multioutput='raw_values')
    EV = explained_variance_score(y_true, y_pred, multioutput='raw_values')
    
    mae_ = f"MAE : {MAE}"
    mape_ = f"MAPE : {MAPE}"
    rmse_ = f"RMSE : {RMSE}"
    r2_ = f"R^2 : {R2}"
    ev_ = f"Explained variation : {EV}"
    
    msg = '\n'.join([mae_, mape_, rmse_, r2_, ev_])
    
    log_path = os.path.join(save_path, f'{phase}.txt')
    
    os.system(f"mkdir -p {save_path}")
    os.system("echo \'{}\' > {}".format(msg, log_path))
    
    print(f'resulting scores have been saved at \"{log_path}\"')


# # Custom criterion

# In[ ]:


from torch.nn.modules.loss import _Loss

class MyCriterion(_Loss):
    def __init__(self):
        super(MyCriterion, self).__init__()
        
    def forward(self, x, y):
        set_trace()
        return nn.MSELoss()(x,y)


# # Setup for distributed computing (dask)

# In[ ]:


from dask.distributed import Client
client = Client('127.0.0.1:8786',
               serializers=['dask', 'pickle'],
                deserializers=['dask', 'msgpack'])
from sklearn.externals import joblib

client


# # Hyperparameter search (random search)

# In[ ]:


from dask_searchcv import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer
from skorch.callbacks import Checkpoint, TrainEndCheckpoint

# for random number generation
from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint 


# In[ ]:


def hyperparams_search(model, super_class, 
                       criterion=nn.modules.loss.MSELoss,
                       ckpt_dir=None, n_splits=5, train_dataset=None, scaler=None):
    
    # init net
    net = super_class(
        model,
        max_epochs=30,
        optimizer=torch.optim.SGD,
        device='cuda',
        criterion=criterion,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        callbacks=[('ealy_stop', callbacks.EarlyStopping()),
                   ('lr_scheduler', callbacks.LRScheduler(policy='CyclicLR',
                                                         mode='exp_range',
                                                         base_lr=1e-3,
                                                         gamma=0.97)),
                   ('MAPE', callbacks.EpochScoring(scoring=make_scorer(mape, 
                                                                       scaler = data_dict['scaler']), lower_is_better=True)),
                   ('R2', callbacks.EpochScoring(scoring='r2', lower_is_better=False)),
                   ]
    )

    params = {'batch_size': sp_randint(4,12),
              'callbacks__lr_scheduler__base_lr': sp_uniform(loc=1e-6, scale=1e-2-1e-6),
              'callbacks__lr_scheduler__gamma': sp_uniform(loc=0.5, scale=1.0-0.5),
              'optimizer__weight_decay': sp_uniform(loc=0.0, scale=1e-2),
              'module__num_units': sp_randint(64,256),
              'module__drop_rate': sp_uniform(loc=0.0, scale=0.5),
             }
    
    search = RandomizedSearchCV(net, params, refit=True, cv=n_splits,n_iter=20, iid=False,
                                scoring=make_scorer(r2_score, multioutput='variance_weighted'))
    
    X, y_true = fetch_samples_from_dataset(train_dataset)
    
    search.fit(X, y_true)  
    
    os.system(f'mkdir -p {ckpt_dir}')
    
    print(f'save best model to {ckpt_dir}...!')
    
    search.best_estimator_.save_params(
        f_params=os.path.join(ckpt_dir, 'params.pt'),
        f_optimizer=os.path.join(ckpt_dir, 'optimizer.pt'),
        f_history=os.path.join(ckpt_dir, 'history.json'))
    
    # save best params also !
    json.dump(search.best_params_, open(os.path.join(ckpt_dir, 'best_params.json'), 'w'))
    
    return search


# # Prepare dataset

# In[ ]:


task = 'regression@pretrained'

input_file = "../../preprocess/data/person_detection_and_tracking_results_drop.pkl"
target_file = "../../preprocess/data/targets_dataframe.pkl"

feature_extraction_model=None
feature_layer='conv1'

data_dict = prepare_dataset(input_file, target_file,
                            feature_extraction_model=feature_extraction_model, layer=feature_layer)

train_dataset = dataset_init(task,
                            data_dict['train_X'], data_dict['train_y'],
                            scaler=data_dict['scaler'], name='train')

# holdouf testset for final evaluation
test_dataset = dataset_init(task, 
                            data_dict['test_X'], data_dict['test_y'],
                            scaler=data_dict['scaler'], name='test')

scaler = data_dict['scaler']


# # Define training configurations & run!

# In[ ]:


super_class = NeuralNetRegressor

if task.split('@')[1]=='fromscratch':
    super_class = AutoEncoderNet

experiments_dict = {}

for name,cr in zip(['MSE', 'MAE', 'Huber'],[nn.modules.loss.MSELoss, nn.modules.loss.L1Loss, nn.modules.loss.SmoothL1Loss]):
    search = hyperparams_search(
                            model=eval('_'.join(task.split('@')).capitalize()),
                            super_class=super_class, criterion=cr,
                            ckpt_dir=f'./models/exp_with_{name}_cost',
                            train_dataset=train_dataset, scaler=scaler)
    experiments_dict[name] = search


# # Performance Evaluation for Testset

# In[ ]:


def visualize_trend(y_true, y_pred, save_dir):
    for ix,col in enumerate(target_columns):
        sampled_y_true = y_true[:,ix]
        sampled_y_pred = y_pred[:,ix]

        sorted_ixs = np.argsort(sampled_y_true)

        sampled_y_true = sampled_y_true[sorted_ixs]
        sampled_y_pred = sampled_y_pred[sorted_ixs]

        fig, axes = plt.subplots(2, figsize=(11,15))
        axes[0].set_title(col+' trace')
        axes[0].plot(sampled_y_true, 'g*')
        axes[0].plot(sampled_y_pred)
        axes[0].axhline(y=sampled_y_true.mean(), color='r', linestyle='--')
        axes[1].set_title(col+' residuals')
        axes[1].scatter(sampled_y_pred, sampled_y_pred-sampled_y_true)
        axes[1].axhline(y=0.0, color='r', linestyle='--')
        
        os.system(f'mkdir -p {save_dir}')
        plt.savefig(os.path.join(save_dir, f'{col.replace("/", "_")}.png'))
        plt.show()


# In[ ]:


def measure_performance(test_dataset, phase, ckpt_dir, score_dir, figure_dir, scaler=None):
    
    f_best_params = os.path.join(ckpt_dir, 'best_params.json')
    f_params = os.path.join(ckpt_dir, 'params.pt')
    
    # best hyperparams dict
    best_params = json.load(open(f_best_params))
    
    # restore net from ckpt
    net = NeuralNetRegressor(
        module=Regression_pretrained,
        device='cuda',
        **best_params,
    )
    net.initialize()  
    net.load_params(f_params=f_params)
    
    X, y_true = fetch_samples_from_dataset(test_dataset)
    y_pred = net.predict(X)

    if scaler:
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)

    record_score(y_pred, y_true, phase=phase, save_path=score_dir)

    report_lerning_process(columns=target_columns,
                           phase=phase, save_dir=figure_dir,
                           y_pred=y_pred,
                           y_true=y_true)
    
    visualize_trend(y_true, y_pred, save_dir=figure_dir)


# In[ ]:


for name in ['MSE', 'MAE', 'Huber']:
    measure_performance(test_dataset, phase='test', 
                         ckpt_dir=f'./models/exp_with_{name}_cost',
                         score_dir=f'./scores/exp_with_{name}_cost',
                         figure_dir=f'./figures/results/exp_with_{name}_cost',
                         scaler=scaler)

