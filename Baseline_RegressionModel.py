#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import os
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


# # Pose Output Format (BODY_25)
# <img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_pose_25.png" width="300">

# # Check cuda.is_available ?

# In[2]:


cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("cuda_available : {}, device : {}".format(cuda_available, device))


# In[3]:


part_pairs = [
              #(1,8),
              #(1,2),
              #(1,5),
              #(2,3),
              #(3,4),
              #(5,6),
              #(6,7),
              (8,9),
              (9,10),
              (10,11),
              (8,12),
              (12,13),
              (13,14),
              #(1,0),
              #(0,15),
              #(15,17),
              #(0,16),
              #(16,18),
              #(2,17),
              #(5,18),
              (14,19),
              (19,20),
              (14,21),
              (11,22),
              (22,23),
              (11,24)]


# # Define Dataset & DataLoader

# In[4]:


class GAITDataset(Dataset):
    def __init__(self, keypoints_csv_file, targets_csv_file, maxlen=200):
        self.keypoints_frame = pd.read_pickle(keypoints_csv_file)
        self.targets_frame = pd.read_pickle(targets_csv_file)
        self.pids = list(set(self.keypoints_frame.index))
        self.maxlen = maxlen
        self.bottom_ixs = list(set(itertools.chain(*part_pairs)))
        
    def __len__(self):
        return len(self.targets_frame)
    
    def __getitem__(self, idx):
        keypoints_seq = np.c_[list(np.asarray(x) for x in self.keypoints_frame.loc[self.pids[idx], 'person_0'].values)]
        
        # filter body parts by its indices
        keypoints_tmp = keypoints_seq.reshape(-1,25,3) # (time_stamps,25,3)
        
        # select bottom parts
        keypoints_seq = keypoints_tmp[:,self.bottom_ixs,:].reshape(len(keypoints_seq), -1)
        
        # zero padding
        keypoints_seq = np.pad(keypoints_seq, ((0,self.maxlen-len(keypoints_seq)),(0,0)),
                                               'constant', constant_values=0).transpose(1,0)
        
        targets = self.targets_frame.loc[self.pids[idx]].values
        
        sample = {'keypoints_seq': torch.tensor(keypoints_seq, dtype=torch.float32).cpu(),
                  'targets': torch.tensor(targets, dtype=torch.float32).cpu()}
        
        return sample

# dataset path
keypoints_csv_file= "./example_data/keypoints/keypoints_df.pkl"
targets_csv_file = "./example_data/targets/targets_df.pkl"


mydataset = GAITDataset(keypoints_csv_file, targets_csv_file)

dataloader = DataLoader(mydataset, 
                        batch_size=5,
                        shuffle=True,
                        num_workers=4)


# In[12]:


mydataset[0]['targets'].size()


# # Define DNN

# In[ ]:


class Conv1d(nn.Conv1d):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1,
                       bias=True,
                       padding_type='same'):
        
        super(Conv1d, self).__init__(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     groups,
                                     bias)
        
        self.padding_type = padding_type
    
    def forward(self, x):
        _, _, input_length = x.size()
        
        if self.padding_type == 'same':
            padding_need = int((input_length * (self.stride[0]-1) + self.kernel_size[0] - self.stride[0]) / 2)
        
        return F.conv1d(x, self.weight, self.bias, self.stride, 
                        padding_need, self.dilation, self.groups)


# In[ ]:


class ResidualBlock(nn.Module):
    def __init__(self, C, highway=True):
        super(ResidualBlock, self).__init__()
        self.highway = highway
        
        # pre-define bottle-neck structure
        
        # conv_1x1_a : reduce number of channels by factor of 4 (output_channel = C/4)
        self.conv_1x1_a = Conv1d(C, int(C/4), kernel_size=1, stride=1, padding_type='same')
        self.bn_1x1_a = nn.BatchNorm1d(int(C/4))
        
        # conv_3x3_b : more wide receptive field (output_channel = C/4)
        self.conv_3x3_b = Conv1d(int(C/4), int(C/4), kernel_size=3, stride=1, padding_type='same')
        self.bn_3x3_b = nn.BatchNorm1d(int(C/4))
        
        # conv_1x1_c : recover org channel C (output_channel = C)
        self.conv_1x1_c = Conv1d(int(C/4), C, kernel_size=1, stride=1, padding_type='same')
        self.bn_1x1_c = nn.BatchNorm1d(C)
        
        # conv_1x1_g : gating for highway network
        self.conv_1x1_g = Conv1d(C, C, kernel_size=1, stride=1, padding_type='same')
        
    
    def forward(self, x):
        '''
            x : size = (batch, C, maxlen)
        '''
        res = x        
    
        # 1x1_a (C/4)
        x = self.conv_1x1_a(x)
        x = self.bn_1x1_a(x)
        x = F.relu(x)
        
        # 3x3_b (C/4)
        x = self.conv_3x3_b(x)
        x = self.bn_3x3_b(x)
        x = F.relu(x)
        
        # 1x1_c (C)
        x = self.conv_1x1_c(x)
        x = self.bn_1x1_c(x)
        x = F.relu(x)
        
        
        if self.highway:
            # gating mechanism from "highway network"
            
            # gating factors controll intensity between x and f(x)
            # gating = 1.0 (short circuit) --> output is identity (same as initial input)
            # gating = 0.0 (open circuit)--> output is f(x) (case of non-residual network)
            gating = F.sigmoid(self.conv_1x1_g(x))
                                   
            # apply gating mechanism
            x = gating * res + (1.0 - gating) * x
            
        else:
            # normal residual ops (addition)
            x = x + res

        # apply relu for final output
        x = F.relu(x)
        
        return x


# In[ ]:


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)
    
class Net(nn.Module):
    def __init__(self, input_size, target_size, n_residual_blocks=4, residual_pooling_indices=range(0,4,2), C=64):
        assert max(residual_pooling_indices) < n_residual_blocks, "maxvalue of residual_pooling_indices ({}) cannot exceed n_residual_blocks ({})".format(max(residual_pooling_indices), n_residual_blocks)
        assert min(residual_pooling_indices) >= 0 , "minvalue of residual_pooling_indices ({}) cannot be smaller than 0".format(min(residual_pooling_indices))

        super(Net, self).__init__()
        
        self.input_size = input_size
        self.target_size = target_size
        
        residual_blocks = []
        pooling_cnt = 0 
        for i in range(n_residual_blocks):
            residual_blocks.append(ResidualBlock(C))
            if i in residual_pooling_indices:
                # every 2 residual, block pooling out!
                residual_blocks.append(nn.MaxPool1d((2,)))
                pooling_cnt += 1

        length_after_pooling = int(mydataset.maxlen/(2**pooling_cnt))

        self.model = nn.Sequential(Conv1d(input_size, C, kernel_size=1, stride=1, padding_type='same'),
                                   nn.BatchNorm1d(C),
                                   nn.ReLU(),
                                   *residual_blocks,
                                   View(-1,64 * length_after_pooling),
                                   nn.Linear(64 * length_after_pooling, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, target_size)
                                   )
        
    def forward(self, x):
        '''
            x : size = (batch, input_size, maxlen)
        '''
        return self.model(x)
            
net = Net(input_size=len(mydataset.bottom_ixs)*3,
          target_size=len(mydataset.targets_frame.columns), n_residual_blocks=4, residual_pooling_indices=(0,2), C=64)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
else:
    print("Single GPU mode")

net.to(device)


# In[ ]:


# define criterion
criterion = nn.MSELoss()

import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=1e-4)

for epoch in range(500):
    running_loss = 0.0
    for idx, batch_item in enumerate(dataloader):
        net.train()
        optimizer.zero_grad()
        
        input, target = batch_item['keypoints_seq'].to(device), batch_item['targets'].to(device)
        
        # feed data to network
        output = net(input)

        # compute loss
        loss = criterion(output, target)
        loss.backward()
        
        running_loss += loss.item()
        
        optimizer.step()

    steps_per_epoch = len(dataloader.dataset)/dataloader.batch_size
    avg_loss = running_loss / steps_per_epoch
    print('========================================================')
    print('EPOCH : {}, AVG_MSE : {:.4f}'.format(epoch, avg_loss))

