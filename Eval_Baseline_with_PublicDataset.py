#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import itertools
from IPython.core.debugger import set_trace
# library
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import collections
import random
import re
from natsort import natsorted


# # Check cuda.is_available ?

# In[2]:


cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("cuda_available : {}, device : {}".format(cuda_available, device))


# In[41]:


class ParkinsonDataset(Dataset):
    def __init__(self, 
                 task_type,
                 train_or_test,
                 split_ratio=(0.8, 0.2),
                 UDysRS_rating_file = './UDysRS_UPDRS_Export/UDysRS.txt',
                 UPDRS_rating_file = './UDysRS_UPDRS_Export/UPDRS.txt',
                 CAPSIT_rating_file = './UDysRS_UPDRS_Export/CAPSIT.txt',
                 sn_file = './UDysRS_UPDRS_Export/sn_numbers.txt',
                 maxlen=600,
                 seed=5
                ):
        
        if task_type not in ['typeA', 'typeB']:
            raise ValueError('Task type is not valid [ typeA | typeB ]')
        
        self.task_type = task_type
        self.maxlen = maxlen

        # support 2 type of tasks ( A : "Communication/Drinking Tasks",
        #                           B : "Leg Agility Task" )
        trajectory_files = ['./UDysRS_UPDRS_Export/Communication_all_export.txt', 
                            './UDysRS_UPDRS_Export/Drinking_all_export.txt']  if task_type=='typeA' else \
                              ['./UDysRS_UPDRS_Export/LA_split_all_export.txt']

        
        def load_data(file_path):
            with open(file_path, 'r') as infile:
                data = json.load(infile)
            return data

        # input data
        traj_data = {}
        for traj_file in trajectory_files:
            data = load_data(traj_file)
            traj_data.update(data)
        
        # target data
        rating_data = {k:load_data(v) for k,v in zip(['UDysRS',
                                                      'UPDRS',
                                                      'CAPSIT'],
                                                      [UDysRS_rating_file,
                                                       UPDRS_rating_file,
                                                       CAPSIT_rating_file])}

        # map : trial number -> subject name
        self.sn_map = load_data(sn_file)

        ## preprocess input data (trajectory)
        input_data = collections.defaultdict(list)
        
        if task_type=='typeB':
            for key,meta_dict in traj_data.items():
                
                all_body_parts = meta_dict['position'].keys()
                
                tmp_dict = collections.defaultdict(list)

                part_active = list(filter(lambda x: x.endswith('_act'), all_body_parts))
                part_rst = list(filter(lambda x: x.endswith('_rst'), all_body_parts))

                # add active data first
                for part in part_active:
                    pure_part = part.split('_')[0]
                    tmp_dict[pure_part] += meta_dict['position'][part]
                    
                # append rst data
                for part in part_rst:
                    pure_part = part.split('_')[0]
                    tmp_dict[pure_part] += meta_dict['position'][part]
                    
                
                # replace with modified tmp_dict
                traj_data[key]['position'] = tmp_dict
        
        
        for key,meta_dict in traj_data.items():
            n_joints = len(meta_dict['position'].keys())
            
            # time-major
            time_series_data = np.asanyarray(list(meta_dict['position'].values())).transpose(1,0,2)
            time_series_data_dt = np.diff(time_series_data, axis=0)   # time difference !!

            time_series_data = time_series_data.reshape(-1, n_joints*2)
            time_series_data_dt = time_series_data_dt.reshape(-1, n_joints*2)  # vectorisze; e.g. seq of position vectors

            input_data["sample_id"].append(key)
            input_data["pose"].append(time_series_data.tolist())
            input_data['pose_dt'].append(time_series_data_dt.tolist())

        ## preprocess target data

        # prepare empty data
        target_data = collections.defaultdict(list)
        all_trials = list(self.sn_map.keys())
        
        # for "UDysRS"
        target_data["trial_nbr"] = all_trials
        for column_name in ['Communication', 'Drinking', 'Higher']:
            for part in [ "Neck", 
                          "Right arm/shoulder", "Left arm/shoulder",
                          "Trunk",
                          "Right leg/hip", "Left leg/hip" ]:
                target_data["UDysRS" + '_' + column_name + '_' + part] = [ np.nan ] * len(target_data["trial_nbr"])

        # for "UPDRS"
        target_data["UPDRS_Total"] = [ np.nan ] * len(target_data["trial_nbr"])            
        
        
        RaitingItem2Name = { 
                            k: v for k,v in zip(['3.1', '3.10', '3.4', '3.5', '3.6', '3.8', '3.9'],
                                                ['SPEECH', 'GAIT', 'FINGER TAPPING', 'HAND MOVEMENTS', 
                                                 'PRONATION-SUPINATION MOVEMENTS OFHANDS', 'LEG AGILITY', 'ARISING FROM CHAIR']
                                               )
                          }
        
        # for CAPSIT
        for column_name in rating_data['CAPSIT'].keys():
            for part in [ "Neck", 
                          "Trunk", 
                          "Upper limb right","Upper limb left",
                          "Lower limb right", "Lower limb left" ]:
                target_data["CAPSIT" + '_' + RaitingItem2Name[column_name] + '_' + part] = [ np.nan ] * len(target_data["trial_nbr"])

        
        # part 1 : 'UDysRS'            
        for column_name, meta_dict in rating_data['UDysRS'].items():
            trial_nbrs = meta_dict.keys()
            for trial_nbr in trial_nbrs:
                try:
                    ix = all_trials.index(trial_nbr)
                except ValueError:
                    # if trial_nbr is not found in all_trials, Skip
                    continue
                    
                for p_ix, part in enumerate([ "Neck", 
                                              "Right arm/shoulder", "Left arm/shoulder",
                                              "Trunk",
                                              "Right leg/hip", "Left leg/hip"]):
                    target_data["UDysRS" + '_' + column_name + '_' + part][ix] = meta_dict[trial_nbr][p_ix]

                
        
            
        # part 2 : 'UPDRS'
        for trial_nbr, val in rating_data['UPDRS']['Total'].items():
            try:
                ix = all_trials.index(trial_nbr)
            except ValueError:
                # if trial_nbr is not found in all_trials, Skip!
                continue

            target_data["UPDRS_Total"][ix] = val

            
        # part 3 : 'CAPSIT'
        for column_name, meta_dict in rating_data['CAPSIT'].items():
            trial_nbrs = meta_dict.keys()
            for trial_nbr in trial_nbrs:
                try:
                    ix = all_trials.index(trial_nbr)
                except ValueError:
                    # if trial_nbr is not found in all_trials, Skip
                    continue
                    
                for p_ix, part in enumerate([ "Neck", 
                                              "Trunk", 
                                             "Upper limb right","Upper limb left",
                                              "Lower limb right", "Lower limb left" ]):
                    target_data["CAPSIT" + '_' + RaitingItem2Name[column_name] + '_' + part][ix] = meta_dict[trial_nbr][p_ix]
        

        # input data frame
        input_df = pd.DataFrame(data=input_data).fillna(0)

        # integratged target data frame
        self.target_df = target_df = pd.DataFrame(data=target_data).fillna(0)        
        
        # valid target indices
        valid_indices = self.target_df[self.target_df[self.target_columns]!=0][self.target_columns].dropna().index
        valid_trial_nbrs = self.target_df['trial_nbr'].iloc[valid_indices].values
        
        if self.task_type=='typeA':
            regex = lambda x: '^{}-.*$'.format(x)
        elif self.task_type=='typeB':
            regex = lambda x: '^{}$'.format(x)
            
        valid_sample_id_regex = '|'.join([regex(x) for x in valid_trial_nbrs])
        input_df = input_df[input_df.sample_id.str.contains(valid_sample_id_regex)]
        target_df = target_df.iloc[valid_indices]
                
        sorted_sample_ids = natsorted(input_df.sample_id.values)
        
        # set random seed, to consistency of performance
        np.random.seed(seed)
        
        # shuffle before split
        input_df = input_df.iloc[np.random.permutation(len(input_df))]
        target_df = target_df.iloc[np.random.permutation(len(target_df))]
        
        if train_or_test=='train':
            self.input_df = input_df[input_df.sample_id.isin(sorted_sample_ids[:int(len(input_df)*split_ratio[0])])]
            self.target_df = target_df
            
        elif train_or_test=='test':
            self.input_df = input_df[input_df.sample_id.isin(sorted_sample_ids[int(len(input_df)*split_ratio[0]):])]
            self.target_df = target_df
            
    @property
    def list_of_ratings(self):
        if self.task_type == 'typeA':
            list_of_ratings = ['UDysRS']
        elif self.task_type == 'typeB':
#             list_of_ratings = ['UPDRS', 'CAPSIT']
            list_of_ratings = ['UPDRS']
            
        return list_of_ratings
    
    @property
    def target_columns(self):
        _target_columns = []
        for rating_name in self.list_of_ratings:
            # filter columns by name
            _target_columns += list(filter(lambda x: x.startswith(rating_name), self.target_df.columns))

        return _target_columns
    
    def __len__(self):
        return len(self.input_df)
    
    def __getitem__(self, idx):
        sample_id = self.input_df.iloc[idx].sample_id
        trial_nbr = re.split("\s|-", sample_id)[0]  

        pose_data = self.input_df[self.input_df.sample_id==sample_id].iloc[:, 1:2].values[0]
        pose_dt_data = self.input_df[self.input_df.sample_id == sample_id].iloc[:, 2:].values[0]
        target_data = self.target_df[self.target_df.trial_nbr==trial_nbr][self.target_columns].values[0]

        pose_data = np.asanyarray(list(np.asanyarray(x) for x in pose_data)).squeeze(axis=0)
        pose_dt_data = np.asanyarray(list(np.asanyarray(x) for x in pose_dt_data)).squeeze(axis=0)
        target_data = np.asanyarray(list(np.asanyarray(x) for x in target_data))
                
        # zero padding
        pose_data = np.pad(pose_data, ((0,self.maxlen-len(pose_data)),(0,0)),
                                               'constant', constant_values=0).transpose(1,0)
        pose_dt_data = np.pad(pose_dt_data, ((0,self.maxlen-len(pose_dt_data)),(0,0)),
                                               'constant', constant_values=0).transpose(1,0)

        sample = {'pose_seq': torch.tensor(pose_data, dtype=torch.float32),
                  'pose_dt_seq': torch.tensor(pose_dt_data, dtype=torch.float32),
                  'targets': torch.tensor(target_data, dtype=torch.float32)
                  }

        return sample


# In[43]:


mydataset = { x : ParkinsonDataset(task_type='typeB', train_or_test=x)                         for x in ['train', 'test'] }

dataloader = { x : DataLoader(mydataset[x],
                        batch_size=5,
                        shuffle=True,
                        num_workers=4) \
                    for x in ['train', 'test'] }


# # Visualize patient distribution

# In[44]:


patient_distribution = collections.defaultdict(None)
patient_distribution['train'] = list(re.split("\s|-", x)[0] for x in  mydataset['train'].input_df.sample_id.values)
patient_distribution['test'] = list(re.split("\s|-", x)[0] for x in  mydataset['test'].input_df.sample_id.values)
venn2([set(patient_distribution['train']), set(patient_distribution['test'])], set_labels=('Train', 'Test'))


# # Define DNN

# In[45]:


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


# In[46]:


class ResidualBlock(nn.Module):
    def __init__(self, C_in, C_out, pool, highway=True):
        super(ResidualBlock, self).__init__()
        self.pool = pool
        self.highway = highway
                
        stride = 1
        
        if C_in != C_out:
            C = C_out
        else:
            C = C_in = C_out
            
        if pool:
            # input dimension matchig
            self.conv_1x1_matching = Conv1d(C_in, C, kernel_size=1, stride=1, padding_type='same')
            self.bn_1x1_matching = nn.BatchNorm1d(C)

            # for pooling of residual path
            stride = 2
            self.conv_2x1_pool = Conv1d(C_in, C, kernel_size=1, stride=2, padding_type='same')
            self.bn_2x1_pool= nn.BatchNorm1d(C)
                
        # conv_1x1_a : reduce number of channels by factor of 4 (output_channel = C/4)
        self.conv_1x1_a = Conv1d(C, int(C/4), kernel_size=1, stride=stride, padding_type='same')
        self.bn_1x1_a = nn.BatchNorm1d(int(C/4))
        
        # conv_3x3_b : more wide receptive field (output_channel = C/4)
        self.conv_3x3_b = Conv1d(int(C/4), int(C/4), kernel_size=3, stride=1, padding_type='same')
        self.bn_3x3_b = nn.BatchNorm1d(int(C/4))
        
        # conv_1x1_c : recover org channel C (output_channel = C)
        self.conv_1x1_c = Conv1d(int(C/4), C, kernel_size=1, stride=1, padding_type='same')
        self.bn_1x1_c = nn.BatchNorm1d(C)
        
        if highway:
            # conv_1x1_g : gating for highway network
            self.conv_1x1_g = Conv1d(C, C, kernel_size=1, stride=1, padding_type='same')
        
        # output
        self.bn_1x1_out = nn.BatchNorm1d(C)
        
    
    def forward(self, x):
        '''
            x : size = (batch, C, maxlen)
        '''
        
        res = x
        
        if self.pool:
            # input dimension matching with 1x1 conv
            x = self.conv_1x1_matching(x)
            x = self.bn_1x1_matching(x)
            
            # pooling of residual path
            res = self.conv_2x1_pool(res)
            res = self.bn_2x1_pool(res)
        
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
        
        if self.highway:
            # gating mechanism from "highway network"
            
            # gating factors controll intensity between x and f(x)
            # gating = 1.0 (short circuit) --> output is identity (same as initial input)
            # gating = 0.0 (open circuit)--> output is f(x) (case of non-residual network)
            gating = F.sigmoid(self.conv_1x1_g(x))
            
            # apply gating mechanism
            x = gating * res + (1.0 - gating) * F.relu(x)

            
        else:
            # normal residual ops (addition)
            x = F.relu(x) + res

            
        x = self.bn_1x1_out(x)
        x = F.relu(x)
        
        return x


# In[47]:


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

class GAP_1D(nn.Module):
    def __init__(self):
        super(GAP_1D, self).__init__()

    def forward(self, x, axis):
        '''
            x : size = (B, C, L)
        '''
        return torch.mean(x, axis)

class ConvNet(nn.Module):
    def __init__(self, input_size, num_layers = [3,4,6], num_filters = [64,128,128]):
        super(ConvNet, self).__init__()
        
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_filters = num_filters

        def res_blocks(residual_blocks, num_layers, num_filters, block_ix, pool_first_layer=True):
            block_layers = num_layers[block_ix]

            for i in range(block_layers):
                # default values
                pool = False
                block_filters = num_filters[block_ix]
                
                C_in = C_out = block_filters
                
                if pool_first_layer and i==0:
                    pool = True
                if i==0 and block_ix > 0:
                    C_in = num_filters[block_ix-1]
                    
                print(f"layer : {i}, block : {block_ix}, C_in/C_out : {C_in}/{C_out}")
                residual_blocks.append(ResidualBlock(C_in=C_in, C_out=C_out,pool=pool, highway=True))
                
        residual_blocks = []

        for i in range(len(num_layers)):
            pool_first_layer = True
            if i == 0:
                pool_first_layer = False
            res_blocks(residual_blocks, num_layers=num_layers, num_filters=num_filters, block_ix=i,
                       pool_first_layer=pool_first_layer)
                
        self.model = nn.Sequential(nn.Conv1d(input_size, num_filters[0], kernel_size=7, stride=2),
                                   nn.BatchNorm1d(num_filters[0]),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=(3,), stride=2,),
                                   nn.Conv1d(num_filters[0], num_filters[0], kernel_size=3, stride=1),
                                   nn.BatchNorm1d(num_filters[0]),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=(3,), stride=2,),
                                   *residual_blocks,
                                   )
        
    def forward(self, x):
        '''
            x : size = (batch, input_size, maxlen)
        '''
        return self.model(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.shape_net = ConvNet(input_size=np.array(mydataset['train'].input_df.pose.values[0]).shape[1],
          num_layers = [3,4], num_filters = [32,64])

        self.flow_net = ConvNet(input_size=np.array(mydataset['train'].input_df.pose_dt.values[0]).shape[1],
          num_layers = [3,4], num_filters = [32,64])

        self.GAP = GAP_1D()

        target_size = len(mydataset['train'].target_columns)

        self.conv_1x1_a = nn.Conv1d(self.shape_net.num_filters[-1], target_size, kernel_size=1,
                                  stride=1)

        self.conv_1x1_b = nn.Conv1d(self.shape_net.num_filters[-1], target_size, kernel_size=1,
                                  stride=1)

        self.linear = nn.Linear(1,1)

    def forward(self, x, dxdt):
        '''
            x : size = (B, C1 (shape), L1)
            dxdt : size = (B, C2 (motion, L2)
        '''

        # Conv features from shapeNet
        shape_conv_feats = self.shape_net(x)

        # Conv features from flowNet
        flow_conv_feats = self.flow_net(dxdt)

        # 1x1 conv (a)
        shape_conv_feats = self.conv_1x1_a(shape_conv_feats)

        # 1x1 conv (b)
        flow_conv_feats = self.conv_1x1_b(flow_conv_feats)

        # gating mechanism ??

        # feature integration
        integrated = torch.cat([shape_conv_feats, flow_conv_feats],dim=1)

        # GAP1
        integrated = self.GAP(integrated, 2)

        # GAP2
        integrated = self.GAP(integrated, 1)[:,None]

        # prediction layer
        pred = self.linear(integrated)

        return pred


net = Net()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
else:
    print("Single GPU mode")
    
net.to(device)


# In[48]:


# define criterion
criterion = nn.MSELoss()
# criterion = nn.L1Loss()

history = {'train': [],
           'test': []}

import torch.optim as optim
from torch.optim import lr_scheduler

# Observe that all parameters are being optimized
optimizer = optim.RMSprop(net.parameters())

epoch_loss = {'train': 0.0, 'test': 0.0}
kk = []
for epoch in range(1,401):
    for phase in ['train', 'test']:
        if phase=='train':
            net.train()
        elif phase=='test':
            net.eval()
        
        running_loss = 0.0

        for idx, batch_item in enumerate(dataloader[phase]):
            x, dxdt, target = batch_item['pose_seq'].to(device), \
                              batch_item['pose_dt_seq'].to(device), \
                              batch_item['targets'].to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase=='train'):
                # feed data to network
                output = net(x, dxdt)

                # compute loss
                loss = criterion(output, target)
                
                if phase=='train':
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            
        avg_loss = running_loss / len(mydataset[phase])
                
        epoch_loss[phase] += avg_loss
        
        if epoch % 10 == 0:
            print('=================={}========================'.format(phase.upper()))
            print('EPOCH : {}, AVG_MSE : {:.4f}'.format(epoch, epoch_loss[phase] / 10))
            history[phase].append(epoch_loss[phase] / 10)
            
            # init epoch_loss at its own phase
            epoch_loss[phase] = 0.0
                
# plot learning curve
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.plot(history['train'], label='train', color='b')
ax.plot(history['test'], label='test', color='orange')

ax.set_title('Learning Curve')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
plt.legend()

print("Saving learning curve...")
fig.savefig('learning.png', dpi=fig.dpi)






# # Random Forest

# In[ ]:


X = {'train': [], 'test': []}
y = {'train': [], 'test': []}

for phase in ['train', 'test']:
    i = 0
    while True:
        try:
            X[phase].append(mydataset[phase][i]['keypoints_seq'].numpy().flatten())
            y[phase].append(mydataset[phase][i]['targets'].numpy())
            i += 1
        except:
            break


# In[ ]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 4, 6, 8],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                          cv = 3, n_jobs=-1, verbose=2)

X_train, y_train, X_test, y_test = np.asanyarray(X['train']), np.asanyarray(y['train']).flatten(),                                     np.asanyarray(X['test']), np.asanyarray(y['test']).flatten()

grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


from sklearn import metrics

best_model = RandomForestRegressor(**grid_search.best_params_, n_jobs=-1)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# In[ ]:




