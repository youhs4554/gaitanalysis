#!/usr/bin/env python
# coding: utf-8

# In[1]:


import easydict
from sklearn.metrics.regression import r2_score, mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import matplotlib.pyplot as plt
import random
from utils import target_columns, visualization
from datasets import gaitregression
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch
import collections
import json
import os
import sys
sys.path.append('/home/hossay/gaitanalysis/src/dev/')
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
with open('../commandline_args.txt', 'r') as f:
    dd = json.load(f)

opt = easydict.EasyDict(dd)
opt.detection_file = '../../../preprocess/data/person_detection_and_tracking_results_drop.pkl'
opt.target_file = '../../../preprocess/data/targets_dataframe.pkl'
opt.sample_size = 128
opt.sample_duration = 128
opt.img_size = 128
opt.delta = 1
opt.batch_size = 8


# In[2]:



# In[3]:


target_columns = target_columns.get_target_columns(opt)

opt.mean = 0.5
opt.std = 0.5


# In[4]:


spatial_transform = TF.Compose([
    TF.ToTensor(),
    TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

temporal_transform = None

# # target transform
# from sklearn.preprocessing import FunctionTransformer

# target_transform = FunctionTransformer(lambda x: x, validate=False)

# target_transform = MinMaxScaler()
target_transform = QuantileTransformer(
    random_state=0, output_distribution="normal"
)


# # Prepare Dataset

# In[5]:


# prepare dataset  (train/test split)
data = gaitregression.prepare_dataset(
    input_file=opt.detection_file,
    target_file=opt.target_file,
    target_columns=target_columns,
    chunk_parts=opt.chunk_parts,
    target_transform=target_transform,
)


# In[6]:


# split train/valid
train_vids, valid_vids = train_test_split(
    np.array(list(set(data["train_X"].vids))), test_size=0.2, random_state=42)

X_train, y_train = (gaitregression.filter_input_df_with_vids(data["train_X"], train_vids),
                    gaitregression.filter_target_df_with_vids(data["train_y"], train_vids))
X_valid, y_valid = (gaitregression.filter_input_df_with_vids(data["train_X"], valid_vids),
                    gaitregression.filter_target_df_with_vids(data["train_y"], valid_vids))


# In[7]:


if opt.with_segmentation:
    ds_class = gaitregression.GAITSegRegDataset
else:
    ds_class = gaitregression.GAITDataset


train_ds = ds_class(
    X=X_train, y=y_train, opt=opt, phase='train', spatial_transform=spatial_transform,
    temporal_transform=temporal_transform
)
valid_ds = ds_class(
    X=X_valid, y=y_valid, opt=opt, phase='valid', spatial_transform=spatial_transform,
    temporal_transform=temporal_transform,
)
test_ds = ds_class(
    X=data["test_X"], y=data["test_y"], opt=opt, phase='test', spatial_transform=spatial_transform,
    temporal_transform=temporal_transform,
)


# In[8]:


# Dataloaders
train_loader = DataLoader(train_ds, batch_size=opt.batch_size,
                          shuffle=True, num_workers=opt.n_threads, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size=opt.batch_size,
                          shuffle=False, num_workers=opt.n_threads, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=opt.batch_size,
                         shuffle=False, num_workers=opt.n_threads, drop_last=True)


# # Define GAN

# In[9]:


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(in_features=100+15, out_features=768)

        self.upconvs = nn.Sequential(
            nn.ConvTranspose3d(in_channels=768, out_channels=384,
                               kernel_size=5, stride=2, padding=0,
                               bias=False),
            nn.BatchNorm3d(num_features=384),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(in_channels=384, out_channels=256,
                               kernel_size=5, stride=2, padding=0,
                               bias=False),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(True),

            nn.ConvTranspose3d(in_channels=256, out_channels=192,
                               kernel_size=5, stride=2, padding=0,
                               bias=False),
            nn.BatchNorm3d(num_features=192),
            nn.ReLU(True),

            nn.ConvTranspose3d(in_channels=192, out_channels=64,
                               kernel_size=5, stride=2, padding=0,
                               bias=False),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(True),

            nn.ConvTranspose3d(in_channels=64, out_channels=3,
                               kernel_size=8, stride=2, padding=0,
                               bias=False),
            nn.Tanh())

    def forward(self, z, c):
        z = torch.cat([z, c], dim=1)
        z = self.linear(z).view(z.size(0), -1, 1, 1, 1)

        return self.upconvs(z)


# In[10]:


class Discriminator(nn.Module):

    def __init__(self, num_classes=15):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Convolution 1
            nn.Conv3d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            # Convolution 2
            nn.Conv3d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            # Convolution 3
            nn.Conv3d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            # Convolution 4
            nn.Conv3d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            # Convolution 5
            nn.Conv3d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

            # Convolution 6
            nn.Conv3d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False))

        # discriminator fc
        self.fc_dis = nn.Linear(13*13*13*512, 1)

        # aux-classifier fc
        self.fc_aux = nn.Linear(13*13*13*512, num_classes)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        dis_out = torch.sigmoid(self.fc_dis(x))
        aux_out = self.fc_aux(x)

        return dis_out, aux_out


# In[11]:


G = Generator().cuda()
D = Discriminator().cuda()


# In[12]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


G.apply(weights_init)
D.apply(weights_init)


# In[13]:


dis_criterion = nn.BCELoss()
aux_criterion = nn.MSELoss()
rec_criterion = nn.MSELoss()

G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))


# # training loop

# In[ ]:


plotter = visualization.VisdomPlotter(env_name='ACGAN_with_reconstructionLoss')


# In[ ]:


def compute_scores(y_pred, y_true, target_transform, scores=['r2', 'mae']):
    score_dict = {
        'r2': r2_score,
        'mae': mean_absolute_error,
        'mse': mean_squared_log_error
    }

    res = []
    for name in scores:
        score_func = score_dict.get(name)
        scores = score_func(
            target_transform.inverse_transform(y_true),
            target_transform.inverse_transform(y_pred),
            multioutput='raw_values',
        )

        res.append(scores)

    return res


# In[ ]:


print('Start training...')

label_real = torch.ones([opt.batch_size, 1]).cuda()
label_fake = torch.zeros([opt.batch_size, 1]).cuda()

# D losses
# 0. total
running_D_loss_tot = 0.0
# 1. real
running_D_loss_real = 0.0
running_D_aux_loss_real = 0.0
# 2. fake
running_D_loss_fake = 0.0
running_D_aux_loss_fake = 0.0

# G losses
# 0. total
running_G_loss_tot = 0.0
# 1. fake
running_G_loss_fake = 0.0
running_G_aux_loss_fake = 0.0
running_G_rec_loss = 0.0   # reconstruction loss

# scores
# 1. real
running_r2_real = [0.0 for _ in range(len(target_columns))]
running_mae_real = [0.0 for _ in range(len(target_columns))]


# 2. fake
running_r2_fake = [0.0 for _ in range(len(target_columns))]
running_mae_fake = [0.0 for _ in range(len(target_columns))]


interval = 10

global_step = 0

for epoch in range(opt.n_iter):
    # training loop
    for step_ix, (inputs, targets, vids) in enumerate(train_loader):
        ### Train D ###
        D.zero_grad()

        # for real data...
        real_data = inputs.cuda()
        dis_out_real, aux_out_real = D(real_data)
        D_loss_real = dis_criterion(dis_out_real, label_real)
        D_aux_loss_real = aux_criterion(aux_out_real, targets.cuda())
        D_loss_realpart = D_loss_real + D_aux_loss_real
        D_loss_realpart.backward()

        # latent vector z
        z = torch.randn((opt.batch_size, 100))

        # for fake data from G...
        fake_data = G(z.cuda(), targets.cuda())
        dis_out_fake, aux_out_fake = D(fake_data.detach())
        D_loss_fake = dis_criterion(dis_out_fake, label_fake)
        D_aux_loss_fake = aux_criterion(aux_out_fake, targets.cuda())
        D_loss_fakepart = D_loss_fake + D_aux_loss_fake
        D_loss_fakepart.backward()

        # D_loss : Discriminator's loss
        D_loss = D_loss_realpart + D_loss_fakepart

        # Update D
        D_optimizer.step()

        # 0. total
        running_D_loss_tot += D_loss.item()
        # 1. real
        running_D_loss_real += D_loss_real.item()
        running_D_aux_loss_real += D_aux_loss_real.item()
        # 2. fake
        running_D_loss_fake += D_loss_fake.item()
        running_D_aux_loss_fake += D_aux_loss_fake.item()

        ### Train G ###
        G.zero_grad()
        dis_out_fake, aux_out_fake = D(fake_data)

        # loss of G for deceiving D
        G_loss_fake = dis_criterion(dis_out_fake, label_real)

        G_aux_loss_fake = aux_criterion(aux_out_fake, targets.cuda())
        G_rec_loss = rec_criterion(fake_data, real_data)  # reconstruction loss
        G_loss = G_loss_fake + G_aux_loss_fake + G_rec_loss

        # Update G
        G_loss.backward()
        G_optimizer.step()

        # 0. total
        running_G_loss_tot += G_loss.item()
        # 1. fake
        running_G_loss_fake += G_loss_fake.item()
        running_G_aux_loss_fake += G_aux_loss_fake.item()
        running_G_rec_loss += G_rec_loss.item()

        # compute scores for real data
        r2_real, mae_real = compute_scores(aux_out_real.detach().cpu().numpy(),
                                           targets.numpy(), target_transform,
                                           scores=['r2', 'mae'])
        # scores (real)
        for n in range(len(target_columns)):
            running_r2_real[n] += r2_real[n]
        for n in range(len(target_columns)):
            running_mae_real[n] += mae_real[n]

        # compute scores for fake data
        r2_fake, mae_fake = compute_scores(aux_out_fake.detach().cpu().numpy(),
                                           targets.numpy(), target_transform,
                                           scores=['r2', 'mae'])

        # scores (real)
        for n in range(len(target_columns)):
            running_r2_fake[n] += r2_fake[n]
        for n in range(len(target_columns)):
            running_mae_fake[n] += mae_fake[n]

        global_step += 1

        if step_ix % interval == 0:
            for win, e in zip(['Fake', 'Real'], [fake_data, real_data]):
                e = e.detach().cpu().numpy()[0][:, ::2]
                e = np.clip(e.transpose(1, 2, 3, 0) *
                            opt.std + opt.mean, 0.0, 1.0)
                plotter.viz.images(
                    e.transpose(0, 3, 1, 2),
                    win=win)

            print('Epoch : ', epoch, 'Step :', step_ix,
                  'D_loss_tot : ', running_D_loss_tot/interval,
                  'D_loss_real : ', running_D_loss_real/interval,
                  'D_aux_loss_real : ', running_D_aux_loss_real/interval,
                  'D_loss_fake : ', running_D_loss_fake/interval,
                  'D_aux_loss_fake : ', running_D_aux_loss_fake/interval,
                  'G_loss_tot : ', running_G_loss_tot/interval,
                  'G_loss_fake : ', running_G_loss_fake/interval,
                  'G_aux_loss_fake : ', running_G_aux_loss_fake/interval,
                  'G_rec_loss : ', running_G_rec_loss/interval,
                  'r2_real : ', np.mean(running_r2_real)/interval,
                  'mae_real : ', np.mean(running_mae_real)/interval,
                  'r2_fake : ', np.mean(running_r2_fake)/interval,
                  'mae_fake : ', np.mean(running_mae_fake)/interval
                  )

            plotter.plot('D_loss_tot', 'train', 'D_loss_tot__trace',
                         global_step, running_D_loss_tot/interval)
            plotter.plot('D_loss_real', 'train', 'D_loss_real__trace',
                         global_step, running_D_loss_real/interval)
            plotter.plot('D_aux_loss_real', 'train', 'D_aux_loss_real__trace',
                         global_step, running_D_aux_loss_real/interval)
            plotter.plot('D_loss_fake', 'train', 'D_loss_fake__trace',
                         global_step, running_D_loss_fake/interval)
            plotter.plot('D_aux_loss_fake', 'train', 'D_aux_loss_fake__trace',
                         global_step, running_D_aux_loss_fake/interval)

            plotter.plot('G_loss_tot', 'train', 'G_loss_tot__trace',
                         global_step, running_G_loss_tot/interval)
            plotter.plot('G_loss_fake', 'train', 'G_loss_fake__trace',
                         global_step, running_G_loss_fake/interval)
            plotter.plot('G_aux_loss_fake', 'train', 'G_aux_loss_fake__trace',
                         global_step, running_G_aux_loss_fake/interval)
            plotter.plot('G_rec_loss', 'train', 'G_rec_loss__trace',
                         global_step, running_G_rec_loss/interval)
            plotter.plot('G_aux_loss_fake', 'train', 'G_aux_loss_fake__trace',
                         global_step, running_G_aux_loss_fake/interval)

            plotter.plot('r2_real', 'train', 'r2_real_avg__trace',
                         global_step, np.mean(running_r2_real)/interval)
            plotter.plot('mae_real', 'train', 'mae_real_avg__trace',
                         global_step, np.mean(running_mae_real)/interval)
            plotter.plot('r2_fake', 'train', 'r2_fake_avg__trace',
                         global_step, np.mean(running_r2_fake)/interval)
            plotter.plot('mae_fake', 'train', 'mae_fake_avg__trace',
                         global_step, np.mean(running_mae_fake)/interval)

            # D losses
            # 0. total
            running_D_loss_tot = 0.0
            # 1. real
            running_D_loss_real = 0.0
            running_D_aux_loss_real = 0.0
            # 2. fake
            running_D_loss_fake = 0.0
            running_D_aux_loss_fake = 0.0

            # G losses
            # 0. total
            running_G_loss_tot = 0.0
            # 1. fake
            running_G_loss_fake = 0.0
            running_G_aux_loss_fake = 0.0
            running_G_rec_loss = 0.0   # reconstruction loss

            # scores
            # 1. real
            running_r2_real = [0.0 for _ in range(len(target_columns))]
            running_mae_real = [0.0 for _ in range(len(target_columns))]

            # 2. fake
            running_r2_fake = [0.0 for _ in range(len(target_columns))]
            running_mae_fake = [0.0 for _ in range(len(target_columns))]


# In[ ]:

