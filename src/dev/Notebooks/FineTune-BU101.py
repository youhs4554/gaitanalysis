#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import natsort
import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import sklearn
from sklearn.model_selection import train_test_split


# In[2]:


import torchvision
import torch
import torch.nn as nn


# In[3]:


from torchvision.datasets import ImageFolder
import torchvision.transforms as TF
import sklearn
from sklearn.model_selection import train_test_split


# In[4]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"


# In[6]:


# define train/test transforms
transforms = {
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[7]:


class BU101(torch.utils.data.Dataset):
    def __init__(self, split, *args, **kwargs):
        # common data attribute
        self.data = ImageFolder(*args, **kwargs)
        
        files = natsort.natsorted(
            glob.glob(os.path.join(kwargs.get("root"), '*', '*')))
        train_files, test_files = train_test_split(files, random_state=0)
        
        # dataset samples to get indices of train(or test) files
        ds_samples = np.array(self.data.samples)[:, 0].tolist()
        self.imgs = test_files if split == 'test' else train_files  # list of image path to use
        self.split = split
        
        # pos of train(or test) files
        self.pos = [ ds_samples.index(file) for file in self.imgs ]
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img, lab = self.data[self.pos[idx]]
        return img, lab


# In[8]:


IMG_ROOT = '/data/torch_data/BU101/images'
VALID_SIZE = 0.2
BATCH_SIZE = 512
NUM_WORKERS = 32
FINE_TUNE = True

datasets = {
    split: BU101(root=IMG_ROOT, transform=transforms[split], split=split) for split in ['train','valid','test']
}


# In[ ]:


import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader

def get_train_valid_test_loader(train_ds, valid_ds, test_ds, 
                                valid_size, batch_size, n_threads):
    
    num_train = len(train_ds)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    # shuffle to prevent memorizing
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=n_threads, pin_memory=True)
    valid_loader = DataLoader(valid_ds,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=n_threads, pin_memory=True)
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=n_threads, pin_memory=True)

    return train_loader, valid_loader, test_loader


# In[ ]:


dataloaders = get_train_valid_test_loader(
    train_ds=datasets['train'], valid_ds=datasets['valid'], test_ds=datasets['test'],
    valid_size=VALID_SIZE, batch_size=BATCH_SIZE, n_threads=NUM_WORKERS)

dataloaders = dict(zip(['train','valid','test'], dataloaders))


# # Check dataset

# In[ ]:


# train_dl_iter = iter(dataloaders['train'])
# imgs, labs = train_dl_iter.next()

# from torchvision.utils import make_grid

# def denormalize_img(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#     shapes = (-1,) + (1,) * (img_tensor.dim()-1)
#     mean = torch.as_tensor(mean).view(shapes)
#     std = torch.as_tensor(std).view(shapes)
#     img_tensor = (img_tensor * std + mean).permute(1,2,0)
    
#     return img_tensor

# plt.figure(figsize=(10,10))
# plt.imshow(denormalize_img(make_grid(imgs[:16])))

# idx2class = { v:k for k,v in dataloaders['train'].dataset.data.class_to_idx.items() }
# class_names = [ idx2class[i] for i in labs.numpy() ]
# print(class_names[:16])


# In[ ]:


# def showClsDistribution(ds):
#     labs, cnts = np.unique([ 
#         os.path.basename(os.path.dirname(p)) for p in ds.imgs ], return_counts=True)

#     title = ds.split
    
#     plt.figure(figsize=(15,15))
#     plt.barh(labs, cnts)
#     plt.title('Class Distribution of ' + title + ' dataset', fontdict={'fontsize': 20})

# showClsDistribution(datasets['train'])
# showClsDistribution(datasets['test'])


# In[ ]:


model = models.resnet101(pretrained=True)

if not FINE_TUNE:
    for parameter in model.parameters():
        parameter.requires_grad = False

# 새로운 fully-connected classifier layer 를 만들어줍니다. (requires_grad 는 True)
n_features = model.fc.in_features
model.fc = nn.Linear(n_features, 101)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

if torch.cuda.is_available():
    model = model.cuda()


# In[ ]:


import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import tqdm

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.000027)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# In[ ]:


dataset_sizes = {'valid': int(np.floor(len(datasets['train'])*VALID_SIZE)),
                 'test': len(datasets['test'])
                }
dataset_sizes['train'] = len(datasets['train'])-dataset_sizes['valid']


# In[ ]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            dl = iter(dataloaders[phase])
            
            for _ in tqdm.tqdm(range(len(dl)), desc='[{}] epoch-{}'.format(phase, epoch)):
                inputs, labels = dl.next()
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if scheduler and phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# In[ ]:


model_ft = train_model(model, criterion, optimizer, scheduler=None, num_epochs=100)


# In[ ]:


visualize_model(model_ft)


# In[ ]:


len(datasets['train'])


# In[ ]:


len(dataloaders['train'])*BATCH_SIZE


# In[ ]:


int(np.floor(len(datasets['train'])*VALID_SIZE))


# In[ ]:




