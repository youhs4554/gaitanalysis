import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader


def get_train_valid_test_loader(train_ds, valid_ds, test_ds, opt):
    num_train = len(train_ds)
    indices = list(range(num_train))
    split = int(np.floor(opt.valid_size * num_train))

    # shuffle to prevent memorizing
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_ds,
                              batch_size=opt.batch_size,
                              sampler=train_sampler,
                              num_workers=opt.n_threads)
    valid_loader = DataLoader(valid_ds,
                              batch_size=opt.batch_size,
                              sampler=valid_sampler,
                              num_workers=opt.n_threads)
    test_loader = DataLoader(test_ds,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=opt.n_threads)

    return train_loader, valid_loader, test_loader
