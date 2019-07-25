import os
import collections
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

plt.ioff()

def scatterplots(target_columns, y_true, y_pred, save_dir=''):
    pred_and_gt = {k: [] for k in target_columns}

    for i, col in enumerate(target_columns):
        pred_and_gt[col].append([y_pred[:, i], y_true[:, i]])

    data = collections.defaultdict(list)

    pp = []
    gg = []
    for i, col in enumerate(pred_and_gt.keys()):
        transposed_data = list(zip(*pred_and_gt[col]))
        preds = np.concatenate(transposed_data[0])
        gts = np.concatenate(transposed_data[1])

        pp.append(preds)
        gg.append(gts)

        for p, g in zip(preds, gts):
            data["name"].append(col)
            data["pred"].append(p)
            data["gt"].append(g)

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))

    axes = axes.flatten()

    for i, col in enumerate(target_columns):
        part_of_df = df.loc[df.name == col]
        ax = axes[i]
        part_of_df.plot.scatter(x="gt", y="pred", c='green', ax=ax, label='data')
        ax.set_title(f'name={col}')

    for i, (preds, gts) in enumerate(zip(pp, gg)):
        ax = axes[i]
        ax.plot([min(gts), max(gts)], [min(gts), max(gts)], 'r--', label='GT=PRED')
        ax.legend()

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    img_path = os.path.join(save_dir, f'scatter_plots.png')
    os.system(f'mkdir -p {save_dir}')

    plt.savefig(img_path)
    plt.close(fig)


def dist_plots(target_columns, group_cols, y_true, y_pred, save_dir, grid_size, figsize, group=None):
    nrow, ncol = grid_size

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)
    axes = axes.flatten()

    for ax, col in zip(axes, group_cols):
        ix = target_columns.index(col)
        sns.distplot(y_pred[:, ix], hist=False, label='y_pred', ax=ax)
        sns.distplot(y_true[:, ix], hist=False, label='y_true', ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.legend()

    fig.tight_layout()

    #     img_path = os.path.join(save_dir, f'trace_plots_{group}.png')
    img_path = os.path.join(save_dir, f'dist_plots_{group}.png')

    os.system(f'mkdir -p {save_dir}')

    plt.savefig(img_path)



def margin_plots(target_columns, group_cols, y_true, y_pred, save_dir, grid_size, figsize, group=None):
    nrow, ncol = grid_size

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)
    axes = axes.flatten()

    for ax,col in zip(axes,group_cols):

        ix = target_columns.index(col)

        sampled_y_true = y_true[:,ix]
        sampled_y_pred = y_pred[:,ix]

        sorted_ixs = np.argsort(sampled_y_true)

        sampled_y_true = sampled_y_true[sorted_ixs]
        sampled_y_pred = sampled_y_pred[sorted_ixs]

        error = sampled_y_true-sampled_y_pred

        ax.set_title(col+' trace')
        ax.plot(sampled_y_true, '^', color='orange', label='y_true')
        ax.plot(sampled_y_pred, label='y_pred')
        ax.fill_between(np.arange(len(sampled_y_true)), sampled_y_true-error, sampled_y_true,
        alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
    linewidth=0, label='margin')
        ax.legend()


    fig.tight_layout()

    #     img_path = os.path.join(save_dir, f'trace_plots_{group}.png')
    img_path = os.path.join(save_dir, f'margin_plots_{group}.png')

    os.system(f'mkdir -p {save_dir}')

    plt.savefig(img_path)