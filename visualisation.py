import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


fontsize1 = 15
fontsize2 = 20
fontsize3 = 25
plt.rc('xtick', labelsize=fontsize1)
plt.rc('ytick', labelsize=fontsize1)
plt.rc('figure', titlesize=fontsize3)
plt.rc('legend', fontsize=fontsize2)
plt.rc('axes', titlesize=fontsize3, labelsize=fontsize2)
GLOB_TITLES = {'LinearNet': '1-layer',
               'NonLinearNet': '2-layer',
               'MoreNonLinearNet': '4-layer',
               }


def plot_decision_boundary(X, y, clf_foo, title, arch, exp_dir):
    # Plotting decision regions
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Zt = clf_foo(torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float())
    # classes
    Z = torch.sigmoid(Zt).detach().numpy().round().reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap='Blues')
    msk = y == 1
    ax.scatter(X[msk, 0], X[msk, 1], c='r',
               s=20, label='class: 1', alpha=0.6)
    ax.scatter(X[~msk, 0], X[~msk, 1], c='green', s=20,
               label='class: 0', alpha=0.6)
    ax.set_title(f'{GLOB_TITLES[arch]} / {title}')
    ax.set_ylabel('x2'), ax.set_xlabel('x1'), ax.legend()
    ax.tick_params(axis='both', which='major')
    plt.savefig(os.path.join(exp_dir, f"boundary-{arch}-{title.split(' ')[0]}.pdf"),
                bbox_inches='tight', pad_inches=0, ax=ax)
    plt.close()


def plot_dist(data, xlabel, arch, exp_dir):
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.distplot(data, bins=10, kde=False, ax=ax)
    mn, md = np.mean(data), np.median(data)
    ax.axvline(mn, c='r', ls='--',
               label=f'mean: {mn:.3f}')
    ax.axvline(md, c='g', ls='--',
               label=f'median: {md:.3f}',)
    ax.set_xlabel(xlabel), ax.set_ylabel('observations')
    ax.set_title(GLOB_TITLES[arch]), ax.grid(), ax.legend()
    plt.savefig(os.path.join(exp_dir, f'dist-{arch}-{xlabel}.pdf'),
                bbox_inches='tight', pad_inches=0, ax=ax)
    plt.close()


def plot_convergence(xs, ys, labels, xlabel, ylabel,
                     title, ax, exp_dir, marker=None):
    for x, y, label in zip(xs, ys, labels):
        sns.lineplot(x, y, label=label, ax=ax, marker=marker)
        ax.set_title(title), ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
    ax.grid()
    if any(label is not None for label in labels):
        ax.legend()
    resfnm = f"{ylabel}-{title.strip(' ')[0]}.pdf" if isinstance(title, str)\
        else f"{ylabel}.pdf"
    plt.savefig(os.path.join(exp_dir, resfnm),
                bbox_inches='tight', pad_inches=0, ax=ax)
    plt.close()


def plot_synthetic_objects(x, y, distilled_x, distilled_y,
                           arch, step_n, exp_dir):
    dmsk = distilled_y == 1
    msk = y == 1
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.scatter(x[msk, 0], x[msk, 1], c='r', s=10,
               edgecolor='k', alpha=0.1)
    ax.scatter(x[~msk, 0], x[~msk, 1], c='lightgreen',
               s=10, edgecolor='k', alpha=0.1)
    ax.scatter(distilled_x[dmsk, 0], distilled_x[dmsk, 1],
               c='orange', marker='*', s=280, edgecolor='k')
    ax.scatter(distilled_x[~dmsk, 0], distilled_x[~dmsk, 1],
               c='cyan', marker='*', s=280, edgecolor='k')
    ax.set_yticks([round(el, 1) for el in np.linspace(-4, 4, 5)])
    # plt.rc('xtick')
    ax.set_xticks([round(el, 1) for el in np.linspace(-4, 4, 5)])
    # plt.rc('ytick')
    ax.set_xlim(-4, 4), ax.set_ylim(-4, 4)
    ax.grid(), ax.set_xlabel('x1'), ax.set_ylabel('x2')
    ax.set_title(f'{GLOB_TITLES[arch]} / step: {step_n + 1}')
    plt.savefig(os.path.join(exp_dir, f"{arch}-data-{step_n}.pdf"),
                bbox_inches='tight', pad_inches=0)
    plt.close()
