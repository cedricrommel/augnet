import os
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from augnet import models

from augnet_training import create_augnet_layer


def plot_shade(logger, ax, color, label="", alpha=0.1, lwd=0.):
    """Plots rotation angle evolution across training
    """
    ax.fill_between(logger.index, logger['lowbd'], logger['upbd'],
                    alpha=alpha, color=color,
                    linewidth=lwd)
    sns.lineplot(
        x=logger.index,
        y='lowbd',
        color=color,
        data=logger,
        label=label
    )
    sns.lineplot(
        x=logger.index,
        y='upbd',
        color=color,
        data=logger
    )


def plot_weights(logger, model, ax, skip=78):
    """Plots transforms weights evolution across AugNet training
    """
    weights = logger.iloc[:, 11:16].values.T
    # skip cols which are not epoch starts
    weights = pd.DataFrame(weights[:, ::skip])
    weights.index = [type(op).__name__ for op in model.aug.augmentations]
    sns.heatmap(weights, vmin=0, vmax=1, ax=ax, cmap="viridis")


def plot_mag(logger, model, ax, skip=78):
    """Plots transforms magnitudes evolution across AugNet training
    """
    mags = logger.iloc[:, 6:11].values.T
    # skip cols which are not epoch starts
    mags = pd.DataFrame(mags[:, ::skip])
    mags.index = [type(op).__name__ for op in model.aug.augmentations]
    for op in model.aug.augmentations:
        mags.loc[type(op).__name__, :] /= op.magnitude_scale
    sns.heatmap(mags, vmin=0, vmax=1, ax=ax, cmap="viridis")


if __name__ == "__main__":
    plt.rc("text", usetex=False)
    plt.rc('font', size=7)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('savefig', bbox='tight')

    figs_dir = "./figures/"

    # Fetch results
    savedir = "saved-outputs/"
    augnet_logger = pd.read_pickle(savedir + "augnet_logger.pkl")
    augnet_inc_logger = pd.read_pickle(
        savedir + "augnet_incomp_reg_logger.pkl")
    augnet_noreg_logger = pd.read_pickle(
        savedir + "augnet_no_reg_logger.pkl")
    augerino_logger = pd.read_pickle(savedir + "augerino_logger.pkl")

    # Scale angles properly
    augerino_logger['lowbd'] = - augerino_logger['width2'] / 2.
    augerino_logger['upbd'] = augerino_logger['width2'] / 2.
    augnet_logger['lowbd'] = - augnet_logger['mag.2'] * np.pi / 180
    augnet_logger['upbd'] = augnet_logger['mag.2'] * np.pi / 180
    augnet_inc_logger['lowbd'] = - augnet_inc_logger['mag.2'] * np.pi / 180
    augnet_inc_logger['upbd'] = augnet_inc_logger['mag.2'] * np.pi / 180
    augnet_noreg_logger['lowbd'] = - augnet_noreg_logger['mag.2'] * np.pi / 180
    augnet_noreg_logger['upbd'] = augnet_noreg_logger['mag.2'] * np.pi / 180

    # Plot rotation angle evolution
    tick_pts = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    tick_labs = [r"-$\pi$/2", r'-$\pi$/4', '0', r'$\pi$/4', r'$\pi$/2']

    # fig, ax0 = plt.subplots(1, 1, figsize=(8, 4), dpi=100)
    fig, ax0 = plt.subplots(1, 1, figsize=(3.25, 1.6), dpi=100)
    fs = 7
    pal = sns.color_palette("tab10")
    col0 = pal[0]
    col1 = pal[1]
    col2 = pal[2]
    col3 = pal[3]

    plot_shade(augerino_logger, ax0, col0, r"Augerino - $\lambda \|\mu\|_2$")
    plot_shade(augnet_logger, ax0, col1,
               r"AugNet - $\lambda \|\mu \odot \omega\|_2$")
    plot_shade(augnet_inc_logger, ax0, col2, r"AugNet - $\lambda \|\mu\|_2$")
    plot_shade(augnet_noreg_logger, ax0, col3, r"AugNet - no reg")

    ax0.set_xlabel("Iterations")
    ax0.set_ylabel("Rotation Width")
    # ax0.tick_params("both")
    # sns.despine()
    ax0.set_xticks([])
    ax0.set_yticks(tick_pts)
    ax0.set_yticklabels(tick_labs)
    ax0.hlines([-np.pi/4, np.pi/4], 0, augerino_logger.shape[0],
               linestyle="--", color="k")

    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig(
        join(figs_dir, "augnet-augerino-and-ablation.pdf"), pad_inches=0)

    # Create dummy AugNet model to use in plot
    aug_layer = create_augnet_layer(
        -1,
        freeze_weights=False
    )

    net = models.SimpleConv(c=32, num_classes=4)

    augnet_model = models.AugAveragedModel(
        net,
        aug_layer,
        ncopies=1
    )

    # Plot weights and magnitudes with two possible regularizers
    fig, ax_list = plt.subplots(
        # 2, 2, figsize=(16, 2*3), sharex=True, sharey=True)
        2, 2, figsize=(6.75, 2), sharex=True, sharey=True)

    plot_weights(augnet_logger, augnet_model, ax_list[0, 0])
    ax_list[0, 0].set_title("Weights", fontsize=fs)
    ax_list[0, 0].set_ylabel(r"$\|\mu \odot \omega\|_2$")

    plot_weights(augnet_inc_logger, augnet_model, ax_list[1, 0])
    ax_list[1, 0].set_ylabel(r"$\|\mu\|_2$")
    ax_list[1, 0].set_xlabel("Iterations")
    ax_list[1, 0].set_xticks([])

    plot_mag(augnet_logger, augnet_model, ax_list[0, 1])
    ax_list[0, 1].set_title("Magnitudes", fontsize=fs)

    plot_mag(augnet_inc_logger, augnet_model, ax_list[1, 1])
    ax_list[1, 1].set_xlabel("Iterations")
    ax_list[1, 1].set_xticks([])

    plt.tight_layout()
    fig.savefig(
        join(figs_dir, "ablation-weights-mags-heatmap.pdf"), pad_inches=0)
