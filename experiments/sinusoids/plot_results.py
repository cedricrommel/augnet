from pathlib import Path
from itertools import groupby

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch

from generate_data import plot_dataset_illustration


def plot_training(training_metrics, title=None,
                  figsize=(3.25, 3)):
    """ Plots training and test loss/accuracy across epochs

    Parameters:
    -----------
    training_metrics : pandas.DataFrame
        Training data from ..._metrics.pkl file.
    """
    metrics_to_keep = [
        col for col in training_metrics
        if "loss" in col or "acc" in col or "epoch" in col
    ]
    reshaped_metrics = pd.melt(
        training_metrics[metrics_to_keep],
        id_vars=["epoch"]
    )
    reshaped_metrics["split"] = reshaped_metrics["variable"].apply(
        lambda x: x.split("_")[1]
    )
    reshaped_metrics["metric"] = reshaped_metrics["variable"].apply(
        lambda x: x.split("_")[-1]
    )
    reshaped_metrics = reshaped_metrics.drop(labels=["variable"], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    sns.lineplot(
        data=reshaped_metrics.query("metric == 'loss'"),
        x="epoch",
        y="value",
        hue="split",
        ax=ax1
    )
    ax1.grid()
    ax1.set_ylabel("Penalized CE Loss")
    ax1.get_legend().remove()
    sns.lineplot(
        data=reshaped_metrics.query("metric == 'acc'"),
        x="epoch",
        y="value",
        hue="split",
        ax=ax2
    )
    ax2.grid()
    ax2.set_ylabel("Accuracy")
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_weights(training_metrics, operations, ax):
    """Plots transforms weights evolution across AugNet training
    """
    weight_cols = [col for col in training_metrics if "weight-" in col]
    assert len(weight_cols) == len(operations)
    weights = training_metrics[weight_cols].T
    weights.index = operations
    # Add col corresponding to initialization
    weights.columns = [c + 1 for c in weights.columns]
    init = pd.DataFrame(
        np.ones(weights.shape[0]) / weights.shape[0],
        columns=[0], index=operations
    )
    weights = pd.concat([init, weights], axis=1)
    sns.heatmap(weights, vmin=0, vmax=1, ax=ax, cmap="viridis")


def plot_mags(training_metrics, operations, ax):
    """Plots transforms magnitudes evolution across AugNet training
    """
    mag_cols = [col for col in training_metrics if "magnitude-" in col]
    assert len(mag_cols) == len(operations)
    mags = training_metrics[mag_cols].T
    mags.index = operations
    # Add col corresponding to initialization
    mags.columns = [c + 1 for c in mags.columns]
    init = pd.DataFrame(
        np.zeros(mags.shape[0]), columns=[0], index=operations)
    mags = pd.concat([init, mags], axis=1)
    sns.heatmap(mags, vmin=0, vmax=1, ax=ax, cmap="viridis")


def plot_params_heatmaps(training_metrics, operations, figsize=(3.25, 3)):
    """Plots boths magnitudes and weights evolution across AugNet training
    in a heatmap
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    plot_weights(training_metrics, operations, ax1)
    ax1.set_title("Weights")
    plot_mags(training_metrics, operations, ax2)
    ax2.set_xlabel("Epoch")
    ax2.set_title("Magnitudes")
    plt.tight_layout()
    return fig


def multi_index_it(df):
    df["tf"] = df.index.to_series().apply(lambda x: x.split("-")[-1])
    df["layer"] = df.index.to_series().apply(lambda x: x.split("-")[1])
    return df.set_index(['tf', 'layer'])


def add_line(ax, xpos, ypos):
    line = plt.Line2D(
        [ypos, ypos + .3],
        [xpos, xpos],
        color='black',
        transform=ax.transAxes
    )
    line.set_clip_on(False)
    ax.add_line(line)


def label_len(my_index, level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k, g in groupby(labels)]


def label_group_bar_table(ax, df):
    """
    source: https://stackoverflow.com/questions/67654744/how-to-arrange-y-
    labels-in-seaborn-clustermap-when-using-a-multiindex-dataframe
    """
    xpos = -.3
    scale = 1./df.index.size
    for level in range(df.index.nlevels):
        pos = df.index.size
        for label, rpos in label_len(df.index, level):
            add_line(ax, pos*scale, xpos)
            pos -= rpos
            lypos = (pos + .5 * rpos)*scale
            ax.text(xpos+.1, lypos, label, ha='center', transform=ax.transAxes)
        add_line(ax, pos * scale, xpos)
        xpos -= .3


def plot_weights_fancy(training_metrics, ax):
    """Plots transforms weights evolution across AugNet training
    """
    weight_cols = [col for col in training_metrics if "weight-" in col]
    weights = training_metrics[weight_cols].T
    # Add col corresponding to initialization
    weights.columns = [c + 1 for c in weights.columns]
    init = pd.DataFrame(
        np.ones(weights.shape[0]) / weights.shape[0],
        columns=[0], index=weights.index
    )
    weights = pd.concat([init, weights], axis=1)

    weights = multi_index_it(weights)
    ax = sns.heatmap(weights, vmin=0, vmax=1, ax=ax, cmap="viridis")
    ax.set_yticks([])
    ax.set_ylabel("")
    label_group_bar_table(ax, weights)


def plot_mags_fancy(training_metrics, ax):
    """Plots transforms magnitudes evolution across AugNet training
    """
    mag_cols = [col for col in training_metrics if "magnitude-" in col]
    mags = training_metrics[mag_cols].T
    # Add col corresponding to initialization
    mags.columns = [c + 1 for c in mags.columns]
    init = pd.DataFrame(
        np.zeros(mags.shape[0]), columns=[0], index=mags.index)
    mags = pd.concat([init, mags], axis=1)

    mags = multi_index_it(mags)
    ax = sns.heatmap(mags, vmin=0, vmax=1, ax=ax, cmap="viridis")
    ax.set_yticks([])
    ax.set_ylabel("")
    label_group_bar_table(ax, mags)


def plot_heatmaps_and_curves(
    training_metrics,
    figsize=(9.75, 6),
    ylim=(-0.6, 0.6),
    n_layers=4
):
    """ Plots heatmap and learned freq shihft for multiple layers"""
    fig = plt.figure(figsize=figsize)

    assert n_layers in [2, 4], "Only 2 and 4 layers allowed"
    half_layers = n_layers // 2

    gs = GridSpec(n_layers, 3)
    ax_w = fig.add_subplot(gs[:half_layers, :-1])
    ax_m = fig.add_subplot(gs[half_layers:, :-1])

    ax_f = [
        fig.add_subplot(gs[layer, -1]) for layer in range(n_layers)
    ]

    plot_weights_fancy(training_metrics, ax_w)
    ax_w.set_title("Weights")
    ax_w.set_xticks([])
    plot_mags_fancy(training_metrics, ax_m)
    ax_m.set_xlabel("Epoch")
    ax_m.set_title("Magnitudes")
    ax_m.set_xticks([])

    for layer, ax in enumerate(ax_f):
        _plot_shift_evolution(
            training_metrics,
            ax,
            layer=layer,
            ylim=ylim,
        )
        ax.set_ylabel(f"Shift layer {layer}")

    plt.tight_layout()
    return fig


def _mag_to_shift(magnitude, mag_range):
    lb, ub = mag_range
    return magnitude * ub + (1 - magnitude) * lb


def plot_shift_evolution(
    training_metrics,
    figsize=(3.25, 1.6),
    **kwargs
):
    """ Plots learned shift within augmentation layer across epochs

    Parameters:
    -----------
    training_metrics : pandas.DataFrame
        Training data from ..._metrics.pkl file.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _plot_shift_evolution(training_metrics, ax, **kwargs)
    ax.set_ylabel("Shift range [Hz]")
    ax.set_xlabel("Iterations")
    ax.set_xticks([])
    return fig


def _plot_shift_evolution(
    training_metrics,
    ax,
    init_mag=0.,
    delta_f=0.5,
    mag_range=(0, 2),
    alpha=0.1,
    color=None,
    ylim=None,
    layer=0,
):
    """ Plots learned shift within augmentation layer across epochs

    Parameters:
    -----------
    training_metrics : pandas.DataFrame
        Training data from ..._metrics.pkl file.
    """
    first_line = training_metrics.iloc[0, :].copy()
    first_line["epoch"] = -1
    first_line["magnitude-0"] = init_mag
    training_metrics = pd.concat([first_line, training_metrics], axis=1)

    # training_metrics["shift_hz"] = training_metrics["magnitude-0"].apply(
    training_metrics["shift_hz"] = training_metrics[
        f"magnitude-l{layer}-DiffFrequencyShift"].apply(
            lambda x: _mag_to_shift(x, mag_range)
        )

    if color is None:
        colors = sns.color_palette("tab10")
        color = colors[1]

    ax.plot(
        training_metrics["epoch"], training_metrics["shift_hz"], color=color
    )
    ax.plot(
        training_metrics["epoch"], - training_metrics["shift_hz"], color=color
    )
    ax.fill_between(
        training_metrics["epoch"],
        - training_metrics["shift_hz"],
        training_metrics["shift_hz"],
        alpha=alpha,
        color=color,
    )
    ax.hlines([-delta_f, delta_f], *ax.get_xlim(), linestyle="--", color="k")

    # plt.ylabel("Frequency Shift Width [Hz]")
    if ylim is not None:
        ax.set_ylim(*ylim)


def get_invariance_res_paths(settings, folder, noise, seed):
    df = pd.DataFrame(settings)

    for tag, res_type in [
        ("inv", "model_invariance.pt"),
        ("metrics", "metrics.pkl")
    ]:
        # baselines' paths
        for label, aug in [("baseline", ""), ("perf_aug", "augmented_")]:
            df[f"{label}_{tag}_path"] = [
                folder / (
                    f"none_{setting['type']}{setting['layers']}"
                    f"{setting['neurons']}_noise{noise}_{seed}_"
                    f"{aug}{res_type}"
                )
                for setting in settings
            ]

        # AugNets' paths
        for ncopies in [1, 4, 10]:
            df[f"augnet_c{ncopies}_{tag}_path"] = [
                folder / (
                    f"augnet-freq-c{ncopies}_{setting['type']}"
                    f"{setting['layers']}{setting['neurons']}_noise{noise}_"
                    f"{seed}_{res_type}"
                )
                for setting in settings
            ]
    return df


def get_invariance_metrics(settings, folder, noise, seed):
    """ Loads into dataframe the invariances from each setting trained
    """
    df = get_invariance_res_paths(settings, folder, noise, seed)

    invariances = list()
    methods = ["baseline", "perf_aug", "augnet_c10", "augnet_c4", "augnet_c1"]
    for method in methods:
        for i in range(df.shape[0]):
            layers = df.loc[i, "layers"]
            neurons = df.loc[i, "neurons"]
            arch = df.loc[i, "type"]
            path = df.loc[i, f"{method}_inv_path"]
            res = torch.load(path).cpu().numpy()
            setting_rows = [
                {
                    "layers": layers,
                    "neurons": neurons,
                    "type": arch,
                    "method": method,
                    "invariance": res_i
                }
                for res_i in res
            ]
            invariances += setting_rows
    return pd.DataFrame(invariances)


def get_training_metrics(settings, folder, noise, seed, mag_range=(0, 2)):
    """ Loads into dataframe the best test accuracy from each setting trained
    """
    df = get_invariance_res_paths(settings, folder, noise, seed)

    metrics = list()
    methods = ["baseline", "perf_aug", "augnet_c10", "augnet_c4", "augnet_c1"]

    for method in methods:
        for i in range(df.shape[0]):
            res = pd.read_pickle(df.loc[i, f"{method}_metrics_path"])
            metrics.append({
                "layers": df.loc[i, "layers"],
                "neurons": df.loc[i, "neurons"],
                "type": df.loc[i, "type"],
                "method": method,
                "best_test_acc": res["epoch_test_acc"].values[-1],
                "learned_shift": _mag_to_shift(
                    res["magnitude-0"].values[-1],
                    mag_range
                ) if method == "augnet" else 0.,
            })
    return pd.DataFrame(metrics)


def _prepare_for_plot(df, arch='mlp', baseline=True,):
    df = df.query("type == @arch and neurons < 3")
    if not baseline:
        df = df.query("method != 'baseline'")
    df["setting"] = df["neurons"].apply(
        lambda x: f"{x} neurons,\n"
    ) + df["layers"].apply(
        lambda x: f"{x} layers"
    )
    df = df.replace({
        "baseline": "Baseline",
        "perf_aug": "Oracle augmentation",
        "augnet_c10": "AugNet (C=10)",
        "augnet_c4": "AugNet (C=4)",
        "augnet_c1": "AugNet (C=1)",
    })
    return df


def plot_invaraince_and_best_test_acc(
    invariance_res,
    test_acc_res,
    arch='mlp',
    median=True,
    baseline=True,
    # figsize=(3.25, 4),
    figsize=(3.25, 3),
):
    """ Plots outputs of get_..._metrics()

    Parameters:
    -----------
    invariance_res : pd.DataFrame
        Invariance metrics from the trainings (fetched using
        get_invariance_metrics).
    test_acc_res : pd.DataFrame
        Test acc metrics from the trainings (fetched using
        get_training_metrics).
    arch : str, optional
        Either 'cnn' or 'mlp'. Corresponds to the backbone architecture to
        plot. Default to 'mlp'.
    median : bool, optional
        Whether to plot median invariances whether then means. Default to
        True.
    baseline : bool, optional
        Whether to plot the baseline (backbone trained without augmentation).
        Default = True.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    invariance_res = _prepare_for_plot(
        invariance_res, arch=arch, baseline=baseline,
    )

    sns.lineplot(
        data=invariance_res,
        x="setting",
        y="invariance",
        hue="method",
        ax=ax1,
        estimator=np.median if median else "mean",
    )
    # ax1.tick_params(axis='x', rotation=45)
    ax1.set_xlabel("")
    ax1.set_ylabel("Invariance")
    ax1.grid()
    ax1.legend(loc="lower right")

    test_acc_res = _prepare_for_plot(
        test_acc_res, arch=arch, baseline=baseline,
    )

    sns.lineplot(
        data=test_acc_res,
        x="setting",
        y="best_test_acc",
        hue="method",
        ax=ax2,
        estimator=np.median if median else "mean",
    )
    # ax2.tick_params(axis='x', rotation=45)
    ax2.set_xlabel("")
    ax2.set_ylabel("Test accuracy")
    ax2.grid()
    ax2.get_legend().remove()

    plt.tight_layout()
    return fig, ax1, ax2


if __name__ == "__main__":
    plt.rc("text", usetex=False)
    plt.rc('font', size=7)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('savefig', bbox='tight')

    seed = 29
    folder = Path("./saved-outputs/results/")
    figs_dir = Path("./figures/test/")
    figs_dir.mkdir(exist_ok=True)

    # >>> Plotting learned shift invariance <<<

    augnet_metrics = pd.read_pickle(
        # "./saved-outputs/augnet-1_cnn22_noise0.5_29_metrics.pkl")
        # "./saved-outputs/augnet-3tf_cnn22_noise0.5_29_metrics.pkl")
        folder / "augnet-1-c4_cnn22_noise0.5_29_metrics.pkl")

    fig1 = plot_shift_evolution(augnet_metrics)
    fig1.savefig(figs_dir / "learned_shift.pdf", pad_inches=0)

    # >>> ... and the corresponding learning curves and params heatmap

    fig2 = plot_training(augnet_metrics)
    fig2.savefig(figs_dir / "training_plot.pdf", pad_inches=0)

    fig3 = plot_params_heatmaps(
        augnet_metrics,
        ["FrequencyShift", "FTSurrogate", "GaussianNoise"]
    )
    fig3.savefig(figs_dir / "params_heatmap.pdf", pad_inches=0)

    # # >>> Plotting 2 and 4 layers results

    augnet_metrics_2l = pd.read_pickle(
        folder / "augnet-2-c4_cnn22_noise0.5_29_metrics.pkl")

    augnet_metrics_4l = pd.read_pickle(
        folder / "augnet-4-c4_cnn22_noise0.5_29_metrics.pkl")

    fig2bis = plot_heatmaps_and_curves(
        augnet_metrics_2l, figsize=(9.75, 4), ylim=(-0.8, 0.8), n_layers=2)
    fig2bis.savefig(figs_dir / "two-aug-layers.pdf", pad_inches=0)

    fig2bisbis = plot_heatmaps_and_curves(augnet_metrics_4l, ylim=(-0.8, 0.8))
    fig2bisbis.savefig(figs_dir / "four-aug-layers.pdf", pad_inches=0)

    # # >>> Plotting model capacity study <<<

    settings = [
        {"layers": 1, "neurons": 1, "type": "mlp"},
        {"layers": 1, "neurons": 2, "type": "mlp"},
        {"layers": 2, "neurons": 2, "type": "mlp"},
        {"layers": 3, "neurons": 2, "type": "mlp"},
    ]

    invariances = get_invariance_metrics(
        settings=settings, folder=folder, noise=0.5, seed=seed)
    best_test_acc = get_training_metrics(
        settings, folder, noise=0.5, seed=seed)
    fig5, _, _ = plot_invaraince_and_best_test_acc(invariances, best_test_acc)
    fig5.savefig(figs_dir / "capacity_study.pdf", pad_inches=0)

    # >>> Plotting dataset illustration <<<

    fig6 = plot_dataset_illustration(
        delta_f=0.5,
        noise=0.5,
        length_s=1,
        random_state=29
    )
    fig6.savefig(figs_dir / "dataset_illustration.pdf", pad_inches=0)
