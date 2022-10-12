import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_concat_results(save_dir, methods):
    """Loads results of CIFAR10 training

    Parameters
    ----------
    save_dir : str
        Path to directory where results are stored.
    methods : dict
        Dictionary mapping methods names (as they will be repported in the plot
        ) to prefix of all result files associated with the metho.

    Returns
    -------
    pandas.DataFrame
        Results of all methods from the `methods` arg which could be found in
        `save_dir`.
    """
    training_res = list()
    for fname in os.listdir(save_dir):
        for method, key in methods.items():
            if key in fname and fname.endswith("metrics.pkl"):
                res = pd.read_pickle(save_dir / fname)
                seed = fname.split("_")[-2]
                res["method"] = method
                res["seed"] = seed
                training_res.append(res)
    return pd.concat(training_res, axis=0, ignore_index=True)


def add_earlystopped_test_acc_for_all(training_res):
    """Add a column to the DataFrame of results containing the earlystopped
    test accuracy based on validation accuracy improvements.

    Parameters
    ----------
    training_res : pandas.DataFrame
        DataFrame containing the results for all methods compared.

    Returns
    -------
    pandas.DataFrame
        Updated dataframe.
    """
    updated_blocks = list()
    for method in training_res["method"].unique():
        for seed in training_res["seed"].unique():
            training_block = training_res.query(
                "seed == @seed and method == @method").reset_index(drop=True)
            best_valid = training_block.loc[0, "epoch_valid_acc"]
            test_based_on_valid = training_block.loc[0, "epoch_test_acc"]
            training_block[
                "epoch_test_acc_based_on_valid"] = test_based_on_valid
            for i in range(1, training_block.shape[0]):
                if training_block.loc[i, "epoch_valid_acc"] > best_valid:
                    test_based_on_valid = training_block.loc[
                        i, "epoch_test_acc"]
                    best_valid = training_block.loc[i, "epoch_valid_acc"]
                training_block.loc[
                    i, "epoch_test_acc_based_on_valid"] = test_based_on_valid
            updated_blocks.append(training_block)
    return pd.concat(updated_blocks, axis=0, ignore_index=True)


def fetch_weights_and_mags(base_path, seed):
    """Loads augmentation parameters learned for each epoch

    Parameters
    ----------
    base_path : str
        Path to the results folder + model prefix
    seed : int
        Random seed to fetch results from.

    Returns
    -------
    pandas.DataFrame
        Learned augmentation weights at each epoch.
    pandas.DataFrame
        Learned augmentation magnitudes at each epoch.
    """
    weights = pd.read_pickle(f"{base_path}_{seed}_augweights.pkl")
    mags = pd.read_pickle(f"{base_path}_{seed}_augmags.pkl")
    return weights, mags


def melt_transforms_evol(weights_or_mags, tag="weights"):
    """Reshapes learned augmentation DataFrame for plotting.
    """
    melted_weights = weights_or_mags.T
    melted_weights["epoch"] = melted_weights.index
    melted_weights = melted_weights.melt(
        id_vars=["epoch"],
        var_name="transform",
        value_name=tag
    )
    melted_weights["layer"] = [
        ind.split('-')[0]
        for ind in melted_weights["transform"]
    ]
    melted_weights["tfs"] = [
        ind.split('-')[1]
        for ind in melted_weights["transform"]
    ]
    return melted_weights


def plot_magnitudes_and_weights(weights, mags, fig_width=15):
    """ Plots learned magnitudes and weights per epoch
    """
    layers = weights["layer"].unique()
    n_layers = len(layers)
    fig, ax_list = plt.subplots(
        n_layers, 2,
        figsize=(fig_width, n_layers * fig_width / 5),
        sharex=True,
        sharey=True,
    )

    for i, layer in enumerate(layers):
        ax = ax_list[i][0]
        sns.lineplot(
            data=weights.query("layer == @layer"),
            x="epoch",
            y="weights",
            hue="tfs",
            ax=ax,
        )
        ax.grid()
        ax.set_ylabel(f"Layer {int(layer) + 1}")
        ax.legend().set_title('')
        if i == 0:
            ax.set_title("Weights")

        ax = ax_list[i][1]
        sns.lineplot(
            data=mags.query("layer == @layer"),
            x="epoch",
            y="magnitudes",
            hue="tfs",
            ax=ax,
        )
        ax.grid()
        ax.legend([], [], frameon=False)
        if i == 0:
            ax.set_title("Magnitudes")
    plt.tight_layout()
    return fig


def fetch_and_plot_weights_mags(base_path, seed, fig_width=15):
    """Loads augmentation parameters learned for each epoch

    Parameters
    ----------
    base_path : str
        Path to the results folder + model prefix
    seed : int
        Random seed to fetch results from.
    fig_width : float
        Desired figure width.

    Returns
    -------
    Figure handle.
    """
    weights, mags = fetch_weights_and_mags(base_path, seed)

    weights = melt_transforms_evol(weights)
    mags = melt_transforms_evol(mags, tag="magnitudes")

    return plot_magnitudes_and_weights(weights, mags, fig_width)


def _load_one_inference_time_res(file_path):
    with open(file_path, "rb") as f:
        res = pickle.load(f)
    entry = {
        "test_time": res["test"][1:],
    }
    entry = pd.DataFrame(entry)
    return entry


def load_inference_time_data(main_dir, ncopies_tested):
    """Loads results of inference time vs C experiment

    Parameters
    ----------
    main_dir : Path
        Path to tidrectory where experiment results are located.
    ncopies_tested : list
        Values of C (ncopies) to look for in results.

    Returns
    -------
    padans.DataFrame
        Table of experimental results containing the inference time (normalized
        wrt baseline with fixed augmentation) and ncopies.
    """
    c_vs_time = list()
    # load augnet times for different values of ncopies
    for ncopies in ncopies_tested:
        file_path = (
            main_dir /
            f"augnet-c{ncopies}-"
            "r0.5-non-aff-aff-aff_resnet18_no_trans_29_timer.pkl"
        )
        entry = _load_one_inference_time_res(file_path)
        entry["ncopies"] = ncopies
        c_vs_time.append(entry)

    # baseline
    file_path = main_dir / "baseline_resnet18_fixed_trans_29_timer.pkl"
    entry = _load_one_inference_time_res(file_path)
    entry["ncopies"] = "baseline"
    c_vs_time.append(entry)

    # group
    c_vs_time = pd.concat(c_vs_time, ignore_index=True)

    # normalize
    ref = c_vs_time.query("ncopies == 'baseline'")["test_time"].median()
    c_vs_time["test_time"] /= ref
    return c_vs_time


if __name__ == "__main__":
    # SETTINGS
    plt.rc("text", usetex=False)
    plt.rc('font', size=7)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('savefig', bbox='tight')

    # FIGURE 6.A - CIFAR10 CURVES
    save_dir = Path("./saved-outputs/results/")
    methods = {
        "No augmentation": "baseline_resnet18_no_trans",
        "Fixed augmentation": "baseline_resnet18_fixed_trans",
        "Augerino": "augerino-c20_resnet18_no_trans",
        "AutoAugment": "baseline_resnet18_autoaug",
        "RandAugment": "baseline_resnet18_randaug",
        "AugNet": "augnet-c20-non-aff-aff-aff_resnet18_no_trans",
    }

    # Fetch results and compute earlystopped scores
    training_res = load_and_concat_results(save_dir, methods)
    training_res = add_earlystopped_test_acc_for_all(training_res)

    # Compute epoch-wise statistics
    temp_m = training_res.groupby(
        ["method", "epoch"]
    )["epoch_test_acc_based_on_valid"].mean()

    temp_q = training_res.groupby(
        ["method", "epoch"]
    )["epoch_test_acc_based_on_valid"].quantile(
        [0., .1, 0.25, .5, 0.75, 0.9, 1.]
    ).unstack()

    # Plot in correct order
    methods_to_plot_in_order = [
        "No augmentation",
        "Fixed augmentation",
        "Augerino",
        "RandAugment",
        "AutoAugment",
        "AugNet",
    ]

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(3.25, 2.25))

    for method in methods_to_plot_in_order:
        mean = temp_m.loc[method]
        quantiles = temp_q.loc[method]
        plt.plot(mean.index, mean, label=method)
        plt.fill_between(
            quantiles.index,
            quantiles[0.1],
            quantiles[0.9],
            alpha=0.3
        )

    plt.ylim(0.75, 0.95)
    plt.legend(loc=0)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.savefig("figures/cifar10-curves.pdf", pad_inches=0)
    plt.show()

    # FIGURE B.5.A - PERFORMANCE SENSITIVITY WRT PENALTY WEIGHT

    # Fetch results and compute earlystopped scores for other values of lambda
    save_dir_reg_study = save_dir / "reg_sensitivity_study"

    reg_study_models = {
        "AugNet 0.05": "augnet-c20-r0.05-non-aff-aff-aff_resnet18_no_trans",
        "AugNet 0.1": "augnet-c20-r0.1-non-aff-aff-aff_resnet18_no_trans",
        "AugNet 1.0": "augnet-c20-r1.0-non-aff-aff-aff_resnet18_no_trans",
    }

    training_res_reg_study = load_and_concat_results(
        save_dir_reg_study,
        reg_study_models
    )
    training_res_reg_study = add_earlystopped_test_acc_for_all(
        training_res_reg_study
    )

    # Keep only the performance at the last epoch
    other_perfs = training_res_reg_study.query("epoch == 299").loc[
        :,
        ["epoch", "seed", "method", "epoch_test_acc_based_on_valid"]
    ].reset_index(drop=True)

    # Also extract final performance of the model trained with the selected
    # lambda and join both dataframes
    selection_perf = training_res.query(
        "method == 'AugNet' and epoch == 299"
    ).replace("AugNet", "AugNet 0.5")
    selection_perf = selection_perf.loc[
        :,
        ["epoch", "seed", "method", "epoch_test_acc_based_on_valid"]
    ].reset_index(drop=True)

    reg_study_perfs = pd.concat(
        [selection_perf, other_perfs],
        ignore_index=True
    )

    # Add a column with lambda value and plot
    reg_study_perfs["lambda"] = reg_study_perfs["method"].apply(
        lambda x: float(x.split(" ")[-1])
    )

    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    g = sns.pointplot(
        data=reg_study_perfs,
        x="lambda",
        y="epoch_test_acc_based_on_valid",
        ci=75,
        capsize=.15,
        linewidth=7,
        ms=3,
        ax=ax,
    )

    ax.grid()
    ylabels = ['{:.3f}'.format(x) for x in g.get_yticks()]
    g.set_yticklabels(ylabels)
    ax.set_xlabel(r"Penalty weight $\lambda$")
    ax.set_ylabel("Test accuracy (earlystopped)")
    plt.savefig(
        "figures/cifar10-lambda-vs-perf.pdf",
        pad_inches=0
    )

    # FIGURE B.6 - INFERENCE TIME VS C

    inference_time_exp_path = save_dir / "timer_experiment"

    #  fetch results
    normalized_c_vs_time = load_inference_time_data(
        inference_time_exp_path,
        [1, 4, 10, 20, 40]
    )

    # plot
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    sns.lineplot(
        data=normalized_c_vs_time.query("ncopies != 'baseline'"),
        x="ncopies",
        y="test_time",
        ci=75,
        estimator=np.median,
        ms=7,
        err_style="bars",
        marker="o",
        ax=ax,
        # label="AugNet"
    )
    plt.grid()
    plt.xlabel(r"Number of copies at inference $C$")
    plt.ylabel("Relative inference time")
    # plt.hlines(
    #     1,
    #     *plt.xlim(),
    #     linestyle='--',
    #     color="black",
    #     label="Baseline"
    # )
    # plt.legend(loc=0)
    plt.savefig(
        "figures/cifar10-c-vs-inftime.pdf",
        pad_inches=0
    )

    # FIGURE B.5.B - PERFORMANCE VS C
    perf_vs_C_path = save_dir / "c_vs_perf" / "c_vs_perf_analysis.pkl"
    c_vs_perf_df = pd.read_pickle(perf_vs_C_path)

    plt.figure(figsize=(3.25, 2.25))
    g = sns.pointplot(
        data=c_vs_perf_df,
        x="ncopies",
        y="epoch_test_acc",
        ci=75,
        capsize=.15,
        ms=3,
        lw=1,
    )
    plt.grid()
    plt.xlabel(r"Number of copies at inference $C$")
    plt.ylabel("Test accuracy")
    ylabels = ['{:,.3f}'.format(x) for x in g.get_yticks()]
    g.set_yticklabels(ylabels)
    plt.savefig("figures/cifar10-c-vs-perf.pdf", pad_inches=0)

    # FIGURES B.7 AND B.8 - LEARNED AUGMENTATIONS

    augnet_path = save_dir / "augnet-c20-non-aff-aff-aff_resnet18_no_trans"

    fig = fetch_and_plot_weights_mags(augnet_path, seed=1, fig_width=6.5)
    fig.savefig("figures/cifar10-weight_mags_seed1.pdf")

    fig = fetch_and_plot_weights_mags(augnet_path, seed=29, fig_width=6.5)
    fig.savefig("figures/cifar10-weight_mags_seed29.pdf")

    # FIGURE B.9 - MODEL INVARIANCE

    invariance_evol_df = pd.read_pickle(
        save_dir / "invariance_study" / "invariance_evol.pkl"
    )

    for seed in [1, 29]:
        fig, ax = plt.subplots(figsize=(3.25, 2.25))

        sns.lineplot(
            data=invariance_evol_df.query(f"seed == {seed}"),
            x="epoch",
            y="f invariance",
            hue="transform",
            style="method",
            ci=80,
            estimator=np.median,
            marker="o",
            ax=ax,
        )

        # plt.ylim(0.8, 1.02)
        plt.ylim(0.55, 1.02)
        plt.grid()
        plt.legend(loc=0)
        plt.savefig(f"figures/cifar10-invariance-seed{seed}.pdf", pad_inches=0)
