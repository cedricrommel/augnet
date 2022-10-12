#!/usr/bin/env python
# coding: utf-8

# Note: This notebook requires to use eeg-augment package, at version
# `augnet-icml` (tag)

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import seaborn as sns
import matplotlib.pyplot as plt

from eeg_augment.plot import get_test_results_df


parser = argparse.ArgumentParser()
parser.add_argument(
    "-b", "--baselines-path",
    type=str,
    default="../../../results/mass_search_l2_p5_b16_24nights_es",
    help="Path where baselines' results should be fetched from."
)
args = parser.parse_args()

# FETCH AND PROCESS BASELINES RESULTS

central_path = Path(args.baselines_path)

to_drop = ["bad_runs", "mass_ref", "noaug-e20", "dada", "new-fraa"]
to_drop += ["gumbel-adda", "gumbel-s-adda-ws20"]
folders = [
    e for e in central_path.iterdir()
    if e.is_dir() and str(e.parts[-1]) not in to_drop
]
methods = [
    {
        "dataset": "mass",
        "method": "es_" + str(e.parts[-1]),
        "folder_path": e,
        "retrain_results_path": e / "search_perf_results.csv",
        "search_history_path": e / "search_history.csv",
    }
    for e in folders
]
methods = pd.DataFrame(methods)

all_results = dict()
for ds in methods["dataset"].unique():
    ds_results = list()
    for method in methods.query("dataset == @ds")["method"]:
        path = methods.query(
            "dataset == @ds and method == @method"
        )["retrain_results_path"]
        df = pd.read_csv(path.values[0]).drop('test_confusion_matrix', axis=1)
        df["method"] = method
        df["dataset"] = ds
        ds_results.append(df)
    all_results[ds] = pd.concat(ds_results, ignore_index=True)

# Creates best_valid_acc and best_test_acc columns for discrete methods
updated_results = list()
for fold in all_results["mass"]["fold"].unique():
    fold_mask = all_results["mass"]["fold"] == fold
    fold_bloc = all_results["mass"][fold_mask].reset_index(drop=True)
    # for method in discrete_methods:
    for method in fold_bloc["method"].unique():
        method_mask = fold_bloc["method"] == method
        method_bloc = fold_bloc[method_mask].reset_index(drop=True)
        method_bloc = method_bloc.sort_values("step_idx")
        method_bloc.loc[0, "best_test_bal_acc"] = method_bloc.loc[
            0, "test_bal_acc"]
        method_bloc.loc[0, "best_valid_bal_acc"] = method_bloc.loc[
            0, "valid_bal_acc"]
        for i in range(1, method_bloc.shape[0]):
            best_test = max(
                method_bloc.loc[i-1, "best_test_bal_acc"],
                method_bloc.loc[i, "test_bal_acc"]
            )
            method_bloc.loc[i, "best_test_bal_acc"] = best_test
            best_valid = max(
                method_bloc.loc[i-1, "best_valid_bal_acc"],
                method_bloc.loc[i, "valid_bal_acc"]
            )
            method_bloc.loc[i, "best_valid_bal_acc"] = best_valid
        updated_results.append(method_bloc)
all_results["mass"] = pd.concat(updated_results, axis=0, ignore_index=True)

# Creates test_bal_acc_based_on_valid column
updated_res = list()
for fold in all_results["mass"]["fold"].unique():
    list_by_method = list()
    fold_mask = all_results["mass"]["fold"] == fold
    fold_bloc = all_results["mass"][fold_mask].reset_index(drop=True)
    for method in all_results["mass"]["method"].unique():
        method_mask = fold_bloc["method"] == method
        method_bloc = fold_bloc[method_mask].reset_index(drop=True)
        curr_best_best_valid = method_bloc.loc[0, "best_valid_bal_acc"]
        method_bloc.loc[0, "test_bal_acc_based_on_valid"] = method_bloc.loc[
            0, "test_bal_acc"]
        for i in range(1, method_bloc.shape[0]):
            best_valid = method_bloc.loc[i, "best_valid_bal_acc"]
            if best_valid > curr_best_best_valid:
                curr_best_best_valid = best_valid
                method_bloc.loc[
                    i,
                    "test_bal_acc_based_on_valid"
                ] = method_bloc.loc[i, "test_bal_acc"]
            else:
                method_bloc.loc[
                    i,
                    "test_bal_acc_based_on_valid"
                ] = method_bloc.loc[i-1, "test_bal_acc_based_on_valid"]
        updated_res.append(method_bloc)
all_results["mass"] = pd.concat(updated_res, axis=0, ignore_index=True)

melted_results = dict()
for ds, res in all_results.items():
    melted_res = res.melt(
        id_vars=[
            'fold',
            'n_fold',
            'subset_ratio',
            'tot_trials',
            'tot_search_duration',
            'step_search_duration',
            'step_idx',
            'method',
            'dataset'
        ],
        value_vars=[
            'test_loss',
            'test_cohen_kappa_score',
            'test_bal_acc',
            'valid_loss',
            'valid_cohen_kappa_score',
            'valid_bal_acc',
            'train_loss',
            'train_cohen_kappa_score',
            'train_bal_acc',
            'best_test_bal_acc',
            # 'median_test_bal_acc',
            # 'median_valid_bal_acc',
            # 'best_median_test_bal_acc',
            # 'best_median_valid_bal_acc',
            'test_bal_acc_based_on_valid',
        ],
        var_name='metric'
    )
    melted_results[ds] = melted_res

mass_results = melted_results["mass"].reset_index(drop=True)


# Add warmstarting time to FAA

FAA_CORRECTION_s = 11.4*46
mass_results.loc[
    mass_results["method"].apply(lambda x: x.endswith("faa")),
    "tot_search_duration"
] += FAA_CORRECTION_s


# Add 20 epoch warmstarting time to other methods

WS20_CORRECTION_s = 11.4*20
for method in mass_results["method"].unique():
    if "ws" in method:
        mass_results.loc[
            mass_results["method"] == method,
            "tot_search_duration"
        ] += WS20_CORRECTION_s


def add_first_point(results_df, baseline_df):
    new_baseline_lines = list()

    n_folds = results_df.loc[0, "n_fold"]
    subset = results_df.loc[0, "subset_ratio"]

    for ds in results_df["dataset"].unique():
        for fold in results_df["fold"].unique():
            for method in results_df["method"].unique():
                for metric in results_df["metric"].unique():
                    relevant_baseline = baseline_df.query(
                        "fold == @fold and subset_ratio == @subset "
                        "and metric == @metric"
                    ).reset_index(drop=True)
                    new_baseline_lines.append({
                        "fold": fold,
                        "n_fold": n_folds,
                        "subset_ratio": subset,
                        "tot_trials": 0,
                        "tot_search_duration": 0,
                        "step_search_duration": 0,
                        "step_idx": -1,
                        "method": method,
                        "dataset": ds,
                        "metric": metric,
                        "value": relevant_baseline.loc[0, "value"],
                    })

    return pd.concat(
        [results_df, pd.DataFrame(new_baseline_lines)],
        ignore_index=True
    )


def make_res_relative(results_df, metric_of_interest="test_bal_acc", ref=0.0):
    results_df = results_df.query(
        "metric == @metric_of_interest"
    ).reset_index(drop=True)
    rel_results = results_df.copy()
    subset = results_df.loc[0, "subset_ratio"]

    for ds in results_df["dataset"].unique():
        for fold in results_df["fold"].unique():
            for method in results_df["method"].unique():
                block_mask = np.logical_and(
                    np.logical_and(
                        rel_results["fold"] == fold,
                        rel_results["subset_ratio"] == subset
                    ), rel_results["method"] == method
                )
                ref_mask = np.logical_and(
                    block_mask,
                    rel_results["tot_search_duration_h"] == 0.0
                )
                ref = rel_results.loc[ref_mask, "value"].values[0]
                rel_results.loc[block_mask, "value"] -= ref
                rel_results.loc[block_mask, "value"] /= ref
    return rel_results


# Add first point, i.e. the no_aug baseline results
def adapt_baseline(mass_baseline):
    to_concat = list()
    for ds in ["test", "valid"]:
        # Build best_test_bal_acc and best_valid_bal_acc
        best_test_ref = mass_baseline.query(
            f"metric == '{ds}_bal_acc'").reset_index(drop=True)
        best_test_ref["metric"] = f"best_{ds}_bal_acc"

        # Build median and best median test ...
        median = mass_baseline.query("metric == @test_acc")["value"].median()
        median_test_ref = best_test_ref.copy()
        median_test_ref["metric"] = f"median_{ds}_bal_acc"
        median_test_ref["value"] = median

        best_median = median_test_ref.copy()
        best_median["metric"] = f"best_median_{ds}_bal_acc"
        to_concat += [best_test_ref, median_test_ref, best_median]

    test_based_valid_ref = best_test_ref.copy()
    test_based_valid_ref["metric"] = "test_bal_acc_based_on_valid"

    return pd.concat(
        [mass_baseline, test_based_valid_ref, *to_concat],
        ignore_index=True
    )


mass_baseline_es = get_test_results_df(central_path / "mass_ref")
mass_baseline_es = adapt_baseline(mass_baseline_es)
new_mass_results = add_first_point(mass_results, mass_baseline_es)


# Check the average pretraining training time
tr_dur = list()
for i in range(1, 6):
    hist = pd.read_json(
        central_path / "mass_ref" / "no-aug" /
        f"0.5-None/fold{i}of5/subset_1.0_samples/history.json"
    )
    tr_dur.append(hist["dur"].mean())

# Convert times from seconds to hours
new_mass_results["tot_search_duration_h"] = (
    new_mass_results["tot_search_duration"] / 3600
)


# Remove times above 12h
new_mass_results = new_mass_results[
    new_mass_results["tot_search_duration_h"] < 12
]


# This is the average re-training time (in hours) with 1 subpolicy
# (equiv to 5 subpolicies)
retraining_offset_h = (new_mass_results.query(
    "method == 'es_randomsearch' and step_search_duration > 0"
)["step_search_duration"] / 3600 / 5).mean()

methods_requiring_retraining = [
    "es_fraa",
    "es_new-relax-adda-ws-5e4",
    "es_new-gumbel-adda-ws-5e4",
    "es_new-adda-ws-5e4",
    "es_randomsearch",
    "es_faa",
    "es_tpe"
]

new_mass_results[
    "tot_search_and_retrain_duration_h"
] = new_mass_results["tot_search_duration_h"]

for method in methods_requiring_retraining:
    new_mass_results.loc[
        new_mass_results["method"] == method,
        "tot_search_and_retrain_duration_h"
    ] += retraining_offset_h


# FETCH AUGNET RESULTS

def create_earlystopped_test_acc(df):
    best_valid = df.loc[0, "valid_bal_acc"]
    df.loc[0, "test_bal_acc_based_on_valid"] = df.loc[0, "test_bal_acc"]
    for i in range(1, df.shape[0]):
        valid = df.loc[i, "valid_bal_acc"]
        if valid > best_valid:
            best_valid = valid
            df.loc[i, "test_bal_acc_based_on_valid"] = df.loc[
                i,
                "test_bal_acc"
            ]
        else:
            df.loc[i, "test_bal_acc_based_on_valid"] = df.loc[
                i-1,
                "test_bal_acc_based_on_valid"
            ]
    return df


def aggregate_mass_augnet_results(exp_path, max_epochs=300):
    if isinstance(exp_path, str):
        exp_path = Path(exp_path)

    dur_sum = np.zeros(max_epochs)
    epoch_counts = np.zeros(max_epochs)

    results_per_fold = list()
    for folder in exp_path.iterdir():
        if folder.is_dir():
            # extract fold index from folder name
            fold = int(str(folder.parts[-1]).replace("fold", "")[0])

            # load history and extract training time and epochs
            fold_history = pd.read_json(
                folder / "subset_1.0_samples" / "history.json"
            ).loc[:, ["epoch", "dur"]]
            epochs = fold_history["epoch"].unique()

            # load test results
            fold_res_test = pd.read_pickle(
                folder / "subset_1.0_samples" /
                "augnet_test_metrics_per_epoch.pkl"
            ).query("epoch in @epochs").loc[:, "test_bal_acc"]

            # load valid results and correct column name
            fold_res_valid = pd.read_pickle(
                folder / "subset_1.0_samples" /
                "augnet_valid_metrics_per_epoch.pkl"
            ).query("epoch in @epochs").loc[:, "test_bal_acc"]

            # concatenate them
            fold_results = pd.concat(
                [fold_history, fold_res_test, fold_res_valid],
                axis=1,
            )
            fold_results.columns = [
                "epoch", "dur", "test_bal_acc", "valid_bal_acc"
            ]
            fold_results["fold"] = fold

            # create test_bal_acc_based_on_valid
            fold_results = create_earlystopped_test_acc(fold_results)

            # sum epoch durations and increment counters
            num_epochs = max(epochs)
            dur_sum[:num_epochs] += fold_results["dur"].values
            epoch_counts[:num_epochs] += 1

            # store for later
            results_per_fold.append(fold_results)

    # compute average epoch durations
    real_max_epoch = sum(epoch_counts > 0)
    ave_epoch_dur = dur_sum[:real_max_epoch] / epoch_counts[:real_max_epoch]

    # compute cumul_dur_h
    cumul_dur_h = np.cumsum(ave_epoch_dur) / 3600

    # replace dur and add cumul_dur_h in results of all folders
    for i in range(len(results_per_fold)):
        fold_res = results_per_fold[i]
        fold_n_epochs = fold_res.shape[0]
        fold_res = pd.concat(
            [fold_res] + [fold_res.tail(1)] * (real_max_epoch - fold_n_epochs),
            ignore_index=True,
        )
        fold_res["dur"] = ave_epoch_dur
        fold_res["cumul_dur_h"] = cumul_dur_h
        results_per_fold[i] = fold_res

    # concatenate and return
    return pd.concat(results_per_fold, axis=0, ignore_index=True)


# With C=4 (not in paper)

# augnet_results_c4 = pd.read_pickle(
#     "../../results/augnet/mass_results/preprocessed_results.pkl"
# )
# reshaped_augnet_results = augnet_results_c4.groupby(
#     ["n_layers", "cumul_dur_h"]
# )["test_bal_acc_based_on_valid"].quantile([.25, .5, .75]).unstack()
# augnet_3l_results = reshaped_augnet_results.loc[3]
# augnet_3l_results = augnet_3l_results.loc[:augnet_3l_results.index[-2], :]

# C=20 (from paper)

augnet_path = Path("saved-outputs/results/")
augnet_results_c20 = aggregate_mass_augnet_results(augnet_path, max_epochs=300)

augnet_3l_results_c20 = augnet_results_c20.groupby(
    ["cumul_dur_h"]
)["test_bal_acc_based_on_valid"].quantile([.25, .5, .75]).unstack()

# Plotting setup

plt.rc("text", usetex=False)
plt.rc('font', size=7)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('savefig', bbox='tight')

# Prepare data for plotting

metric_of_interest = "test_bal_acc_based_on_valid"
methods_to_plot = [
    "es_tpe",
    "es_faa",
    "es_fraa",
    "es_new-dada",
    "es_new-relax-adda-ws-5e4"
]

colors_baselines = sns.color_palette("tab10")
colors_cadda = sns.color_palette("crest")
adda_variants_colors = sns.cubehelix_palette(
    n_colors=6,
    start=2,
    rot=0,
    dark=0.2,
    light=.8,
    reverse=True
)

HUE_MAP = {
    "ADDA": colors_baselines[2],
    "DADA": colors_baselines[0],
    "Faster AA": colors_baselines[1],
    "AutoAugment": colors_baselines[-5],
    "AA": colors_baselines[-5],
    "Fast AA": colors_baselines[-4],
    "RandomSearch": colors_baselines[-2],
    "AugNet": colors_baselines[3],
    "AugNet*": colors_baselines[3],
}

labels = {
    "fraa": "Faster AA",
    "new-dada": "DADA",
    "new-relax-adda-ws-5e4": "ADDA",
    "tpe": "AutoAugment",
    "faa": "Fast AA",
    "randomsearch": "RandomSearch",
}
labels = {"es_" + k: v for k, v in labels.items()}

ordered_res = pd.concat([
    new_mass_results.query(
        "metric == @metric_of_interest and method == @method"
    )
    for method in methods_to_plot
])

figures_dir = Path("./figures")
figures_dir.mkdir(exist_ok=True)

# FIGURE 6.b - GPU TIME VS PERFORMANCE PLOT

fig, ax1 = plt.subplots(1, 1, figsize=(3.25, 2.5), sharey=True)
sns.lineplot(
    data=ordered_res.replace(labels),
    x="tot_search_and_retrain_duration_h",
    y="value",
    hue="method",
    palette=HUE_MAP,
    style="method",
    estimator=np.median,
    lw=1.5,
    markers=True,
    dashes=False,
    ms=5,
    ci=None,
    ax=ax1,
)

ax1.plot(
    augnet_3l_results_c20.index,
    augnet_3l_results_c20[0.5],
    color=colors_baselines[3],
    label="AugNet*"
)

ax1.grid()
ax1.set_xlabel("GPU hours", fontsize=7)
ax1.set_ylabel("Test balanced accuracy", fontsize=7)
ax1.legend(loc="lower right", fontsize=6)
ax1.set_ylim(0.8215, 0.838)

plt.savefig(
    figures_dir / "benchmark_mass_noshades.pdf",
    pad_inches=0
)

# FIGURE B.10 - SHADED GPU TIME VS PERFORMANCE PLOT

fig2, ax2 = plt.subplots(1, 1, figsize=(3.25, 2.5), sharey=True)

sns.lineplot(
    data=ordered_res.replace(labels),
    x="tot_search_and_retrain_duration_h",
    y="value",
    hue="method",
    palette=HUE_MAP,
    style="method",
    estimator=np.median,
    lw=1.5,
    markers=True,
    dashes=False,
    ms=5,
    ci=75,
    ax=ax2,
)

ax2.plot(
    augnet_3l_results_c20.index,
    augnet_3l_results_c20[0.5],
    color=colors_baselines[3],
    label="AugNet*"
)
ax2.fill_between(
    augnet_3l_results_c20.index,
    augnet_3l_results_c20[0.25],
    augnet_3l_results_c20[0.75],
    color=colors_baselines[3],
    alpha=0.3,
)

ax2.grid()
ax2.set_xlabel("GPU hours", fontsize=7)
ax2.set_ylabel("Test balanced accuracy", fontsize=7)
ax2.set_ylim(0.8215, 0.838)
ax2.legend(loc="lower right", fontsize=6)

plt.vlines(
    2,
    *ax1.get_ylim(),
    color="k",
    linestyle="--",
)
plt.text(2.5, 0.823, "2h budget")

plt.savefig(
    figures_dir / "benchmark_mass_with_shades.pdf",
    pad_inches=0
)

# FIGURE B.11 - FOLD-WISE IMPROVEMENTS WRT ADDA WITH 2H BUDGET

# Interpolate scores at common grid of training times
interpolated_results = list()
for method in methods_to_plot:
    for fold in ordered_res["fold"].unique():
        method_res_in_fold = ordered_res.query(
            "fold == @fold and method == @method"
        ).reset_index(drop=True).copy()
        measured_times = method_res_in_fold[
            "tot_search_and_retrain_duration_h"].values
        interp_func = interp1d(measured_times, method_res_in_fold["value"])
        new_times = np.hstack([
            measured_times.min(),
            np.arange(2, int(measured_times.max()) + 1, step=1)
        ])
        interpolated_res = pd.DataFrame({
            "tot_search_and_retrain_duration_h": new_times,
            "value": interp_func(new_times),
        })
        interpolated_res["fold"] = fold
        interpolated_res["method"] = method
        interpolated_res["metric"] = "test_bal_acc_based_on_valid"
        interpolated_results.append(interpolated_res)

for fold in augnet_results_c20["fold"].unique():
    method_res_in_fold = augnet_results_c20.query(
        "fold == @fold"
    ).reset_index(drop=True).copy()
    measured_times = method_res_in_fold["cumul_dur_h"].values
    interp_func = interp1d(
        measured_times,
        method_res_in_fold["test_bal_acc_based_on_valid"]
    )
    new_times = np.hstack([
        measured_times.min(),
        np.arange(2, int(measured_times.max()) + 1, step=1)
    ])
    interpolated_res = pd.DataFrame({
        "tot_search_and_retrain_duration_h": new_times,
        "value": interp_func(new_times),
    })
    interpolated_res["fold"] = fold
    interpolated_res["method"] = "AugNet*"
    interpolated_res["metric"] = "test_bal_acc_based_on_valid"
    interpolated_results.append(interpolated_res)

interpolated_results = pd.concat(interpolated_results, ignore_index=True)

# Compute the fold-wise relative improvements wrt to reference method (ADDA)
reference_method = "es_new-relax-adda-ws-5e4"

interpolated_results_w_ref = list()
for method in interpolated_results["method"].unique():
    for fold in interpolated_results["fold"].unique():
        ref_res_in_fold = interpolated_results.query(
            "fold == @fold and method == @reference_method"
        ).reset_index(drop=True).copy()
        min_time = ref_res_in_fold["tot_search_and_retrain_duration_h"].min()
        method_res_in_fold = interpolated_results.query(
            "fold == @fold and method == @method "
            "and tot_search_and_retrain_duration_h >= @min_time"
        ).reset_index(drop=True).copy()
        method_res_in_fold["relative_improvement"] = (
            method_res_in_fold["value"] - ref_res_in_fold["value"]
        ) * 100 / ref_res_in_fold["value"]
        interpolated_results_w_ref.append(method_res_in_fold)
interpolated_results_w_ref = pd.concat(
    interpolated_results_w_ref,
    ignore_index=True
)

# Fetch results at 2h of training
labels["es_tpe"] = "AA"
budget2h_relative_results = interpolated_results_w_ref.replace(labels).query(
    "tot_search_and_retrain_duration_h == 2.0 and method != 'ADDA'"
)

fig3, ax3 = plt.subplots(1, 1, figsize=(3.25, 2.5), sharey=True)
sns.boxplot(
    data=budget2h_relative_results,
    x="method",
    y="relative_improvement",
    palette=HUE_MAP,
    ax=ax3,
)

ax3.grid()
ax3.set_xlabel("")
ax3.set_ylabel(
    "Test balanced accuracy improvement\nover ADDA with 2h budget (%)",
    fontsize=7
)
ax3.hlines(0., *ax3.get_xlim(), linestyle="--", lw=2., color=HUE_MAP["ADDA"])
ax3.text(3.1, -0.4, "ADDA (ref.)", color=HUE_MAP["ADDA"])
plt.savefig(
    figures_dir / "benchmark_mass_foldwise_rel_adda_rebuttal_2hbudget.pdf",
    pad_inches=0
)
