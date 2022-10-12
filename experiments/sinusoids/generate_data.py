import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

import torch

from braindecode.augmentation.base import AugmentedDataLoader


def make_sinusoid(f, a=1, length_s=10, sfreq=100, phase=0):
    """Creates a sinusoidal signal

    Parameters
    ----------
    f : float
        Frequency of sinusoid.
    a : float, optional
        Amplitude, by default 1.
    length_s : float, optional
        Lenght of the signal, in seconds. By default 10.
    sfreq : float, optional
        Sampling frequency, by default 100.
    phase : float, optional
        Phase of the signal. Default = 0.

    Returns
    -------
    numpy.ndarray
        Sinusoid with desired properties.
    """
    t = np.arange(0, length_s, 1 / sfreq)
    return a * np.sin(2 * np.pi * f * t + phase)


def make_dataset(
    n_per_class,
    batch_size,
    train_size=2/3,
    delta_f=1,
    freqs=[2, 4, 6, 8],
    a=1,
    length_s=10,
    sfreq=100,
    noise=0.,
    transform=None,
    random_state=None,
    verbose=False,
):
    """Creates dataset of sinusoids made from generative waves whose frequency
    has been shifted randomly within a desired range

    Parameters
    ----------
    n_per_class : int
        Number of examples per class (generative wave).
    batch_size : int
        Size of batches.
    train_size : float, optional
        Fraction of data to use for training. By default 2/3.
    delta_f : float, optional
        Allowed magnitude to shift the generator frequency up and down.
        By default 1.
    freqs : list, optional
        List of frequencies of generator sinusoids (in Hz). Equal to the number
        of classes in the problam., by default [2, 4, 6, 8]
    a : float, optional
        Amplitude, by default 1.
    length_s : float, optional
        Lenght of the signal, in seconds. By default 10.
    sfreq : float, optional
        Sampling frequency, by default 100.
    noise : float, optional
        Std of the Gaussian distirbution from which additive noise is sampled.
        Defaults to 0.
    transform : braindecode.augmentation.Transform | None, optional
        Augmentation to apply when sampling training data. Default None.
    random_state : int | None, optional
        Random state to sample frequencies, by default None
    verbose : bool, optional
        Whether to report dataset built at the end or not.
        By default False.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        Training and test data loaders.
    """
    rng = check_random_state(random_state)

    # Populate dataset by sampling sinusoids using the generating frequencies
    waves = list()
    labels = list()
    for c, f in enumerate(freqs):
        for _ in range(n_per_class):
            df = rng.rand() * 2 * delta_f - delta_f
            phase = rng.rand() * 4 * np.pi * (1 / f)  # 2 periods
            wave = make_sinusoid(
                f + df,
                phase=phase,
                a=a,
                length_s=length_s,
                sfreq=sfreq
            )
            wave += noise * rng.normal(loc=0., scale=1., size=wave.shape)
            waves.append(wave)
            labels.append(c)

    # Consolidate
    waves = np.stack(waves)
    labels = np.array(labels)

    # Shuffle and turn into correctly shaped tensors
    ds_size = n_per_class * len(freqs)
    indices = np.arange(ds_size)
    rng.shuffle(indices)
    X = torch.as_tensor(waves[indices, :]).float().view(ds_size, 1, -1)
    y = torch.as_tensor(labels[indices])

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        random_state=rng,
        stratify=y
    )

    # Create TensorDatasets
    traindata = torch.utils.data.TensorDataset(X_train, y_train)
    testdata = torch.utils.data.TensorDataset(X_test, y_test)

    # And DataLoaders
    trainloader = AugmentedDataLoader(
        traindata, batch_size=batch_size, transforms=transform)
    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=batch_size)

    # Report
    if verbose:
        print("### Dataset ready ###")
        print("- Possible classes: ")
        for i, f in enumerate(freqs):
            print(f"\t{i}: {f} Hz")
        print(f"- Delta frequency: \t{delta_f} Hz")
        print(f"- Sampling frequency: \t{sfreq} Hz")
        print(f"- Inputs dimension: \t{X_train.shape[1]}")
        print(f"- Training set size: \t{X_train.shape[0]}")
        print(f"- Test set size: \t{X_test.shape[0]}")

    return trainloader, testloader


def plot_dataset_illustration(
    freqs=[2, 4, 6, 8],
    delta_f=1,
    length_s=1,
    sfreq=100,
    noise=0.5,
    lw=1.2,
    random_state=None,
    figsize=(6.5, 5),
):
    """ Plots extremal waves from each category, next to each generating wave.
    """
    plt.rc("text", usetex=False)
    plt.rc('font', size=7)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('savefig', bbox='tight')

    rng = check_random_state(random_state)

    t = np.arange(0, length_s, 1 / sfreq)
    n_classes = len(freqs)
    fig, ax_arr = plt.subplots(n_classes, 4, figsize=figsize)

    cpal = sns.color_palette("tab10")

    for i, f in enumerate(freqs):
        ax = ax_arr[i, 0]
        ax.plot(
            t,
            make_sinusoid(f, length_s=length_s, sfreq=sfreq),
            color=cpal[0],
            linewidth=lw
        )
        ax.set_ylabel(f"Class {i} ({f} Hz)")
        if i == 0:
            ax.set_title("Generator", fontsize=7)

        ax = ax_arr[i, 1]
        max_s = make_sinusoid(f + delta_f, length_s=length_s, sfreq=sfreq)
        ax.plot(
            t, max_s, label=f"f + {delta_f}Hz", color=cpal[2], linewidth=lw)
        min_s = make_sinusoid(f - delta_f, length_s=length_s, sfreq=sfreq)
        ax.plot(
            t, min_s, label=f"f - {delta_f}Hz", color=cpal[1], linewidth=lw)
        if i == 0:
            ax.set_title("Random frequency shift", fontsize=7)

        ax = ax_arr[i, 2]
        phase_max = rng.rand() * 4 * np.pi
        max_s = make_sinusoid(
            f + delta_f, length_s=length_s, sfreq=sfreq, phase=phase_max)
        ax.plot(
            t, max_s, label=f"f + {delta_f}Hz", color=cpal[2], linewidth=lw)
        phase_min = rng.rand() * 4 * np.pi
        min_s = make_sinusoid(
            f - delta_f, length_s=length_s, sfreq=sfreq, phase=phase_min)
        ax.plot(
            t, min_s, label=f"f - {delta_f}Hz", color=cpal[1], linewidth=lw)
        if i == 0:
            ax.set_title("+ Random phase", fontsize=7)

        ax = ax_arr[i, 3]
        max_s += noise * rng.randn(*max_s.shape)
        ax.plot(
            t, max_s, label=f"f + {delta_f}Hz", color=cpal[2], linewidth=lw)
        min_s += noise * rng.randn(*max_s.shape)
        ax.plot(
            t, min_s, label=f"f - {delta_f}Hz", color=cpal[1], linewidth=lw)
        if i == 0:
            ax.set_title("+ Noise", fontsize=7)

    plt.tight_layout()
    return fig
