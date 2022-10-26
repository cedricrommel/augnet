""" Differentiable augmentations for EEG tasks
"""

from typing import Optional
from numbers import Real
from collections.abc import Iterable

import numpy as np
from sklearn.utils import check_random_state
import torch
from torch import Tensor, nn
from torch.distributions import RelaxedBernoulli, Bernoulli

from braindecode.augmentation.base import Output, Transform
from braindecode.augmentation.functional import time_reverse
from braindecode.augmentation.functional import sign_flip
from braindecode.augmentation.functional import ft_surrogate
from braindecode.augmentation.functional import channels_dropout
from braindecode.augmentation.functional import gaussian_noise
from braindecode.augmentation.functional import channels_permute
from braindecode.augmentation.functional import smooth_time_mask
from braindecode.augmentation.functional import frequency_shift
from braindecode.augmentation.functional import sensors_rotation
from braindecode.augmentation.transforms import _get_standard_10_20_positions

from .diff_functional import diff_channels_shuffle


class DiffTransform(Transform):
    """ Basic class used for implementing differentiable data augmentations
    (where probability and magnitude can be learned with gradient descent)

    As proposed in [1]_

    Code copied and modified from https://github.com/moskomule/dda/

    Parameters
    ----------
    initial_probability : float | None, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. When set to None,
        the initial probability will be drawn from a uniform distribution.
        Set to None by default.
    initial_magnitude : float | None, optional
        Initial value for the magnitude. Defines the strength of the
        transformation applied between 0 and 1 and depends on the nature of
        the transformation and on its range. Some transformations don't
        have any magnitude. It can be equivalent to another argument of
        object with more meaning. In case both are passed, magnitude will
        override the latter. Defaults to None (uniformly sampled between 0
        and 1).
    mag_range : tuple of two floats | None, optional
        Valid range of the argument mapped by `magnitude` (e.g. standard
        deviation, number of sample, etc.):
        ```
        argument = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0].
        ```
        If `magnitude` is None it is ignored. Defaults to None.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults
        to 0.05.
    random_state: int, optional
        Seed to be used to instatiate numpy random number generator
        instance. Used to decide whether or not to transform given the
        probability argument. Also used for initializing the magnitudes and
        probabilities (independently of the forward) Defaults to None.
    **kwargs:
        Keyword arguments to be passed to operation.

    References
    ----------
    .. [1] Hataya R., Zdenek J., Yoshizoe K., Nakayama H. (2020) Faster
    AutoAugment: Learning Augmentation Strategies Using Backpropagation.
    In: Vedaldi A., Bischof H., Brox T., Frahm JM. (eds) Computer Vision –
    ECCV 2020. ECCV 2020. Lecture Notes in Computer Science, vol 12370.
    Springer, Cham. https://doi.org/10.1007/978-3-030-58595-2_1
    """
    operation = None

    def __init__(
        self,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=None,
        temperature=0.05,
        random_state=None,
        **kwargs
    ):

        # Important to separate the init RNG from the forward one, so that
        # one can init a new policy and load a checkpoint without modifying
        # the forward rng state
        self.init_rng = check_random_state(random_state)
        if initial_probability is None:
            initial_probability = self.init_rng.uniform()
        if initial_magnitude is None:
            initial_magnitude = self.init_rng.uniform()
        self.initial_probability = initial_probability
        self.initial_magnitude = initial_magnitude

        # Standard Transform init (not parametrized by magnitude yet)
        super().__init__(
            probability=initial_probability,
            random_state=random_state,
            **kwargs,
        )

        # Make probability and magnitude into learnable parameters
        self._probability = nn.Parameter(
            torch.empty(1).fill_(initial_probability)
        )
        self.mag_range = mag_range
        if self.mag_range:
            self._magnitude = nn.Parameter(
                torch.empty(1).fill_(initial_magnitude)
            )
        else:
            self._magnitude = None

        # Save useful parameters
        self.temperature = temperature

    def forward(self, X: Tensor, y: Tensor = None) -> Output:
        """General forward pass for an differentiable augmentation
        transform.

        Parameters
        ----------
        X : torch.Tensor
            EEG input example or batch.
        y : torch.Tensor | None
            EEG labels for the example or batch. Defaults to None.

        Returns
        -------
        torch.Tensor
            Transformed inputs.
        torch.Tensor, optional
            Transformed labels. Only returned when y is not None.
        """
        if self.training:
            X = torch.as_tensor(X).float()

            # Default outputs, used when mask is False everywhere
            if y is not None:
                y = torch.as_tensor(y)
                return_y = True
            else:  # For the case when inputs don't include y
                y = torch.zeros(X.shape[0])
                return_y = False

            # Samples a mask setting for each example whether they should
            # stay inchanged or not
            mask = self._get_mask(X.shape[0]).to(X.device)

            # Transform whole batch
            tr_X, tr_y = self.operation(
                X, y,
                **self.get_augmentation_params(X, y)
            )
            # Create the convex combination
            out_X = mask * tr_X + (1 - mask) * X

            # Return (X, y) or only X according to the input length
            if return_y:
                out_y = mask.view(-1) * tr_y + (1 - mask.view(-1)) * y
                return out_X, out_y
            else:
                return out_X
        else:
            # At inference, the regular forward from Transform can be used
            return super().forward(X, y)

    def _map_magnitude(self, X):
        if self.mag_range is None:
            return
        lb, ub = self.mag_range
        return (self.magnitude * ub + (1 - self.magnitude) * lb).to(X.device)

    def get_augmentation_params(self, X, y):
        return super().get_augmentation_params(X, y)

    def _get_mask(self, batch_size=None) -> torch.Tensor:
        if self.training:
            size = (batch_size, 1)
            mask = RelaxedBernoulli(
                temperature=self.temperature, probs=self.probability
            ).rsample(size)
            return mask
        else:
            size = (batch_size,)
            return Bernoulli(
                probs=self.probability).sample(size).squeeze().bool()

    @property
    def probability(self) -> torch.Tensor:
        return self._probability.clamp(0, 1)

    @property
    def magnitude(self) -> Optional[torch.Tensor]:
        if self._magnitude is None:
            return None
        return self._magnitude.clamp(0, 1)


class DiffTimeReverse(DiffTransform):
    """ Flip the time axis of each feature sample with a given probability

    Parameters
    ----------
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : object, optional
        Always ignored, exists for compatibility.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """
    operation = staticmethod(time_reverse)

    def __init__(
        self,
        initial_probability=None,
        initial_magnitude=None,
        temperature=0.05,
        random_state=None
    ):
        super().__init__(
            initial_probability=initial_probability,
            temperature=temperature,
            random_state=random_state
        )


class DiffSignFlip(DiffTransform):
    """ Flip the sign axis of each feature sample with a given probability

    Parameters
    ----------
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : object, optional
        Always ignored, exists for compatibility.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """

    operation = staticmethod(sign_flip)

    def __init__(
        self,
        initial_probability=None,
        initial_magnitude=None,
        temperature=0.05,
        random_state=None
    ):
        super().__init__(
            initial_probability=initial_probability,
            temperature=temperature,
            random_state=random_state
        )


class DiffFTSurrogate(DiffTransform):
    """ FT surrogate augmentation of a single EEG channel, as proposed in [1]_

    Parameters
    ----------
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : float | None, optional
        Initial value for the magnitude. Defines the strength of the
        transformation applied between 0 and 1 and depends on the nature of the
        transformation and on its range. Some transformations don't have any
        magnitude (=None). It can be equivalent to another argument of object
        with more meaning. In case both are passed, magnitude will override the
        latter. Defaults to 0.5.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.

    References
    ----------
    .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
       Clifford, G. D. (2018). Addressing Class Imbalance in Classification
       Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
       preprint arXiv:1806.08675.
    """
    operation = staticmethod(ft_surrogate)

    def __init__(
        self,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=(0, 1),
        temperature=0.05,
        channel_indep=False,
        random_state=None
    ):
        super().__init__(
            initial_probability=initial_probability,
            initial_magnitude=initial_magnitude,
            mag_range=mag_range,
            temperature=temperature,
            random_state=random_state
        )
        self.channel_indep = channel_indep

    def get_augmentation_params(self, X, y):
        return {
            "phase_noise_magnitude": self._map_magnitude(X),
            "channel_indep": self.channel_indep,
            "random_state": self.rng,
        }


class DiffChannelsDropout(DiffTransform):
    """ Randomly set channels to flat signal

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : float | None, optional
        Initial value for the magnitude. Defines the strength of the
        transformation applied between 0 and 1 and depends on the nature of the
        transformation and on its range. Some transformations don't have any
        magnitude (=None). It can be equivalent to another argument of object
        with more meaning. In case both are passed, magnitude will override the
        latter. Defaults to 0.5.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument and to sample channels to erase. Defaults to None.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """
    operation = staticmethod(channels_dropout)

    def __init__(
        self,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=(0, 1),
        temperature=0.05,
        random_state=None
    ):
        super().__init__(
            initial_probability=initial_probability,
            initial_magnitude=initial_magnitude,
            mag_range=mag_range,
            temperature=temperature,
            random_state=random_state
        )

    def get_augmentation_params(self, X, y):
        return {
            "p_drop": self._map_magnitude(X),
            "random_state": self.rng,
        }


class DiffChannelsShuffle(DiffTransform):
    """ Randomly shuffle channels in EEG data matrix

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : float | None, optional
        Initial value for the magnitude. Defines the strength of the
        transformation applied between 0 and 1 and depends on the nature of the
        transformation and on its range. Some transformations don't have any
        magnitude (=None). It can be equivalent to another argument of object
        with more meaning. In case both are passed, magnitude will override the
        latter. Defaults to 0.5.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """
    operation = staticmethod(diff_channels_shuffle)

    def __init__(
        self,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=(0, 1),
        temperature=0.05,
        random_state=None
    ):
        super().__init__(
            initial_probability=initial_probability,
            initial_magnitude=initial_magnitude,
            mag_range=mag_range,
            temperature=temperature,
            random_state=random_state
        )

    def get_augmentation_params(self, X, y):
        return {
            "p_shuffle": self._map_magnitude(X),
            "random_state": self.rng,
        }


class DiffGaussianNoise(DiffTransform):
    """Randomly add white noise to all channels

    Suggested e.g. in [1]_, [2]_ and [3]_

    Parameters
    ----------
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : float | None, optional
        Initial value for magnitude. Float between 0 and 1 encoding the
        standard deviation to use for the additive noise:
        ```
        std = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to 0.5.
    mag_range : tuple of two floats | None, optional
        Std range when set using the magnitude (see `magnitude`).
        If omitted, the range (0, 0.2) will be used.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Wang, F., Zhong, S. H., Peng, J., Jiang, J., & Liu, Y. (2018). Data
       augmentation for eeg-based emotion recognition with deep convolutional
       neural networks. In International Conference on Multimedia Modeling
       (pp. 82-93).
    .. [2] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [3] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.

    """
    operation = staticmethod(gaussian_noise)

    def __init__(
        self,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=(0, 0.2),
        random_state=None
    ):
        super().__init__(
            initial_probability=initial_probability,
            initial_magnitude=initial_magnitude,
            mag_range=mag_range,
            random_state=random_state,
        )

    def get_augmentation_params(self, X, y):
        return {
            "std": self._map_magnitude(X),
            "random_state": self.rng,
        }


class DiffChannelsSymmetry(DiffTransform):
    """Permute EEG channels inverting left and right-side sensors

    Suggested e.g. in [1]_

    Parameters
    ----------
    ordered_ch_names : list
        Ordered list of strings containing the names (in 10-20
        nomenclature) of the EEG channels that will be transformed. The
        first name should correspond the data in the first row of X, the
        second name in the second row and so on.
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : object, optional
        Always ignored, exists for compatibility.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.

    References
    ----------
    .. [1] Deiss, O., Biswal, S., Jin, J., Sun, H., Westover, M. B., & Sun, J.
       (2018). HAMLET: interpretable human and machine co-learning technique.
       arXiv preprint arXiv:1803.09702.

    """

    operation = staticmethod(channels_permute)

    def __init__(
        self,
        ordered_ch_names,
        initial_probability=None,
        initial_magnitude=None,
        temperature=0.05,
        random_state=None
    ):
        assert (
            isinstance(ordered_ch_names, Iterable) and
            all(isinstance(ch, str) for ch in ordered_ch_names)
        ), "ordered_ch_names should be a list of str."
        self.ordered_ch_names = ordered_ch_names

        permutation = list()
        for idx, ch_name in enumerate(ordered_ch_names):
            new_position = idx
            # Find digits in channel name (assuming 10-20 system)
            d = ''.join(list(filter(str.isdigit, ch_name)))
            if len(d) > 0:
                d = int(d)
                if d % 2 == 0:  # pair/right electrodes
                    sym = d - 1
                else:  # odd/left electrodes
                    sym = d + 1
                new_channel = ch_name.replace(str(d), str(sym))
                if new_channel in ordered_ch_names:
                    new_position = ordered_ch_names.index(new_channel)
            permutation.append(new_position)

        self.permutation = permutation

        super().__init__(
            initial_probability=initial_probability,
            random_state=random_state,
        )

    def get_augmentation_params(self, X, y):
        return {"permutation": self.permutation}


class DiffTimeMask(DiffTransform):
    """Replace part of all channels by zeros

    Suggested e.g. in [1]_ and [2]_
    Similar to the time variant of SpecAugment for speech signals [3]_

    Parameters
    ----------
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : float, optional
        Initial value for the magnitude. Float between 0 and 1 encoding the
        number of consecutive samples within `mag_range` to set to 0:
        ```
        mask_len_samples = int(round(magnitude * mag_range[1] +
            (1 - magnitude) * mag_range[0]))
        ```
        Defaults to 0.5.
    mag_range : tuple of two floats | None, optional
        Range of possible values for `mask_len_samples` settable using the
        magnitude (see `magnitude`). If omitted, the range (0, 100) samples
        will be used.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [2] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.
    .. [3] Park, D.S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E.D.,
       Le, Q.V. (2019) SpecAugment: A Simple Data Augmentation Method for
       Automatic Speech Recognition. Proc. Interspeech 2019, 2613-2617

    """
    operation = staticmethod(smooth_time_mask)

    def __init__(
        self,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=(0, 100),
        temperature=0.05,
        random_state=None
    ):
        super().__init__(
            initial_probability=initial_probability,
            initial_magnitude=initial_magnitude,
            mag_range=mag_range,
            temperature=temperature,
            random_state=random_state,
        )

    def get_augmentation_params(self, X, y):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains two elements:
            - a tensor (mask_start_per_sample) of integers representing
            the position to start masking the signal (has hence the same size
            as the first dimension of X, i.e. one start position per example in
            the batch).
            - the number of consecutive samples to zero out (mask_len_samples).
        """
        seq_length = torch.as_tensor(X.shape[-1], device=X.device)
        mask_len_samples = self._map_magnitude(X)
        mask_start = torch.as_tensor(self.rng.uniform(
            low=0, high=1, size=X.shape[0],
        ), device=X.device) * (seq_length - mask_len_samples)
        return {
            "mask_start_per_sample": mask_start,
            "mask_len_samples": mask_len_samples,
        }


class DiffFrequencyShift(DiffTransform):
    """Add a random shift in the frequency domain to all channels.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the signals to be transformed.
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : float, optional
        Initial  value for the magnitude. Float between 0 and 1 encoding the
        `max_shift` parameter:
        ```
        max_shift = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Random frequency shifts will be samples uniformly in the interval
        `[0, max_shift]`. Defaults to 0.5.
    mag_range : tuple of two floats | None, optional
        Range of possible values for `max_shift` settable using the magnitude
        (see `magnitude`). If omitted the range (0 Hz, 5 Hz) will be used.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.
    """
    operation = staticmethod(frequency_shift)

    def __init__(
        self,
        sfreq,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=(0, 5),
        temperature=0.05,
        random_state=None,
    ):
        assert isinstance(sfreq, Real) and sfreq > 0,\
            "sfreq should be a positive float."
        self.sfreq = sfreq

        super().__init__(
            initial_probability=initial_probability,
            initial_magnitude=initial_magnitude,
            mag_range=mag_range,
            temperature=temperature,
            random_state=random_state,
        )

    def get_augmentation_params(self, X, y):
        u = torch.as_tensor(
            self.rng.uniform(size=X.shape[0]),
            device=X.device
        )
        max_delta_freq = self._map_magnitude(X)
        if isinstance(max_delta_freq, torch.Tensor):
            max_delta_freq = max_delta_freq.to(X.device)
        delta_freq = u * 2 * max_delta_freq - max_delta_freq
        return {
            "delta_freq": delta_freq,
            "sfreq": self.sfreq,
        }


class DiffSensorsRotation(DiffTransform):
    """Interpolates EEG signals over sensors rotated around the desired axis
    with an angle sampled uniformly between 0 and `max_degree`.

    Suggested in [1]_

    Parameters
    ----------
    sensors_positions_matrix : numpy.ndarray
        Matrix giving the positions of each sensor in a 3D cartesian coordiante
        systemsof. Should have shape (3, n_channels), where n_channels is the
        number of channels. Standard 10-20 positions can be obtained from
        `mne` through:
        ```
        ten_twenty_montage = mne.channels.make_standard_montage(
            'standard_1020'
        ).get_positions()['ch_pos']
        ```
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : float, optional
        Initial value for the magnitude. Float between 0 and 1 encoding the
        `max_degree` parameter:
        ```
        max_degree = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to 0.5.
    mag_range : tuple of two floats | None, optional
        Range of possible values for `max_degree` settable using the magnitude
        (see `magnitude`). If omitted, the range (0, 30 degrees) will be used.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    axis : 'x' | 'y' | 'z', optional
        Axis around which to rotate. Defaults to 'z'.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        `False`, standard scipy.interpolate.Rbf (with quadratic kernel) will be
        used (as in the original paper). Defaults to True.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Krell, M. M., & Kim, S. K. (2017). Rotational data augmentation for
       electroencephalographic data. In 2017 39th Annual International
       Conference of the IEEE Engineering in Medicine and Biology Society
       (EMBC) (pp. 471-474).
    """
    operation = staticmethod(sensors_rotation)

    def __init__(
        self,
        sensors_positions_matrix,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=(0, 30),
        temperature=0.05,
        axis='z',
        spherical_splines=True,
        random_state=None
    ):
        if isinstance(sensors_positions_matrix, (np.ndarray, Iterable)):
            sensors_positions_matrix = torch.as_tensor(
                sensors_positions_matrix
            )
        assert isinstance(sensors_positions_matrix, torch.Tensor),\
            "sensors_positions should be an Tensor"
        assert isinstance(axis, str) and axis in ['x', 'y', 'z'],\
            "axis can be either x, y or z."
        assert sensors_positions_matrix.shape[0] == 3,\
            "sensors_positions_matrix shape should be 3 x n_channels."
        assert isinstance(spherical_splines, bool),\
            "spherical_splines should be a boolean"
        self.sensors_positions_matrix = sensors_positions_matrix
        self.axis = axis
        self.spherical_splines = spherical_splines

        super().__init__(
            initial_probability=initial_probability,
            initial_magnitude=initial_magnitude,
            mag_range=mag_range,
            random_state=random_state
        )

    def get_augmentation_params(self, X, y):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        sensors_positions_matrix : numpy.ndarray
            Matrix giving the positions of each sensor in a 3D cartesian
            coordinate system. Should have shape (3, n_channels), where
            n_channels is the number of channels.
        axis : 'x' | 'y' | 'z'
            Axis around which to rotate.
        angles : array-like
            Array of float of shape ``(batch_size,)`` containing the rotation
            angles (in degrees) for each element of the input batch, sampled
            uniformly between ``-max_degrees``and ``max_degrees``.
        spherical_splines : bool
            Whether to use spherical splines for the interpolation or not. When
            `False`, standard scipy.interpolate.Rbf (with quadratic kernel)
            will be used (as in the original paper).
        """
        u = self.rng.uniform(
            low=0,
            high=1,
            size=X.shape[0]
        )
        max_degrees = self._map_magnitude(X)
        random_angles = torch.as_tensor(
            u, device=X.device) * 2 * max_degrees - max_degrees
        return {
            "sensors_positions_matrix": self.sensors_positions_matrix,
            "axis": self.axis,
            "angles": random_angles,
            "spherical_splines": self.spherical_splines
        }


class DiffSensorsZRotation(DiffSensorsRotation):
    """Interpolates EEG signals over sensors rotated around the Z axis
    with an angle sampled uniformly between
    `- (magnitude * mag_range[1] + (1-magnitude) * mag_range[0])` and
    `magnitude * mag_range[1] + (1-magnitude) * mag_range[0]`

    Suggested in [1]_

    Parameters
    ----------
    ordered_ch_names : list
        List of strings representing the channels of the montage considered.
        Has to be in standard 10-20 style. The order has to be consistent with
        the order of channels in the input matrices that will be fed to the
        transform. This channel will be used to compute approximate sensors
        positions from a standard 10-20 montage.
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : float, optional
        Initial value for the magnitude. Float between 0 and 1 encoding the
        `max_degree` parameter:
        ```
        max_degree = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to 0.5.
    mag_range : tuple of two floats | None, optional
        Range of possible values for `max_degree` settable using the magnitude
        (see `magnitude`). If omitted, the range (0, 30 degrees) will be used.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        `False`, standard scipy.interpolate.Rbf (with quadratic kernel) will be
        used (as in the original paper). Defaults to True.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Krell, M. M., & Kim, S. K. (2017). Rotational data augmentation for
       electroencephalographic data. In 2017 39th Annual International
       Conference of the IEEE Engineering in Medicine and Biology Society
       (EMBC) (pp. 471-474).
    """
    def __init__(
        self,
        ordered_ch_names,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=(0, 30),
        temperature=0.05,
        spherical_splines=True,
        random_state=None
    ):
        self.ordered_ch_names = ordered_ch_names
        sensors_positions_matrix = torch.as_tensor(
            _get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            initial_probability=initial_probability,
            initial_magnitude=initial_magnitude,
            mag_range=mag_range,
            axis='z',
            sensors_positions_matrix=sensors_positions_matrix,
            spherical_splines=spherical_splines,
            random_state=random_state
        )


class DiffSensorsYRotation(DiffSensorsRotation):
    """Interpolates EEG signals over sensors rotated around the Z axis
    with an angle sampled uniformly between
    `- (magnitude * mag_range[1] + (1-magnitude) * mag_range[0])` and
    `magnitude * mag_range[1] + (1-magnitude) * mag_range[0]`

    Suggested in [1]_

    Parameters
    ----------
    ordered_ch_names : list
        List of strings representing the channels of the montage considered.
        Has to be in standard 10-20 style. The order has to be consistent with
        the order of channels in the input matrices that will be fed to the
        transform. This channel will be used to compute approximate sensors
        positions from a standard 10-20 montage.
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : float, optional
        Initial value for the magnitude. Float between 0 and 1 encoding the
        `max_degree` parameter:
        ```
        max_degree = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to 0.5.
    mag_range : tuple of two floats | None, optional
        Range of possible values for `max_degree` settable using the magnitude
        (see `magnitude`). If omitted, the range (0, 30 degrees) will be used.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        `False`, standard scipy.interpolate.Rbf (with quadratic kernel) will be
        used (as in the original paper). Defaults to True.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Krell, M. M., & Kim, S. K. (2017). Rotational data augmentation for
       electroencephalographic data. In 2017 39th Annual International
       Conference of the IEEE Engineering in Medicine and Biology Society
       (EMBC) (pp. 471-474).
    """

    def __init__(
        self,
        ordered_ch_names,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=(0, 30),
        temperature=0.05,
        spherical_splines=True,
        random_state=None
    ):
        self.ordered_ch_names = ordered_ch_names
        sensors_positions_matrix = torch.as_tensor(
            _get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            initial_probability=initial_probability,
            initial_magnitude=initial_magnitude,
            mag_range=mag_range,
            axis='y',
            sensors_positions_matrix=sensors_positions_matrix,
            spherical_splines=spherical_splines,
            random_state=random_state
        )


class DiffSensorsXRotation(DiffSensorsRotation):
    """Interpolates EEG signals over sensors rotated around the Z axis
    with an angle sampled uniformly between
    `- (magnitude * mag_range[1] + (1-magnitude) * mag_range[0])` and
    `magnitude * mag_range[1] + (1-magnitude) * mag_range[0]`

    Suggested in [1]_

    Parameters
    ----------
    ordered_ch_names : list
        List of strings representing the channels of the montage considered.
        Has to be in standard 10-20 style. The order has to be consistent with
        the order of channels in the input matrices that will be fed to the
        transform. This channel will be used to compute approximate sensors
        positions from a standard 10-20 montage.
    initial_probability : float, optional
        Initial value for probability. Float between 0 and 1 defining the
        uniform probability of applying the operation. Set to 0.5 by default.
    initial_magnitude : float, optional
        Initial value for the magnitude. Float between 0 and 1 encoding the
        `max_degree` parameter:
        ```
        max_degree = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to 0.5.
    mag_range : tuple of two floats | None, optional
        Range of possible values for `max_degree` settable using the magnitude
        (see `magnitude`). If omitted, the range (0, 30 degrees) will be used.
    temperature : float, optional
        Temperature parameter of the RelaxedBernouilli distribution used to
        decide whether to apply the operation to the input or not. Defaults to
        0.05.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        `False`, standard scipy.interpolate.Rbf (with quadratic kernel) will be
        used (as in the original paper). Defaults to True.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Krell, M. M., & Kim, S. K. (2017). Rotational data augmentation for
       electroencephalographic data. In 2017 39th Annual International
       Conference of the IEEE Engineering in Medicine and Biology Society
       (EMBC) (pp. 471-474).
    """

    def __init__(
        self,
        ordered_ch_names,
        initial_probability=None,
        initial_magnitude=None,
        mag_range=(0, 30),
        temperature=0.05,
        spherical_splines=True,
        random_state=None
    ):
        self.ordered_ch_names = ordered_ch_names
        sensors_positions_matrix = torch.as_tensor(
            _get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            initial_probability=initial_probability,
            initial_magnitude=initial_magnitude,
            mag_range=mag_range,
            axis='x',
            sensors_positions_matrix=sensors_positions_matrix,
            spherical_splines=spherical_splines,
            random_state=random_state
        )


# Maps each differentiable transform to the parameters it requires to be
# instantiated, other than the regular ones (initial_probability,
# initial_magnitude, temperature, random_state). Useful to instantiate them
# functionally
POSSIBLE_DIFF_TRANSFORMS = {
    DiffTimeReverse: [],
    DiffSignFlip: [],
    DiffFTSurrogate: [],
    DiffChannelsDropout: [],
    DiffChannelsShuffle: [],
    DiffGaussianNoise: [],
    DiffChannelsSymmetry: ["ordered_ch_names"],
    DiffTimeMask: [],
    DiffFrequencyShift: ["sfreq"],
    DiffSensorsXRotation: ["ordered_ch_names"],
    DiffSensorsYRotation: ["ordered_ch_names"],
    DiffSensorsZRotation: ["ordered_ch_names"],
}
