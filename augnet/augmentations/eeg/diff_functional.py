import torch

from braindecode.augmentation.functional import _pick_channels_randomly
from braindecode.augmentation.functional import _make_permutation_matrix

from ..straight_through import ste


def diff_channels_shuffle(X, y, p_shuffle, random_state):
    mask = _pick_channels_randomly(X, 1-p_shuffle, random_state)
    batch_permutations = ste(
        _make_permutation_matrix(X, mask, random_state),
        p_shuffle
    )
    return torch.matmul(batch_permutations, X), y
