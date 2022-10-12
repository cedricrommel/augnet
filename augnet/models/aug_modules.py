import torch.nn as nn
import torch.nn.functional as F
import torch


class AugAveragedModel(nn.Module):
    """Augment-Forward-Aggregate super-model

    It encapsulates a trunk neural network with an augmentation module in the
    front-end and an averaging layer on the back-end. Useable for both
    Augerino and AugNet models.

    Parameters
    ----------
    model: torch.nn.Module
        Trunk network.
    aug: torch.nn.Moduel
        Augmentation module, e.g. ``augnet.models.UniformAug`` or
        ``augnet.models.AugmentationModule``.
    n_copies: int, optional
        Number of batch copies made before augmentation. Only used for
        inference, not for training. Defaults to 4.
    ncopies_tr: int, optional
        Number of batch copies made before augmentation at **training**.
        Defaults to 1.
    """
    def __init__(self, model, aug, ncopies=4, ncopies_tr=1):
        super().__init__()
        self.aug = aug
        self.model = model
        self.ncopies = ncopies
        self.ncopies_tr = ncopies_tr

    def forward(self, x):
        # Decide the number of copies depending on if we are training or not
        curr_ncopies = self.ncopies_tr if self.training else self.ncopies

        if curr_ncopies > 1:
            # Replicate the batch n_copies times, creating a super batch of
            # size curr_ncopies * batch_size
            super_x = torch.cat([x for _ in range(curr_ncopies)], dim=0)

            # Augment and forward it through the network
            bs = x.shape[0]
            out = self.model(self.aug(super_x))

            # Compute the logits and averages over curr_ncopies
            return sum(
                torch.split(F.log_softmax(out, dim=-1), bs)
            ) / curr_ncopies
        else:
            return self.model(self.aug(x))
