import torch
from torch import nn, Tensor
from torch.distributions import Categorical


class AugmentationLayer(nn.Module):
    """Differentiable augmentation layer, as proposed in [1]_

    Parameters
    ----------
    augmentations: list
        Possible augmentations that the layer can select.
    temperature: float
        Temperature of the softmax of augmentations' weights.
    initial_weights: torch.Tensor, optional
        Initial weights for each candidate augmentation. Defaults to None,
        which corresponds to equal weights.
    data_dim: int, optional
        Dimensionality of the input data, e.g. 3 for images (H, W, C) and 2 for
        EEG windows (C, T). Defaults to 3.

    References
    ----------
    .. [1] C. Rommel, T. Moreau, A. Gramfort. Deep invariant networks with
    differentiable augmentation layers, in Advances on Neural Information
    Processing Systems (NeurIPS), 2022.
    """
    def __init__(
        self,
        augmentations: list,
        temperature: float,
        initial_weights: torch.Tensor = None,
        data_dim: int = 3,
        fixed_probability: int = 1.0,
    ):
        super(AugmentationLayer, self).__init__()
        self.augmentations = nn.ModuleList(augmentations)
        self._weights = nn.Parameter(torch.ones(len(self.augmentations)))
        if initial_weights is not None:
            self._weights = nn.Parameter(initial_weights)
        self.temperature = temperature
        self.data_dim = data_dim
        self._weights_shape = [1] * self.data_dim

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            return (
                torch.stack(
                    [aug(input) for aug in self.augmentations]
                ) * self.weights.view(-1, 1, *self._weights_shape)
            ).sum(0)
        else:
            sampled_op_idx = Categorical(self.weights).sample()
            return self.augmentations[sampled_op_idx](input)

    @property
    def weights(self):
        return self._weights.div(self.temperature).softmax(0)

    @property
    def magnitudes(self):
        return torch.concat([
            aug.magnitude if aug.magnitude is not None else aug.probability
            for aug in self.augmentations
        ])

    @property
    def _magnitudes(self):
        return torch.concat([
            aug._magnitude if aug._magnitude is not None else aug._probability
            for aug in self.augmentations
        ])

    @property
    def _mag_grads(self):
        grads = list()
        for aug in self.augmentations:
            if aug.magnitude is not None and aug._magnitude.grad is not None:
                grads.append(aug._magnitude.grad)
            elif (
                aug.probability is not None and
                aug._probability.grad is not None
            ):
                grads.append(aug._probability.grad)
            else:
                grads.append(torch.zeros(1))
        return torch.concat(grads)

    @property
    def augmentations_names(self):
        return [type(aug).__name__ for aug in self.augmentations]

    def freeze_probs(self):
        """Freezes the probability parameter of all augmentations in the layer
        """
        for aug in self.augmentations:
            aug._probability.requires_grad = False


class AugmentationModule(nn.Module):
    """AugNet augmentation module, as described in [1]_

    Parameters
    ----------
    layers: list
        List of augmentation layers to apply sequentially.

    References
    ----------
    .. [1] C. Rommel, T. Moreau, A. Gramfort. Deep invariant networks with
    differentiable augmentation layers, in Advances on Neural Information
    Processing Systems (NeurIPS), 2022.
    """
    def __init__(
        self,
        layers: list,
    ):
        super(AugmentationModule, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.freeze_probs()

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    @property
    def weights(self):
        return torch.concat([layer.weights for layer in self.layers])

    @property
    def _weights(self):
        return torch.concat([layer._weights for layer in self.layers])

    @property
    def magnitudes(self):
        return torch.concat([layer.magnitudes for layer in self.layers])

    @property
    def _magnitudes(self):
        return torch.concat([layer._magnitudes for layer in self.layers])

    @property
    def _mag_grads(self):
        _mag_grads = [layer._mag_grads for layer in self.layers]
        if len(_mag_grads) > 0:
            return torch.concat(_mag_grads)

    @property
    def _weight_grads(self):
        _w_grads = [
            layer._weights.grad
            if layer._weights.grad is not None
            else torch.zeros_like(layer.weights)
            for layer in self.layers
        ]
        return torch.concat(_w_grads)

    @property
    def augmentations_names(self):
        layer_op_names = list()
        for layer_index, layer in enumerate(self.layers):
            layer_op_names += [
                f"{layer_index}-{aug}" for aug in layer.augmentations_names]
        return layer_op_names

    def freeze_probs(self):
        """Freezes the probability parameter of all augmentations in the module
        """
        for layer in self.layers:
            layer.freeze_probs()
