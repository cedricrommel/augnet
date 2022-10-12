import pytest

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss


from augnet.models import (
    AugAveragedModel,
    AugerinoAugModule,
    AugmentationLayer,
    AugmentationModule,
    SimpleMLP,
)
from augnet.augmentations.vision import (
    TranslateX,
    TranslateY,
    ShearX,
    ShearY,
    Rotate,
)
from augnet.augmentations.vision.augmentations import _Operation


@pytest.fixture(scope="function")
def dummy_data():
    torch.manual_seed(555)
    return torch.randn(16, 10, 10, 3), torch.randint(0, 2, (16,))


def create_aug_layers(n_layers, init_mag=0.05, fixed_prob=1.0):
    aug_layers = list()
    for _ in range(n_layers):
        # create augmentations
        operations = [
            TranslateX(
                initial_magnitude=init_mag,
                initial_probability=fixed_prob,
            ),
            TranslateY(
                initial_magnitude=init_mag,
                initial_probability=fixed_prob
            ),
            Rotate(
                initial_magnitude=init_mag,
                initial_probability=fixed_prob,
                magnitude_scale=180
            ),
            ShearX(
                initial_magnitude=init_mag,
                initial_probability=fixed_prob
            ),
            ShearY(
                initial_magnitude=init_mag,
                initial_probability=fixed_prob
            ),
        ]

        # create layers
        aug_layer = AugmentationLayer(
            augmentations=operations,
            temperature=0.01,
        )
        aug_layers.append(aug_layer)
    return aug_layers


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_augnet(n_layers, dummy_data):
    # create layers
    aug_layers = create_aug_layers(n_layers=n_layers)

    # create aug module
    aug_module = AugmentationModule(aug_layers)

    # create trunk model
    X, y = dummy_data
    trunk_model = SimpleMLP(
        n_neurons=3,
        n_layers=2,
        in_shape=np.prod(X.shape[1:]).item(),
        num_classes=2,
    )

    # create augnet
    augnet = AugAveragedModel(
        model=trunk_model,
        aug=aug_module,
        ncopies=4,
    )

    # check train forward
    augnet.train()
    pred_y = augnet(X)
    assert pred_y.shape == F.one_hot(y, 2).shape, (
        "Incompatible output shape found after AugNet forward pass in "
        "training mode."
    )

    # check backward pass leads to gradients...
    loss = CrossEntropyLoss()(pred_y, y)
    loss.backward()
    # both for the trunk model...
    for p in augnet.model.parameters():
        assert p.grad is not None, (
            "None gradient found in AugNet's trunk model"
        )
    # ... and the aug layers
    for layer in augnet.aug.layers:
        assert layer._weights.grad is not None, (
            "None gradient found in AugNet's augmentation layers."
        )
        assert layer._magnitudes is not None, (
            "None gradient found in AugNet's augmentation layers."
        )

    # check eval forward
    augnet.eval()
    pred_y = augnet(X)
    assert pred_y.shape == F.one_hot(y, 2).shape, (
        "Incompatible output shape found after AugNet forward pass "
        "in eval mode."
    )


def test_augerino(dummy_data):
    # create aug module
    aug_module = AugerinoAugModule()

    # create trunk model
    X, y = dummy_data
    trunk_model = SimpleMLP(
        n_neurons=3,
        n_layers=2,
        in_shape=np.prod(X.shape[1:]).item(),
        num_classes=2,
    )

    # create augnet
    augerino = AugAveragedModel(
        model=trunk_model,
        aug=aug_module,
        ncopies=4,
    )

    # check train forward
    augerino.train()
    pred_y = augerino(X)
    assert pred_y.shape == F.one_hot(y, 2).shape, (
        "Incompatible output shape found after Augerino forward pass in "
        "training mode."
    )

    # check backward pass leads to gradients...
    loss = CrossEntropyLoss()(pred_y, y)
    loss.backward()
    # both for the trunk model...
    for p in augerino.model.parameters():
        assert p.grad is not None, (
            "None gradient found in Augerino's trunk model"
        )
    # ... and the aug layers
    assert augerino.aug.width.grad is not None, (
        "None gradient found in Augerino's augmentation module"
    )

    # check eval forward
    augerino.eval()
    pred_y = augerino(X)
    assert pred_y.shape == F.one_hot(y, 2).shape, (
        "Incompatible output shape found after Augerino forward pass "
        "in eval mode."
    )


def dummy_augmentation(X, mag):
    batch_size = X.shape[0]
    multipliers = torch.randint(0, 2, size=(batch_size,))
    return multipliers.view(batch_size, 1, 1, 1) * torch.ones_like(X)


class DummyAug(_Operation):
    def __init__(self):
        super().__init__(
            dummy_augmentation,
            initial_magnitude=1.0,
            initial_probability=1.0,
        )


class DummyTrunkModel(torch.nn.Module):
    def forward(self, X):
        if len(X.shape) < 4:
            X = X.unsqueeze(0)
        batch_size = X.shape[0]
        return X.view(batch_size, -1).max(dim=1)[0]


def test_ncopies(dummy_data):
    # create dummy augmentation module
    aug_module = DummyAug()

    # create dummy trunk model
    trunk_model = DummyTrunkModel()

    # create augnet
    augnet = AugAveragedModel(
        model=trunk_model,
        aug=aug_module,
        ncopies=10,
    )

    # forward in training mode (should do only one input copy)
    X, _ = dummy_data
    torch.manual_seed(555)
    augnet.train()
    y_pred_train = augnet(X)
    assert torch.logical_or(
        y_pred_train == 1.0,
        y_pred_train < 1e-6  # testing for 0 exactly can fail
    ).all().item(), "Possible problem on number of copies made in train mode."

    # forward in eval mode (should do many input copies)
    torch.manual_seed(555)
    augnet.eval()
    y_pred_eval = augnet(X)
    assert torch.logical_or(
        y_pred_eval < 1.0,
        y_pred_eval > 0.0,
    ).all().item(), "Possible problem on number of copies made in eval mode."

    # forward in eval mode with ncopies=1 (should do only one input copy)
    augnet.ncopies = 1
    torch.manual_seed(555)
    y_pred_eval_1 = augnet(X)
    assert torch.logical_or(
        y_pred_eval_1 == 1.0,
        y_pred_eval_1 < 1e-6,  # testing for 0 exactly can fail
    ).all().item(), "Possible problem on number of copies made in eval mode."
