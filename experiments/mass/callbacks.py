import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import log_loss

import torch

from skorch.callbacks import Callback


def compute_reg(model, X, y, reg):
    """ Can be used in EpochScoring callback to track regularization value
    """
    l2_magnitude_weight = (
        model.module_.aug._magnitudes.clone().detach().clamp(0, 1) *
        model.module_.aug.weights.clone().detach()
    ).norm().item()
    return - reg * l2_magnitude_weight


def compute_class_weights_dict(y):
    """ Computes frequency per class
    """
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    return {i: w for i, w in enumerate(class_weights)}


def compute_balanced_log_loss(model, X, y):
    """ Can be used in EpochScoring callback to track balanced CE loss
    """
    class_weights = compute_class_weights_dict(y)
    sample_weights = compute_sample_weight(class_weights, y)
    with torch.no_grad():
        y_proba = model.predict_proba(X)
    return log_loss(y, y_proba, sample_weight=sample_weights)


def fetch_weights_and_mags(model, mag_log=None, w_log=None):
    epoch = 0
    if w_log is not None:
        epoch = w_log.columns[-1] + 1

    layer_op_names = [
        [type(op).__name__ for op in layer.augmentation]
        for layer in model.aug.layers
    ]
    indices = list()
    for names in layer_op_names:
        indices += names

    new_weights = pd.DataFrame(
        model.aug.weights.clone().detach().cpu().numpy().reshape(-1, 1),
        index=indices,
        columns=[epoch],
    )

    epoch = 0
    if mag_log is not None:
        epoch = mag_log.columns[-1] + 1

    _new_magnitudes = pd.DataFrame(
        model.aug._magnitudes.clone().detach().cpu().numpy().reshape(-1, 1),
        index=indices,
        columns=[epoch],
    )
    if w_log is None or mag_log is None:
        return new_weights, _new_magnitudes
    return (
        pd.concat([w_log, new_weights], axis=1),
        pd.concat([mag_log, _new_magnitudes], axis=1)
    )


class AugParamsSaver(Callback):
    """ Skorch callback which allows to save to checkpoint weights and
    magnitudes learned by the augmentations layers at each epoch
    """
    def __init__(self, root_path):
        self.root_path = root_path

    def initialize(self):
        self.weights = None
        self.magnitudes = None
        return self

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        """Called at the beginning of training."""
        weights, mags = fetch_weights_and_mags(
            net.module_, mag_log=self.magnitudes, w_log=self.weights)
        self.weights = weights
        self.magnitudes = mags

    def on_epoch_end(
        self,
        net,
        dataset_train=None,
        dataset_valid=None,
        **kwargs
    ):
        """Called at the end of each epoch."""
        weights, mags = fetch_weights_and_mags(
            net.module_, mag_log=self.magnitudes, w_log=self.weights)
        self.weights = weights
        self.magnitudes = mags
        self.weights.to_pickle(self.root_path + "augweights.pkl")
        self.magnitudes.to_pickle(self.root_path + "augmags.pkl")


class TestEpochScoring(Callback):
    """ Skorch callback which allows to checkpoint test metrics during training
    """
    def __init__(self, root_path, test_set):
        self.root_path = root_path
        self.test_set = test_set

    def initialize(self):
        self.epoch = 1
        self.results_per_epoch = list()
        self.y_true = np.array([y for _, y, _ in self.test_set])
        self.class_weights = compute_class_weights_dict(self.y_true)

    def on_epoch_end(
        self,
        net,
        dataset_train=None,
        dataset_valid=None,
        **kwargs
    ):
        """Called at the end of each epoch."""
        with torch.no_grad():
            y_proba = net.predict_proba(self.test_set)
        y_pred = np.argmax(y_proba, axis=1)

        # Because drop_last=True in dataloaders, predictions size don't
        # always match labels
        dropping = self.y_true.shape[0] - y_pred.shape[0]
        # assert dropping < self.model_params["batch_size"]
        y_true = self.y_true[:-dropping]
        sample_weights = compute_sample_weight(self.class_weights, y_true)

        self.results_per_epoch.append({
            'epoch': self.epoch,
            'test_bal_acc': balanced_accuracy_score(y_true, y_pred),
            'test_confusion_matrix': confusion_matrix(y_true, y_pred),
            'test_wCE_loss': log_loss(y_true, y_proba,
                                      sample_weight=sample_weights),
            'test_cohen_kappa_score': cohen_kappa_score(y_true, y_pred),
        })
        self.epoch += 1
        pd.DataFrame(self.results_per_epoch).to_pickle(
            self.root_path + "metrics_per_epoch.pkl")
