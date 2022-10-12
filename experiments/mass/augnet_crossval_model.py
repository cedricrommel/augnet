from os.path import join
import copy

import numpy as np
from torch.utils.data import Subset
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics import log_loss, cohen_kappa_score
from sklearn.utils.class_weight import compute_sample_weight

from eeg_augment.training_utils import CrossvalModel, fit_and_predict
from eeg_augment.training_utils import make_training_specific_callbacks
from eeg_augment.training_utils import set_random_seeds, set_deterministic_mode

from callbacks import AugParamsSaver, TestEpochScoring


class AugnetCrossvalModel(CrossvalModel):
    def __init__(
        self,
        training_dir,
        model,
        **kwargs,
    ):
        super().__init__(training_dir, model, **kwargs)
        # TOOD: set tensorboard here

    def _fit_and_score(
        self,
        split,
        epochs,
        windows_dataset,
        model_params,
        ckpt_prefix,
        only_history,
        random_state,
        **kwargs
    ):
        """Train and tests a copy of self.model on the desired split

        Parameters
        ----------
        split : tuple
            Tuple containing the fold index, the training set proportion and
            the indices of the training, validation and test set.
        epochs : int
            Maximum number of epochs for the training.
        windows_dataset : torch.utils.data.Dataset
            Dataset that will be split and used for training, validation and
            tetsing.
        model_params : dict
            Modified copy of self.model_params.
        ckpt_prefix : str
            Prefix to add to checkpoint files.
        only_history : bool
            Whether to only checkpoint the training history or not.
        random_state : int | None
            Seed to use for RNGs.

        Returns
        -------
        dict
            Dictionary containing the balanced accuracy, the loss, the kappa
            score and the confusion matrix for the training, validationa and
            test sets.
        """

        fold, subset_ratio, train_subset_idx, valid_idx, test_idx = split

        set_random_seeds(
            seed=random_state,
            cuda=self.model_params['device'].type == "cuda"
        )
        set_deterministic_mode()

        fold_path = join(self.training_dir, f'fold{fold}of{self.n_folds}')
        subset_path = join(fold_path, f'subset_{subset_ratio}_samples')

        train_subset = Subset(windows_dataset, train_subset_idx)
        test_set = Subset(windows_dataset, test_idx)
        valid_set = Subset(windows_dataset, valid_idx)

        print(
            f"---------- Fold {fold} out of {self.n_folds} |",
            f"Training size: {len(train_subset)} ----------"
        )

        callbacks = list()
        callbacks += self.shared_callbacks

        callbacks += make_training_specific_callbacks(
            subset_path,
            metric_to_monitor=self.monitor,
            should_checkpoint=self.should_checkpoint,
            should_load_state=self.should_load_state,
            ckpt_prefix=ckpt_prefix,
        )

        # Add AugParamsSaver callback with fold path
        params_saver = AugParamsSaver(root_path=join(subset_path, "augnet_"))
        test_scoring = TestEpochScoring(
            root_path=join(subset_path, "augnet_test_"), test_set=test_set)
        valid_scoring = TestEpochScoring(
            root_path=join(subset_path, "augnet_valid_"), test_set=valid_set)
        callbacks += [
            ('params_saver', params_saver),
            ('test_scoring', test_scoring),
            ('valid_scoring', valid_scoring),
        ]

        # TODO: Add tensorboard callback

        predicted_probas, class_weights = fit_and_predict(
            model=copy.deepcopy(self.model),
            train_set=train_subset,
            valid_set=valid_set,
            test_set=test_set,
            epochs=epochs,
            model_params=model_params,
            balanced_loss=self.balanced_loss,
            callbacks=callbacks,
            **kwargs
        )

        results_per_subset = {
            'fold': fold, 'n_fold': self.n_folds, 'subset_ratio': subset_ratio,
        }
        keys_and_ds = zip(
            ["train", "valid", "test"],
            [train_subset, valid_set, test_set]
        )
        for key, ds in keys_and_ds:
            # Evaluate metrics on all 3 datasets and add to results returned
            y_true = np.array([y for _, y, _ in ds])
            y_proba = predicted_probas[key]
            y_pred = np.argmax(y_proba, axis=1)

            # Because drop_last=True in dataloaders, predictions size don't
            # always match labels
            dropping = y_true.shape[0] - y_pred.shape[0]
            assert dropping < self.model_params["batch_size"]
            y_true = y_true[:-dropping]

            ds_weights = compute_sample_weight(class_weights, y_true)

            results_per_subset.update({
                f'{key}_bal_acc': balanced_accuracy_score(y_true, y_pred),
                f'{key}_confusion_matrix': confusion_matrix(y_true, y_pred),
                f'{key}_wCE_loss': log_loss(y_true, y_proba,
                                            sample_weight=ds_weights),
                f'{key}_cohen_kappa_score': cohen_kappa_score(y_true, y_pred),
            })
        return results_per_subset
