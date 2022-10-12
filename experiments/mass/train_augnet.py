import argparse

from sklearn.utils import check_random_state

from eeg_augment.training_utils import find_device, set_random_seeds
from eeg_augment.training_utils import set_deterministic_mode
from eeg_augment.training_utils import make_vanilla_model_params
from eeg_augment.training_utils import make_shared_callbacks
from eeg_augment.training_utils import load_preproc_mass
from eeg_augment.training_utils import make_sleep_stager_model

from augnet.augmentations.eeg import DiffTimeReverse
from augnet.augmentations.eeg import DiffSignFlip
from augnet.augmentations.eeg import DiffChannelsSymmetry
from augnet.augmentations.eeg import DiffFTSurrogate
from augnet.augmentations.eeg import DiffChannelsDropout
from augnet.augmentations.eeg import DiffFrequencyShift
from augnet.augmentations.eeg import DiffGaussianNoise
from augnet.augmentations.eeg import DiffTimeMask
from augnet.augmentations.eeg import DiffChannelsShuffle
from augnet.augmentations.eeg import DiffSensorsZRotation
from augnet.augmentations.eeg import DiffSensorsYRotation
from augnet.augmentations.eeg import DiffSensorsXRotation

from augnet import models
from augnet.losses import InvariancePromotingLoss

from augnet_crossval_model import AugnetCrossvalModel


def create_all_transformations(
    sfreq,
    ordered_ch_names,
    init_mag=0.5,
    freeze_prob=True,
    random_state=None
):
    """ Builds all EEG transforms
    """
    rng = check_random_state(random_state)

    operations = [
        DiffTimeReverse(
            initial_magnitude=1.0, initial_probability=init_mag,
            random_state=rng),
        DiffSignFlip(
            initial_magnitude=1.0, initial_probability=init_mag,
            random_state=rng),
        DiffChannelsSymmetry(
            initial_magnitude=1.0, initial_probability=init_mag,
            random_state=rng, ordered_ch_names=ordered_ch_names),
        DiffFTSurrogate(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng),
        DiffChannelsDropout(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng),
        DiffFrequencyShift(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng, sfreq=sfreq),
        DiffGaussianNoise(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng),
        DiffTimeMask(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng),
        DiffChannelsShuffle(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng),
        DiffSensorsZRotation(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng, ordered_ch_names=ordered_ch_names),
        DiffSensorsYRotation(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng, ordered_ch_names=ordered_ch_names),
        DiffSensorsXRotation(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng, ordered_ch_names=ordered_ch_names),
    ]

    if freeze_prob:
        for op in operations:
            if op._magnitude is not None:
                op._probability.requires_grad = False
    return operations


def create_time_freq_transformations(
    sfreq,
    ordered_ch_names,
    init_mag=0.5,
    freeze_prob=True,
    random_state=None
):
    """ Builds EEG time-frequency transforms
    """
    rng = check_random_state(random_state)

    operations = [
        DiffTimeReverse(
            initial_magnitude=1.0, initial_probability=init_mag,
            random_state=rng),
        DiffGaussianNoise(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng),
        DiffTimeMask(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng),
        DiffFTSurrogate(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng),
        DiffFrequencyShift(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng, sfreq=sfreq),
    ]

    if freeze_prob:
        for op in operations:
            if op._magnitude is not None:
                op._probability.requires_grad = False
    return operations


def create_spatial_transformations(
    sfreq,
    ordered_ch_names,
    init_mag=0.5,
    freeze_prob=True,
    random_state=None
):
    """ Builds EEG spatial transforms
    """
    rng = check_random_state(random_state)

    operations = [
        DiffSignFlip(
            initial_magnitude=1.0, initial_probability=init_mag,
            random_state=rng),
        DiffChannelsSymmetry(
            initial_magnitude=1.0, initial_probability=init_mag,
            random_state=rng, ordered_ch_names=ordered_ch_names),
        DiffChannelsDropout(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng),
        DiffChannelsShuffle(
            initial_magnitude=init_mag, initial_probability=1.0,
            random_state=rng),
    ]

    if freeze_prob:
        for op in operations:
            if op._magnitude is not None:
                op._probability.requires_grad = False
    return operations


def create_augnet_layer(
    sfreq,
    ordered_ch_names,
    transforms_type,
    init_mag=0.5,
    freeze_weights=False,
    freeze_prob=True,
    random_state=None
):
    """ Builds EEG-compatible augmentation layer with all transforms
    implemented
    """
    rng = check_random_state(random_state)

    if transforms_type == "all":
        operations_builder = create_all_transformations
    elif transforms_type == "time-freq":
        operations_builder = create_time_freq_transformations
    elif transforms_type == "spatial":
        operations_builder = create_spatial_transformations
    else:
        raise ValueError(
            "Possible values for transforms_type are 'all', 'time-freq' and "
            f"'spatial'. Got {transforms_type}."
        )

    operations = operations_builder(
        sfreq=sfreq,
        ordered_ch_names=ordered_ch_names,
        init_mag=init_mag,
        freeze_prob=freeze_prob,
        random_state=rng,
    )

    aug_layer = models.AugmentationLayer(
        augmentations=operations, temperature=0.01, data_dim=2,
    )

    # In order to simplify the optimization, it is possible to freeze
    # weights (and probabilities, but those should always be frozen at 1.)
    if freeze_weights:
        aug_layer._weights.requires_grad = False
    return aug_layer


def create_augnet_module(
    sfreq,
    ordered_ch_names,
    transforms_types=None,
    n_layers=1,
    init_mag=0.5,
    freeze_weights=False,
    freeze_prob=True,
    random_state=None
):
    """ Creates an EEG compatible augmentation module with the desired number
    of layers
    """
    rng = check_random_state(random_state)

    if transforms_types is None or len(transforms_types) == 0:
        return models.AugmentationModule(
            [
                create_augnet_layer(
                    sfreq,
                    ordered_ch_names,
                    init_mag=init_mag,
                    freeze_weights=freeze_weights,
                    freeze_prob=freeze_prob,
                    random_state=rng,
                    transforms_type="all",
                ) for _ in range(n_layers)
            ]
        )
    else:
        assert len(transforms_types) == n_layers,\
            f"Got {n_layers} layers, but only type f{transforms_types}"
        return models.AugmentationModule(
            [
                create_augnet_layer(
                    sfreq,
                    ordered_ch_names,
                    init_mag=init_mag,
                    freeze_weights=freeze_weights,
                    freeze_prob=freeze_prob,
                    random_state=rng,
                    transforms_type=t,
                ) for t in transforms_types
            ]
        )


def train_augnet(
    dir,
    dataset,
    ch_names,
    sfreq,
    n_classes,
    n_jobs,
    device,
    seed,
    n_layers,
    transforms_types,
    init_mag,
    ncopies,
    lr,
    batch_size,
    num_workers,
    aug_reg,
    early_stop,
    patience,
    n_folds,
    train_size_over_valid,
    epochs,
    data_ratio,
    grouped_subset,
):
    # Detect device and set global rngs for model init
    device, cuda = find_device(device)
    set_random_seeds(seed=seed, cuda=cuda)
    set_deterministic_mode()

    # Create augmentation layers
    aug_module = create_augnet_module(
        sfreq, ch_names, n_layers=n_layers, random_state=seed,
        init_mag=init_mag, transforms_types=transforms_types,
    )

    # Create back-bone
    net = make_sleep_stager_model(
        windows_dataset=dataset,
        device=device,
        sfreq=sfreq,
        n_classes=n_classes,
    )

    # and the averaged model
    model = models.AugAveragedModel(net, aug_module, ncopies=ncopies)

    # Create vanilla model params for skorch training
    model_params = make_vanilla_model_params(
        lr=lr,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    # Set criterion with regularization
    model_params['criterion'] = InvariancePromotingLoss
    model_params['criterion__reg'] = aug_reg
    model_params['criterion__model'] = model

    # Make sure drop_last is correctly set in dataloaders
    model_params['iterator_valid__drop_last'] = True

    # Create standard callbacks for this dataset (such as earlystopping)
    shared_callbacks = make_shared_callbacks(
        early_stop=early_stop,
        patience=patience,
    )

    # Add custom callbacks to be able to track what is happenning with the
    # augmentation layers during training
    # train_ce_loss = EpochScoring(
    #     scoring=compute_balanced_log_loss,
    #     on_train=True, name='train_ce_loss',
    #     lower_is_better=True,
    # )
    # valid_ce_loss = EpochScoring(
    #     scoring=compute_balanced_log_loss,
    #     on_train=False, name='valid_ce_loss',
    #     lower_is_better=True,
    # )
    # train_reg = EpochScoring(
    #     scoring=partial(compute_reg, reg=aug_reg), on_train=False,
    #     name='train_reg', lower_is_better=True,
    # )
    # shared_callbacks += [
    #     ('train_ce_loss', train_ce_loss),
    #     ('valid_ce_loss', valid_ce_loss),
    #     ('train_reg', train_reg),
    # ]

    # Instatiate wrapper around model and dataset allowing to train and
    # evaluate with cross-validation
    cross_val_training = AugnetCrossvalModel(
        training_dir=dir,
        model=model,
        model_params=model_params,
        shared_callbacks=shared_callbacks,
        balanced_loss=True,
        monitor='valid_bal_acc_best',
        should_checkpoint=True,  # XXX
        log_tensorboard=False,  # XXX
        random_state=seed,
        n_folds=n_folds,
        train_size_over_valid=train_size_over_valid,
    )

    cross_val_training.learning_curve(
        windows_dataset=dataset,
        epochs=epochs,
        data_ratios=data_ratio,
        grouped_subset=grouped_subset,
        n_jobs=n_jobs,
        verbose=False,  # XXX
    )


def main(args):
    # Load dataset
    dataset, ch_names, sfreq, n_classes = load_preproc_mass(
        n_subjects=args.n_subj,
        preload=args.preload,
        n_jobs=args.n_jobs,
        bids_root=args.data_dir,
    )

    train_augnet(
        dir=args.dir,
        dataset=dataset,
        ch_names=ch_names,
        sfreq=sfreq,
        n_classes=n_classes,
        n_jobs=args.n_jobs,
        device=args.device,
        seed=args.seed,
        n_layers=args.n_layers,
        transforms_types=args.tf_types,
        init_mag=args.init_mag,
        ncopies=args.ncopies,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_reg=args.aug_reg,
        early_stop=args.early_stop,
        patience=args.patience,
        n_folds=args.n_folds,
        train_size_over_valid=args.train_size_over_valid,
        epochs=args.epochs,
        data_ratio=args.data_ratio,
        grouped_subset=args.grouped_subset,
    )


def make_argparser():
    parser = argparse.ArgumentParser(description="MASS experiment")

    parser.add_argument(
        "--dir",
        type=str,
        default='./saved-outputs/augnet/',
        help="training directory (default: ./saved-outputs/augnet/)",
    )
    parser.add_argument(
        "--data_dir",
        help="directory for MASS dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="initial learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--aug_reg",
        type=float,
        default=5,
        help="augmentation regularization weight",
    )
    parser.add_argument(
        "--init_mag",
        type=float,
        default=0.1,
        help="Initial magnitude in augmentation module (default:0.1)",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=3,
        help="number of layers in augmentation module (default:3)",
    )
    parser.add_argument(
        "--tf_types",
        action="append",
        type=str,
        help="Types of augmentation layers.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        metavar="N",
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--ncopies",
        type=int,
        default=4,
        metavar="N",
        help="number of augmentations in network (defualt: 4)"
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda:1",
        help="Device (default: cuda:1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=29,
        metavar="N",
        help="Random seed."
    )
    parser.add_argument(
        "--n_subj",
        type=int,
        default=60,
        help="Number of subjects nights to load."
    )
    parser.add_argument(
        "--data_ratio",
        type=float,
        default=1.,
        help="Ratio of training set to use for training."
    )
    parser.add_argument(
        "--train_size_over_valid",
        type=float,
        default=0.5,
        help="Train set size wrt validation set size."
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds to cross-validate results."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of processes to parallelize across."
    )
    parser.add_argument(
        "--grouped_subset",
        action="store_true",
        help="When subsetting the training set (data_ratio < 1), will only "
             "allow to keep whole nights."
    )
    parser.add_argument(
        '--no_early_stop',
        action='store_false',
        dest='early_stop',
        help="When this argument is passed, no early stopping will be used."
    )
    parser.add_argument(
        "--preload", action='store_true',
        help="Whether to preload data to the RAM"
    )
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of workers used for data loading'
    )
    parser.add_argument(
        '--patience',
        default=30,
        type=int,
        help='Patience to use for earlystopping.'
    )
    return parser


if __name__ == '__main__':

    parser = make_argparser()
    args = parser.parse_args()

    main(args)
