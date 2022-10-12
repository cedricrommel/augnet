import argparse
import pickle
import os
from os.path import join
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms

from augnet import models, losses
from augnet.utils import set_global_rngs, set_deterministic_mode

from augnet.augmentations.vision import Identity
from augnet.augmentations.vision import MagHorizontalFlip
from augnet.augmentations.vision import TranslateX, TranslateY
from augnet.augmentations.vision import ShearX, ShearY
from augnet.augmentations.vision import Rotate
from augnet.augmentations.vision import Contrast
from augnet.augmentations.vision import Brightness
from augnet.augmentations.vision import SamplePairing


def compute_metrics(model, data, use_cuda, epoch, args):
    """ Computes loss and accuracy for given batch
    """
    kwargs = {}
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    # Correct loss function depends on method
    if args.method == "augerino":
        criterion = losses.safe_unif_aug_loss
        kwargs["reg"] = args.aug_reg
        kwargs["model"] = model
    elif args.method == "augnet":
        criterion = losses.aug_layer_loss
        kwargs["reg"] = args.aug_reg
        if args.pen_schedule:
            if epoch > 40:
                kwargs["reg"] /= 10
                print(f"Reducing penalty weight to {kwargs['reg']}")
            elif epoch > 150:
                kwargs["reg"] /= 20
                print(f"Reducing penalty weight to {kwargs['reg']}")
        kwargs["model"] = model
    elif args.method == "none":
        criterion = ce_loss_fn
    else:
        raise ValueError(
                "Unknown value for method argument. "
                "Possible values are 'augnet', 'augerino' or 'none'."
                f"Got {args.method}."
            )

    inputs, labels = data

    if use_cuda:
        inputs, labels = inputs.to(args.device), labels.to(args.device)

    try:
        outputs = model(inputs)
    except RuntimeError:
        outputs = model(inputs)
    loss = criterion(outputs, labels, **kwargs)

    with torch.no_grad():
        y_pred = np.argmax(outputs.clone().detach().cpu(), axis=1)
        acc = accuracy_score(
            labels.clone().detach().cpu(), y_pred, normalize=False)
        ce_loss = ce_loss_fn(outputs, labels)
    return loss, acc, ce_loss


def aggreg_and_print_metric(running_loss, running_acc, running_ce, loader, tag,
                            metrics=None, writer=None, epoch=None):
    """ Aggregates running metrics into epoch metrics, stores them in
    dictionary and prints
    """
    if metrics is None:
        metrics = dict()
    metrics[f"epoch_{tag}_loss"] = running_loss / len(loader)
    metrics[f"epoch_{tag}_acc"] = running_acc / len(loader.dataset)
    metrics[f"epoch_{tag}_ce"] = running_ce / len(loader)
    running_reg = running_loss - running_ce
    metrics[f"epoch_{tag}_reg"] = - running_reg / len(loader)

    print(f"Epoch {tag} loss = ", metrics[f"epoch_{tag}_loss"])
    print(f"Epoch {tag} CE_loss = ", metrics[f"epoch_{tag}_ce"])
    print(f"Epoch {tag} reg = ", metrics[f"epoch_{tag}_reg"])
    print(f"Epoch {tag} acc = ", metrics[f"epoch_{tag}_acc"])

    log_this(writer, tag, epoch, metrics)
    return metrics


def create_geometric_operations(init_mag, freeze_prob):
    """ Instatiates a new list of geometric operations to create an
    augmentation layer
    """
    operations = [
        # Geometric
        TranslateX(initial_magnitude=init_mag, initial_probability=1.0),
        TranslateY(initial_magnitude=init_mag, initial_probability=1.0),
        Rotate(initial_magnitude=init_mag, initial_probability=1.0,),
        ShearX(initial_magnitude=init_mag, initial_probability=1.0),
        ShearY(initial_magnitude=init_mag, initial_probability=1.0),
        MagHorizontalFlip(initial_magnitude=init_mag),
    ]

    # Probabilities are all frozen to the value 1.
    if freeze_prob:
        for op in operations:
            op._probability.requires_grad = False
    return operations


def create_non_geom_operations(init_mag, freeze_prob):
    """ Instatiates a new list of non_geometric operations to create an
    augmentation layer
    """
    operations = [
        Contrast(initial_magnitude=init_mag, initial_probability=1.0),
        Brightness(initial_magnitude=init_mag, initial_probability=1.0),
        SamplePairing(initial_magnitude=init_mag, initial_probability=1.0),
    ]

    # Probabilities are all frozen to the value 1.
    if freeze_prob:
        for op in operations:
            op._probability.requires_grad = False
    return operations


def create_affine_operations(init_mag, freeze_prob):
    """ Instatiates a new list of geometric operations to create an
    augmentation layer
    """
    operations = [
        # Geometric
        TranslateX(initial_magnitude=init_mag, initial_probability=1.0),
        TranslateY(initial_magnitude=init_mag, initial_probability=1.0),
        Rotate(initial_magnitude=init_mag, initial_probability=1.0,),
        ShearX(initial_magnitude=init_mag, initial_probability=1.0),
        ShearY(initial_magnitude=init_mag, initial_probability=1.0),
    ]

    # Probabilities are all frozen to the value 1.
    if freeze_prob:
        for op in operations:
            op._probability.requires_grad = False
    return operations


def create_non_aff_operations(init_mag, freeze_prob):
    """ Instatiates a new list of non-affine operations to create an
    augmentation layer
    """
    operations = [
        Contrast(initial_magnitude=init_mag, initial_probability=1.0),
        Brightness(initial_magnitude=init_mag, initial_probability=1.0),
        SamplePairing(initial_magnitude=init_mag, initial_probability=1.0),
        MagHorizontalFlip(initial_magnitude=init_mag),
    ]

    # Probabilities are all frozen to the value 1.
    if freeze_prob:
        for op in operations:
            op._probability.requires_grad = False
    return operations


def create_identity_operations(init_mag, freeze_prob):
    """ Instatiates a new list with an identity operation for sanity checking
    """
    operations = [
        Identity(initial_magnitude=init_mag, initial_probability=1.0)
    ]

    # Probabilities are all frozen to the value 1.
    if freeze_prob:
        for op in operations:
            op._probability.requires_grad = False
    return operations


OPERATIONS_BUILDERS = {
    "geom": create_geometric_operations,
    "non-geom": create_non_geom_operations,
    "aff": create_affine_operations,
    "non-aff": create_non_aff_operations,
    "id": create_identity_operations,
}


def create_augmentation_layer(
    type="geom",
    init_mag=0.5,
    freeze_weights=False,
    freeze_prob=True,
    temperature=0.01,
):
    """ Instantiates a new augmentation layer
    """
    aug_layer = models.AugmentationLayer(
        augmentations=OPERATIONS_BUILDERS[type](init_mag, freeze_prob),
        temperature=temperature,
    )
    # In order to simplify the optimization, it is possible to freeze
    # weights
    if freeze_weights:
        aug_layer._weights.requires_grad = False
    return aug_layer


def create_augnet_layers(
    init_mag=0.5,
    freeze_weights=False,
    freeze_prob=True,
    n_layers=1,
    types=None,
    temperature=0.01,
):
    """ Creates an augmentation module, made of several augmentation layers
    """

    if types is None:
        aug_module = models.AugmentationModule([
            create_augmentation_layer(
                init_mag=init_mag,
                freeze_weights=freeze_weights,
                freeze_prob=freeze_prob,
                temperature=temperature,
            )
            for _ in range(n_layers)
        ])
    else:
        assert len(types) == n_layers,\
            f"Got {n_layers} layers, but only type f{types}"
        aug_module = models.AugmentationModule([
            create_augmentation_layer(
                init_mag=init_mag,
                freeze_weights=freeze_weights,
                freeze_prob=freeze_prob,
                type=t,
                temperature=temperature,
            )
            for t in types
        ])

    return aug_module


def create_optimal_module(
    init_mag="perf",
    freeze_mags=False,
    temperature=0.01,
):
    """ Instatiates augmentation module for sanity checking, containing
    the optimal baseline augmentations: flip, translate-x, translate-y
    """
    if init_mag == "perf":
        init_mag_flip = 1.
        init_mag_translate = 0.25
    else:
        init_mag_flip = init_mag_translate = float(init_mag)
    flip = MagHorizontalFlip(
        initial_magnitude=init_mag_flip,
        magnitude_scale=0.5,
    )
    aug_layer_1 = models.AugmentationLayer(
        augmentations=[flip],
        temperature=temperature,
    )
    tx = TranslateX(
        initial_probability=1.,
        initial_magnitude=init_mag_translate,
    )
    tx._probability.requires_grad = False
    aug_layer_2 = models.AugmentationLayer(
        augmentations=[tx],
        temperature=temperature,
    )
    ty = TranslateY(
        initial_probability=1.,
        initial_magnitude=init_mag_translate,
    )
    ty._probability.requires_grad = False
    aug_layer_3 = models.AugmentationLayer(
        augmentations=[ty],
        temperature=temperature,
    )

    module = models.AugmentationModule([
        aug_layer_1,
        aug_layer_2,
        aug_layer_3,
    ])
    if freeze_mags:
        for layer in module.layers:
            for op in layer.augmentations:
                op._magnitude.requires_grad = False
    return module


def create_stmx_optimal_module(
    init_mag="perf",
    perf_weights=True,
    freeze_mags=False,
    freeze_weights=False,
    weights_eps=0.,
    temperature=0.01,
):
    """Similar to `create_optimal_module`. Instatiates augmentation module for
    sanity checking, containing 3 geometric layers with optimal weights and
    magnitudes baseline augmentations: flip, translate-x, translate-y

    Parameters
    ----------
    init_mag: float | "perf", optional
        How to initialize magnitues of Hflip and Tx,y. "perf" means using the
        optimal magnitudes correpsonding to the baseline.
    perf_weights: bool, optional
        Whether to use optimal weights.
    freeze_mags : bool, optional
        Freezes magnitudes, by default False
    freeze_weights : bool, optional
        Freezes weights, by default False
    weights_eps : _type_, optional
        Perturbation on weights, by default 0.
    temperature : float, optional
        Used for the softmax.
    """
    aug_module = create_augnet_layers(
        freeze_weights=freeze_weights,
        freeze_prob=True,
        n_layers=3,
        types=["geom"] * 3,
        temperature=temperature,
    )

    # Freeze magnitudes if necessary
    if freeze_mags:
        for layer in aug_module.layers:
            for op in layer.augmentations:
                op._magnitude.requires_grad = False

    if init_mag == "perf":
        init_mag_flip = 1.
        init_mag_translate = 0.25
    else:
        init_mag_flip = init_mag_translate = float(init_mag)

    with torch.no_grad():
        # Set first layer weights to select HFlip
        if perf_weights:
            hflip_weights = torch.zeros(6) + weights_eps
            hflip_weights[5] = 1 - weights_eps
            # The following barbarian expression allows compensate the softmax
            aug_module.layers[0]._weights = torch.nn.Parameter(
                torch.log1p(hflip_weights / 12)
            )

            # Set HFlip magnitude
            aug_module.layers[0].augmentations[5]._magnitude.fill_(
                init_mag_flip)
        else:
            # Set all magnitudes when not using perfect weights
            for operation in aug_module.layers[0].augmentations:
                operation._magnitude.fill_(init_mag_flip)

        # Set t-x and t-y weights on two next layers
        for i in range(2):
            if perf_weights:
                translate_weights = torch.zeros(6) + weights_eps
                translate_weights[i] = 1 - weights_eps
                aug_module.layers[1 + i]._weights = torch.nn.Parameter(
                    torch.log1p(translate_weights / 12)
                )
                aug_module.layers[1 + i].augmentations[i]._magnitude.fill_(
                    init_mag_translate)
            else:
                # Set all magnitudes when not using perfect weights
                for operation in aug_module.layers[i + 1].augmentations:
                    operation._magnitude.fill_(init_mag_translate)

    return aug_module


def fetch_weights_and_mags(model, mag_log=None, w_log=None):
    epoch = 1
    if w_log is not None:
        epoch = w_log.columns[-1] + 1

    new_weights = pd.DataFrame(
        model.aug.weights.clone().detach().cpu().numpy().reshape(-1, 1),
        index=model.aug.operation_names,
        columns=[epoch],
    )

    epoch = 1
    if mag_log is not None:
        epoch = mag_log.columns[-1] + 1

    _new_magnitudes = pd.DataFrame(
        model.aug._magnitudes.clone().detach().cpu().numpy().reshape(-1, 1),
        index=model.aug.operation_names,
        columns=[epoch],
    )
    if w_log is None or mag_log is None:
        return new_weights, _new_magnitudes
    return (
        pd.concat([w_log, new_weights], axis=1),
        pd.concat([mag_log, _new_magnitudes], axis=1)
    )


def prep_dataloaders(args):
    # Set desired fixed data augmentation
    if args.transform == 'none':
        test_transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor(),
        ])
        transform = test_transform
    elif args.transform == "fixed":
        test_transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif args.transform == "ra":
        test_transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform = transforms.Compose([
            transforms.RandAugment(num_ops=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif args.transform == "aa":
        test_transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform = transforms.Compose([
            transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.CIFAR10,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        raise ValueError(f"args.transform not recognized: {args.transform}")

    unsplit_trainset = torchvision.datasets.CIFAR10(
        args.data_dir,
        train=True,
        download=False,
        transform=transform
    )

    unsplit_validset = torchvision.datasets.CIFAR10(
        args.data_dir,
        train=True,
        download=False,
        transform=test_transform,
    )

    testset = torchvision.datasets.CIFAR10(
        args.data_dir,
        train=False,
        download=False,
        transform=test_transform,
    )

    # Make split for training and validation sets
    i_train, i_valid = train_test_split(
        np.arange(len(unsplit_trainset)),
        train_size=0.8,
        random_state=args.seed,
        stratify=[y for _, y in unsplit_trainset]
    )

    # Only use a fraction of the training set
    if args.tr_fraction < 1.:
        i_train,  _ = train_test_split(
            i_train,
            train_size=args.tr_fraction,
            random_state=args.seed,
            stratify=[
                y for i, (_, y) in enumerate(unsplit_trainset) if i in i_train]
        )

    trainset = Subset(unsplit_trainset, i_train)
    validset = Subset(unsplit_validset, i_valid)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size)
    validloader = DataLoader(validset, batch_size=args.test_batch_size)
    testloader = DataLoader(testset, batch_size=args.test_batch_size)
    return trainloader, validloader, testloader


def create_tensorboard(args):
    """Creates tensorboard summary writer attached to a log folder
    IF args request so
    """
    if args.tensorboard:
        logs_dir = join(args.dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return SummaryWriter(logs_dir)
    return None


def log_this(writer, identifier, iter, *args):
    """Logs metric (loss, acc, aug params, ...) into tensorboard if the latter
    if set-up

    Parameters
    ----------
    writer : torch.utils.tensorboard.SummaryWriter
        Tensorboard logger.
    identifier : "aug_params", "train", "valid" or "test"
        Tags what is being logged.
    iter : int
        Epoch or iteration.
    """
    # Do nothing if writer is None
    if writer is not None:
        # Special tratment when logging augmentation parameters
        if identifier == "aug_params":
            metrics = ["weight", "magnitude"]
            for metric, table in zip(metrics, args):
                table = table.loc[:, iter].to_dict()
                writer.add_scalars(
                    f"{metric}",
                    {
                        name: value
                        for name, value in table.items()
                    },
                    iter
                )
        elif identifier == "grads":
            aug_module, averaged_grads = args
            operations_names = aug_module.operation_names
            for metric, tensor in averaged_grads.items():
                writer.add_scalars(
                    metric,
                    {
                        name: value.item()
                        for name, value in zip(operations_names, tensor)
                    },
                    iter
                )
        else:
            table = args[0]
            if identifier == "train":
                # Log the loss and its decomposition for the training set
                metrics = ["loss", "ce", "reg"]
            else:
                metrics = ["loss"]

            writer.add_scalars(
                f"{identifier}-loss",
                {
                    metric: table[f"epoch_{identifier}_{metric}"]
                    for metric in metrics
                },
                iter
            )

            writer.add_scalars(
                "acc",
                {identifier: table[f"epoch_{identifier}_acc"]},
                iter
            )


def train_model(args, dataloaders):
    print(f"Using seed: {args.seed}")
    set_global_rngs(args.seed, all_gpus=True)
    # Set deterministic mode and handle special case of AugNet
    # (kornia is unfortunately not deterministic)
    set_deterministic_mode(det_algos=args.method != "augnet")

    trainloader, validloader, testloader = dataloaders

    # Set desired backbone network
    if args.backbone == "layer13":
        net = models.layer13s(in_channels=3, num_targets=10)
    elif args.backbone == "resnet18":
        net = models.make_resnet18k(num_classes=10)
    else:
        raise ValueError(
            "Unknown value for backbone argument. "
            "Possible values are 'layer13' or 'resnet18'."
            f"Got {args.backbone}."
        )

    # Set desired augmentation layer
    if args.method == 'none':
        print("Training directly with the backbone NN")
        model = net
        root_name = f"baseline_{args.backbone}_"
    else:
        if args.method == "augerino":
            print("Training with Augerino")
            aug_module = models.AugerinoAugModule()
            root_name = f"{args.method}-c{args.ncopies}_{args.backbone}_"
        elif args.method == "augnet":
            print(f"Training with AugNet {args.n_layers} layers")
            root_name = f"{args.method}-c{args.ncopies}-r{args.aug_reg}"
            if args.sanity_check:
                root_name += "-hflip-tx-ty"
                aug_module = create_optimal_module(
                    init_mag=args.init_mag,
                    freeze_mags=args.freeze_mags,
                    temperature=args.temp,
                )
            elif args.sanity_check_weights:
                root_name += "-geom-geom-geom-perf"
                aug_module = create_stmx_optimal_module(
                    init_mag=args.init_mag,
                    perf_weights=args.perf_weights,
                    freeze_mags=args.freeze_mags,
                    freeze_weights=args.freeze_weights,
                    weights_eps=args.weights_eps,
                    temperature=args.temp,
                )
            else:
                if args.l_types is not None:
                    for ltype in args.l_types:
                        root_name += f"-{ltype}"
                else:
                    root_name += f"{args.n_layers}"
                aug_module = create_augnet_layers(
                    n_layers=args.n_layers,
                    types=args.l_types,
                    init_mag=float(args.init_mag),
                    temperature=args.temp,
                )
            root_name += f"_{args.backbone}_"
        else:
            raise ValueError(
                "Unknown value for method argument. "
                "Possible values are 'augnet', 'augerino' or 'none'."
                f"Got {args.method}."
            )
        net = torch.nn.Sequential(
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            net,
        )
        model = models.AugAveragedModel(net, aug_module, ncopies=args.ncopies)
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}.")

    # Fetch datasets
    if args.transform == 'none':
        root_name += "no_trans_"
    elif args.transform == 'fixed':
        root_name += "fixed_trans_"
    elif args.transform == 'ra':
        root_name += "randaug_"
    elif args.transform == 'aa':
        root_name += "autoaug_"
    else:
        raise ValueError(f"args.transform not recognized: {args.transform}")

    # Add seed to the root name to be able to check reproducibility
    # root_name += f"{args.seed}_with-norm_"  # TODO: change this
    root_name += f"{args.seed}_"
    metrics_fname = root_name + "metrics.pkl"
    weights_fname = root_name + "augweights.pkl"
    mags_fname = root_name + "augmags.pkl"
    if args.timeit:
        timer_fname = root_name + "timer.pkl"

    # Create optimizer with common learning rate but different decays for
    # the augmentation layer parameters and for the backbone parameters
    if args.method == 'none':
        optimizers = [torch.optim.SGD(
            model.parameters(),
            weight_decay=args.wd,
            lr=args.lr,
            momentum=0.9,
        )]
    else:
        optimizers = [torch.optim.AdamW(
            [
                {
                    'name': 'model',
                    'params': model.model.parameters(),
                    "weight_decay": args.wd
                },
                {
                    'name': 'aug',
                    'params': model.aug.parameters(),
                    "weight_decay": args.aug_wd
                }
            ], lr=args.lr
        )]

    # Add scheduler (as mentioned un Augerino paper)
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.cosann,
    ) for optimizer in optimizers]

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.to(args.device)
        print("Using Cuda")

    # save init model
    os.makedirs(args.dir, exist_ok=True)
    fname = root_name + "init.pt"
    torch.save(model.state_dict(), args.dir + fname)

    # Create folder for saving checkpoints of every epoch
    if args.save_all:
        checkpoints_dir = args.dir + "all_checkpoints/"
        os.makedirs(args.dir + "all_checkpoints/", exist_ok=True)

    # Set global RNGs
    set_global_rngs(args.seed, all_gpus=True)

    # Start tensorboard if requested
    writer = create_tensorboard(args)

    # Training loop
    training_metrics = list()
    weights = magnitudes = None
    if args.timeit:
        timer = {
            "train": [],
            "test": [],
            "n_train_batches": len(trainloader),
            "n_test_batches": len(testloader),
        }
    for epoch in range(args.epochs):

        tr_loss = valid_loss = test_loss = 0
        tr_ce_loss = valid_ce_loss = test_ce_loss = 0
        tr_acc = valid_acc = test_acc = 0
        if args.method == "augnet":
            averaged_grads = {
                "mag_grad": model.aug._mag_grads.cpu(),
                "weight_grad": model.aug._weight_grads.cpu(),
            }
        model.train()
        if args.timeit:
            start = time()
        for _, data in enumerate(trainloader, 0):

            # zero the parameter gradients
            [optimizer.zero_grad() for optimizer in optimizers]

            # forward
            loss, acc, ce_loss = compute_metrics(
                model, data, use_cuda, epoch, args)

            # backward + optimize
            loss.backward()
            [optimizer.step() for optimizer in optimizers]
            if args.timeit:
                timer["train"].append(time() - start)
            tr_loss += loss.item()
            tr_acc += acc
            tr_ce_loss += ce_loss.item()
            if args.method == "augnet":
                averaged_grads["mag_grad"] += model.aug._mag_grads.cpu()
                averaged_grads["weight_grad"] += model.aug._weight_grads.cpu()
            if args.timeit:
                start = time()

        # Saving learned augmentation parameters
        if args.method == "augnet":
            weights, magnitudes = fetch_weights_and_mags(
                model, mag_log=magnitudes, w_log=weights)
            weights.to_pickle(join(args.dir, weights_fname))
            magnitudes.to_pickle(join(args.dir, mags_fname))
            log_this(writer, "aug_params", epoch + 1, weights, magnitudes)
            averaged_grads["mag_grad"] /= len(trainloader.dataset)
            averaged_grads["weight_grad"] /= len(trainloader.dataset)
            log_this(writer, "grads", epoch + 1, model.aug, averaged_grads)

        # Report training set metrics
        print("Epoch = ", epoch)
        metrics = aggreg_and_print_metric(
            running_loss=tr_loss,
            running_acc=tr_acc,
            running_ce=tr_ce_loss,
            loader=trainloader,
            tag="train",
            metrics={"epoch": epoch},
            writer=writer,
            epoch=epoch + 1,
        )

        # Compute valid and test metrics to track training
        model.eval()
        with torch.no_grad():
            for _, data in enumerate(validloader, 0):
                loss, acc, ce_loss = compute_metrics(
                    model, data, use_cuda, epoch, args)
                valid_loss += loss.item()
                valid_acc += acc
                valid_ce_loss += ce_loss.item()

            metrics = aggreg_and_print_metric(
                running_loss=valid_loss,
                running_acc=valid_acc,
                running_ce=valid_ce_loss,
                loader=validloader,
                tag="valid",
                metrics=metrics,
                writer=writer,
                epoch=epoch + 1,
            )

            for _, data in enumerate(testloader, 0):
                if args.timeit:
                    start = time()
                loss, acc, ce_loss = compute_metrics(
                    model, data, use_cuda, epoch, args)
                if args.timeit:
                    timer["test"].append(time() - start)
                test_loss += loss.item()
                test_acc += acc
                test_ce_loss += ce_loss.item()

            metrics = aggreg_and_print_metric(
                running_loss=test_loss,
                running_acc=test_acc,
                running_ce=test_ce_loss,
                loader=testloader,
                tag="test",
                metrics=metrics,
                writer=writer,
                epoch=epoch + 1,
            )

        print("\n")
        # Store all metrics for saving
        training_metrics.append(metrics)

        [scheduler.step() for scheduler in schedulers]

        # Save updated training metrics
        pd.DataFrame(training_metrics).to_pickle(
            args.dir + metrics_fname
        )

        if args.timeit:
            with open(args.dir + timer_fname, "wb+") as f:
                pickle.dump(timer, f)

        if args.save_all:
            # Save model
            model_fname = root_name + f"e{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoints_dir + model_fname)

    # Save trained model
    model_fname = root_name + "trained.pt"
    torch.save(model.state_dict(), args.dir + model_fname)
    return pd.DataFrame(training_metrics), weights, magnitudes


def make_args_parser():
    parser = argparse.ArgumentParser(description="CIFAR10 experiment")

    parser.add_argument(
        "--dir",
        type=str,
        default='./saved-outputs/',
        help="training directory (default: None)",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default='none',
        help="default transforms, options = 'none' or 'fixed'",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/cifar10",
        help="directory for CIFAR10 dataset",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 128)",
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for validation and test (default: 1000)",
    )

    parser.add_argument(
        "--tr-fraction",
        type=float,
        default=1.,
        help="portion of the training set to use."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="initial learning rate for model (default: 0.1)",
    )
    parser.add_argument(
        "--aug_lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="initial learning rate for augmentation module (default: 5e-4)",
    )
    parser.add_argument(
        "--cosann",
        type=int,
        default=200,
        help="Number of epochs for cosine annealing (default: 200).",
    )
    parser.add_argument(
        "--aug_reg",
        type=float,
        default=0.01,
        help="augmentation regularization weight",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="Number of augmentation layers (default: 1)",
    )
    parser.add_argument(
        "--l_types",
        action='append',
        type=str,
        help="Layer types.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=5e-4,
        metavar="weight_decay",
        help="Model backbone weight decay",
    )
    parser.add_argument(
        "--aug-wd",
        type=float,
        default=0.,
        metavar="weight_decay",
        help="Augmentation weight decay",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 200)",
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
        default=42,
        metavar="N",
        help="Random seed."
    )
    parser.add_argument(
        "--method",
        default="augnet",
        help="Method to use: either 'augnet' (default), 'augerino' or 'none'."
    )
    parser.add_argument(
        "--backbone",
        default="layer13",
        help="Back-bone NN to use: either 'layer13' (default) or 'resnet18'."
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Whether to train augnet with baseline augmentations."
    )
    parser.add_argument(
        "--sanity-check-weights",
        action="store_true",
        help="Whether to train augnet with **softmax** and baseline "
             "augmentations."
    )
    parser.add_argument(
        "--init-mag",
        default=0.5,
        help="Used to initialiaze augmentations magnitudes."
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Whether to log metrics into tensorboard."
    )
    parser.add_argument(
        "--freeze-mags",
        action="store_true",
        help="Whether to use fixed magnitudes."
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether to use fixed weights."
    )
    parser.add_argument(
        "--perf-weights",
        action="store_true",
        help="Whether to init with perfect weights."
    )
    parser.add_argument(
        "--weights-eps",
        type=float,
        default=0.,
        help="Used for sanity check with weights. Pertubation added to optimal"
             " weights."
    )
    parser.add_argument(
        "--checkpoint",
        help="Path used to initialiaze model. If none, creates new model."
    )
    parser.add_argument(
        "--pen-schedule",
        action="store_true",
        help="Whether to schedule penalty weight decrease."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.01,
        help="Temperature of softmax in augmentation layers."
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Whether to checkpoint the model at the end of every epoch"
    )
    parser.add_argument(
        "--timeit",
        action="store_true",
        help="Whether to record iteration times."
    )
    return parser


if __name__ == '__main__':
    parser = make_args_parser()
    args = parser.parse_args()

    dataloaders = prep_dataloaders(args)
    _ = train_model(args, dataloaders)
