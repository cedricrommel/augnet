import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

import torch

from braindecode.augmentation.transforms import FrequencyShift

from augnet.augmentations.eeg import DiffFrequencyShift
from augnet.augmentations.eeg import DiffFTSurrogate
from augnet.augmentations.eeg import DiffGaussianNoise

from augnet import losses, models
from augnet.utils import set_global_rngs, assess_invariance

from .generate_data import make_dataset


DELTA_F_HZ = 0.5
FREQS = [2, 4, 6, 8]
AMPLITUDE = 1
LENGTH_S = 10
SFREQ = 100


def compute_metrics(model, data, use_cuda, args):
    """ Computes loss and accuracy for given batch
    """
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    # Correct loss function depends on method
    if args.method == "augnet":
        criterion = losses.aug_layer_loss
        kwargs = {
            "model": model,
            "reg": args.reg,
        }
    elif args.method == "none":
        criterion = ce_loss_fn
        kwargs = {}
    else:
        raise ValueError(
                "Unknown value for method argument. "
                "Possible values are 'augnet' or 'none'."
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
        acc = accuracy_score(labels.clone().detach().cpu(), y_pred)
        ce_loss = ce_loss_fn(outputs, labels)
    return loss, acc, ce_loss


def fetch_weights_and_mags(model, metrics):
    new_weights = {}
    new_magnitudes = {}
    for i, layer in enumerate(model.aug.layers):
        weights = layer.weights.clone().detach().cpu().numpy()
        for o, aug in enumerate(layer.augmentations):
            layer_op_name = f"l{i}-{type(aug).__name__}"
            new_weights[f"weight-{layer_op_name}"] = weights[o]
            new_magnitudes[
                f"magnitude-{layer_op_name}"
            ] = aug.magnitude.clone().detach().cpu().item()

    metrics.update(new_weights)
    metrics.update(new_magnitudes)

    return metrics


def aggreg_and_print_metric(running_loss, running_acc, running_ce, loader, tag,
                            metrics=None):
    """ Aggregates running metrics into epoch metrics, stores them in
    dictionary and prints
    """
    if metrics is None:
        metrics = dict()
    metrics[f"epoch_{tag}_loss"] = running_loss / len(loader)
    metrics[f"epoch_{tag}_acc"] = running_acc / len(loader)
    metrics[f"epoch_{tag}_ce"] = running_ce / len(loader)
    metrics[f"epoch_{tag}_reg"] = (running_loss - running_ce) / len(loader)

    print(f"Epoch {tag} loss = ", metrics[f"epoch_{tag}_loss"])
    print(f"Epoch {tag} CE_loss = ", metrics[f"epoch_{tag}_ce"])
    print(f"Epoch {tag} reg = ", metrics[f"epoch_{tag}_reg"])
    print(f"Epoch {tag} acc = ", metrics[f"epoch_{tag}_acc"])
    return metrics


def freeze_augnet_layer_ops_probs(
    operations,
    freeze_prob=True,
):
    if freeze_prob:
        # for aug in aug_layer.transforms:
        for aug in operations:
            aug._probability.requires_grad = False
    return operations


def create_augnet_layer_freq_shift_only(
    sfreq=SFREQ,
    init_mag=0.5,
    freeze_prob=True,
    random_state=None
):
    rng = check_random_state(random_state)

    operations = [
        DiffFrequencyShift(
            initial_magnitude=init_mag,
            initial_probability=1.0,
            random_state=rng,
            sfreq=sfreq,
            mag_range=(0, 2),
        ),
    ]

    return freeze_augnet_layer_ops_probs(
        operations,
        freeze_prob=freeze_prob
    )


def create_augnet_layer_3_tfs(
    sfreq=SFREQ,
    init_mag=0.5,
    freeze_prob=True,
    random_state=None
):
    rng = check_random_state(random_state)

    operations = [
        DiffFrequencyShift(
            initial_magnitude=init_mag,
            initial_probability=1.0,
            random_state=rng,
            sfreq=sfreq,
            mag_range=(0, 2),
        ),
        DiffFTSurrogate(
            initial_magnitude=init_mag,
            initial_probability=1.0,
            random_state=rng,
            mag_range=(0, 0.1),
        ),
        DiffGaussianNoise(
            initial_magnitude=init_mag,
            initial_probability=1.0,
            random_state=rng,
            mag_range=(0, 0.1),
        ),
    ]

    return freeze_augnet_layer_ops_probs(
        operations,
        freeze_prob=freeze_prob
    )


OPERATIONS_BUILDERS = {
    "freq": create_augnet_layer_freq_shift_only,
    "3tfs": create_augnet_layer_3_tfs,
}


def create_augmentation_layer(
    type="3tfs",
    init_mag=0.5,
    freeze_weights=False,
    freeze_prob=True,
    random_state=None,
):
    """ Instantiates a new augmentation layer
    """
    aug_layer = models.AugmentationLayer(
        augmentations=OPERATIONS_BUILDERS[type](
            init_mag=init_mag,
            freeze_prob=freeze_prob,
            random_state=random_state,
        ),
        temperature=0.01,
        data_dim=2,
    )
    # In order to simplify the optimization, it is possible to freeze
    # weights
    if freeze_weights:
        aug_layer._weights.requires_grad = False
    return aug_layer


def create_augnet_module(
    init_mag=0.5,
    freeze_weights=False,
    freeze_prob=True,
    n_layers=1,
    types=None,
    random_state=None,
):
    """ Creates an augmentation module, made of several augmentation layers
    """

    if types is None:
        aug_module = models.AugmentationModule([
            create_augmentation_layer(
                init_mag=init_mag,
                freeze_weights=freeze_weights,
                freeze_prob=freeze_prob,
                random_state=random_state,
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
                random_state=random_state,
            )
            for t in types
        ])

    return aug_module


def trainer(trainloader, testloader, model, args):
    training_metrics = list()
    # logger = list()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd,
    )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.to(args.device)

    for epoch in range(args.epochs):
        tr_loss = test_loss = 0
        tr_acc = test_acc = 0
        tr_ce_loss = test_ce_loss = 0
        model.train()
        for data in trainloader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            loss, acc, ce_loss = compute_metrics(model, data, use_cuda, args)

            # backward + optimize
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_acc += acc
            tr_ce_loss += ce_loss.item()

        # Report training set metrics
        print("Epoch = ", epoch)
        metrics = aggreg_and_print_metric(
            tr_loss,
            tr_acc,
            tr_ce_loss,
            trainloader,
            tag="train",
            metrics={"epoch": epoch},
        )

        if args.method == "augnet":
            metrics = fetch_weights_and_mags(model, metrics)

        # Compute valid and test metrics to track training
        model.eval()
        with torch.no_grad():
            for _, data in enumerate(testloader, 0):
                loss, acc, ce_loss = compute_metrics(
                    model, data, use_cuda, args)
                test_loss += loss.item()
                test_acc += acc
                test_ce_loss += ce_loss.item()

            metrics = aggreg_and_print_metric(
                test_loss,
                test_acc,
                test_ce_loss,
                testloader,
                tag="test",
                metrics=metrics,
            )

        print("\n")
        # Store all metrics for saving
        training_metrics.append(metrics)
    return pd.DataFrame(training_metrics)


def main(args):
    set_global_rngs(args.seed)

    # Set backbone network
    if args.backbone == "cnn":
        first_filter_width_s = args.n_periods_in_rf * 1 / np.min(FREQS)
        first_filter_width = int(first_filter_width_s * SFREQ)
        net = models.FreqNet(
            first_filter_width=first_filter_width,
            n_filters=args.num_channels,
            n_layers=args.num_layers,
        )
    elif args.backbone == "mlp":
        net = models.SimpleMLP(
            n_neurons=args.num_channels,
            n_layers=args.num_layers,
            in_shape=int(SFREQ * LENGTH_S),
        )
    else:
        raise ValueError(
            "Unknown value for backbone argument. "
            "Possible values are 'cnn' or 'mlp'."
            f"Got {args.backbone}."
        )
    print(
        f"Using {args.backbone} backbone, with {args.num_channels} neurons"
        f" and {args.num_layers} layers."
    )

    # Set desired augmentation layer
    root_name = f"{args.method}"
    if args.method == 'none':
        print("Training directly with the backbone NN")
        model = net
        # root_name = f"baseline_"
    elif args.method == 'augnet':
        aug_module = create_augnet_module(
            n_layers=args.n_aug_layers,
            types=args.l_types,
            init_mag=args.init_mag,
            random_state=args.seed,
        )
        if args.l_types is not None:
            for ltype in args.l_types:
                root_name += f"-{ltype}"
        else:
            root_name += f"-{args.n_aug_layers}"
        model = models.AugAveragedModel(net, aug_module, ncopies=args.ncopies,
                                        ncopies_tr=args.ncopies_tr)
        root_name += f"-c{args.ncopies}"
        if args.ncopies_tr > 1:
            root_name += f"{args.ncopies_tr}"
    else:
        raise ValueError(
            "Unknown value for method argument. "
            "Possible values are 'augnet' or 'none'."
            f"Got {args.method}."
        )

    root_name += f"_{args.backbone}{args.num_layers}{args.num_channels}_"
    root_name += f"noise{args.noise}_"

    # Add seed to the root name to be able to check reproducibility
    root_name += f"{args.seed}_"

    # Create transform when applicable
    transform = None
    if args.augment:
        transform = FrequencyShift(
            probability=0.5,
            sfreq=SFREQ,
            max_delta_freq=DELTA_F_HZ,
            random_state=args.seed
        )
        root_name += "augmented_"

    # Create dataset
    trainloader, testloader = make_dataset(
        args.n_per_class,
        args.batch_size,
        train_size=args.train_size,
        delta_f=DELTA_F_HZ,
        freqs=FREQS,
        a=AMPLITUDE,
        length_s=LENGTH_S,
        sfreq=SFREQ,
        noise=args.noise,
        transform=transform,
        random_state=args.seed,
        verbose=False,
    )

    # Train model
    set_global_rngs(args.seed)
    training_metrics = trainer(trainloader, testloader, model, args)

    # Save trained model and training metrics
    model_fname = root_name + "trained.pt"
    os.makedirs(args.dir, exist_ok=True)
    torch.save(model.state_dict(), join(args.dir, model_fname))

    metrics_fname = root_name + "metrics.pkl"
    training_metrics = pd.DataFrame(training_metrics).to_pickle(
        join(args.dir, metrics_fname)
    )

    # Assess invariance to the correct transform
    freqshift_tf = FrequencyShift(
        probability=1.,  # <---
        sfreq=SFREQ,
        max_delta_freq=DELTA_F_HZ,
        random_state=args.seed
    )
    model_invariance = assess_invariance(
        model=model,
        dataloader=testloader,
        transform=freqshift_tf,
        seed=args.seed,
        use_embedding=args.use_embedding,
        method=args.method,
        device=args.device,
    )
    if args.use_embedding:
        root_name += 'emb_'
    invariances_fname = root_name + 'model_invariance.pt'
    torch.save(model_invariance, join(args.dir, invariances_fname))
    print(f"Median invariance: {model_invariance.median().item()}")


def make_parser():
    parser = argparse.ArgumentParser(description="mario-iggy experiment")

    parser.add_argument(
        "--dir",
        type=str,
        default='./saved-outputs',
        help="training directory (default: None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size (default: 128)",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=150,
        metavar="N",
        help="number of samples per class (default: 150)",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=2/3,
        help="Size of training set, wrt test set (default: 2/3)",
    )

    parser.add_argument(
        "--num_channels",
        type=int,
        default=1,
        help="number of channels for network (default: 1)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="number of hidden layers for network (default: 1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        metavar="LR",
        help="initial learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        metavar="weight_decay",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--ncopies",
        type=int,
        default=4,
        metavar="N",
        help="number of augmentations in network at inference (defualt: 4)"
    )
    parser.add_argument(
        "--ncopies-tr",
        type=int,
        default=1,
        metavar="N",
        help="number of augmentations in network at training (defualt: 1)"
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=0.3,
        help="regularization weight (default: 0.3)"
    )
    parser.add_argument(
        "--init-mag",
        type=float,
        default=0.,
        help="Initial augmentations magnitude (default: 0.)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Turns on verbose."
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda:1",
        help="Device (default: cuda:1)"
    )
    parser.add_argument(
        "--method",
        default="augnet",
        help="Method to use: either 'augnet' (default) or 'none'."
    )
    parser.add_argument(
        "--n_aug_layers",
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
        "--backbone",
        default="cnn",
        help="Back-bone NN to use: either 'cnn' (default) or 'mlp'."
    )
    parser.add_argument(
        "--n-periods-in-rf",
        type=int,
        default=2,
        help="Minimum number of periods captured by first layer receptive "
             "field in case backbone is cnn (default: 2)."
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.,
        help="Std of the noise added to the signals (default: 0)."
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Whether to apply the perfect augmentation."
    )
    parser.add_argument(
        "--use-embedding",
        action="store_true",
        help="Whether to compute invariance with embeddings."
    )
    return parser


if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()

    main(args)
