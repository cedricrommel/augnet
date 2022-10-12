import os
from os.path import join
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from augnet import models, losses
from augnet.utils import set_global_rngs
from augnet.augmentations.vision import TranslateX, TranslateY
from augnet.augmentations.vision import ShearX, ShearY
from augnet.augmentations.vision import Rotate

from experiments.mario_iggy.generate_data import generate_mario_data


def trainer(
    model,
    trainloader,
    reg=0.5,
    epochs=20,
    criterion=losses.aug_layer_loss,
    lr=5e-4,
    decay=1.,
    device="cuda:1",
    verbose=False
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    print(optimizer)
    print(f"reg={reg}")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device(device)
        model = model.to(device)

    logger = []
    ce_loss = torch.nn.CrossEntropyLoss()

    for _ in tqdm(range(epochs)):  # loop over the dataset multiple times
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if use_cuda:
                inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(
                outputs, labels, model, reg=reg
            )
            loss.backward()
            optimizer.step()

            acc = torch.sum(torch.argmax(outputs, -1) == labels)

            log = [op.probability.item() for op in model.aug.augmentations]
            log += [op.magnitude.item() for op in model.aug.augmentations]
            log += model.aug.weights.tolist()
            # TODO: These should be properties of AugmentationLayer
            log += [
                op._probability.grad.item()
                if op._probability.grad is not None else 0.
                for op in model.aug.augmentations
            ]
            log += [
                op._magnitude.grad.item()
                if op._magnitude.grad is not None else 0.
                for op in model.aug.augmentations
            ]
            log += model.aug._weights.grad.tolist() \
                if model.aug._weights.grad is not None \
                else torch.zeros_like(model.aug.weights).tolist()
            log += [loss.item()]
            log += [ce_loss(outputs, labels).item()]
            log += [acc.item()]

            logger.append(log)

            if verbose:
                av_other_mag = np.mean([
                    op.magnitude.item()
                    for i, op in enumerate(model.aug.augmentations)
                    if i != 2
                ])
                av_other_weights = np.mean([
                    w
                    for i, w in enumerate(
                        model.aug.weights.detach().cpu().numpy()
                    )
                    if i != 2
                ])
                print(
                    f"angle: {model.aug.augmentations[2].magnitude.item()} | "
                    f"rot w: {model.aug.weights[2]} | "
                    f"av. other mags: {av_other_mag} | "
                    f"av. other weights: {av_other_weights}"
                )

    logdf = pd.DataFrame(logger)
    logdf.columns = ['prob.' + str(i) for i in range(5)] +\
        ['mag.' + str(i) for i in range(5)] +\
        ['w.' + str(i) for i in range(5)] +\
        ['grad' + str(i) for i in range(15)] +\
        ['loss', 'CE_loss', 'acc']
    logdf["acc"] /= 128.
    logdf = logdf.reset_index()
    return logdf


def create_augnet_layer(init_mag=1., freeze_weights=False, freeze_prob=True):
    operations = [
        TranslateX(initial_magnitude=init_mag, initial_probability=1.0),
        TranslateY(initial_magnitude=init_mag, initial_probability=1.0),
        Rotate(
            initial_magnitude=init_mag,
            initial_probability=1.0,
            magnitude_scale=180
        ),
        ShearX(initial_magnitude=init_mag, initial_probability=1.0),
        ShearY(initial_magnitude=init_mag, initial_probability=1.0),
    ]

    aug_layer = models.AugmentationLayer(
        augmentations=operations, temperature=0.01
    )

    # In order to simplify the optimization, it is possible to freeze
    # weights (and probabilities, but those should always be frozen at 1.)
    if freeze_weights:
        aug_layer._weights.requires_grad = False
    if freeze_prob:
        for op in aug_layer.augmentations:
            op._probability.requires_grad = False
    return aug_layer


def main(args):
    set_global_rngs(args.seed)

    trainloader, _ = generate_mario_data(
        ntrain=args.ntrain, ntest=args.ntest, batch_size=args.batch_size,
        dpath=args.data_path,
    )

    if args.init == "lower":
        init_magnitude = 1 / 8
    elif args.init == "higher":
        init_magnitude = 1 / 2
    else:
        raise ValueError(
            "Unknown value for 'init'."
            f"Possible values are 'lower' or 'higher'. Got {args.init}."
        )

    print(f"init with {init_magnitude}")
    aug_layer = create_augnet_layer(
        init_magnitude,
        freeze_weights=False
    )

    # Create neural net back-bone, made of 5-layer CNN where the number of
    # channels is doubled at each layer.
    net = models.SimpleConv(c=args.num_channels, num_classes=4)

    augnet_model = models.AugAveragedModel(
        net,
        aug_layer,
        ncopies=args.ncopies
    )

    if args.pen == "correct":
        loss = losses.aug_layer_loss
    elif args.pen == "incomplete":
        print("Using partial penalty")
        loss = losses.partial_aug_layer_loss
    else:
        raise ValueError(
            "Unknown value for 'pen'. Possible values are 'correct',"
            f"or 'incomplete'. Got {args.pen}."
        )

    try:
        set_global_rngs(args.seed)
        augnet_logger = trainer(
            model=augnet_model,
            trainloader=trainloader,
            reg=args.reg,
            criterion=loss,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            decay=args.wd,
            verbose=args.verbose,
        )
    except RuntimeError:
        # There seems to be a problem with kornia that makes the first try to
        # train the model raise a CUDA error. The latter is not well documented
        # and it seems tha tjust relaunching the exp work...
        augnet_logger = trainer(
            model=augnet_model,
            trainloader=trainloader,
            reg=args.reg,
            criterion=loss,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            decay=args.wd,
            verbose=args.verbose,
        )

    os.makedirs(args.dir, exist_ok=True)
    torch.save(
        augnet_model.state_dict(),
        join(args.dir, f"{args.prefix}_model.pt")
    )
    augnet_logger.to_pickle(join(args.dir, f"{args.prefix}_logger.pkl"))
    return


def make_parser():
    parser = argparse.ArgumentParser(description="mario-iggy experiment")

    parser.add_argument(
        "--dir",
        type=str,
        default='./saved-outputs',
        help="training directory (default: None)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default='mario',
        help="prefix to use to save model and logs (default: 'mario')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 128)",
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=10000,
        metavar="N",
        help="number of training points (default: 10000)",
    )
    parser.add_argument(
        "--ntest",
        type=int,
        default=5000,
        metavar="N",
        help="number of test points (default: 5000)",
    )

    parser.add_argument(
        "--num_channels",
        type=int,
        default=32,
        help="number of channels for network (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="initial learning rate (default: 5e-4)",
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
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--pen",
        default="correct",
        help="Augmentation penalty to use:"
             "either 'correct' (default), 'incomplete'",
    )
    parser.add_argument(
        "--init",
        default="lower",
        help="Whether to initialize with an angle 'lower' (default)"
             "or 'higher' than pi/4.",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=25,
        metavar="N",
        help="save frequency (default: 25)",
    )
    parser.add_argument(
        "--ncopies",
        type=int,
        default=1,
        metavar="N",
        help="number of augmentations in network (defualt: 1)"
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=0.5,
        help="regularization weight (default: 0.5)"
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
        "--data_path",
        default="",
        help="Where to look for the initial picture to use for data generation"
    )
    return parser


if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()

    main(args)
