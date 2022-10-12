import os
from os.path import join
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import torch

from augnet import models, losses
from augnet.utils import set_global_rngs

from experiments.mario_iggy.generate_data import generate_mario_data


def trainer(
    model,
    trainloader,
    reg=0.05,
    epochs=20,
    criterion=losses.unif_aug_loss,
    lr=0.01,
    decay=0.,
    device="cuda:1",
    verbose=False
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device(device)
        model = model.to(device)

    logger = []
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    softplus = torch.nn.Softplus()

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

            log = softplus(model.aug.width).tolist()
            log += model.aug.width.grad.data.tolist()
            log += [loss.item()]
            log += [ce_loss_fn(outputs, labels).item()]
            log += [acc.item()]

            logger.append(log)

            if verbose:
                print(model.aug.width[2].item() / 2 * 180 / np.pi)

    logdf = pd.DataFrame(logger)
    logdf.columns = ['width' + str(i) for i in range(6)] +\
        ['grad' + str(i) for i in range(6)] +\
        ['loss', 'CE_loss', 'acc']
    logdf["acc"] /= 128.
    logdf = logdf.reset_index()
    return logdf


def plot_shade(logger, ax, color, label="", alpha=0.1, lwd=0.):
    ax.fill_between(logger.index, logger['lowbd'], logger['upbd'],
                    alpha=alpha, color=color,
                    linewidth=lwd)
    sns.lineplot(
        x=logger.index,
        y='lowbd',
        color=color,
        data=logger,
        label=label
    )
    sns.lineplot(
        x=logger.index,
        y='upbd',
        color=color,
        data=logger
    )


def main(args):
    set_global_rngs(args.seed)

    trainloader, _ = generate_mario_data(
        ntrain=args.ntrain, ntest=args.ntest, batch_size=args.batch_size,
        dpath=args.data_path,
    )

    if args.init == "lower":
        init_magnitude = np.pi / 8
    elif args.init == "higher":
        init_magnitude = 3 * np.pi / 8
    else:
        raise ValueError(
            "Unknown value for 'init'."
            f"Possible values are 'lower' or 'higher'. Got {args.init}."
        )

    augerino = models.AugerinoAugModule()

    # Create neural net back-bone, made of 5-layer CNN where the number of
    # channels is doubled at each layer.
    net = models.SimpleConv(c=args.num_channels, num_classes=4)

    model = models.AugAveragedModel(
        net,
        augerino,
        ncopies=args.ncopies
    )

    init_magnitude = np.log(np.exp(init_magnitude * 2) - 1)
    start_widths = torch.ones(6) * init_magnitude
    model.aug.set_width(start_widths)

    try:
        augnet_logger = trainer(
            model=model,
            trainloader=trainloader,
            reg=args.reg,
            criterion=losses.unif_aug_loss,
            epochs=args.epochs,
            lr=args.lr,
            decay=args.wd,
            device=args.device,
            verbose=args.verbose,
        )
    except RuntimeError:
        # There seems to be a problem with kornia that makes the first try to
        # train the model raise a CUDA error. The latter is not well documented
        # and it seems tha tjust relaunching the exp work...
        augnet_logger = trainer(
            model=model,
            trainloader=trainloader,
            reg=args.reg,
            criterion=losses.unif_aug_loss,
            epochs=args.epochs,
            lr=args.lr,
            decay=args.wd,
            device=args.device,
            verbose=args.verbose,
        )

    os.makedirs(args.dir, exist_ok=True)
    torch.save(
        model.state_dict(),
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
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.,
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
        default=0.05,
        help="regularization weight (default: 0.05)"
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
