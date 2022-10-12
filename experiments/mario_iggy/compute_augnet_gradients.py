import os
from os.path import join
import argparse

import pandas as pd
from tqdm import tqdm

import torch

from augnet import models, losses
from augnet.utils import set_global_rngs

from .generate_data import generate_mario_data
from augnet_training import create_augnet_layer


def pretrain_model(
    model,
    trainloader,
    reg=0.5,
    epochs=20,
    criterion=losses.aug_layer_loss,
    lr=5e-4,
    decay=1.,
    device="cuda:1",
):
    """Pretains AugNet model for 100 iterations
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    print(optimizer)
    print(f"reg={reg}")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device(device)
        model = model.to(device)

    logger = []

    count = 0
    for _ in tqdm(range(epochs)):  # loop over the dataset multiple times
        for _, data in enumerate(trainloader):
            count += 1
            if count < 100:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                if use_cuda:
                    # inputs, labels = inputs.cuda(), labels.cuda()
                    inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels, model,
                                 reg=reg)
                loss.backward()
                optimizer.step()

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
                log += [loss.item(), count]
                logger.append(log)
    logdf = pd.DataFrame(logger)
    logdf.columns = ['prob.' + str(i) for i in range(5)] +\
        ['mag.' + str(i) for i in range(5)] +\
        ['w.' + str(i) for i in range(5)] +\
        ['grad' + str(i) for i in range(15)] +\
        ['loss', 'count']
    logdf = logdf.reset_index()
    return model, logdf


def compute_gradients(
    model,
    trainloader,
    reg,
    criterion=losses.aug_layer_loss,
    device="cuda:1"
):
    # train_widths = model.aug.width.data.clone()
    train_mags = [op.magnitude.item() for op in model.aug.augmentations]
    # widths = torch.linspace(-5, 5, 20)
    mags = torch.linspace(0, 1, 20)

    grad_logger = []
    for batch_idx, batch in tqdm(enumerate(trainloader), total=25):
        if batch_idx < 25:
            x, y = batch
            x, y = x.to(device), y.to(device)
            for idx, ww in enumerate(mags):
                temp_mags = train_mags.copy()
                temp_mags[2] = ww
                for i, m in enumerate(temp_mags):
                    model.aug.augmentations[i]._magnitude.data = torch.Tensor(
                        [m],
                    ).to(model.aug.augmentations[i]._magnitude.device)
                preds = model(x)

                acc = torch.sum(torch.argmax(preds, -1) == y)

                loss = criterion(model(x), y, model, reg=reg)

                loss.backward()

                rot_mag_grad = torch.zeros(1)
                if model.aug.augmentations[2]._magnitude.grad is not None:
                    rot_mag_grad = model.aug.augmentations[2]._magnitude.grad

                grad_logger.append([
                    batch_idx,
                    model.aug.augmentations[2].magnitude.item(),
                    rot_mag_grad.item(),
                    acc.item(),
                    loss.item()
                ])
    grad_logger = pd.DataFrame(grad_logger)
    grad_logger.columns = ['idx', 'magnitude', 'grad', 'acc', 'loss']
    grad_logger['acc'] /= 128.
    return model, grad_logger


def main(args):
    set_global_rngs(args.seed)

    trainloader, _ = generate_mario_data(
        ntrain=args.ntrain, ntest=args.ntest, batch_size=args.batch_size,
        dpath="experiments/mario_iggy/"
    )

    aug_layer = create_augnet_layer(
        1/8,
        freeze_weights=False
    )

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
        pretrained_model, _ = pretrain_model(
            augnet_model,
            trainloader,
            reg=args.reg,
            epochs=args.epochs,
            criterion=loss,
            lr=args.lr,
            decay=args.wd,
            device=args.device,
        )
    except RuntimeError:
        set_global_rngs(args.seed)
        pretrained_model, _ = pretrain_model(
            augnet_model,
            trainloader,
            reg=args.reg,
            epochs=args.epochs,
            criterion=loss,
            lr=args.lr,
            decay=args.wd,
            device=args.device,
        )

    _, grad_logger = compute_gradients(
        pretrained_model,
        trainloader,
        reg=args.reg,
        criterion=loss,
        device=args.device,
    )

    os.makedirs(args.dir, exist_ok=True)
    grad_logger.to_pickle(join(args.dir, f"{args.prefix}_grads_logger.pkl"))


if __name__ == '__main__':

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
    args = parser.parse_args()

    main(args)
