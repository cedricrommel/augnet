from pathlib import Path

import pandas as pd

import torch
from torchvision import transforms

from augnet import models
from augnet.utils import set_global_rngs
from train_augerino_new import (
    make_args_parser,
    prep_dataloaders,
    create_augnet_layers,
    compute_metrics,
    aggreg_and_print_metric
)


def create_model(args, ncopies):
    """ Creates AugNet model with desired number of copies
    """
    net = models.make_resnet18k(num_classes=10)

    aug_module = create_augnet_layers(
        n_layers=args.n_layers,
        types=args.l_types,
        init_mag=float(args.init_mag),
        temperature=args.temp,
    )
    net = torch.nn.Sequential(
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        net,
    )
    return models.AugAveragedModel(net, aug_module, ncopies=ncopies)


def eval_test(model, testloader, args):
    """ Runs a single evaluation epoch of the model over test set
    """
    test_loss = 0
    test_ce_loss = 0
    test_acc = 0

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(testloader, 0):
            loss, acc, ce_loss = compute_metrics(
                model, data, use_cuda, 300, args)
            test_loss += loss.item()
            test_acc += acc
            test_ce_loss += ce_loss.item()

        metrics = aggreg_and_print_metric(
            running_loss=test_loss,
            running_acc=test_acc,
            running_ce=test_ce_loss,
            loader=testloader,
            tag="test",
            metrics=None,
            writer=None,
            epoch=None,
        )
    return metrics


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()

    master_path = Path("./saved-outputs/results/")

    # create test set
    _, _, testloader = prep_dataloaders(args)
    # init holding dict
    c_vs_perf = list()

    # loop over ncopies
    for ncopies in [4, 10, 20]:
        # create model
        model = create_model(args, ncopies)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            model = model.to(args.device)

        # loop over seeds
        for seed in [1, 29, 42, 666, 777]:
            # build correct path to weights
            weights_path = (
                master_path /
                f"augnet-c20-non-aff-aff-aff_resnet18_no_trans_{seed}_"
                "trained.pt"
            )

            # load weights
            checkpoint = torch.load(weights_path)
            model.load_state_dict(checkpoint)

            # evaluate on whole test set
            set_global_rngs(seed, all_gpus=True)
            metrics = eval_test(model, testloader, args)

            # store results
            metrics["ncopies"] = ncopies
            metrics["seed"] = seed
            c_vs_perf.append(metrics)

    c_vs_perf_df = pd.DataFrame(c_vs_perf)

    # Save  results
    res_dir = master_path / "c_vs_perf"
    res_dir.mkdir(exist_ok=True)
    c_vs_perf_df.to_pickle(res_dir / "c_vs_perf_analysis.pkl")
