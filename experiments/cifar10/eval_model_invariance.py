from pathlib import Path

from tqdm import tqdm

import torch
from torchvision import transforms

import pandas as pd

from augnet import models
from augnet.utils import assess_invariance

from augnet.augmentations.vision import MagHorizontalFlip
from augnet.augmentations.vision import TranslateX, TranslateY
from augnet.augmentations.vision import Rotate
from augnet.augmentations.vision import Brightness

from train_augerino_new import (
    prep_dataloaders,
    create_augnet_layers,
    make_args_parser,
)


def create_model(args):
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
    model = models.AugAveragedModel(net, aug_module, ncopies=args.ncopies)
    return model.to(args.device)


def evaluate_invariance(
    model,
    transform,
    dataloader,
    seed,
    use_embedding,
    device
):
    meta_invariances = assess_invariance(
        model=model,
        dataloader=dataloader,
        transform=transform,
        seed=seed,
        use_embedding=use_embedding,
        method="augnet",
        device=device,
    )
    trunk_invariances = assess_invariance(
        model=model.model,
        dataloader=dataloader,
        transform=transform,
        seed=seed,
        use_embedding=use_embedding,
        method="augnet",
        device=device,
    )
    results = {
        "augnet invariance": meta_invariances,
        "f invariance": trunk_invariances
    }
    return pd.DataFrame(results)


def evaluate_invariance_baseline(
    model,
    transform,
    dataloader,
    seed,
    use_embedding,
    device,
):
    trunk_invariances = assess_invariance(
        model=model,
        dataloader=dataloader,
        transform=transform,
        seed=seed,
        use_embedding=use_embedding,
        method="none",
        device=device,
    )
    results = {
        "f invariance": trunk_invariances
    }
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = make_args_parser()
    parser.add_argument(
        "--use-embedding",
        action="store_true",
        help="Whether to compute invariance with embeddings."
    )
    args = parser.parse_args()

    transforms_to_test_seed29 = [
        MagHorizontalFlip(
            initial_magnitude=1.,
        ).to(args.device),
        TranslateY(
            initial_magnitude=0.5,
            initial_probability=1.0,
        ).to(args.device),
        Rotate(
            initial_magnitude=1.0,
            initial_probability=1.0,
        ).to(args.device),
    ]

    transforms_to_test_seed1 = [
        Brightness(
            initial_magnitude=1.0,
            initial_probability=1.0,
        ).to(args.device),
        TranslateX(
            initial_magnitude=1.0,
            initial_probability=1.0,
        ).to(args.device),
        TranslateY(
            initial_magnitude=1.0,
            initial_probability=1.0,
        ).to(args.device),
    ]

    transforms_to_test = {
        1: transforms_to_test_seed1,
        29: transforms_to_test_seed29,
    }

    trainloader, _, testloader = prep_dataloaders(args)

    master_dir = Path("./saved-outputs/results/")
    checkpoints_dir = master_dir / "all_checkpoints"

    # Init storage
    invariance_evol = list()

    # Create model on cuda
    model = create_model(args)

    # Epochs to evaluate on:
    eval_epochs = list(range(11, 30, 10)) + list(range(1, 300, 30))

    # Loop over seeds
    for seed, selected_transforms in transforms_to_test.items():
        # Loop over epochs
        for epoch in eval_epochs:
            # Load weights
            weights_path = (
                checkpoints_dir /
                "augnet-c20-non-aff-aff-aff_resnet18_no_trans_"
                f"{seed}_e{epoch}.pt"
            )
            checkpoint = torch.load(weights_path)
            model.load_state_dict(checkpoint)

            # Loop over transforms
            for transform in tqdm(
                selected_transforms,
                desc=f"epoch={epoch}"
            ):
                # Evaluate invariance wrt transform
                invariance_metrics = evaluate_invariance(
                    model=model,
                    transform=transform,
                    dataloader=testloader,
                    seed=args.seed,
                    use_embedding=args.use_embedding,
                    device=args.device,
                )
                invariance_metrics["epoch"] = epoch
                invariance_metrics["seed"] = seed
                invariance_metrics["transform"] = type(transform).__name__
                invariance_metrics["method"] = "augnet"

                # Store
                invariance_evol.append(invariance_metrics)

    # Save to disk (preliminar)
    invariance_evol_df = pd.concat(invariance_evol, ignore_index=True)
    inv_exp_dir = master_dir / "invariance_study"
    inv_exp_dir.mkdir(exist_ok=True)
    invariance_evol_df.to_pickle(inv_exp_dir / "invariance_evol.pkl")

    # XXX
    args.transform = "fixed"
    args.method = "none"

    # Create model on cuda
    base_model = models.make_resnet18k(num_classes=10)
    base_model.to(args.device)

    # Loop over seeds
    for seed, selected_transforms in transforms_to_test.items():
        # Loop over epochs
        for epoch in eval_epochs:
            # Load weights
            weights_path = (
                checkpoints_dir /
                f"baseline_resnet18_fixed_trans_{seed}_e{epoch}.pt"
            )
            checkpoint = torch.load(weights_path)
            base_model.load_state_dict(checkpoint)

            # Loop over transforms
            for transform in tqdm(
                selected_transforms,
                desc=f"epoch={epoch}"
            ):
                # Evaluate invariance wrt transform
                invariance_metrics = evaluate_invariance_baseline(
                    model=base_model,
                    transform=transform,
                    dataloader=testloader,
                    seed=args.seed,
                    use_embedding=args.use_embedding,
                    device=args.device,
                )
                invariance_metrics["epoch"] = epoch
                invariance_metrics["seed"] = seed
                invariance_metrics["transform"] = type(transform).__name__
                invariance_metrics["method"] = "baseline"

                # Store
                invariance_evol.append(invariance_metrics)

    invariance_evol_df = pd.concat(invariance_evol, ignore_index=True)
    invariance_evol_df.to_pickle(inv_exp_dir / "invariance_evol.pkl")
