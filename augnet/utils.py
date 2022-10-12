import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from augnet import models

from torchdiffeq import odeint


def expm(A, rtol=1e-4):
    """ assumes A has shape (bs,d,d)
        returns exp(A) with shape (bs,d,d)
    """
    identity = torch.eye(
        A.shape[-1],
        device=A.device,
        dtype=A.dtype
    )[None].repeat(A.shape[0], 1, 1)
    return odeint(
        lambda t, x: A @ x,
        identity,
        torch.tensor([0., 1.]).to(A.device, A.dtype),
        rtol=rtol
    )[-1]


def set_global_rngs(seed, all_gpus=False):
    torch.manual_seed(seed)
    if all_gpus:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_deterministic_mode(det_algos=False):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(det_algos)
    torch.backends.cudnn.benchmark = False


def compute_cos_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Return cos distance defined as 1 - cos similarity"""
    return 1 - F.cosine_similarity(x1, x2)


def get_model_without_last_layer(model):
    new_model = deepcopy(model)
    new_model.embedding = True
    return new_model


def get_aug_model_without_last_layer(model):
    # new_model = torch.nn.Sequential(*(list(model.model.children())[:-1]))
    new_model = get_model_without_last_layer(model.model)
    return models.AugAveragedModel(
        new_model,
        deepcopy(model.aug),
        model.ncopies
    )


def inv_measure(X, embedding, transform):
    with torch.no_grad():
        Z = embedding(X)
        Z_t = embedding(transform(X))

    perms = torch.randperm(Z.shape[0])
    Z_shuffled = Z[perms, :]
    baseline = compute_cos_distance(Z, Z_shuffled)
    transformed_distance = compute_cos_distance(Z, Z_t)
    # for numerical stability
    epsilon = 1e-8
    return torch.divide(
        baseline - transformed_distance, baseline + epsilon
    )


def assess_invariance(
    model,
    dataloader,
    transform,
    seed,
    use_embedding,
    method,
    device,
):
    set_global_rngs(seed)

    av_invariance = list()
    model.eval()

    if use_embedding:
        if method == "none":
            embedding = get_model_without_last_layer(model)
        else:
            embedding = get_aug_model_without_last_layer(model)
    else:
        embedding = model

    for X, _ in dataloader:
        X = X.to(device)
        batch_inv = inv_measure(X, embedding, transform)
        av_invariance.append(batch_inv)
    return torch.cat(av_invariance).cpu()
