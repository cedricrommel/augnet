import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


def safe_unif_aug_loss(
    outputs,
    labels,
    model,
    base_loss_fn=torch.nn.CrossEntropyLoss(),
    reg=0.01
):
    """Safe penalized loss for Augerino

    This loss has the same shape as `unif_aug_loss`, but shuts down when one
    of the transforms' magntude reaches a threshold (10 here). It is used by
    Augerino in the CIFAR10 experiment for example.

    Parameters
    ----------
    outputs : torch.Tensor
        Augmentation network output.
    labels : torch.Tensor
        Ground truth.
    model : torch.nn.Module
        Augerino model.
    base_loss_fn : callable, optional
        Loss function to be penalized. By default torch.nn.CrossEntropyLoss().
    reg : float, optional
        Parameter setting the strength of the regularization. By default 0.1.

    Returns
    -------
    torch.Tensor - size (1,)
        Regularized loss.
    """
    base_loss = base_loss_fn(outputs, labels)
    sp = torch.nn.Softplus()
    width = sp(model.aug.width)
    aug_loss = (width).norm()
    # shuts down regularizer when one of the transforms' magnitude reaches 10
    shutdown = torch.all(width < 10)

    return base_loss - reg * aug_loss * shutdown


def unif_aug_loss(
    outputs,
    labels,
    model,
    base_loss_fn=torch.nn.CrossEntropyLoss(),
    reg=0.1
):
    """Penalized loss for Augerino

    It biases the model toward broad augmentation distributions by maximizing
    the L2-norm of the augmentations range vector.

    Parameters
    ----------
    outputs : torch.Tensor
        Augmentation network output.
    labels : torch.Tensor
        Ground truth.
    model : torch.nn.Module
        Augerino model.
    base_loss_fn : callable, optional
        Loss function to be penalized. By default torch.nn.CrossEntropyLoss().
    reg : float, optional
        Parameter setting the strength of the regularization. By default 0.1.

    Returns
    -------
    torch.Tensor - size (1,)
        Regularized loss.
    """

    base_loss = base_loss_fn(outputs, labels)

    sp = torch.nn.Softplus()
    width = sp(model.aug.width)

    # XXX: While the paper says their regularizer is L2-norm squared,
    # it is acually the L2-norm
    aug_loss = (width).norm()

    return base_loss - reg * aug_loss


def aug_layer_loss(
    outputs,
    labels,
    model,
    base_loss_fn=torch.nn.CrossEntropyLoss(),
    reg=0.5
):
    """Penalized loss for AugNet

    It biases the model toward broad augmentation distributions by maximizing
    the L2-norm of the Haddamar product of augmentations weights and
    magnitudes.

    Parameters
    ----------
    outputs : torch.Tensor
        Augmentation network output.
    labels : torch.Tensor
        Ground truth.
    model : torch.nn.Module
        Model including an AugNet layer in front-end.
    base_loss_fn : callable, optional
        Loss function to be penalized. By default torch.nn.CrossEntropyLoss().
    reg : float, optional
        Parameter setting the strength of the regularization. By default 0.5.

    Returns
    -------
    torch.Tensor - size (1,)
        Regularized loss.
    """

    base_loss = base_loss_fn(outputs, labels)

    l2_magnitude_weight = (
        model.aug._magnitudes.clamp(0, 1) * model.aug.weights
    ).norm()

    return base_loss - reg * l2_magnitude_weight


def partial_aug_layer_loss(
    outputs,
    labels,
    model,
    base_loss_fn=torch.nn.CrossEntropyLoss(),
    reg=0.5
):
    """Incomplete penalized loss for AugNet (weights penalty missing)

    It biases the model toward augmentations with large magnitudes,
    by maximizing the L2-norm of the magnitudes vector.

    Parameters
    ----------
    outputs : torch.Tensor
        Augmentation network output.
    labels : torch.Tensor
        Ground truth.
    model : torch.nn.Module
        Model including an AugNet layer in front-end.
    base_loss_fn : callable, optional
        Loss function to be penalized. By default torch.nn.CrossEntropyLoss().
    reg : float, optional
        Parameter setting the strength of the regularization. By default 0.5.

    Returns
    -------
    torch.Tensor - size (1,)
        Regularized loss.
    """

    base_loss = base_loss_fn(outputs, labels)

    l2_magnitude = model.aug._magnitudes.clamp(0, 1).norm()

    return base_loss - reg * l2_magnitude


class InvariancePromotingLoss(_Loss):
    """Class encapsulating aug_layer_loss

    This is useful when wanting to use skorch, as in the MASS experiment.

    Parameters
    ----------
    reg : float
        Regularization weight.
    model : torch.nn.Module
        Averaged model with augmentation layer.
    weight : torch.Tensor | None, optional
        Tensor of size n_classes, used to balance the base loss function.
        Defaults to None.
    base_loss_cls : torch.nn.Module, optional
        Torch loss class which we want to regularize. Defaults to
        torch.nn.CrossEntropyLoss.
    """
    def __init__(
        self,
        reg,
        model,
        *args,
        weight=None,
        base_loss_cls=torch.nn.CrossEntropyLoss,
        **kwargs,
    ):
        super(InvariancePromotingLoss, self).__init__(*args, **kwargs)
        self.reg = reg
        self.base_loss_cls = base_loss_cls
        self.model = model
        self.weight = weight
        self.base_loss_fn = self.base_loss_cls(weight=self.weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return aug_layer_loss(input, target, model=self.model, reg=self.reg,
                              base_loss_fn=self.base_loss_fn)
