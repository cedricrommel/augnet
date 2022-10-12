import pytest

from experiments.sinusoids.train_augnet import main, make_parser


@pytest.fixture(scope="module")
def mock_args(default_args):
    sinusoids_parser = make_parser()
    args = default_args(sinusoids_parser)
    args.batch_size = 16
    args.n_per_class = 10
    args.num_channels = 2
    args.epochs = 2
    args.noise = 0.2
    args.device = "cpu"
    return args


@pytest.mark.parametrize("backbone", ["cnn", "mlp"])
@pytest.mark.parametrize(
    "method,augment",
    [
        ('none', False),
        ('none', True),
        ('augnet', False),
    ]
)
def test_main(mock_dir, mock_args, backbone, method, augment):
    mock_args.dir = mock_dir
    mock_args.method = method
    mock_args.backbone = backbone
    mock_args.augment = augment
    main(mock_args)
