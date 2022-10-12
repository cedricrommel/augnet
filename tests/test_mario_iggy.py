import pytest

from experiments.mario_iggy import augnet_training, augerino_training


@pytest.fixture(scope="module")
def mock_augnet_args(default_args):
    mario_parser = augnet_training.make_parser()
    args = default_args(mario_parser)
    args.epochs = 2
    args.batch_size = 16
    args.ntrain = 32
    args.ntest = 16
    args.device = "cpu"
    args.data_path = "experiments/mario_iggy/"
    return args


@pytest.fixture(scope="module")
def mock_augerino_args(default_args):
    mario_parser = augerino_training.make_parser()
    args = default_args(mario_parser)
    args.epochs = 2
    args.batch_size = 16
    args.ntrain = 32
    args.ntest = 16
    args.device = "cpu"
    args.data_path = "experiments/mario_iggy/"
    return args


@pytest.mark.parametrize(
    "pen,reg",
    [
        ("correct", 0.5),
        ("correct", 0.),
        ("incomplete", 0.5),
    ]
)
def test_main_augnet(mock_dir, mock_augnet_args, pen, reg):
    mock_augnet_args.dir = mock_dir
    mock_augnet_args.pen = pen
    mock_augnet_args.reg = reg
    augnet_training.main(mock_augnet_args)


@pytest.mark.parametrize("reg", [0., 0.2])
def test_main_augerino(mock_dir, mock_augerino_args, reg):
    mock_augerino_args.dir = mock_dir
    mock_augerino_args.reg = reg
    augerino_training.main(mock_augerino_args)
