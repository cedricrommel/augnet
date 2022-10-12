import pytest
import tempfile


class MockArgs:
    def __init__(self, args_parser):
        for action in args_parser._actions:
            setattr(self, action.dest, action.default)


@pytest.fixture(scope="session")
def default_args():
    return MockArgs


@pytest.fixture(scope="function")
def mock_dir():
    with tempfile.TemporaryDirectory() as directory:
        yield directory
