from torch import nn

from augnet.models.layer13 import Expression


def ConvBNrelu(in_channels, out_channels, filter_width=3, stride=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, filter_width, padding=1,
                  stride=stride),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


class FreqNet(nn.Module):
    """
    Very small CNN
    """
    def __init__(
        self,
        first_filter_width,
        n_filters=10,
        n_layers=2,
        in_channels=1,
        num_targets=4,
        dropout=False,
        embedding=False,
    ):
        super().__init__()
        self.first_filter_width = first_filter_width
        self.num_targets = num_targets
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.embedding = embedding

        self.repeated_layers = nn.ModuleList([
            ConvBNrelu(self.n_filters, self.n_filters)
            for _ in range(self.n_layers - 1)
        ])
        self.first_layer = ConvBNrelu(
            in_channels, self.n_filters, self.first_filter_width
        )
        self.mpool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(.3) if dropout else nn.Sequential()
        self.mean = Expression(lambda u: u.mean(-1))
        self.lin = nn.Linear(self.n_filters, self.num_targets)

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.repeated_layers:
            x = layer(x)
        x = self.mpool(x)
        x = self.drop(x)
        x = self.mean(x)
        if self.embedding:
            return x
        return self.lin(x)


def FCrelu(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
    )


class SimpleMLP(nn.Module):
    """ Small MLP
    """
    def __init__(
        self,
        n_neurons=1,
        n_layers=1,
        in_shape=1000,
        num_classes=4,
        embedding=False,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.num_targets = num_classes
        self.in_shape = in_shape
        self.embedding = embedding

        self.first_layer = FCrelu(in_shape, self.n_neurons)
        self.hidden = nn.ModuleList([
            FCrelu(self.n_neurons, self.n_neurons)
            for _ in range(self.n_layers - 1)
        ])
        self.out_layer = nn.Linear(self.n_neurons, self.num_targets)

    def forward(self, x):
        x = self.first_layer(x.view(-1, self.in_shape))
        for layer in self.hidden:
            x = layer(x)
        if self.embedding:
            return x
        return self.out_layer(x)
