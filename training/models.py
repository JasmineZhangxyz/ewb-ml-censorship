import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, num_classes, in_dim, hid_dim, *dims, dropout=0., activation=F.relu):
        """
        Args:
            num_classes (int): dimension of the network output
            in_dim (int): dimension of the network input
            hid_dim (int): first hidden dimension of the network
            dims (list of ints): optional hidden dimensions
        """
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.in_dim, hid_dim))

        prev_dim = hid_dim
        for dim in dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim

        self.output_layer = nn.Linear(prev_dim, num_classes)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.output_layer(x)
