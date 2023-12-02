import torch
from torch.nn import Dropout
from torch_geometric.nn import SAGEConv, Linear, to_hetero
from torch_geometric.utils import dropout_edge

metadata = ['user', 'item'], [('user', 'rates', 'item'), ('item', 'rev_rates', 'user')]


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # these convolutions have been replicated to match the number of edge types\
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, n_factors, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * n_factors, hidden_channels)
        self.dropout1 = Dropout(p=0.5)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.dropout2 = Dropout(p=0.5)
        self.lin3 = Linear(hidden_channels, hidden_channels)
        self.dropout3 = Dropout(p=0.25)
        self.lin4 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        # concat user and movie embeddings
        z = torch.cat([z_dict['user'][row], z_dict['item'][col]], dim=-1)
        # concatenated embeddings passed to linear layer
        z = self.lin1(z).relu()
        z = self.dropout1(z)
        z = self.lin2(z).relu()
        z = self.dropout2(z)
        z = self.lin3(z).relu()
        z = self.dropout3(z)
        z = self.lin4(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, n_factors, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(n_factors)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(n_factors, hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # z_dict contains dictionary of movie and user embeddings returned from GraphSage
        edge_label_index, mask = dropout_edge(edge_label_index, p=0.02, training=self.training)
        z_dict = self.encoder(x_dict, edge_index_dict)
        output = self.decoder(z_dict, edge_label_index)
        output = torch.sigmoid(output)
        output = output * 4 + 1
        return output, mask