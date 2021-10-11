import torch
from torch import nn
import math

class GraphGather(nn.Module):
    def __init__(self, node_features, out_features,
                 att_depth=2, att_hidden_dim=100, att_dropout_p=0.0,
                 emb_depth=2, emb_hidden_dim=100, emb_dropout_p=0.0):
        super(GraphGather, self).__init__()

        # denoted i and j in GGNN, MPNN and PotentialNet papers
        self.att_nn = FeedForwardNetwork(
            node_features * 2, [att_hidden_dim] * att_depth, out_features, dropout_p=att_dropout_p, bias=False)
        self.emb_nn = FeedForwardNetwork(
            node_features, [emb_hidden_dim] * emb_depth, out_features, dropout_p=emb_dropout_p, bias=False)

    def forward(self, hidden_nodes, input_nodes, node_mask):
        cat = torch.cat([hidden_nodes, input_nodes], dim=2)

        energy_mask = (node_mask == 0).float() * 1e6
        energies = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention = torch.sigmoid(energies)
        embedding = self.emb_nn(hidden_nodes)
        return torch.sum(attention * embedding, dim=1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, hidden_layer_sizes, out_features, activation='SELU', bias=False, dropout_p=0.0):
        super(FeedForwardNetwork, self).__init__()

        if activation == 'SELU':
            Activation = nn.SELU
            Dropout = nn.AlphaDropout
            init_constant = 1.0
        elif activation == 'ReLU':
            Activation = nn.ReLU
            Dropout = nn.Dropout
            init_constant = 2.0
        layer_sizes = [in_features] + hidden_layer_sizes + [out_features]
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(Dropout(dropout_p))
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias))
            layers.append(Activation())

        layers.append(Dropout(dropout_p))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias))

        self.seq = nn.Sequential(*layers)
        count=0
        for i in range(1, len(layers), 3):
            nn.init.normal_(layers[i].weight, std=math.sqrt(init_constant / layers[i].weight.size(1)))
            count+=1


    def forward(self, input):
        return self.seq(input)

    # I'm probably *supposed to* override extra_repr but then self.seq (unreadable) will be printed too
    def __repr__(self):
        ffnn = type(self).__name__
        in_features = self.seq[1].in_features
        hidden_layer_sizes = [linear.out_features for linear in self.seq[1:-1:3]]
        out_features = self.seq[-1].out_features
        if len(self.seq) > 2:
            activation = str(self.seq[2])
        else:
            activation = 'None'
        bias = self.seq[1].bias is not None
        dropout_p = self.seq[0].p
        return '{}(in_features={}, hidden_layer_sizes={}, out_features={}, activation={}, bias={}, dropout_p={})'.format(
            ffnn, in_features, hidden_layer_sizes, out_features, activation, bias, dropout_p
        )



