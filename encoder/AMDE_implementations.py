import torch
import numpy as np
from torch import nn
from encoder.AMDE import AMDE
from encoder.modules import FeedForwardNetwork, GraphGather


class Graph_encoder(AMDE):

    def __init__(self, node_features_1, edge_features_1, node_features_2, edge_features_2, message_size, message_passes,out_features,
                 msg_depth=4, msg_hidden_dim=200, att_depth=3, att_hidden_dim=200,
                 gather_width=100, gather_att_depth=3, gather_att_hidden_dim=100,
                 gather_emb_depth=3, gather_emb_hidden_dim=100,
                 out_depth=2, out_hidden_dim=100,out_layer_shrinkage=1.0,):
        super(Graph_encoder, self).__init__(node_features_1, edge_features_1, node_features_2, edge_features_2,
                                            message_size, message_passes, out_features)

        self.msg_nns_1 = nn.ModuleList()
        self.att_nns_1 = nn.ModuleList()
        self.msg_nns_2 = nn.ModuleList()
        self.att_nns_2 = nn.ModuleList()
        self.droupout=0.0
        for _ in range(edge_features_1):
            self.msg_nns_1.append(
                FeedForwardNetwork(node_features_1, [msg_hidden_dim] * msg_depth, message_size, dropout_p=self.droupout,bias=False))
            self.att_nns_1.append(
                FeedForwardNetwork(node_features_1, [att_hidden_dim] * att_depth, message_size, dropout_p=self.droupout,bias=False))
        for _ in range(edge_features_2):
            self.msg_nns_2.append(
                FeedForwardNetwork(node_features_2, [msg_hidden_dim] * msg_depth, message_size, dropout_p=self.droupout,bias=False))
            self.att_nns_2.append(
                FeedForwardNetwork(node_features_2, [att_hidden_dim] * att_depth, message_size, dropout_p=self.droupout,bias=False))


        self.gru_1 = nn.GRUCell(input_size=message_size, hidden_size=node_features_1, bias=False)
        self.gru_2 = nn.GRUCell(input_size=message_size, hidden_size=node_features_2, bias=False)

        self.gather_1 = GraphGather(
            node_features_1, gather_width,
            gather_att_depth, gather_att_hidden_dim, self.droupout,
            gather_emb_depth, gather_emb_hidden_dim, self.droupout)
        self.gather_2 = GraphGather(
            node_features_2, gather_width,
            gather_att_depth, gather_att_hidden_dim, self.droupout,
            gather_emb_depth, gather_emb_hidden_dim, self.droupout)

        out_layer_sizes = [round(out_hidden_dim * (out_layer_shrinkage ** (i / (out_depth - 1 + 1e-9)))) for i in range(out_depth)]
            # example: depth 5, dim 50, shrinkage 0.5 => out_layer_sizes [50, 42, 35, 30, 25]
        self.out_nn = FeedForwardNetwork(gather_width * 2, out_layer_sizes, out_features, dropout_p=self.droupout)

    def aggregate_message_1(self, nodes, node_neighbours, edges, node_neighbour_mask):

        energy_mask = (node_neighbour_mask == 0).float() * 1e6
        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns_1[i](node_neighbours) for i in range(self.edge_fts_1)
        ]
        embedding = sum(embeddings_masked_per_edge)
        energies_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.att_nns_1[i](node_neighbours) for i in range(self.edge_fts_1)
        ]
        energies = sum(energies_masked_per_edge) - energy_mask.unsqueeze(-1)
        attention = torch.softmax(energies, dim=1)
        return torch.sum(attention * embedding, dim=1)

    def aggregate_message_2(self, nodes, node_neighbours, edges, node_neighbour_mask):

        energy_mask = (node_neighbour_mask == 0).float() * 1e6
        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns_2[i](node_neighbours) for i in range(self.edge_fts_2)
        ]
        embedding = sum(embeddings_masked_per_edge)
        energies_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.att_nns_2[i](node_neighbours) for i in range(self.edge_fts_2)
        ]
        energies = sum(energies_masked_per_edge) - energy_mask.unsqueeze(-1)
        attention = torch.softmax(energies, dim=1)
        return torch.sum(attention * embedding, dim=1)

    def update_1(self, nodes, messages):
        return self.gru_1(messages)

    def readout_1(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather_1(hidden_nodes, input_nodes, node_mask)
        return graph_embeddings

    def update_2(self, nodes, messages):
        return self.gru_2(messages)


    def readout_2(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather_2(hidden_nodes, input_nodes, node_mask)
        return graph_embeddings

    def readout(self, input_nodes, node_mask):
        graph_embeddings = []
        for i in range(input_nodes.shape[0]):
            emb = input_nodes[i][0]
            for j in range(input_nodes.shape[1] - 1):
                emb = torch.cat([emb, input_nodes[i][j + 1]], dim=0)

            emb = emb.detach().numpy()
            graph_embeddings.append(emb)
        graph_embeddings = np.array(graph_embeddings)
        graph_embeddings = torch.from_numpy(graph_embeddings)

        return graph_embeddings

    def final_layer(self, connected_vector):
        return self.out_nn(connected_vector)

