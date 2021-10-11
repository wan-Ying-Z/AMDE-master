import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch.utils import data
from encoder.graph_features import atom_features
from collections import defaultdict
from subword_nmt.apply_bpe import BPE
import codecs


class MolGraphDataset(data.Dataset):
    def __init__(self, path, prediction=False):
        print(path)
        file=pd.read_csv(path,sep=',')
        n_cols=file.shape[1]
        self.header_cols = np.genfromtxt(path, delimiter=',', usecols=range(0, n_cols), dtype=np.str, comments=None)
        self.target_names = self.header_cols[0:1, -1]
        self.smiles1 = np.genfromtxt(path,delimiter=',',skip_header=1,usecols=[3],dtype=np.str,comments=None)
        self.smiles2 = np.genfromtxt(path,delimiter=',',skip_header=1,usecols=[1],dtype=np.str,comments=None)

        self.drug_id_1 = np.genfromtxt(path,delimiter=',',skip_header=1,usecols=[2],dtype=np.str,comments=None)
        self.drug_id_2 = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[0], dtype=np.str, comments=None)

        if prediction:
            self.targets = np.empty((len(self.smiles1),1))
        else:
            self.targets = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[4],dtype=np.int, comments=None)



    def __getitem__(self, index):
        adj_1, nd_1, ed_1 = smile_to_graph(self.smiles1[index])
        adj_2, nd_2, ed_2 = smile_to_graph(self.smiles2[index])

        d1,mask_1=drug2emb_encoder(self.smiles1[index])
        d2,mask_2 = drug2emb_encoder(self.smiles2[index])
        targets = self.targets[index]

        return (adj_1, nd_1, ed_1),(adj_2, nd_2, ed_2),targets,d1,d2,mask_1,mask_2

    def __len__(self):
        return len(self.smiles1)


def drug2emb_encoder(x):


    ## Sequence encoder parameter
    vocab_path = './ESPF/drug_codes_chembl.txt'
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
    max_d = 50
    t1 = dbpe.process_line(x).split()
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])
    except:
        i1 = np.array([0])
        print('error:', x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d
    return i, np.asarray(input_mask)

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])
    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return adjacency, node_features, edge_features

BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)

def molgraph_collate_fn(data):
    n_samples = len(data)
    (adj_1, node_fts_1, edge_fts_1),(adj_2, node_fts_2, edge_fts_2),targets_0,d1,d2 ,mask_1,mask_2= data[0]
    n_nodes_largest_graph_1 = max(map(lambda sample: sample[0][0].shape[0], data))
    n_nodes_largest_graph_2 = max(map(lambda sample: sample[1][0].shape[0], data))

    n_node_fts_1 = node_fts_1.shape[1]
    n_edge_fts_1 = edge_fts_1.shape[2]
    n_node_fts_2 = node_fts_2.shape[1]
    n_edge_fts_2 = edge_fts_2.shape[2]
    n_targets = 1
    n_emb= d1.shape[0]
    n_mask=mask_1.shape[0]

    adjacency_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1)
    node_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_node_fts_1)
    edge_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1, n_edge_fts_1)

    adjacency_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2)
    node_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_node_fts_2)
    edge_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2, n_edge_fts_2)

    target_tensor = torch.zeros(n_samples, n_targets)
    d1_emb_tensor=torch.zeros(n_samples, n_emb)
    d2_emb_tensor = torch.zeros(n_samples, n_emb)
    mask_1_tensor=torch.zeros(n_samples, n_mask)
    mask_2_tensor = torch.zeros(n_samples, n_mask)


    for i in range(n_samples):
        (adj_1, node_fts_1, edge_fts_1),(adj_2, node_fts_2, edge_fts_2),target,d1,d2,mask_1,mask_2= data[i]
        n_nodes_1 = adj_1.shape[0]
        n_nodes_2 = adj_2.shape[0]


        adjacency_tensor_1[i, :n_nodes_1, :n_nodes_1] = torch.Tensor(adj_1)
        node_tensor_1[i, :n_nodes_1, :] = torch.Tensor(node_fts_1)
        edge_tensor_1[i, :n_nodes_1, :n_nodes_1, :] = torch.Tensor(edge_fts_1)

        adjacency_tensor_2[i, :n_nodes_2, :n_nodes_2] = torch.Tensor(adj_2)
        node_tensor_2[i, :n_nodes_2, :] = torch.Tensor(node_fts_2)
        edge_tensor_2[i, :n_nodes_2, :n_nodes_2, :] = torch.Tensor(edge_fts_2)

        target_tensor[i]=torch.tensor(target)
        d1_emb_tensor[i] = torch.IntTensor(d1)
        d2_emb_tensor[i] = torch.IntTensor(d2)
        mask_1_tensor[i] = torch.tensor(mask_1)
        mask_2_tensor[i] = torch.tensor(mask_2)


    return adjacency_tensor_1, node_tensor_1, edge_tensor_1, adjacency_tensor_2, node_tensor_2, edge_tensor_2,target_tensor,d1_emb_tensor,d2_emb_tensor,mask_1_tensor,mask_2_tensor

