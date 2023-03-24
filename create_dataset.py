"""Script to process flow-driven spatial networks."""

import os
import random
import math
import argparse
from pathlib import Path
import pickle

from ogb.io import DatasetSaver
from ogb.linkproppred import PygLinkPropPredDataset
import networkit as nk
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, remove_isolated_nodes


def positive_train_test_split_edges(data, val_ratio: float = 0.05, test_ratio: float = 0.1):
    assert 'batch' not in data  # No batch-mode.

    row, col = data.edge_index
    data.edge_index = None

    n_v = int(math.floor(val_ratio * row.size(0)))  # num val samples
    n_t = int(math.floor(test_ratio * row.size(0))) # num test samples

    # Positive edges
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm] # shuffle edges

    r, c = row[:n_v], col[:n_v] # get val data
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]   # get test data
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:] # get train data
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    # edge_index should only contain positive train edges; val and test should be excluded!
    data.edge_index = to_undirected(data.train_pos_edge_index)
    return data

def sample(num_samples, set_nodes, delta, device):
    edge_list = []
    num_nodes = len(set_nodes)

    # randomly sample num_sample nodes
    sampled_ids = torch.randint(num_nodes, size=(num_samples,))
    sampled_nodes = set_nodes[sampled_ids]

    # create filter conditions
    x_min = sampled_nodes[:, 1] - delta
    x_max = sampled_nodes[:, 1] + delta
    y_min = sampled_nodes[:, 2] - delta
    y_max = sampled_nodes[:, 2] + delta
    z_min = sampled_nodes[:, 3] - delta
    z_max = sampled_nodes[:, 3] + delta

    min_cons = torch.column_stack((x_min, y_min, z_min))
    max_cons = torch.column_stack((x_max, y_max, z_max))
    conditions = torch.cat((sampled_ids.unsqueeze(-1), min_cons, max_cons), dim=-1).to(device=device)
    nodes = set_nodes[:, 1:].to(device=device)

    edge_list = []
    for batch_conditions in tqdm(torch.split(conditions, 50), mininterval=5):
        min_matches = (batch_conditions[:, 1:4].unsqueeze(1) < nodes).all(dim=-1)
        max_matches = (batch_conditions[:, 4:].unsqueeze(1) > nodes).all(dim=-1)
        matches = torch.stack((min_matches, max_matches)).all(dim=0)
        
        for ind_matches, point_id in zip(matches, batch_conditions[:, 0]):
            match_ids = torch.where(ind_matches)[0]
            sampled_match_id = torch.randint(len(match_ids), size=(1,))
            sampled_point_id = match_ids[sampled_match_id]

            sampled_edge = (point_id.item(), sampled_point_id.item())

            # avoid self loops
            if sampled_edge[0] != sampled_edge[1]:
                edge_list.append(sampled_edge)

    return torch.tensor(edge_list)

def negative_sampling(data, n_train, n_val, n_test, device):
    num_samples = n_train + n_test + n_val
    distances = data.dist
    all_undirectional_edges = data.edge_index_undirected

    # seed random functions
    np.random.seed(123)
    torch.manual_seed(123)
    random.seed(123)

    # determine spatial criteria
    mean = distances.mean()
    two_sigma = 2 * distances.std()
    delta = mean + two_sigma

    # determine possible node ids including their coords
    set_nodes = all_undirectional_edges.unique()
    set_nodes = torch.cat((set_nodes[None].t(), data.x[set_nodes]), dim=1)

    sampled_edges = sample(int(num_samples * 1.5), set_nodes, delta, device)  # *1.5 to counteract sampling of existing links  
    sampled_edges = torch.sort(sampled_edges.t(), dim=0)[0]

    all_edges = torch.cat((all_undirectional_edges, sampled_edges), dim=1)

    # obtain just negative edges; we could have also sampled existing positive edges
    _, indices = np.unique(all_edges, return_index=True, axis=1)
    indices = torch.tensor(indices).sort()[0]
    indices = indices[all_undirectional_edges.shape[1]:]    # remove the indices of positive edges
    sampled_edges = all_edges[:, indices][:, :num_samples]  # only select specified number of samples
    assert sampled_edges.shape[1] == num_samples

    perm = torch.randperm(sampled_edges.shape[1])           # shuffle neg edges
    sampled_edges = sampled_edges[:, perm]

    data.train_neg_edge_index = sampled_edges[:, :n_train].long()
    data.test_neg_edge_index = sampled_edges[:, n_train:n_train + n_val].long()
    data.val_neg_edge_index = sampled_edges[:, n_train + n_val:].long()

    return data

def create_dataset(data, name):
    # create data for ogbl-like dataset
    ogbl_saver = DatasetSaver(dataset_name=name, is_hetero=False, version=1)

    # save graph
    graph_list = []
    graph = dict()
    graph['num_nodes'] = int(data.num_nodes)
    graph['node_feat'] = np.array(data.x)
    graph['edge_index'] = data.edge_index.numpy() # only train pos edge index, but both directions / undirected
    graph_list.append(graph)
    ogbl_saver.save_graph_list(graph_list)

    # save splits
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t() # these are only one directional
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    ogbl_saver.save_split(split_edge, split_name='spatial')

    # copy mapping dir
    mapping_path = 'mapping/'
    os.makedirs(mapping_path,exist_ok=True)
    try:
        os.mknod(os.path.join(mapping_path, 'README.md'))
    except:
        print("Readme.md already exists.")
    ogbl_saver.copy_mapping_dir(mapping_path)

    # save task info
    ogbl_saver.save_task_info(task_type='link prediction', eval_metric ='acc')

    # get meta dict
    meta_dict = ogbl_saver.get_meta_dict()

    dataset = PygLinkPropPredDataset(name, meta_dict=meta_dict)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--dataset_name', required=True, type=str)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 else 'cpu')

    path = Path(args.path)
    files = [file for file in path.glob('**/*') if file.is_file() and file.suffix in ['.npy', '.graph', '.mat', '.csv']]
    files.sort()

    if (len(files) == 2) and (files[0].suffix == '.csv'): # vessel graph
        nodes = torch.tensor(pd.read_csv(str(files[1]), sep=';')[['pos_x', 'pos_y', 'pos_z']].values).float()
        edges = torch.tensor(pd.read_csv(str(files[0]), sep=';')[['node1id', 'node2id']].values)
    elif len(files) == 2: # road networks
        metis_reader = nk.graphio.METISGraphReader()
        nk_graph = metis_reader.read(str(path / 'edges.graph'))

        edges = torch.tensor(list(nk_graph.iterEdges())).long()
        nodes = torch.tensor(pd.read_csv(path / 'nodes.graph', ' ', header=None).to_numpy()).float()

    # construct pyg graph
    data = Data(x=nodes, edge_index=edges.t().contiguous())

    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _, node_mask = remove_isolated_nodes(edge_index=data.edge_index, num_nodes=data.num_nodes)
    data.x = data.x[node_mask]

    # ensure that graph is directed to begin with
    data.edge_index = torch.sort(data.edge_index, dim=0)[0].unique(dim=1)

    # determine euclidean distances between nodes for each edge
    nodes_id1, nodes_id2 = data.edge_index
    assert (nodes_id1 < nodes_id2).all()

    data.dist = ((data.x[nodes_id1] - data.x[nodes_id2])**2).sum(dim=1).sqrt()

    # convert to undirected graph in a specific format; get edges in both directions
    edge_index_swapped = data.edge_index.detach().clone()
    edge_index_swapped[[0, 1]] = edge_index_swapped[[1, 0]]
    edge_index_undirected = torch.stack((data.edge_index, edge_index_swapped), dim=0).view(2, data.edge_index.shape[1]*2).t().contiguous().view(2, data.edge_index.shape[1]*2)
    data.edge_index_undirected = edge_index_undirected

    data = positive_train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
    n_train = data.train_pos_edge_index.shape[1]
    n_test = data.test_pos_edge_index.shape[1]
    n_val = data.val_pos_edge_index.shape[1]

    data = negative_sampling(data, n_train=n_train, n_test=n_test, n_val=n_val, device=device)

    dataset = create_dataset(data, args.dataset_name)

    dataset_name = args.dataset_name.split('-')[-1]
    with open(f'./dataset/{dataset_name}.pickle', 'wb') as f:
        pickle.dump(dataset, f)

