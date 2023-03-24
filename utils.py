"""Helper functions for graph extraction."""

import numpy as np
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    negative_sampling, add_self_loops, to_undirected,
    coalesce, remove_self_loops, scatter
)


def neighbors(fringe, A, outgoing=True):
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)
    return res


def k_hop_subgraph(src, dst, A, node_features=None, node_pos=None, y=1, hops=None):
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, hops+1):

        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)

        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)

    # Create subgraph and add target link
    subgraph = A[nodes, :][:, nodes]
    subgraph[0, 1] = 1
    subgraph[1, 0] = 1

    if node_features is not None:
        node_features = node_features[nodes]

    # Create data obj
    subgraph = ssp.triu(subgraph)   # make graph directed before line graph generations
    u, v, _ = ssp.find(subgraph)
    edge_index = torch.stack([torch.tensor(u).long(), torch.tensor(v).long()], 0)
    
    src_edges = (edge_index == 0).any(dim=0).int()
    dst_edges = (edge_index == 1).any(dim=0).int()

    num_nodes = len(nodes)
    y = torch.tensor([y])

    edge_attr = node_features[edge_index[1]] - node_features[edge_index[0]]
    edge_attr /= 0.01
    edge_attr = torch.cat((edge_attr, src_edges.unsqueeze(1), dst_edges.unsqueeze(1)), dim=-1)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)
    data_line_graph = gen_line_graph(data.clone())

    if data_line_graph.is_directed():
        data_line_graph.edge_index = to_undirected(data_line_graph.edge_index)

    if data_line_graph.x.shape[0] == 1: # add self loop to graphs of originally 2 nodes / 1 edge
        data_line_graph.edge_index = torch.tensor([[0], [0]])

    assert data_line_graph.x.shape[0] == data_line_graph.edge_index.unique().shape[0]

    return data_line_graph


def gen_line_graph(data):
    N = data.num_nodes
    edge_index, edge_attr = data.edge_index, data.edge_attr
    edge_index_orig, edge_attr_orig = coalesce(edge_index, edge_attr, N)

    edge_index = to_undirected(edge_index_orig)
    row, col = edge_index

    # Compute node indices
    mask = row < col
    row, col = row[mask], col[mask]
    i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

    (row, col), i = coalesce(
        torch.stack([
            torch.cat([row, col], dim=0),
            torch.cat([col, row], dim=0)
        ], dim=0),
        torch.cat([i, i], dim=0),
        N,
    )

    # Compute new edge indices according to `i`.
    count = scatter(torch.ones_like(row), row, dim=0,
                    dim_size=data.num_nodes, reduce='sum')
    joints = torch.split(i, count.tolist())

    def generate_grid(x):
        row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
        col = x.repeat(x.numel())
        return torch.stack([row, col], dim=0)

    joints = [generate_grid(joint) for joint in joints]
    joints = torch.cat(joints, dim=1)
    joints, _ = remove_self_loops(joints)
    N = row.size(0) // 2
    joints = coalesce(joints, num_nodes=N)

    edge_index = joints.sort(dim=0)[0].unique(dim=1)

    data.x = edge_attr_orig
    data.edge_index = edge_index
    data.num_nodes = edge_attr_orig.size(0)
    data.edge_attr = None

    assert data.x.shape[0] == data.num_nodes
    return data


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()

        if 'edge_neg' in split_edge['train']:
            # use presampled  negative training edges for ogbl-vessel
            neg_edge = split_edge[split]['edge_neg'].t()

        else:
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))

        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                target_neg.view(-1)])

    return pos_edge, neg_edge