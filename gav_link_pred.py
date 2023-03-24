"""Script train and evaluate GAV."""

import argparse
import time
import os, sys
import random
from shutil import copy
from tqdm import tqdm
import pickle

import numpy as np
import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, DataLoader

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from utils import *
from models import *


class DynamicDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train', transform=None, **kwargs):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.sub_transform = transform
        super().__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.percent)

        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)

        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        self.A_csc = None
 
    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]

        # Subgraph extraction module
        data = k_hop_subgraph(
            src, dst, self.A, node_features=self.data.x, node_pos=self.data.pos, y=y, hops=self.num_hops
        )

        return data


def train():
    model.train()

    total_loss = 0
    print('Training...')
    pbar = tqdm(train_loader, ncols=70, mininterval=5)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.batch)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)

@torch.no_grad()
def test():
    model.eval()

    y_pred, y_true = [], []
    total_loss_val = 0
    print('Validating...')
    for data in tqdm(val_loader, ncols=70, mininterval=5):
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        total_loss_val += loss.item() * data.num_graphs

        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

    loss_val = total_loss_val / len(val_dataset)
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    y_pred, y_true = [], []
    print('Testing...')
    for data in tqdm(test_loader, ncols=70, mininterval=5):
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true==1]
    neg_test_pred = test_pred[test_true==0]

    results_hits = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    results = evaluate_ogb_rocauc(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)

    return results, loss_val, results_hits

def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator_hits.K = K
        valid_hits = evaluator_hits.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator_hits.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results

def evaluate_ogb_rocauc(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    valid_rocauc = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
        })[f'rocauc']

    test_rocauc = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'rocauc']

    results = {}
    results['rocauc'] = (valid_rocauc, test_rocauc)
    return results


if __name__ == "__main__":
    # Data settings
    parser = argparse.ArgumentParser(description='Graph Attentive Vectors (GAV)')
    parser.add_argument('--path_to_py', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ogbl-vessel')

    # GNN settings
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)

    # Subgraph extraction settings
    parser.add_argument('--num_hops', type=int, default=1)

    # Training settings
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--train_percent', type=float, default=100)
    parser.add_argument('--val_percent', type=float, default=100)
    parser.add_argument('--test_percent', type=float, default=100)
    parser.add_argument('--num_workers', type=int, default=16, help="number of workers ")

    # Testing settings
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--data_appendix', type=str, default='', help="an appendix to the data directory")
    parser.add_argument('--save_appendix', type=str, default='', help="an appendix to the save directory")
    parser.add_argument('--keep_old', action='store_true', help="do not overwrite old files in the save directory")
    parser.add_argument('--continue_from', type=int, default=None, help="from which epoch's checkpoint to continue training")
    parser.add_argument('--only_test', action='store_true', help="only test without training")
    args = parser.parse_args()

    if args.path_to_py is not None:
        os.chdir(args.path_to_py)

    # To get reproducable results
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enabled = False  # https://github.com/pytorch/pytorch/issues/27588

    # Set name for save dir
    if args.save_appendix == '':
        args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
    
    # Create result dir
    args.res_dir = os.path.join('results/{}{}'.format(args.dataset, args.save_appendix))
    print('Results will be saved in ' + args.res_dir)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir) 
    if not args.keep_old:
        # Backup python files.
        copy('gav_link_pred.py', args.res_dir)
        copy('utils.py', args.res_dir)
        copy('models.py', args.res_dir)
    log_file = os.path.join(args.res_dir, 'log.txt')

    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')
    with open(log_file, 'a') as f:
        f.write('\n' + cmd_input)

    # Load raw data
    if args.dataset.startswith('ogbl'): # ogbl-vessel
        dataset = PygLinkPropPredDataset(name=args.dataset)
        split_edge = dataset.get_edge_split()
        data = dataset[0]
    else:   # others
        with open(f'./dataset/{args.dataset}.pickle', 'rb') as f:
            dataset = pickle.load(f)
        split_edge = dataset.get_edge_split()
        data = dataset[0]

    # Scale node features
    data.x = (data.x - data.x.min(dim=0)[0]) / (data.x.max(dim=0)[0] - data.x.min(dim=0)[0])
    data.x[data.x != data.x] = 0    # replace nan with 0 for datasets with no z coordinate

    # Assign metric and evaluator
    evaluator = Evaluator(name='ogbl-vessel')
    evaluator_hits = Evaluator(name='ogbl-collab')

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    path = dataset.root + args.data_appendix
    
    # Load train, val, and test datasets
    train_dataset = DynamicDataset(
        path,
        data,
        split_edge,
        num_hops=args.num_hops,
        percent=args.train_percent,
        split='train'
    ) 

    val_dataset = DynamicDataset(
        path,
        data,
        split_edge,
        num_hops=args.num_hops,
        percent=args.val_percent,
        split='valid'
    )

    test_dataset = DynamicDataset(
        path,
        data,
        split_edge,
        num_hops=args.num_hops,
        percent=args.test_percent,
        split='test'
    )

    # Get dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers)

    for run in range(args.runs):
        # Init Graph Attentive Vectors
        model = GAV(args.num_layers).to(device)
        parameters = list(model.parameters())

        # Init optim and tensorboard
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
        tb_writer = SummaryWriter(
            log_dir=args.res_dir,
            filename_suffix=time.strftime("%Y%m%d%H%M%S")
        )

        start_epoch = 1
        if args.continue_from is not None:
            ckpt_path = os.path.join(args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run+1, args.continue_from))
            state_dict = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state_dict)   #https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/3
            model.to(device)
            # optimizer.load_state_dict(state_dict) #TODO

            start_epoch = args.continue_from + 1
            args.epochs -= args.continue_from
        
        if args.only_test:
            results, _, results_hits = test()
            results.update(results_hits)
            for key, result in results.items():
                valid_res, test_res = result
                print(key)
                print(f'Run: {run + 1:02d}, '
                    f'Valid: {100 * valid_res:.2f}%, '
                    f'Test: {100 * test_res:.2f}%')
            exit()

        # Training starts
        for epoch in range(start_epoch, start_epoch + args.epochs):
            loss = train()
            tb_writer.add_scalar('loss/train', loss, epoch)

            if epoch % args.eval_steps == 0:
                results, loss_val, results_hits = test()
                tb_writer.add_scalar('loss/val', loss_val, epoch)
                results.update(results_hits)

                for key, result in results.items():
                    tb_writer.add_scalar(f'{key}/val', result[0], epoch)
                    tb_writer.add_scalar(f'{key}/test', result[1], epoch)

                if epoch % args.log_steps == 0:
                    model_name = os.path.join(
                        args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run+1, epoch))
                    optimizer_name = os.path.join(
                        args.res_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run+1, epoch))
                    torch.save(model.state_dict(), model_name)
                    torch.save(optimizer.state_dict(), optimizer_name)

                    for key, result in results.items():
                        valid_res, test_res = result
                        to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                                    f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                                    f'Test: {100 * test_res:.2f}%')
                        print(key)
                        print(to_print)
                        with open(log_file, 'a') as f:
                            print(key, file=f)
                            print(to_print, file=f)

    print(f'Results are saved in {args.res_dir}')