import os
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Model
from datasets import KnowledgeGraph, KGDataset


def main(args):
    if args.mode == 'train':
        device = torch.device(args.device)
        if args.save_path is None:
            args.save_path = os.path.join('save', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        # if not os.path.isdir(args.save_path):
        #    os.makedirs(args.save_path)
        dataset = KnowledgeGraph(args.data_path, args.dataset)
        model = Model(dataset.num_entities, dataset.num_relations, args.model_name, args.part, args.dimension, args.regularization, args.alpha).to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        train(args, device, dataset, model, optimizer)
    elif args.mode == 'test':
        device = torch.device(args.device)
        dataset = KnowledgeGraph(args.data_path, args.dataset)
        model = Model(dataset.num_entities, dataset.num_relations, args.model_name, args.part, args.dimension, args.regularization, args.alpha).to(device)
        state_file = os.path.join(args.test_path, 'epoch_best.pth')
        if not os.path.isfile(state_file):
            raise RuntimeError('file {} is not found'.format(state_file))
        print('load checkpoint {}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        test(args, device, dataset, model, epoch, is_test=True)
    else:
        raise RuntimeError('wrong mode')


def train(args, device, dataset, model, optimizer):
    best_mrr, best_epoch = 0.0, 0
    stop = 0
    data_loader = DataLoader(KGDataset(dataset.train_data), batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=KGDataset.test_collate_fn)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total_loss = 0.
        model.train()
        for data in data_loader:
            heads, relations, tails = data[0].to(device), data[1].to(device), data[2].to(device)
            scores, factor1, factor2, factor3, factor4 = model(heads, relations, tails)
            loss = F.cross_entropy(scores, tails) + args.lambda1 * factor1 + args.lambda2 * factor2 + args.lambda3 * factor3 + args.lambda4 * factor4
            total_loss += loss.item() * heads.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(dataset.train_data)
        t1 = time.time()
        print('\n[train: epoch {}], loss: {}, time: {}s'.format(epoch, total_loss, t1 - t0))
        if not (epoch % args.valid_interval):
            metrics = test(args, device, dataset, model, epoch, is_test=False)
            _ = test(args, device, dataset, model, epoch, is_test=True)
            if metrics['mrr'] > best_mrr:
                best_mrr, best_epoch = metrics['mrr'], epoch
                stop = 0
                # save(args.save_path, epoch, model)
            else:
                stop += 1
        if stop >= 20:
            break
    print('best mrr: {} at epoch {}'.format(best_mrr, best_epoch))


def test(args, device, dataset, model, epoch, is_test=True):
    if is_test:
        data_loader = DataLoader(KGDataset(dataset.test_data), batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=KGDataset.test_collate_fn)
    else:
        data_loader = DataLoader(KGDataset(dataset.valid_data), batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=KGDataset.test_collate_fn)
    metrics = defaultdict(float)
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            heads, relations, tails = data[0].to(device), data[1].to(device), data[2].to(device)
            scores = model(heads, relations, tails)[0]

            for i, score in enumerate(scores):
                target = score[tails[i].item()].clone()
                score[dataset.hr_vocab[(heads[i].item(), relations[i].item())]] = -1e8
                rank = torch.ge(score, target).float().sum() + 1
                metrics['mrr'] += (1.0 / rank).item()
                metrics['hit@1'] += (rank <= 1).float().item()
                metrics['hit@3'] += (rank <= 3).float().item()
                metrics['hit@10'] += (rank <= 10).float().item()
                metrics['number'] += 1

    for metric in ['mrr', 'hit@1', 'hit@3', 'hit@10']:
        metrics[metric] /= metrics['number']
    t1 = time.time()
    print('[test: epoch {}], mrr: {}, hit@1: {}, hit@3: {}, hit@10: {}, time: {}s'.format(epoch, metrics['mrr'], metrics['hit@1'], metrics['hit@3'], metrics['hit@10'], t1 - t0))
    return metrics


def save(save_path, epoch, model):
    state = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    state_path = os.path.join(save_path, 'epoch_best.pth')
    torch.save(state, state_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Graph Completion by Intermediate Variable Regularization')
    parser.add_argument('--mode', type=str, default='train',  choices=['train', 'test'], help='mode')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'], help='device')
    parser.add_argument('--dataset', type=str, default='WN18RR', choices=['WN18RR', 'FB15k-237', 'YAGO3-10', 'Kinship'], help='dataset')
    parser.add_argument('--model_name', type=str, default='ComplEx', choices=['CP', 'ComplEx', 'SimplE', 'ANALOGY', 'QuatE', 'TuckER'], help='model name')
    parser.add_argument('--regularization', type=str, default='w/o', choices=['w/o', 'F2', 'N3', 'DURA', 'TNRR'], help='regularization')
    parser.add_argument('--data_path', type=str, default='datasets', help='data path')
    parser.add_argument('--save_path', type=str, default=None, help='save path')
    parser.add_argument('--test_path', type=str, default=None, help='test path')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--valid_interval', type=int, default=1, help='number of epochs to valid')

    parser.add_argument('--part', type=int, default=2, help='number of parts')
    parser.add_argument('--dimension', type=int, default=1000, help='number of dimension of each part')

    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1,  help='learning rate')
    parser.add_argument('--alpha', type=float, default=3, help='power coefficient')
    parser.add_argument('--lambda1', type=float, default=0.0, help='regularization coefficient')
    parser.add_argument('--lambda2', type=float, default=0.0, help='regularization coefficient')
    parser.add_argument('--lambda3', type=float, default=0.0, help='regularization coefficient')
    parser.add_argument('--lambda4', type=float, default=0.0, help='regularization coefficient')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    parse_args = parser.parse_args()
    random.seed(parse_args.seed)
    np.random.seed(parse_args.seed)
    torch.manual_seed(parse_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(parse_args.seed)

    print(parse_args.__dict__)
    main(parse_args)