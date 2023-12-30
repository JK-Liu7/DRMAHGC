import timeit
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from data_preprocess import *
from model import MAEModel
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='B-dataset', help='dataset')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=5, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument("--epochs", type=int, default=800, help='number of training epochs')
    parser.add_argument("--num_heads", type=int, default=4, help='number of hidden attention heads')
    parser.add_argument("--num_out_heads", type=int, default=1, help='number of output attention heads')
    parser.add_argument("--num_layers", type=int, default=2, help='number of hidden layers')
    parser.add_argument("--num_hidden", type=int, default=64, help='number of hidden dimension')
    parser.add_argument("--residual", action="store_true", default=False, help='use residual connection')
    parser.add_argument("--in_drop", type=float, default=0.2,  help='input feature dropout')
    parser.add_argument("--attn_drop", type=float, default=0.1, help='attention dropout')
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3,  help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=1e-4,  help='weight decay')
    parser.add_argument("--negative_slope", type=float, default=0.2, help='the negative slope of leaky relu for GAT')
    parser.add_argument("--activation", type=str, default='prelu')
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.1)
    parser.add_argument("--encoder", type=str, default='gat')
    parser.add_argument("--decoder", type=str, default='gat')
    parser.add_argument("--loss_fn", type=str, default='sce')


    args = parser.parse_args()
    args.data_dir = '../data/' + args.dataset + '/'
    args.GAE_data_dir = 'Embedding/' + args.dataset + '/'
    args.result_dir = 'Embedding/' + args.dataset + '/'

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)

    drug_graph, disease_graph, protein_graph, data = dgl_similarity_graph(data, args)

    drug_graph = drug_graph.to(device)
    disease_graph = disease_graph.to(device)
    protein_graph = protein_graph.to(device)
    drug_feature = torch.FloatTensor(data['drs']).to(device)
    disease_feature = torch.FloatTensor(data['dis']).to(device)
    protein_feature = torch.FloatTensor(data['prs']).to(device)

    start = timeit.default_timer()

    print('Dataset:', args.dataset)
    entity = ['drug', 'disease', 'protein']

    for item in entity:
        entity_graph = eval(str(item) + '_graph')
        entity_feature = eval(str(item) + '_feature')
        entity_num = eval('args.' + str(item) + '_number')
        model = MAEModel(args, in_dim=entity_num)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_loss = 1
        best_epoch = 0
        best_model = None

        for epoch in range(args.epochs):
            loss_list = []
            model.train()
            loss, loss_dict, enc_rep = model(entity_graph, entity_feature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            end = timeit.default_timer()
            time = end - start
            loss_epoch = loss.item()
            show = [epoch + 1, round(time, 2), round(loss_epoch, 4)]
            print('\t\t'.join(map(str, show)))
            if loss_epoch < best_loss:
                best_epoch = epoch + 1
                best_loss = loss_epoch
                best_model = copy.deepcopy(model)

        print('Entity:', item, ';\tBest epoch:', best_epoch, ';\tBest loss:', best_loss)

        with torch.no_grad():
            rep = best_model.embed(entity_graph, entity_feature)
        rep = rep.detach().cpu().numpy()
        rep1 = pd.DataFrame(data=rep)
        rep1.to_csv(args.result_dir + item + '.csv')