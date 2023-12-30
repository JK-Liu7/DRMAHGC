import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from Metapath_Augmentation.path_aug_model import Metapath_Augmentation
from Structural_Augmentation.struc_aug import Structural_Augmentation
from pos_contrast import mp_pos, mp_data


from model import MAHGCL
from metric import *
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=5, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout of HGT')
    parser.add_argument('--augmentation_path', default=['RDR', 'RPR', 'DRD', 'DPD'], type=list, help='augmentation_path')
    parser.add_argument('--augmentation_intra_graph_num', default=1, type=int, help='augmentation_intra_graph_num')
    parser.add_argument('--augmentation_inter_graph_num', default=0, type=int, help='augmentation_inter_graph_num')
    parser.add_argument('--resolution', default=1000, type=int, help='resolution of graphon')
    parser.add_argument('--graphon_method', default='USVT', help='method of graphon estimation')
    parser.add_argument('--threshold_usvt', default='0.1', type=float, help='threshold of usvt')
    parser.add_argument('--lam_r', default='0.5', type=float, help='coefficient of mixup')
    parser.add_argument('--lam_d', default='0.5', type=float, help='coefficient of mixup')
    parser.add_argument('--n', default='2', type=int, help='n power of the adjacency matrix')
    parser.add_argument('--p_drug', default='30', type=int, help='threshold of drug node degree')
    parser.add_argument('--p_disease', default='15', type=int, help='threshold of disease node degree')
    parser.add_argument('--t', default='0.5', type=float, help='threshold of feature similarity')
    parser.add_argument('--pos_num', default='5', type=int, help='threshold of positive samples selection')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='4', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--embedding_dim', default='512', type=int, help='GAE embedding dimension')
    parser.add_argument('--hgt_in_dim', default='128', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='32', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='128', type=int, help='heterogeneous graph transformer output dimension')
    parser.add_argument('--tau_cross', default='0.8', type=float, help='tau in cross-view contrastive loss')
    parser.add_argument('--tau_mp', default='0.8', type=float, help='tau_mp in intra-view contrastive loss')
    parser.add_argument('--tau_sc', default='0.8', type=float, help='tau_sc in intra-view contrastive loss')
    parser.add_argument('--lam_cross', default='0.5', type=float, help='lam in cross-view contrastive loss')
    parser.add_argument('--lam_intra', default='0.5', type=float, help='lam in intra-view contrastive loss')
    parser.add_argument('--loss_rate', default='0.5', type=float, help='loss rate of unsupervised learning and training')
    parser.add_argument('--n_neg', default='20', type=int, help='number of negative candidates')
    parser.add_argument('--pool', default='mean', help='method of pooling in negative sampling')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    args.GAE_data_dir = 'SGMAE/Embedding/' + args.dataset + '/'

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, prpr_graph, data = dgl_similarity_graph(data, args)

    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)
    drug_feature = torch.FloatTensor(data['drug_gae']).to(device)
    disease_feature = torch.FloatTensor(data['disease_gae']).to(device)
    protein_feature = torch.FloatTensor(data['protein_gae']).to(device)

    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tLoss_con\t\tLoss_cls\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs = [], []

    print('Dataset:', args.dataset)

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        model = MAHGCL(args)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.8, patience=100, verbose=True, min_lr=1e-3)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        X_train_p = torch.LongTensor(data['X_train_p'][i]).to(device)
        X_train_n = torch.LongTensor(data['X_train_n'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        X_test_p = torch.LongTensor(data['X_test_p'][i]).to(device)
        X_test_n = torch.LongTensor(data['X_test_n'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, meta_paths, edge_types, data = dgl_heterograph(data, data['X_train_p'][i], args)

        g_s_drug, feature_graph_drug = Structural_Augmentation(data, args, 'drug', drdipr_graph, edge_types)
        g_s_disease, feature_graph_disease = Structural_Augmentation(data, args, 'disease', drdipr_graph, edge_types)
        g_m_drug, g_m_disease, g_adj_r, g_adj_d = Metapath_Augmentation(drdipr_graph, meta_paths, edge_types, args)

        rdr, drd = mp_data(data['X_train_p'][i], data, args)
        pos_r, pos_d = mp_pos(rdr, drd, feature_graph_drug, feature_graph_disease, args)
        drdipr_graph = drdipr_graph.to(device)
        g_s_drug = g_s_drug.to(device)
        g_s_disease = g_s_disease.to(device)
        g_m_drug = g_m_drug.to(device)
        g_m_disease = g_m_disease.to(device)

        pos_r = torch.IntTensor(pos_r.toarray()).to(device)
        pos_d = torch.IntTensor(pos_d.toarray()).to(device)

        drug_nei2 = searching_2hop(args, data, 'drug', drdipr_graph, drdr_graph, rdr, drd, data['X_train_p'][i])
        disease_nei2 = searching_2hop(args, data, 'disease', drdipr_graph, didi_graph, rdr, drd, data['X_train_p'][i][:, [1, 0]])

        for epoch in range(args.epochs):

            neg_candidates_train_r = sampling_2hop(args, 'drug', drug_nei2, data['X_train_p'][i], data['X_train_n'][i])
            neg_candidates_test_r = sampling_2hop(args, 'drug', drug_nei2, data['X_test_p'][i], data['X_test_n'][i])
            neg_candidates_train_d = sampling_2hop(args, 'disease', disease_nei2, data['X_train_p'][i][:, [1, 0]],
                                                   data['X_train_n'][i][:, [1, 0]])
            neg_candidates_test_d = sampling_2hop(args, 'disease', disease_nei2, data['X_test_p'][i][:, [1, 0]],
                                                  data['X_test_n'][i][:, [1, 0]])

            model.train()

            l_con, train_output, _, _ = model(g_m_drug, g_m_disease, g_s_drug, g_s_disease, pos_r, pos_d,
                                        drug_feature, disease_feature, protein_feature, X_train_p, X_train_n, neg_candidates_train_r, neg_candidates_train_d)
            l_train = cross_entropy(train_output, torch.flatten(Y_train))
            l_all = args.loss_rate * l_con + (1 - args.loss_rate) * l_train
            optimizer.zero_grad()
            l_all.backward()
            optimizer.step()

            l_con = l_con.detach().cpu().numpy()
            l_train = l_train.detach().cpu().numpy()

            with torch.no_grad():
                model.eval()

                _, test_score, _, _ = model(g_m_drug, g_m_disease, g_s_drug, g_s_disease,
                                      pos_r, pos_d, drug_feature, disease_feature, protein_feature, X_test_p, X_test_n, neg_candidates_test_r, neg_candidates_test_d)
            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            scheduler.step(AUC)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), l_con, l_train, round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)

    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')



