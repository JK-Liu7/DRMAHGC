import numpy as np
import random
import torch
import pandas as pd
import dgl
import networkx as nx
from sklearn.model_selection import StratifiedKFold
import copy

device = torch.device('cuda')

def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    adj = adj.numpy()
    return adj


def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def get_data(args):
    data = dict()

    drf = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
    drg = pd.read_csv(args.data_dir + 'DrugGIP.csv').iloc[:, 1:].to_numpy()

    dip = pd.read_csv(args.data_dir + 'DiseasePS.csv').iloc[:, 1:].to_numpy()
    dig = pd.read_csv(args.data_dir + 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()

    prs = pd.read_csv(args.data_dir + 'Protein_sequence.csv').iloc[:, 1:].to_numpy()
    prg_r = pd.read_csv(args.data_dir + 'ProteinGIP_Drug.csv').iloc[:, 1:].to_numpy()
    prg_d = pd.read_csv(args.data_dir + 'ProteinGIP_Disease.csv').iloc[:, 1:].to_numpy()

    data['drug_number'] = int(drf.shape[0])
    data['disease_number'] = int(dig.shape[0])
    data['protein_number'] = int(prg_r.shape[0])

    data['drf'] = drf
    data['drg'] = drg
    data['dip'] = dip
    data['dig'] = dig
    data['prs'] = prs
    data['prgr'] = prg_r
    data['prgd'] = prg_d

    data['drdi'] = pd.read_csv(args.data_dir + 'DrugDiseaseAssociationNumber.csv', dtype=int).to_numpy()
    data['drpr'] = pd.read_csv(args.data_dir + 'DrugProteinAssociationNumber.csv', dtype=int).to_numpy()
    data['dipr'] = pd.read_csv(args.data_dir + 'ProteinDiseaseAssociationNumber.csv', dtype=int).to_numpy()

    data['didr'] = data['drdi'][:, [1, 0]]
    data['prdr'] = data['drpr'][:, [1, 0]]
    data['prdi'] = data['dipr'][:, [1, 0]]

    drug_GAE = pd.read_csv(args.GAE_data_dir + 'drug.csv').iloc[:, 1:].to_numpy()
    disease_GAE = pd.read_csv(args.GAE_data_dir + 'disease.csv').iloc[:, 1:].to_numpy()
    protein_GAE = pd.read_csv(args.GAE_data_dir + 'protein.csv').iloc[:, 1:].to_numpy()
    data['drug_gae'] = drug_GAE
    data['disease_gae'] = disease_GAE
    data['protein_gae'] = protein_GAE
    return data


def data_processing(data, args):
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)
    # unsamples=[]

    unsamples = zero_index[int(args.negative_rate * len(one_index)):]
    data['unsample'] = np.array(unsamples)

    zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    index = np.array(one_index + zero_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]

    drs_mean = (data['drf'] + data['drg']) / 2
    dis_mean = (data['dip'] + data['dig']) / 2

    drs = np.where(data['drf'] == 0, data['drg'], drs_mean)
    dis = np.where(data['dip'] == 0, data['dip'], dis_mean)

    prg = (data['prgr'] + data['prgd']) / 2
    prs_mean = (data['prs'] + prg) / 2
    prs = np.where(data['prs'] == 0, prg, prs_mean)

    data['drs'] = drs
    data['dis'] = dis
    data['prs'] = prs
    data['all_samples'] = samples
    data['all_drdi'] = samples[:, :2]
    data['all_drdi_p'] = drdi_p
    data['all_drdi_n'] = drdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p
    return data


def k_fold(data, args):
    k = args.k_fold
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    X = data['all_drdi']
    Y = data['all_label']
    X_train_all, X_train_p_all, X_test_all, X_test_p_all, Y_train_all, Y_test_all = [], [], [], [], [], []
    X_train_n_all, X_test_n_all = [], []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_p = X_train[Y_train[:, 0] == 1, :]
        X_train_n = X_train[Y_train[:, 0] == 0, :]
        X_test_p = X_test[Y_test[:, 0] == 1, :]
        X_test_n = X_test[Y_test[:, 0] == 0, :]
        X_train_all.append(X_train)
        X_train_p_all.append(X_train_p)
        X_train_n_all.append(X_train_n)
        X_test_all.append(X_test)
        X_test_p_all.append(X_test_p)
        X_test_n_all.append(X_test_n)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    data['X_train'] = X_train_all
    data['X_train_p'] = X_train_p_all
    data['X_train_n'] = X_train_n_all
    data['X_test'] = X_test_all
    data['X_test_p'] = X_test_p_all
    data['X_test_n'] = X_test_n_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    return data


def sampling(args, type, pos_sample, neg_sample):
    if type == 'drug':
        n_items = args.disease_number
    else:
        n_items = args.drug_number
    pos_anchor = neg_sample[:, 0]
    neg_candidates = []
    for pos in pos_anchor.cpu().numpy():
        pos = int(pos)
        neg_items = []
        for i in range(args.n_neg):
            while True:
                negitem = random.choice(range(n_items))
                if negitem not in pos_sample[pos_sample[:, 0] == pos, -1]:
                    break
            neg_items.append(negitem)
        neg_candidates.append(neg_items)
    return neg_candidates


def sampling_2hop(args, type, nei2, pos_sample, neg_sample):
    if type == 'drug':
        n_anchors = args.drug_number
        n_items = args.disease_number
    else:
        n_anchors = args.disease_number
        n_items = args.drug_number
    pos_anchor = neg_sample[:, 0]
    neg_candidates, neg_candidates_all = [], []
    for i in range(n_anchors):
        nei2_candidates = nei2[str(i)]
        nei2_unique = set(nei2_candidates) - set(pos_sample[pos_sample[:, 0] == i, -1])
        nei2_unique = np.array(list(nei2_unique))
        if len(nei2_unique) >= args.n_neg:
            neg_items = np.random.choice(nei2_unique, size=args.n_neg)
            neg_candidates.append(neg_items)
        else:
            nei2_unique = set(np.arange(n_items)) - set(pos_sample[pos_sample[:, 0] == i, -1])
            nei2_unique = np.array(list(nei2_unique))
            neg_items = np.random.choice(nei2_unique, size=args.n_neg)
            neg_candidates.append(neg_items)

    for pos in pos_anchor:
        pos = int(pos)
        neg_item = neg_candidates[pos]
        neg_candidates_all.append(neg_item)

    return neg_candidates_all


def searching_2hop(args, data, type, g, sim_g, rdr, drd, pos_sample):
    if type == 'drug':
        new_g = dgl.metapath_reachable_graph(g, ['drug-disease', 'disease-drug'])
        metapaths = rdr
        pr = data['drpr']
        num = args.drug_number
    else:
        new_g = dgl.metapath_reachable_graph(g, ['disease-drug', 'drug-disease'])
        metapaths = drd
        pr = data['dipr']
        num = args.disease_number

    edges_meta = new_g.edges(order='eid')

    edges_sim = sim_g.edges(order='eid')

    edge_types = data['edge_types']
    hetero_dic = {}
    for edge in edge_types:
        hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()

    hetero_dic[(type, 'meta', type)] = (edges_meta[0], edges_meta[1])
    new_g = dgl.heterograph(hetero_dic)
    neighbour_2 = {}

    for i in range(num):
        sg = dgl.sampling.sample_neighbors(new_g, {type: [i]}, -1)
        sg_edges1 = sg.all_edges(form='all', etype=(type, 'meta', type))
        nei1 = sg_edges1[0]
        nei2 = []
        drdi = pos_sample

        for item in nei1:
            item = item.item()
            nei2_item1 = drdi[drdi[:, 0] == item, 1]
            nei2.append(nei2_item1)

        nei2 = np.array(nei2)
        nei2 = np.array([i for item in nei2 for i in item])
        nei2 = np.unique(nei2)
        nei2 = set(nei2) - set(drdi[drdi[:, 0] == i, 1])
        nei2 = np.array(list(nei2))
        neighbour_2.update({str(i): nei2})
    return neighbour_2


def dgl_similarity_graph(data, args):
    drdr_matrix = k_matrix(data['drs'], args.neighbor)
    didi_matrix = k_matrix(data['dis'], args.neighbor)
    prpr_matrix = k_matrix(data['prs'], args.neighbor)
    drdr_nx = nx.from_numpy_matrix(drdr_matrix)
    didi_nx = nx.from_numpy_matrix(didi_matrix)
    prpr_nx = nx.from_numpy_matrix(prpr_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)
    prpr_graph = dgl.from_networkx(prpr_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['drs'])
    didi_graph.ndata['dis'] = torch.tensor(data['dis'])
    prpr_graph.ndata['prs'] = torch.tensor(data['prs'])

    return drdr_graph, didi_graph, prpr_graph, data


def dgl_heterograph(data, drdi, args):
    drdi_list, drpr_list, dipr_list = [], [], []
    didr_list, prdr_list, prdi_list = [], [], []
    didr = drdi[:, [1, 0]]

    for i in range(drdi.shape[0]):
        drdi_list.append(drdi[i])
        didr_list.append(didr[i])
    for i in range(data['drpr'].shape[0]):
        drpr_list.append(data['drpr'][i])
        prdr_list.append(data['prdr'][i])
    for i in range(data['dipr'].shape[0]):
        dipr_list.append(data['dipr'][i])
        prdi_list.append(data['prdi'][i])

    node_dict = {
        'drug': args.drug_number,
        'disease': args.disease_number,
        'protein': args.protein_number
    }

    heterograph_dict = {
        ('drug', 'drug-disease', 'disease'): (drdi_list),
        ('disease', 'disease-drug', 'drug'): (didr_list),
        ('drug', 'drug-protein', 'protein'): (drpr_list),
        ('protein', 'protein-drug', 'drug'): (prdr_list),
        ('disease', 'disease-protein', 'protein'): (dipr_list),
        ('protein', 'protein-disease', 'disease'): (prdi_list)
    }

    data['feature_dict'] = {
        'drug': torch.tensor(data['drug_gae']),
        'disease': torch.tensor(data['disease_gae']),
        'protein': torch.tensor(data['protein_gae'])
    }

    drdipr_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)
    drdipr_graph.ndata['h'] = data['feature_dict']

    meta_paths = {'RDR':['drug-disease', 'disease-drug'],
                  'RPR':['drug-protein', 'protein-drug'],
                  'DRD':['disease-drug', 'drug-disease'],
                  'DPD':['disease-protein', 'protein-disease']}

    edge_types = {
        'drug-disease': ['drug', 'disease'],
        'drug-protein': ['drug', 'protein'],
        'disease-protein': ['disease', 'protein']
    }

    data['drdipr_graph'] = drdipr_graph
    data['meta_paths'] = meta_paths
    data['edge_types'] = edge_types

    return drdipr_graph, meta_paths, edge_types, data





