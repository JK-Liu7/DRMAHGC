import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity as cos
import pandas as pd
import scipy
import dgl
from scipy import sparse
from numpy.linalg import matrix_power
import random


def feature_cos(data, target, t):

    features = data['feature_dict'][target]
    cos_graph = cos(features)

    for i in range(len(cos_graph)):
        for j in range(len(cos_graph)):
            if cos_graph[i][j] >= t:
                cos_graph[i][j] = 1
            else:
                cos_graph[i][j] = 0

    cos_graph = sparse.csr_matrix(cos_graph)
    return cos_graph


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def Structural_Augmentation(data, args, target, g, edge_types):
    n = args.n
    t = args.t
    # p = args.p

    homograph = dgl.to_homogeneous(g)
    adj = homograph.adjacency_matrix().to_dense()

    adj_n = matrix_power(adj, n)
    if target == 'drug':
        target_adj = adj_n[:args.drug_number, :args.drug_number]
        p = args.p_drug
    else:
        target_adj = adj_n[args.drug_number:args.drug_number + args.disease_number, args.drug_number:args.drug_number + args.disease_number]
        p = args.p_disease

    for i in range(len(target_adj)):
        for j in range(len(target_adj)):
            if target_adj[i][j] >= p:
                pass
            else:
                target_adj[i][j] = 0
    padj = target_adj

    feature_graph = feature_cos(data, target, t)
    padj = padj * feature_graph

    padj = normalize_adj(padj)
    padj_sp = sparse.coo_matrix(padj)


    # augmentation
    hetero_dic = {}
    for edge in edge_types:
        hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
    hetero_dic[(target, 'AUG_S', target)] = (padj_sp.row, padj_sp.col)
    new_g = dgl.heterograph(hetero_dic)
    new_g.nodes[target].data['h'] = g.ndata['h'][target]

    return new_g, feature_graph
