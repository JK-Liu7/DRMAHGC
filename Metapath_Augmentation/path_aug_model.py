import numpy as np
import dgl
from Metapath_Augmentation import learner
from Metapath_Augmentation import simulator
from scipy import sparse
import itertools


class Path_Augmentation:
    def __init__(self, g, meta_path_dic, method, args):
        super().__init__()

        self.meta_path_dic = meta_path_dic
        self.g = g
        self.meta_path_reach = {}
        self.method = method
        self.args = args

        for meta_path in meta_path_dic:
            self.meta_path_reach[meta_path] = dgl.metapath_reachable_graph(g, meta_path_dic[meta_path])


    def estimate_graphon(self, meta_path):

        mat = self.meta_path_reach[meta_path].adj(scipy_fmt='coo').toarray()
        _, graphon = learner.estimate_graphon([mat], self.method, self.args)
        return graphon


    def generate_graph(self, node_num, num_graphs, graphon):
        num_nodes = node_num
        generated_graphs = simulator.simulate_graphs(graphon, num_graphs=num_graphs, num_nodes=num_nodes, graph_size="fixed")
        for ind, item in enumerate(generated_graphs):
            generated_graphs[ind] = item + np.eye(node_num)
        return generated_graphs


def metapath_augmentation(g, path_augmentation, metapath_type, args, target_category, edge_types):
    # graphon estimator
    graphons = {}
    node_num = g.ndata['h'][target_category].size()[0]
    for path in args.augmentation_path:
        graphons[path] = path_augmentation.estimate_graphon(path)

    augmentated_graphs = []
    # intra-path augmentation
    for path in metapath_type:
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,
                                                                                   args.augmentation_intra_graph_num,
                                                                                   graphons[path])

    # inter-path augmentation
    arg_W = []
    combinations = list(itertools.combinations(metapath_type, 2))
    for com1, com2 in combinations:

        if target_category == 'drug':
            new_graphon = args.lam_r * graphons[com1] + (1 - args.lam_r) * graphons[com2]
        else:
            new_graphon = args.lam_d * graphons[com1] + (1 - args.lam_d) * graphons[com2]
        arg_W.append(new_graphon)
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,
                                                                                   args.augmentation_inter_graph_num,
                                                                                   new_graphon)

    # augmentation
    hetero_dic, hetero_dic1, hetero_dic2 = {}, {}, {}
    for edge in edge_types:
        hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
        hetero_dic1[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()
        hetero_dic2[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()

    for ind, item in enumerate(augmentated_graphs):
        adj_mat = sparse.coo_matrix(item)
        hetero_dic[(target_category, "AUG_" + str(ind), target_category)] = (adj_mat.row, adj_mat.col)

    adj_mat = sparse.coo_matrix(augmentated_graphs[0])
    hetero_dic1[(target_category, 'AUG_M1', target_category)] = (adj_mat.row, adj_mat.col)
    adj_mat = sparse.coo_matrix(augmentated_graphs[1])
    hetero_dic2[(target_category, 'AUG_M2', target_category)] = (adj_mat.row, adj_mat.col)

    new_g = dgl.heterograph(hetero_dic)
    new_g.nodes[target_category].data['h'] = g.ndata['h'][target_category]

    return new_g, augmentated_graphs


def Metapath_Augmentation(drdipr_graph, meta_paths, edge_types, args):
    path_augmentation = Path_Augmentation(drdipr_graph, meta_paths, args.graphon_method, args)
    target_category = 'drug'
    metapath_type = ['RDR', 'RPR']
    g_m_drug, g_adj_r = metapath_augmentation(drdipr_graph, path_augmentation, metapath_type, args, target_category, edge_types)
    target_category = 'disease'
    metapath_type = ['DRD', 'DPD']
    g_m_disease, g_adj_d = metapath_augmentation(drdipr_graph, path_augmentation, metapath_type, args, target_category, edge_types)

    return g_m_drug, g_m_disease, g_adj_r, g_adj_d
