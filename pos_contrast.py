import numpy as np
import scipy.sparse as sp
from collections import Counter


def mp_data(X_train_p, data, args):
    r = args.drug_number
    d = args.disease_number
    p = args.protein_number
    drpr = data['drpr']
    dipr = data['dipr']

    label_r = np.array([1] * (len(X_train_p) + len(drpr)), dtype=int)
    drpr_pr = drpr[:, -1] + d
    drpr = np.array([drpr[:, 0], drpr_pr]).T
    dr_dipr = np.concatenate((X_train_p, drpr), axis=0)
    dr_dipr = np.concatenate((dr_dipr, np.expand_dims(label_r, axis=1)), axis=1)
    dr_dipr = sp.coo_matrix((np.ones(dr_dipr.shape[0]),(dr_dipr[:,0], dr_dipr[:, 1])), shape=(r, d + p)).toarray()
    rdr = np.matmul(dr_dipr, dr_dipr.T)
    rdr = sp.coo_matrix(rdr)

    label_d = np.array([1] * (len(X_train_p) + len(dipr)), dtype=int)
    dipr_pr = dipr[:, -1] + r
    dipr = np.array([dipr[:, 0], dipr_pr]).T
    di_drpr = np.concatenate((X_train_p[:, (1, 0)], dipr), axis=0)
    di_drpr = np.concatenate((di_drpr, np.expand_dims(label_d, axis=1)), axis=1)
    di_drpr = sp.coo_matrix((np.ones(di_drpr.shape[0]), (di_drpr[:, 0], di_drpr[:, 1])), shape=(d, r + p)).toarray()
    drd = np.matmul(di_drpr, di_drpr.T)
    drd = sp.coo_matrix(drd)
    return rdr, drd


def mp_pos(rdr, drd, feature_graph_drug, feature_graph_disease, args):
    r = args.drug_number
    d = args.disease_number
    p = args.protein_number
    pos_num = args.pos_num
    # drug
    rdr = rdr.A.astype("float32")
    dia_r = sp.dia_matrix((np.ones(r), 0), shape=(r, r)).toarray()

    rr = np.ones((r, r)) - dia_r
    rdr = rdr * rr
    pos_r_mp = np.zeros((r, r))
    k = 0
    for i in range(r):
        pos_r_mp[i, i] = 1
        one = rdr[i].nonzero()[0]
        if len(one) > pos_num - 1:
            oo = np.argsort(-rdr[i, one])
            sele = one[oo[:pos_num - 1]]
            pos_r_mp[i, sele] = 1
            k += 1
        else:
            pos_r_mp[i, one] = 1
    pos_r = pos_r_mp * feature_graph_drug.A
    pos_r = sp.coo_matrix(pos_r)

    # disease
    drd = drd.A.astype("float32")
    dia_d = sp.dia_matrix((np.ones(d), 0), shape=(d, d)).toarray()
    dd = np.ones((d, d)) - dia_d
    drd = drd * dd
    pos_d_mp = np.zeros((d, d))
    k = 0
    for j in range(d):
        pos_d_mp[j, j] = 1
        # i = j + r
        one = drd[j].nonzero()[0]
        if len(one) > pos_num - 1:
            oo = np.argsort(-drd[j, one])
            sele = one[oo[:pos_num - 1]]
            pos_d_mp[j, sele] = 1
            k += 1
        else:
            pos_d_mp[j, one] = 1
    pos_d = pos_d_mp * feature_graph_disease.A
    pos_d = sp.coo_matrix(pos_d)
    return pos_r, pos_d

