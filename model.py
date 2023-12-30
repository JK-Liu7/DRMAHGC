import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from Contrast import Contrast
from Metapath_Augmentation.mp_encoder import Mp_encoder
from Structural_Augmentation.st_encoder import St_encoder
import os


class MAHGCL(nn.Module):
    def __init__(self, args):
        super(MAHGCL, self).__init__()
        self.args = args
        self.mp = Mp_encoder(self.args)
        self.st = St_encoder(self.args)
        self.contrast_r = Contrast(self.args)
        self.contrast_d = Contrast(self.args)
        self.n_neg = self.args.n_neg
        self.pool = self.args.pool

        self.drug_linear = nn.Linear(args.embedding_dim, args.hgt_in_dim)
        self.disease_linear = nn.Linear(args.embedding_dim, args.hgt_in_dim)
        self.protein_linear = nn.Linear(args.embedding_dim, args.hgt_in_dim)

        self.hidden_dim = args.hgt_out_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.args.hgt_out_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )


    def forward(self, g_rm, g_dm, g_r, g_d, pos_r, pos_d,
                drug_feature, disease_feature, protein_feature, pos_sample, neg_sample, neg_candidates_r, neg_candidates_d):

        drug_feature = F.elu(self.drug_linear(drug_feature))
        disease_feature = F.elu(self.disease_linear(disease_feature))
        protein_feature = F.elu(self.protein_linear(protein_feature))

        r_mp, d_mp = self.mp(g_rm, g_dm, drug_feature, disease_feature, protein_feature)
        r_st, d_st = self.st(g_r, g_d, drug_feature, disease_feature, protein_feature)
        l_r = self.contrast_r(r_mp, r_st, pos_r)
        l_d = self.contrast_d(d_mp, d_st, pos_d)

        emb_neg_r = self.negative_sampling(r_mp, d_mp, pos_sample, neg_sample, neg_candidates_r)
        emb_neg_d = self.negative_sampling(d_mp, r_mp, pos_sample[:, [1, 0]], neg_sample[:, [1, 0]], neg_candidates_d)

        r_emb = r_mp
        d_emb = d_mp

        drdi_emb_pos = torch.mul(r_emb[pos_sample[:, 0]], d_emb[pos_sample[:, 1]])

        drdi_emb_neg = torch.mul(emb_neg_r, emb_neg_d)

        drdi_emb = torch.cat((drdi_emb_pos, drdi_emb_neg), dim=0)
        output = self.mlp(drdi_emb)
        return l_r + l_d, output, drdi_emb_pos, drdi_emb_neg


    def negative_sampling(self, emb_r, emb_d, pos_sample, neg_sample, neg_candidates):

        neg_num = neg_sample.shape[0]

        pos_anchor = neg_sample[:, 0]
        pos_target = pos_sample[:, 1]

        emb_anchor = emb_r[pos_anchor]
        emb_pos = emb_d[pos_target]
        emb_neg = emb_d[neg_candidates]

        if self.pool != 'concat':
            emb_anchor = self.pooling(emb_anchor).unsqueeze(dim=1)

        if self.args.negative_rate != 1.0:
            for i in range(int(self.args.negative_rate) - 1):
                emb_pos = torch.cat((emb_pos, emb_pos), dim=0)

        seed = torch.rand(neg_num, 1, 1).to(emb_anchor.device)
        emb_neg = seed * emb_pos.unsqueeze(dim=1) + (1 - seed) * emb_neg

        scores = (emb_anchor.unsqueeze(dim=1) * emb_neg).sum(dim=-1)
        indices = torch.max(scores, dim=1)[1].detach().unsqueeze(dim=1)

        neg_items_emb_ = emb_neg.permute(0, 2, 1)
        emb_neg = neg_items_emb_[[[i] for i in range(neg_num)], range(neg_items_emb_.shape[1]), indices]
        return emb_neg

    def pooling(self, emb):
        if self.pool == 'mean':
            return emb.mean(dim=1)
        elif self.pool == 'sum':
            return emb.sum(dim=1)
        elif self.pool == 'concat':
            return emb.view(emb.shape[0], -1)
        else:  # final
            return emb[:, -1, :]









