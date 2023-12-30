import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch


class HGT(nn.Module):
    def __init__(self, args):
        super(HGT, self).__init__()
        self.layer = args.hgt_layer
        self.head = args.hgt_head
        self.in_dim = args.hgt_in_dim
        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, int(args.hgt_in_dim / args.hgt_head), args.hgt_head,
                                                   3, 4, args.dropout)
        self.hgt = nn.ModuleList()
        for _ in range(args.hgt_layer):
            self.hgt.append(self.hgt_dgl)

        self.drug_linear = nn.Linear(args.hgt_in_dim, args.hgt_in_dim)
        self.disease_linear = nn.Linear(args.hgt_in_dim, args.hgt_in_dim)
        self.protein_linear = nn.Linear(args.hgt_in_dim, args.hgt_in_dim)

    def forward(self, g, drug_feature, disease_feature, protein_feature):

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }

        g.ndata['h'] = feature_dict
        g = dgl.to_homogeneous(g, ndata='h')
        feature = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)
        hgt_out = 0
        for layer in self.hgt:
            hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        return hgt_out


class St_encoder(nn.Module):
    def __init__(self, args):
        super(St_encoder, self).__init__()
        self.HGT_encoder = nn.ModuleList([HGT(args) for _ in range(2)])
        self.drug_num = args.drug_number
        self.disease_num = args.disease_number
        self.protein_num = args.protein_number

    def forward(self, g_r, g_d, drug_feature, disease_feature, protein_feature):
        embeds_r = self.HGT_encoder[0](g_r, drug_feature, disease_feature, protein_feature)
        embeds_d = self.HGT_encoder[1](g_d, drug_feature, disease_feature, protein_feature)

        emb_r = embeds_r[:self.drug_num, :]
        emb_d = embeds_d[self.drug_num:self.drug_num + self.disease_num, :]

        return emb_r, emb_d
