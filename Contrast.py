import torch
import torch.nn as nn


class Contrast(nn.Module):
    def __init__(self, args):
        super(Contrast, self).__init__()
        self.hidden_dim = args.hgt_out_dim
        self.proj_cross = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.proj_intra_mp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.proj_intra_sc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.tau_cross = args.tau_cross
        self.tau_mp = args.tau_mp
        self.tau_sc = args.tau_sc
        self.lam_cross = args.lam_cross
        self.lam_intra = args.lam_intra

        for model in self.proj_cross:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.proj_intra_mp:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.proj_intra_sc:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2, tau):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        z_mp_cross = self.proj_cross(z_mp)
        z_sc_cross = self.proj_cross(z_sc)
        z_mp_intra = self.proj_intra_mp(z_mp)
        z_sc_intra = self.proj_intra_sc(z_sc)
        matrix_mp2sc = self.sim(z_mp_cross, z_sc_cross, self.tau_cross)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        l_mp_cross = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        l_sc_cross = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1)).mean()

        l_cross = self.lam_cross * l_mp_cross + (1 - self.lam_cross) * l_sc_cross

        matrix_mp2mp = self.sim(z_mp_intra, z_mp_intra, self.tau_mp)
        matrix_sc2sc = self.sim(z_sc_intra, z_sc_intra, self.tau_sc)

        matrix_mp2mp = matrix_mp2mp / (torch.sum(matrix_mp2mp, dim=1).view(-1, 1) + 1e-8)
        l_mp_intra = -torch.log(matrix_mp2mp.mul(pos).sum(dim=-1)).mean()

        matrix_sc2sc = matrix_sc2sc / (torch.sum(matrix_sc2sc, dim=1).view(-1, 1) + 1e-8)
        l_sc_intra = -torch.log(matrix_sc2sc.mul(pos).sum(dim=-1)).mean()
        l_intra = self.lam_intra * l_mp_intra + (1 - self.lam_intra) * l_sc_intra

        return l_cross + l_intra

