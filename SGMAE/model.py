import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch
from functools import partial
from loss import sce_loss
from GAT import GAT


class MAEModel(nn.Module):
    def __init__(self, args, in_dim):
        super(MAEModel, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = self.args.num_layers
        self.num_hidden = self.args.num_hidden
        self.num_heads = self.args.num_heads
        self.num_out_heads = self.args.num_out_heads
        self.mask_rate = self.args.mask_rate
        self.replace_rate = self.args.replace_rate
        self.mask_token_rate = 1 - self.replace_rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.in_dim))

        self.criterion = partial(sce_loss, alpha=2)

        enc_num_hidden = self.num_hidden // self.num_heads
        enc_nhead = self.num_heads
        dec_in_dim = self.num_hidden
        dec_num_hidden = self.num_hidden // self.num_out_heads

        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        self.encoder = GAT(self.args, self.in_dim, enc_num_hidden, enc_num_hidden, self.num_layers, self.num_heads, encoding=True)
        self.decoder = GAT(self.args, dec_in_dim, self.in_dim, dec_num_hidden, num_layers=1, num_out_heads=1, encoding=False)


    def forward(self, g, x):
        loss, enc_rep = self.mask_attr_prediction(g, x)
        loss_item = {"loss": loss.item()}

        return loss, loss_item, enc_rep


    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep


    def mask_attr_prediction(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self.mask_rate)
        use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        rep[mask_nodes] = 0
        recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)

        return loss, enc_rep


    def encoding_mask_noise(self, g, x, mask_rate):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)