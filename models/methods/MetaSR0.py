# https://github.com/HeZongyao/DIIF/blob/main/models/modules/metasr.py
# CVPR'2019
# Meta-SR: A Magnification-Arbitrary Network for Super-Resolution

import torch
import torch.nn as nn
import torch.nn.functional as F


import models
from models import register
# import models.networks as networks

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
        shape: [H,W]
        ranges: h and w coord range[-1, 1]
        ret: H*W coord
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        # i: index, n: coordinate num
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)

        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def get_coord_cell(img_shape, scale = 1, device=None):
    """
        img_shape: (H, W)
        scale: upscale
    """
    h_w = [int(img_shape[-2] * scale), int(img_shape[-1] * scale)]
    # coord, rgb = to_pixel_samples(img.contiguous())

    single_coord = make_coord(h_w).to(device)
    coord = single_coord.unsqueeze(0).repeat(img_shape[0], 1, 1)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / h_w[-2]
    cell[:, :, 1] *= 2 / h_w[-1]
    return coord, cell

@register('metasr')
class MetaSR(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256, training=True):
        super().__init__()

        # self.encoder = networks.define_module(opt_encoder)
        # self.imnet = networks.define_module(opt_imnet)
        self.training = training
        
        self.encoder = models.make(encoder_spec)
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        feat = F.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device)
        feat_coord[:, :, 0] -= (2 / feat.shape[-2]) / 2
        feat_coord[:, :, 1] -= (2 / feat.shape[-1]) / 2
        feat_coord = feat_coord.permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        coord_ = coord.clone()
        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_q = (coord_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat = F.grid_sample(
            feat, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2] / 2
        rel_coord[:, :, 1] *= feat.shape[-1] / 2

        r_rev = cell[:, :, 0] * (feat.shape[-2] / 2)
        inp = torch.cat([rel_coord, r_rev.unsqueeze(-1)], dim=-1)

        bs, q = coord.shape[:2]
        pred = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
        pred = torch.bmm(q_feat.contiguous().view(bs * q, 1, -1), pred)
        pred = pred.view(bs, q, 3)
        return pred

    def batched_query_rgb(self, coord, cell, bsize):
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        return torch.cat(preds, dim=1)

    def forward(self, inp, h = 0, w = 0):
        # inp: lr
        if h == 0 or w == 0:
            out_shape = [inp.shape[0], inp.shape[1], 4 * inp.shape[-2], 4 * inp.shape[-1]]  # B, C, H, W
        else:
            out_shape = [inp.shape[0], inp.shape[1], h, w]  # B, C, H, W
        coord, cell = get_coord_cell(out_shape, device=inp.device)

        self.gen_feat(inp)
        if self.training:
            out = self.query_rgb(coord, cell)
        else:
            out = self.batched_query_rgb(coord, cell, 512 * 512)
        return out