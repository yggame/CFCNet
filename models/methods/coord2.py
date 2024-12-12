
# https://github.com/yinboc/liif/blob/main/models/liif.py
# CVPR'2021

import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

def normalize(x, eps=1e-10):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class CoordAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.q_conv = nn.Linear(in_dim, hidden_dim*3, 1)
        
        self.dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.dis_conv = nn.Linear(hidden_dim, 1, 1)
        self.x_conv = nn.Linear(hidden_dim, 1, 1)
        self.y_conv = nn.Linear(hidden_dim, 1, 1)
        
        self.k_conv = nn.Linear(in_dim, hidden_dim, 1)
        self.v_conv = nn.Linear(in_dim, in_dim, 1)
        # self.proj = nn.Linear(in_dim, in_dim, 1)

    def forward(self, q_feat, q_coord, coord):
        '''
        q_feat: (B, N, C)
        q_coord: (B, N, 2)
        coord: (B, N, 2)
        '''
        identity = q_feat
        B, N, C = q_feat.shape
        # q_coord = q_coord.view(B, N, 1, 1, 2)
        # coord = coord.view(B, 1, N, 1, 2)
        # q_coord = q_coord.repeat(1, 1, N, 1, 1)
        # coord = coord.repeat(1, N, 1, 1, 1)
        
        coord_diff = q_coord - coord
        coord_dist = coord_diff.pow(2).sum(dim=-1, keepdim=True).sqrt()
        coord_feat = torch.cat([coord_diff, coord_dist], dim=-1)

        x_q_c = q_coord[:, :, 0].unsqueeze(-1).repeat(1, 1, N)
        y_q_c = q_coord[:, :, 1].unsqueeze(-1).repeat(1, 1, N)

        x_coord = coord[:, :, 0].unsqueeze(-1).repeat(1,1, N)
        y_coord = coord[:, :, 1].unsqueeze(-1).repeat(1,1, N)

        # x_coord的后两维进行转置
        x_coord = x_coord.transpose(1, 2)
        y_coord = y_coord.transpose(1, 2)

        x_diff = torch.abs(x_q_c - x_coord)
        y_diff = torch.abs(y_q_c - y_coord)
        eps = 1e-10

        diff = x_diff + y_diff + eps

        # 对diff的（-2， -1）两维，通过softmax进行归一化
        diff = F.softmax(diff, dim=-2) # B N N
        atten = torch.ones_like(diff) - diff
        # atten = atten / atten.sum(dim=-1, keepdim=True) # B N N

        # atten 与 q_feat矩阵相乘
        coord_feat = torch.einsum('bnc,bnn->bnc', q_feat, atten)
        
 
        return coord_feat + identity

class DirectionAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim): # 576 512
        super().__init__()
        self.q_conv = nn.Linear(in_dim, hidden_dim, 1)
        self.k_conv = nn.Linear(in_dim, hidden_dim, 1)  
        
  
        self.v_conv = nn.Linear(in_dim, in_dim, 1)
        self.proj = nn.Linear(in_dim, 3, 1)
        

    def forward(self, preds, rel_coords, areas) :
        '''
        preds: [4 (B, N, C)]
        rel_coords: [4 (B, N, 2)]
        areas: [4 (B, N)]
        '''
        preds_all = torch.stack(preds, dim=0)
        
        # 坐标偏差作为权重
        # 设一个areas同尺寸的全1矩阵
        areas = F.softmax(torch.ones_like(torch.stack(areas,dim=0))-torch.stack(areas,dim=0), dim=0)
        attn_lst = []
        
        ret = 0
        for i in range(len(preds)):
            attn = preds[i] * areas[i].unsqueeze(-1)
            ret = ret + attn
        ret = self.proj(ret)
        return ret
        
        
    
@register('coord2')
class coord(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None
        
        self.coord_attention = CoordAttention(in_dim=3, hidden_dim=32)
        self.direction_attention = DirectionAttention(in_dim=576, hidden_dim=512)

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        rel_coords = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                    
                # 以q_coord为key，q_feat为value，coord为query，计算相对坐标
                pred_feat = self.coord_attention(q_feat, q_coord, coord) # 8 2304 576
                
                # 计算相对坐标,作为每一个方向的注意力输入key
                rel_coord = coord - q_coord # 8 2304 2
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])

                # rel_coord = coord - q_coord
                # rel_coord[:, :, 0] *= feat.shape[-2]
                # rel_coord[:, :, 1] *= feat.shape[-1]
                # inp = torch.cat([q_feat, rel_coord], dim=-1)

                # inp = pred_feat
                # if self.cell_decode:
                #     rel_cell = cell.clone()
                #     rel_cell[:, :, 0] *= feat.shape[-2]
                #     rel_cell[:, :, 1] *= feat.shape[-1]
                #     inp = torch.cat([inp, rel_cell], dim=-1)

                # bs, q = coord.shape[:2]
                # pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                # preds.append(pred)

                # area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                # areas.append(area + 1e-9)
                
                rel_coords.append(rel_coord)
                areas.append(area + 1e-9)
                preds.append(pred_feat)
        
        preds_feat = self.direction_attention(preds, rel_coords, areas) # 结合方向大小注意力进行预测preds_feat: 8 2304 576
        
        ret = preds_feat
        
        # if self.cell_decode:
        #     rel_cell = cell.clone()
        #     rel_cell[:, :, 0] *= feat.shape[-2]
        #     rel_cell[:, :, 1] *= feat.shape[-1]
        #     preds_feat_rel_cell = torch.cat([preds_feat, rel_cell], dim=-1)  # 8 2304 576+2
        #     ret = self.imnet(preds_feat_rel_cell.view(coord.shape[0] * coord.shape[1], -1)).view(coord.shape[0], coord.shape[1], -1)
            
        
        # # concat cell等信息
        
        # tot_area = torch.stack(areas).sum(dim=0)
        # if self.local_ensemble:
        #     t = areas[0]; areas[0] = areas[3]; areas[3] = t
        #     t = areas[1]; areas[1] = areas[2]; areas[2] = t
        # ret = 0
        # for pred, area in zip(preds, areas):
        #     ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

# 测试代码，测试coord attention
if __name__ == '__main__':
    model = CoordAttention(in_dim=3, hidden_dim=32)
    q_feat = torch.randn(8, 2304, 576)
    q_coord = torch.randn(8, 2304, 2)
    coord = torch.randn(8, 2304, 2)
    out = model(q_feat, q_coord, coord)
    print(out.shape)
    
    print(out)
    
    # 测试 DirectionAttention 代码
    # model = DirectionAttention(in_dim=3, hidden_dim=32)
    
    # preds = [torch.randn(8, 2304, 576) for i in range(4)]
    # rel_coords = [torch.randn(8, 2304, 2) for i in range(4)]
    # areas = [torch.randn(8, 2304) for i in range(4)]
    
    # out = model(preds, rel_coords, areas)
   
    # print(out)