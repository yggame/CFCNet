
# https://github.com/yinboc/liif/blob/main/models/liif.py
# CVPR'2021

import sys, os, math
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

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.dim = in_dim
        self.hidden_dim = hidden_dim
    
        self.conv1 = nn.Linear(3, hidden_dim*3, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim*3)
        self.act = h_swish()
        
        self.dis_conv = nn.Linear(hidden_dim, 1, 1)
        self.x_conv = nn.Linear(hidden_dim, 1, 1)
        self.y_conv = nn.Linear(hidden_dim, 1, 1)
        
        self.q_conv = nn.Linear(in_dim, hidden_dim, 1)
        self.k_conv = nn.Linear(in_dim, hidden_dim, 1)
        self.v_conv = nn.Linear(in_dim, in_dim, 1)
        self.proj = nn.Linear(576, 576, 1)
        
        self.pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_y = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_dis = nn.AdaptiveAvgPool2d((None, 1))

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
        
        coord_diff = (coord - q_coord) * math.sqrt(q_coord.shape[1])
        coord_dist = coord_diff.pow(2).sum(dim=-1, keepdim=True).sqrt()
        coord_feat = torch.cat([coord_diff, coord_dist], dim=-1)

        # x_q_c = q_coord[:, :, 0].unsqueeze(-1).repeat(1, 1, N)
        # y_q_c = q_coord[:, :, 1].unsqueeze(-1).repeat(1, 1, N)

        # x_coord = coord[:, :, 0].unsqueeze(-1).repeat(1,1, N)
        # y_coord = coord[:, :, 1].unsqueeze(-1).repeat(1,1, N)

        # # x_coord的后两维进行转置
        # x_coord = x_coord.transpose(1, 2)
        # y_coord = y_coord.transpose(1, 2)

        # x_diff = (x_q_c - x_coord).pow(2)
        # y_diff = (y_q_c - y_coord).pow(2)
        # eps = 1e-10

        # diff = x_diff + y_diff + eps
        # q = self.q_conv(coord_feat) # B N hidden_dim
        # k = self.k_conv(coord_feat) # B N hidden_dim
        
        # qk = torch.einsum('bnc,bmc->bnm', q, k) # / (self.hidden_dim ** 0.5) # B N N
        # qk = F.softmax(qk, dim=-1) # B N N
        
        # # v = self.v_conv(coord_feat) # B N H

        # # # 对diff的（-2， -1）两维，通过softmax进行归一化
        # # diff = F.softmax(diff, dim=-2) # B N N
        # # atten = torch.ones_like(diff) - diff
        # # # atten = atten / atten.sum(dim=-1, keepdim=True) # B N N

        # # atten 与 q_feat矩阵相乘
        # coord_feat = torch.einsum('bnc,bnn->bnc', q_feat, qk)
        
        # coord_feat = self.proj(coord_feat)
 
        # return coord_feat + identity
        
        xyd = self.conv1(coord_feat)
        xyd = self.bn1(xyd.view(-1, self.hidden_dim*3))
        xyd = self.act(xyd).view(B, N, self.hidden_dim*3)
        
        x_dif, y_dif, dist = torch.split(xyd, [self.hidden_dim, self.hidden_dim, self.hidden_dim], dim=-1)
        
        x_dif = self.x_conv(x_dif)
        y_dif = self.y_conv(y_dif)
        dist = self.dis_conv(dist)
        
        return (x_dif + y_dif + dist) * identity
               
        
        
# class CoordAttention2(nn.Module):
#     def __init__(self, in_dim, hidden_dim):
#         super().__init__()
#         self.q_conv = nn.Linear(in_dim, hidden_dim, 1)
#         self.k_conv = nn.Linear(in_dim, hidden_dim, 1)
#         self.v_conv = nn.Linear(in_dim, in_dim, 1)
#         self.proj = nn.Linear(in_dim, in_dim, 1)

#     def forward(self, q_feat, q_coord, coord):
#         '''
#         q_feat: (B, N, C)
#         q_coord: (B, N, 2)
#         coord: (B, N, 2)
#         '''
#         identity = q_feat
#         B, N, C = q_feat.shape
#         q_coord = q_coord.view(B, N, 1, 2)
#         coord = coord.view(B, 1, N, 2)
#         q_coord = q_coord.repeat(1, 1, N, 1)
#         coord = coord.repeat(1, N, 1, 1)
        
#         coord_diff = q_coord - coord
#         coord_dist = coord_diff.pow(2).sum(dim=-1, keepdim=True).sqrt()
#         coord_feat = torch.cat([coord_diff, coord_dist], dim=-1)
#         coord_feat = self.q_conv(coord_feat) 
        
        
        
#         coord_feat = F.relu(coord_feat)
#         coord_feat = self.k_conv(coord_feat)
#         coord_feat = F.relu(coord_feat)
#         coord_feat = self.v_conv(coord_feat)
#         coord_feat = F.relu(coord_feat)
#         coord_feat = coord_feat.view(B, N, N, C)
#         attn = F.softmax(coord_feat, dim=-1)
#         attn = attn / attn.sum(dim=-1, keepdim=True)
#         attn = attn.unsqueeze(-1)
#         x = (attn * q_feat.unsqueeze(1)).sum(dim=2)
#         x = self.proj(x)
#         return x + identity
        
        
        # B, C, H, W = q.shape
        # q = self.q_conv(q).view(B, C, -1)
        # k = self.k_conv(k).view(B, C, -1)
        # v = self.v_conv(v).view(B, C, -1)
        # attn = (q @ k.transpose(-2, -1)) / C**0.5
        # attn = F.softmax(attn, dim=-1)
        # x = (attn @ v).view(B, C, H, W)
        # x = self.proj(x)
        # return x

# class DirectionAttention(nn.Module):
#     def __init__(self, in_dim, hidden_dim): # 576 512
#         super().__init__()
#         self.q_conv = nn.Conv2d(in_dim, hidden_dim, 1)
#         self.k_conv = nn.Conv2d(in_dim, hidden_dim, 1)
#         self.v_conv = nn.Conv2d(in_dim, in_dim, 1)
#         self.proj = nn.Conv2d(in_dim, in_dim, 1)

#     def forward(self, q_lst, k_lst, area_lst):
#         B, C, H, W = q_lst[0].shape
#         q_lst = [self.q_conv(q).view(B, C, -1) for q in q_lst]
#         k_lst = [self.k_conv(k).view(B, C, -1) for k in k_lst]
#         v_lst = [self.v_conv(v).view(B, C, -1) for v in q_lst]
#         attn_lst = []
#         for q, k, v in zip(q_lst, k_lst, v_lst):    
#             attn = (q @ k.transpose(-2, -1)) / C**0.5
#             attn = F.softmax(attn, dim=-1)
#             attn_lst.append(attn)
#         attn_lst = [attn * area.unsqueeze(-1) for attn, area in zip(attn_lst, area_lst)]
#         attn_lst = [attn / attn.sum(dim=-1, keepdim=True) for attn in attn_lst]
#         x = sum([attn.unsqueeze(-1) * v for attn, v in zip(attn_lst, v_lst)])
#         x = x.view(B, C, H, W)
#         x = self.proj(x)
#         return x

class DirectionAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim): # 576 512
        super().__init__()
        self.q_conv = nn.Linear(in_dim, hidden_dim, 1)
        self.k_conv = nn.Linear(in_dim, hidden_dim, 1)  
        
  
        self.v_conv = nn.Linear(in_dim, in_dim, 1)
        self.proj = nn.Linear(in_dim, out_dim, 1)
        
        self.conv1 = nn.Linear(in_dim, hidden_dim*4, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.act = h_swish()
        
        self.conv2_1 = nn.Linear(hidden_dim, out_dim, 1)
        self.conv2_2 = nn.Linear(hidden_dim, out_dim, 1)
        self.conv2_3 = nn.Linear(hidden_dim, out_dim, 1)
        self.conv2_4 = nn.Linear(hidden_dim, out_dim, 1)
        

    def forward(self, preds, rel_coords, areas) :
        '''
        preds: [4 (B, N, C)]
        rel_coords: [4 (B, N, 2)]
        areas: [4 (B, N)]
        '''
        # preds_all = torch.stack(preds, dim=0)
        
        # 坐标偏差作为权重
        # 设一个areas同尺寸的全1矩阵
        # areas = F.softmax(torch.ones_like(torch.stack(areas,dim=0))-torch.stack(areas,dim=0), dim=0)
        # attn_lst = []
        
        # ret = 0
        # for i in range(len(preds)):
        #     attn = preds[i] * areas[i].unsqueeze(-1)
        #     ret = ret + attn
        # ret = self.proj(ret)
        preds_all = torch.stack(preds, dim=-1)
        areas_all = torch.stack(areas, dim=-1)
        
        B, N, C, K = preds_all.shape
        
        areas_all = areas_all.view(-1, 4)
        
        dire_1234 = self.conv1(areas_all)
        dire_1234 = self.bn1(dire_1234)
        dire_1234 = self.act(dire_1234).view(B, N, self.hidden_dim*4)
        
        dire_1, dire_2, dire_3, dire_4 = torch.split(dire_1234, [self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim], dim=-1)
        
        dire_1 = self.conv2_1(dire_1)
        dire_2 = self.conv2_2(dire_2)
        dire_3 = self.conv2_3(dire_3)
        dire_4 = self.conv2_4(dire_4)
        
        ret = preds[0] * dire_1 + preds[1] * dire_2 + preds[2] * dire_3 + preds[3] * dire_4
        return ret
        
        
    
@register('coord')
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
            # imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None
            
        self.kannet = KAN(layers_hidden=[576+2, 256, 256, 3])
        
        self.coord_attention = CoordAttention(in_dim=3, hidden_dim=8)
        # self.direction_attention = DirectionAttention(in_dim=4, hidden_dim=48, out_dim=3)
        hidden_dim = 8
        
        
        self.conv1_c = nn.Linear(3+2, hidden_dim*3, 1)
        self.bn1_c = nn.BatchNorm1d(hidden_dim*3)
        self.act_c = h_swish()
        
        self.dis_conv_c = nn.Linear(hidden_dim, 1, 1)
        self.x_conv_c = nn.Linear(hidden_dim, 1, 1)
        self.y_conv_c = nn.Linear(hidden_dim, 1, 1)
        
        in_dim = 4
        # hidden_dim = 8
        self.hidden_dim = hidden_dim
        out_dim = 1
        self.conv1 = nn.Linear(in_dim, hidden_dim*4, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.act = h_swish()
        if self.cell_decode:
            # self.conv1_c = nn.Linear(3, hidden_dim*3, 1)
            
            self.conv2_1 = nn.Linear(hidden_dim+2, out_dim, 1)
            self.conv2_2 = nn.Linear(hidden_dim+2, out_dim, 1)
            self.conv2_3 = nn.Linear(hidden_dim+2, out_dim, 1)
            self.conv2_4 = nn.Linear(hidden_dim+2, out_dim, 1)
        else:
            self.conv2_1 = nn.Linear(hidden_dim, out_dim, 1)
            self.conv2_2 = nn.Linear(hidden_dim, out_dim, 1)
            self.conv2_3 = nn.Linear(hidden_dim, out_dim, 1)
            self.conv2_4 = nn.Linear(hidden_dim, out_dim, 1)

    def gen_feat(self, inp):
        self.inp = inp
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
                # pred_feat = self.coord_attention(q_feat, q_coord, coord) # 8 2304 576
                identity = q_feat
                B, N, C = q_feat.shape
                
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    # inp = torch.cat([inp, rel_cell], dim=-1)
                    
                    coord_diff = (coord - q_coord)
                    coord_diff[:, :, 0] *= feat.shape[-2]
                    coord_diff[:, :, 1] *= feat.shape[-1]
                    coord_dist = coord_diff.pow(2).sum(dim=-1, keepdim=True)
                    coord_feat = torch.cat([coord_diff, coord_dist,rel_cell], dim=-1)
                else:
                    coord_diff = (coord - q_coord)
                    coord_diff[:, :, 0] *= feat.shape[-2]
                    coord_diff[:, :, 1] *= feat.shape[-1]
                    coord_dist = coord_diff.pow(2).sum(dim=-1, keepdim=True)
                    coord_feat = torch.cat([coord_diff, coord_dist], dim=-1)
        
                xyd = self.conv1_c(coord_feat)
                xyd = self.bn1_c(xyd.view(-1, self.hidden_dim*3))
                xyd = self.act_c(xyd).view(B, N, self.hidden_dim*3)
                
                x_dif, y_dif, dist = torch.split(xyd, [self.hidden_dim, self.hidden_dim, self.hidden_dim], dim=-1)
                
                # x_dif = self.x_conv_c(torch.concat([x_dif,rel_cell], dim=-1))
                # y_dif = self.y_conv_c(torch.concat([y_dif,rel_cell], dim=-1))
                # dist = self.dis_conv_c(torch.concat([dist,rel_cell], dim=-1))
                
                x_dif = self.x_conv_c(x_dif)
                y_dif = self.y_conv_c(y_dif)
                dist = self.dis_conv_c(dist)
                
                pred_feat = (x_dif + y_dif + dist) * identity

                # pred_feat = self.imnet(torch.concat([pred_feat, rel_cell], dim=-1)).view(coord.shape[0], coord.shape[1], -1)
                
                # 计算相对坐标,作为每一个方向的注意力输入key
                rel_coord = (q_coord - coord) * feat.shape[-2] # 8 2304 2
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
        
        # preds_feat = self.direction_attention(preds, rel_coords, areas) # 结合方向大小注意力进行预测preds_feat: 8 2304 576
        
        # ret = preds_feat
        
        # if self.cell_decode:
        #     rel_cell = cell.clone()
        #     rel_cell[:, :, 0] *= feat.shape[-2]
        #     rel_cell[:, :, 1] *= feat.shape[-1]
        #     ret = torch.cat([ret, rel_cell], dim=-1)  # 8 2304 576+2
        # #     ret = self.imnet(preds_feat_rel_cell.view(coord.shape[0] * coord.shape[1], -1)).view(coord.shape[0], coord.shape[1], -1)
        # ret = self.imnet(ret.view(coord.shape[0] * coord.shape[1], -1)).view(coord.shape[0], coord.shape[1], -1)    
        
        # # # concat cell等信息
        
        # # tot_area = torch.stack(areas).sum(dim=0)
        # # if self.local_ensemble:
        # #     t = areas[0]; areas[0] = areas[3]; areas[3] = t
        # #     t = areas[1]; areas[1] = areas[2]; areas[2] = t
        # # ret = 0
        # # for pred, area in zip(preds, areas):
        # #     ret = ret + pred * (area / tot_area).unsqueeze(-1)
        # return ret
        
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        # (areas / tot_area)
    
        preds_all = torch.stack(preds, dim=-1)
        areas_all = torch.stack(areas, dim=-1) /tot_area.unsqueeze(-1)

        B, N, C, K = preds_all.shape
        
        areas_all = areas_all.view(-1, 4)
        
        dire_1234 = self.conv1(areas_all)
        dire_1234 = self.bn1(dire_1234)
        dire_1234 = self.act(dire_1234).view(B, N, self.hidden_dim*4)
        
        dire_1, dire_2, dire_3, dire_4 = torch.split(dire_1234, [self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim], dim=-1)
        
        if self.cell_decode:
            rel_cell = cell.clone()
            rel_cell[:, :, 0] *= feat.shape[-2]
            rel_cell[:, :, 1] *= feat.shape[-1]
            dire_1 = self.conv2_1(torch.cat([dire_1, rel_cell], dim=-1))
            dire_2 = self.conv2_2(torch.cat([dire_2, rel_cell], dim=-1))
            dire_3 = self.conv2_3(torch.cat([dire_3, rel_cell], dim=-1))
            dire_4 = self.conv2_4(torch.cat([dire_4, rel_cell], dim=-1))
            
        else:
            dire_1 = self.conv2_1(dire_1)
            dire_2 = self.conv2_2(dire_2)
            dire_3 = self.conv2_3(dire_3)
            dire_4 = self.conv2_4(dire_4)
            
        # res_inp = F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
        #               padding_mode='border', align_corners=False)[:, :, 0, :] \
        #               .permute(0, 2, 1)
        
        ret = preds[0] * dire_1 + preds[1] * dire_2 + preds[2] * dire_3 + preds[3] * dire_4
        ret = self.imnet(torch.cat([ret, rel_cell], dim=-1)).view(coord.shape[0], coord.shape[1], -1)
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        # ret = self.kannet(torch.cat([ret, rel_cell], dim=-1).view(coord.shape[0] * coord.shape[1], -1)).view(coord.shape[0], coord.shape[1], -1)
        # ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
        #               padding_mode='border', align_corners=False)[:, :, 0, :] \
        #               .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
    

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        # assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
        
# 测试代码，测试coord attention
if __name__ == '__main__':
    # model = CoordAttention(in_dim=3, hidden_dim=32)
    # q_feat = torch.randn(8, 2304, 576)
    # q_coord = torch.randn(8, 2304, 2)
    # coord = torch.randn(8, 2304, 2)
    # out = model(q_feat, q_coord, coord)
    # print(out.shape)
    
    # print(out)
    
    # 测试 DirectionAttention 代码
    model = DirectionAttention(in_dim=3, hidden_dim=32)
    
    preds = [torch.randn(8, 2304, 576) for i in range(4)]
    rel_coords = [torch.randn(8, 2304, 2) for i in range(4)]
    areas = [torch.randn(8, 2304) for i in range(4)]
    
    out = model(preds, rel_coords, areas)
   
    print(out)