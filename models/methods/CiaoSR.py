# https://github.com/caojiezhang/CiaoSR
# CVPR'2023
# CiaoSR: Continuous Implicit Attention-in-Attention Network for Arbitrary-Scale Image Super-Resolution

"""
Modified from https://github.com/caojiezhang/CiaoSR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
# from models.arch_ciaosr.arch_csnln import CrossScaleAttention

def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

#cross-scale non-local attention
class CrossScaleAttention(nn.Module):
    def __init__(self, channel=64, reduction=2, ksize=3, scale=2, stride=1, softmax_scale=10, average=True, conv=default_conv):
        super(CrossScaleAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale

        self.scale = scale
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_1 = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())  
        self.conv_match_2 = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())  
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())    
        #self.register_buffer('fuse_weight', fuse_weight)

        if 3 in scale:
            self.downx3 = nn.Conv2d(channel, channel, ksize, 3, 1)
        if 4 in scale:
            self.downx4 = nn.Conv2d(channel, channel, ksize, 4, 1)

        self.down = nn.Conv2d(channel, channel, ksize, 2, 1)    

    def forward(self, input):
        _, _, H, W = input.shape

        if not isinstance(self.scale, list):
            self.scale = [self.scale]

        res_y = []
        for s in self.scale:
            
            # if (H%2 != 0):
            #     input = F.pad(input, (0, 0, 0, 1), "constant", 0)
            # if (W%2 != 0):
            #     input = F.pad(input, (0, 1, 0, 0), "constant", 0)

            mod_pad_h, mod_pad_w = 0, 0
            if H % s != 0:
                mod_pad_h = s - H % s
            if W % s != 0:
                mod_pad_w = s - W % s
            input_pad = F.pad(input, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

            #get embedding
            embed_w = self.conv_assembly(input_pad)     # [16, 64, 48, 48]
            match_input = self.conv_match_1(input_pad)  # [16, 32, 48, 48]

            # b*c*h*w
            shape_input = list(embed_w.size())      # b*c*h*w
            input_groups = torch.split(match_input, 1, dim=0)  # 16x[1, 32, 48, 48]
            # kernel size on input for matching
            kernel = s * self.ksize

            # raw_w is extracted for reconstruction
            raw_w = extract_image_patches(embed_w, ksizes=[kernel, kernel],
                                        strides=[self.stride * s, self.stride * s],
                                        rates=[1, 1],
                                        padding='same') # [16, 2304, 576], 2304=64*6*6, 576=48*48/(2*2), [N, C*k*k, L] 

            # raw_shape: [N, C, k, k, L]
            raw_w = raw_w.view(shape_input[0], shape_input[1], kernel, kernel, -1) # [16, 64, 6, 6, 576]
            raw_w = raw_w.permute(0, 4, 1, 2, 3).contiguous()    # [16, 576, 64, 6, 6] raw_shape: [N, L, C, k, k]
            raw_w_groups = torch.split(raw_w, 1, dim=0)  # 16x[1, 576, 64, 6, 6]


            # downscaling X to form Y for cross-scale matching
            ref = F.interpolate(input_pad, scale_factor=1./s, mode='bilinear')  # [16, 64, 24, 24]
            ref = self.conv_match_2(ref)        # [16, 32, 24, 24]
            w = extract_image_patches(ref, ksizes=[self.ksize, self.ksize],
                                    strides=[self.stride, self.stride],
                                    rates=[1, 1],
                                    padding='same')   # [16, 288, 576], 288=32*3*3, 576=24*24
            shape_ref = ref.shape
            
            # w shape: [N, C, k, k, L]
            w = w.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1) # [16, 32, 3, 3, 576]
            w = w.permute(0, 4, 1, 2, 3).contiguous()    # [16, 576, 32, 3, 3] w shape: [N, L, C, k, k]
            w_groups = torch.split(w, 1, dim=0)     # 16x[1, 576, 32, 3, 3]

            y = []
            # 1*1*k*k
            #fuse_weight = self.fuse_weight

            for xi, wi, raw_wi in zip(input_groups, w_groups, raw_w_groups):
                # normalize
                wi = wi[0]  # [576, 32, 3, 3] [L, C, k, k]
                max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                    axis=[1, 2, 3], keepdim=True)), self.escape_NaN) # 
                wi_normed = wi/ max_wi # 
                
                # Compute correlation map
                xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # [1, 32, 50, 50]  xi: 1*c*H*W
                yi = F.conv2d(xi, wi_normed, stride=1)   # [1, 576, 48, 48] [1, L, H, W] L = shape_ref[2]*shape_ref[3]
                # yi = F.conv2d(xi.cpu(), wi_normed.cpu(), stride=1)  #TODO

                yi = yi.view(1, shape_ref[2] * shape_ref[3], shape_input[2], shape_input[3])  # [1, 576, 48, 48]  (B=1, C=32*32, H=32, W=32)
                # rescale matching score
                yi = F.softmax(yi*self.softmax_scale, dim=1)     # [1, 576, 48, 48]
                if self.average == False:
                    yi = (yi == yi.max(dim=1,keepdim=True)[0]).float()

                # deconv for reconsturction
                wi_center = raw_wi[0]   # [576, 64, 6, 6]
                yi = F.conv_transpose2d(yi, wi_center, stride=self.stride*s, padding=s)   #[1, 64, 96, 96]
                # yi = F.conv_transpose2d(yi, wi_center.cpu(), stride=self.stride*s, padding=s).cuda()  #TODO

                # add down
                if s == 2:
                    yi = self.down(yi)  #[1, 64, 48, 48]
                elif s == 3:
                    yi = self.downx3(yi)
                elif s == 4:
                    yi = self.downx4(yi)

                yi =yi/6.
                y.append(yi)

            y = torch.cat(y, dim=0)
            y = y[:, :, :H, :W]

            res_y.append(y)
        
        res_y = torch.cat(res_y, dim=1)

        return res_y  #y

@register('ciaosr')
class CiaoSR(nn.Module):
    """
    The subclasses should define `generator` with `encoder` and `imnet`,
        and overwrite the function `gen_feature`.
    If `encoder` does not contain `mid_channels`, `__init__` should be
        overwrite.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet_q,
                 imnet_k,
                 imnet_v,
                 local_size=2,
                 feat_unfold=True,
                 non_local_attn=True,
                 multi_scale=[2],
                 softmax_scale=1,
                 ):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.local_size = local_size
        self.non_local_attn = non_local_attn
        self.multi_scale = multi_scale
        self.softmax_scale = softmax_scale

        # imnet
        self.encoder = models.make(encoder, args={'no_upsampling': True})
        imnet_dim = self.encoder.out_dim # self.encoder.embed_dim if hasattr(self.encoder, 'embed_dim') else self.encoder.out_dim
        if self.feat_unfold:
            imnet_q_in_dim = imnet_dim * 9
            imnet_k_in_dim = imnet_k_out_dim = imnet_dim * 9
            imnet_v_in_dim = imnet_v_out_dim = imnet_dim * 9
        else:
            imnet_q_in_dim= imnet_dim
            imnet_k_in_dim = imnet_k_out_dim = imnet_dim
            imnet_v_in_dim = imnet_v_out_dim = imnet_dim

        imnet_k_in_dim += 4
        imnet_v_in_dim += 4

        if self.non_local_attn:
            imnet_q_in_dim += imnet_dim * len(multi_scale)
            imnet_v_in_dim += imnet_dim * len(multi_scale)
            imnet_v_out_dim += imnet_dim * len(multi_scale)

        self.imnet_q = models.make(imnet_q, args={'in_dim': imnet_q_in_dim})
        self.imnet_k = models.make(imnet_k, args={'in_dim': imnet_k_in_dim, 'out_dim': imnet_k_out_dim})
        self.imnet_v = models.make(imnet_v, args={'in_dim': imnet_v_in_dim, 'out_dim': imnet_v_out_dim})

        if self.non_local_attn:
            self.non_local_attn_dim = imnet_dim * len(multi_scale)
            self.cs_attn = CrossScaleAttention(channel=imnet_dim, scale=multi_scale)

        self.feat_coord = None

    def gen_feat(self, inp):
        self.inp = inp
        feat = self.encoder(inp)
        '''
        if hasattr(self.encoder, 'embed_dim'):
            # SwinIR
            feat = self.encoder.check_image_size(inp)
            feat = self.encoder.conv_first(feat)
            feat = self.encoder.conv_after_body(self.encoder.forward_features(feat)) + feat
        else:
            feat = self.encoder(inp)
        '''

        if self.training or self.feat_coord is None or self.feat_coord.shape[-2] != feat.shape[-2] \
                or self.feat_coord.shape[-1] != feat.shape[-1]:
            self.feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
                .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        B, C, H, W = feat.shape
        if self.non_local_attn:
            crop_h, crop_w = 48, 48
            if H * W > crop_h * crop_w:
                # Fixme: generate cross attention by image patches
                self.non_local_feat_v = torch.zeros(B, self.non_local_attn_dim, H, W).cuda()
                for i in range(H // crop_h):
                    for j in range(W // crop_w):
                        i1, i2 = i * crop_h, ((i + 1) * crop_h if i < H // crop_h - 1 else H)
                        j1, j2 = j * crop_w, ((j + 1) * crop_w if j < W // crop_w - 1 else W)

                        padding = 3 // 2
                        pad_i1, pad_i2 = (padding if i1 - padding >= 0 else 0), (
                            padding if i2 + padding <= H else 0)
                        pad_j1, pad_j2 = (padding if j1 - padding >= 0 else 0), (
                            padding if j2 + padding <= W else 0)

                        crop_feat = feat[:, :, i1 - pad_i1:i2 + pad_i2, j1 - pad_j1:j2 + pad_j2]
                        crop_non_local_feat = self.cs_attn(crop_feat)
                        self.non_local_feat_v[:, :, i1:i2, j1:j2] = crop_non_local_feat[:, :,
                                                               pad_i1:crop_non_local_feat.shape[-2] - pad_i2,
                                                               pad_j1:crop_non_local_feat.shape[-1] - pad_j2]
            else:
                self.non_local_feat_v = self.cs_attn(feat)  # [16, 64, 48, 48]

        self.feats = [feat]
        return self.feats

    def query_rgb(self, coord, scale=None):
        """Query RGB value of GT.

        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """

        res_features = []
        for feature in self.feats:
            B, C, H, W = feature.shape  # [16, 64, 48, 48]

            if self.feat_unfold:
                feat_q = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
                feat_k = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
                if self.non_local_attn:
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
                    feat_v = torch.cat([feat_v, self.non_local_feat_v], dim=1)  # [16, 576+64, 48, 48]
                else:
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
            else:
                feat_q = feat_k = feat_v = feature

            # query
            query = F.grid_sample(feat_q, coord.flip(-1).unsqueeze(1), mode='nearest',
                                  align_corners=False).permute(0, 3, 2, 1).contiguous()  # [16, 2304, 1, 576]

            #feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1) \
            #    .unsqueeze(0).expand(B, 2, *feature.shape[-2:])  # [16, 2, 48, 48]
            #feat_coord = feat_coord.to(coord)
            feat_coord = self.feat_coord

            if self.local_size == 1:
                v_lst = [(0, 0)]
            else:
                v_lst = [(i, j) for i in range(-1, 2, 4 - self.local_size) for j in range(-1, 2, 4 - self.local_size)]
            eps_shift = 1e-6
            preds_k, preds_v = [], []

            for v in v_lst:
                vx, vy = v[0], v[1]
                # project to LR field
                tx = ((H - 1) / (1 - scale[:, 0, 0])).view(B, 1)  # [16, 1]
                ty = ((W - 1) / (1 - scale[:, 0, 1])).view(B, 1)  # [16, 1]
                rx = (2 * abs(vx) - 1) / tx if vx != 0 else 0  # [16, 1]
                ry = (2 * abs(vy) - 1) / ty if vy != 0 else 0  # [16, 1]

                bs, q = coord.shape[:2]
                coord_ = coord.clone()  # [16, 2304, 2]
                if vx != 0:
                    coord_[:, :, 0] += vx / abs(vx) * rx + eps_shift  # [16, 2304]
                if vy != 0:
                    coord_[:, :, 1] += vy / abs(vy) * ry + eps_shift  # [16, 2304]
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # key and value
                key = F.grid_sample(feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()  # [16, 2304, 576]
                value = F.grid_sample(feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                      align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()  # [16, 2304, 576]

                # Interpolate K to HR resolution
                coord_k = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),
                                        mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2,
                                                                                                 1)  # [16, 2304, 2]

                Q, K = coord, coord_k  # [16, 2304, 2]
                rel = Q - K  # [16, 2304, 2]
                rel[:, :, 0] *= feature.shape[-2]  # without mul
                rel[:, :, 1] *= feature.shape[-1]
                inp = rel  # [16, 2304, 2]

                scale_ = scale.clone()  # [16, 2304, 2]
                scale_[:, :, 0] *= feature.shape[-2]
                scale_[:, :, 1] *= feature.shape[-1]

                inp_v = torch.cat([value, inp, scale_], dim=-1)  # [16, 2304, 580]
                inp_k = torch.cat([key, inp, scale_], dim=-1)  # [16, 2304, 580]

                inp_k = inp_k.contiguous().view(bs * q, -1)
                inp_v = inp_v.contiguous().view(bs * q, -1)

                weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous()  # [16, 2304, 576]
                pred_k = (key * weight_k).view(bs, q, -1)  # [16, 2304, 576]

                weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()  # [16, 2304, 576]
                pred_v = (value * weight_v).view(bs, q, -1)  # [16, 2304, 576]

                preds_v.append(pred_v)
                preds_k.append(pred_k)

            preds_k = torch.stack(preds_k, dim=-1)  # [16, 2304, 576, 4]
            preds_v = torch.stack(preds_v, dim=-2)  # [16, 2304, 4, 576]

            attn = (query @ preds_k)  # [16, 2304, 1, 4]
            x = ((attn / self.softmax_scale).softmax(dim=-1) @ preds_v)  # [16, 2304, 1, 576]
            x = x.view(bs * q, -1)  # [16*2304, 576]

            res_features.append(x)

        result = torch.cat(res_features, dim=-1)  # [16, 2304, 576x2]
        result = self.imnet_q(result)  # [16, 2304, 3]
        result = result.view(bs, q, -1)

        result += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',
                                padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        return result

    def batched_predict(self, x, coord, cell, eval_bsize):
        """Batched predict.

        Args:
            x (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """
        with torch.no_grad():
            if coord is None and cell is None:
                # Evaluate encoder efficiency
                feat = self.encoder(x)
                return None

            self.gen_feat(x)
            n = coord.shape[1]
            left = 0
            preds = []
            while left < n:
                right = min(left + eval_bsize, n)
                pred = self.query_rgb(coord[:, left:right, :], cell[:, left:right, :])
                preds.append(pred)
                left = right
            pred = torch.cat(preds, dim=1)
        return pred

    def forward(self, x, coord, cell, bsize=None):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """
        if bsize is not None:
            pred = self.batched_predict(x, coord, cell, bsize)
        else:
            self.gen_feat(x)
            pred = self.query_rgb(coord, cell)

        return pred