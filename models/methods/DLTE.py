# https://github.com/HeZongyao/DIIF
# ICME 2024
# Dynamic Implicit Image Function for Efficient Arbitrary-Scale Super-Resolution

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.networks as networks


def make_cell_info(out_shape, in_shape, v_range=None, device=None):
    """ Make cell info of each latent code. A cell contains the nearest output pixels of an latent code.
            out_shape: [s*H,s*W]
            in_shape: [H,W]
            stack_unit: stack coordinate nums
            cell_decode: True or False
            v_range: coord range[-1, 1]
        return:
            cell_info: (top_coord, left_coord, bottom_coord, right_coord, h_cell, w_cell, h_num. w_num, h_max_num. w_max_num, startIndex, endIndex) * H * W
            coord_num_seqs: [h axis coordinate nums, w axis coordinate nums]
    """

    device = device if device is not None else torch.device('cpu')  # use gpu or cpu

    if v_range is None:
        v0, v1 = -1, 1
    else:
        v0, v1 = v_range
    [in_h, in_w] = in_shape
    [out_h, out_w] = out_shape
    in_radius_h, in_radius_w = (v1 - v0) / (2 * in_h), (v1 - v0) / (2 * in_w)
    out_radius_h, out_radius_w = (v1 - v0) / (2 * out_h), (v1 - v0) / (2 * out_w)
    h_ratio, w_ratio = out_h / in_h, out_w / in_w

    # input index sequences on each axis
    in_h_axis_seq, in_w_axis_seq = torch.arange(in_h).float(), torch.arange(in_w).float()

    # output index sequences on each axis
    start_index_seqs = [
        (torch.arange(in_h).float() * h_ratio - 0.5).ceil(),
        (torch.arange(in_w).float() * w_ratio - 0.5).ceil()
    ]
    end_index_seqs = [
        ((torch.arange(in_h).float() + 1) * h_ratio - 0.5).ceil(),
        ((torch.arange(in_w).float() + 1) * w_ratio - 0.5).ceil()
    ]
    # coordinate numbers of each cell, (h_num, w_num)
    coord_num_seqs = [
        end_index_seqs[0] - start_index_seqs[0],
        end_index_seqs[1] - start_index_seqs[1]
    ]

    # cell center coordinates
    in_h_coord_seq, in_w_coord_seq = in_radius_h + 2 * in_radius_h * in_h_axis_seq, in_radius_w + 2 * in_radius_w * in_w_axis_seq
    # output border coordinates relative to cell center, (top, left, bottom, right)
    start_coord_seqs = [
        in_h * (out_radius_h + 2 * out_radius_h * start_index_seqs[0] - in_h_coord_seq),
        in_w * (out_radius_w + 2 * out_radius_w * start_index_seqs[1] - in_w_coord_seq)
    ]
    end_coord_seqs = [
        in_h * (out_radius_h + 2 * out_radius_h * (end_index_seqs[0] - 1) - in_h_coord_seq),
        in_w * (out_radius_w + 2 * out_radius_w * (end_index_seqs[1] - 1) - in_w_coord_seq),
    ]

    start_borders = torch.stack(torch.meshgrid(*start_coord_seqs), dim=-1).to(device)
    end_borders = torch.stack(torch.meshgrid(*end_coord_seqs), dim=-1).to(device)
    #num_grids = torch.stack(torch.meshgrid(*coord_num_seqs), dim=-1).to(device)

    # area of coordinate, (h_area, w_area)
    cell_areas = torch.ones(in_h, in_w, 2).to(device)
    cell_areas[:, :, 0] = 2 * in_h / out_h
    cell_areas[:, :, 1] = 2 * in_w / out_w

    return cell_areas.permute(2, 0, 1), start_borders.permute(2, 0, 1), end_borders.permute(2, 0, 1), coord_num_seqs


# Dynamic LTE framework
class DLTE(nn.Module):
    def __init__(self, opt_encoder, opt_imnet, unfold_num=1, single_slice=False, coarse_cell_decode=True,
                 fine_cell_decode=False, local_ensemble=False, feat_unfold=False, cell_decode=True, fixed_area=False):
        super(DLTE, self).__init__()

        # slicing mode
        self.single_slice = single_slice
        # cell docoding
        self.fine_cell_decode = cell_decode and fine_cell_decode
        self.coarse_cell_decode = cell_decode and coarse_cell_decode
        self.cell_decode = cell_decode
        # slice ensemble
        self.local_ensemble = local_ensemble
        # feature unfolding
        self.feat_unfold = feat_unfold
        self.unfold_num = unfold_num
        self.unfold_num2 = unfold_num * unfold_num

        # testing config
        self.batch_unit = 512 * 384  # 256 * 256 | 512 * 384
        self.fixed_area = fixed_area
        self.max_scale = 4
        self.fixed_border = False
        # self.device = torch.device(('cuda:' + str(gpu_ids[0])) if gpu_ids is not None else 'cpu')  # use gpu or cpu

        # encoder for output latent code
        if opt_imnet is not None:
            opt_encoder['scale'] = 1
        self.encoder = networks.define_module(opt_encoder)

        hidden_dim = opt_imnet['in_dim']
        ensemble_num = opt_imnet['ensemble_num']

        self.coef = nn.Conv2d(opt_encoder['out_nc'] if 'out_nc' in opt_encoder else 64, hidden_dim // (ensemble_num * ensemble_num), 3, padding=1)
        self.freq = nn.Conv2d(opt_encoder['out_nc'] if 'out_nc' in opt_encoder else 64, hidden_dim // (ensemble_num * ensemble_num), 3, padding=1)

        #opt_imnet['in_dim'] = hidden_dim
        opt_imnet['local_ensemble'] = self.local_ensemble
        opt_imnet['unfold_num'] = self.unfold_num if self.local_ensemble else 0
        opt_imnet['ensemble_num'] = opt_imnet['ensemble_num'] if self.local_ensemble else 0
        self.imnet = networks.define_module(opt_imnet)

    def unfold_feat(self, feat, coef, freq, w_seq=None):
        """
        Unfold feature maps

        :param feat:
        :param coef:
        :param freq:
        :param w_seq:
        :return:
        """

        [in_b, in_c, in_h, in_w] = feat.shape

        if w_seq is None:
            w_start, w_end = 0, in_w
        else:
            padding = self.unfold_num // 2
            w_start = w_seq[0] - padding if w_seq[0] - padding >= 0 else 0
            w_end = w_seq[1] + padding if w_seq[1] + padding <= in_w else in_w
            in_w = w_end - w_start

        if self.feat_unfold:
            coef = F.unfold(coef[:, :, :, w_start: w_end], self.unfold_num, padding=self.unfold_num // 2).view(
                in_b, in_c, self.unfold_num, self.unfold_num, in_h, in_w)
            freq = F.unfold(freq[:, :, :, w_start: w_end], self.unfold_num, padding=self.unfold_num // 2).view(
                in_b, in_c, self.unfold_num, self.unfold_num, in_h, in_w)

        if w_seq is None:
            return feat, coef, freq
        else:
            start = padding if w_seq[0] - padding > 0 else 0
            end = -padding if w_seq[1] + padding <= feat.shape[-1] else feat.shape[-1]
            return feat[:, :, :, w_seq[0]:w_seq[1]], coef[:, :, :, :, :, start:end], freq[:, :, :, :, :, start:end]

    def query_rgb(self, feat, out_size, cell_areas=None, start_borders=None, end_borders=None, coord_num_seqs=None):
        """
        Upscale with mlp network

        feat: input feature map, (B, C, H, W)
        out_size: output size, (B, 3, sH, sW)
        cell_info: cell info, (stack_unit*2 (+2), H, W)
        coord_num_seqs: coordinate nums of h and w axis, list
        """

        # sample upscale without mlp network
        if self.imnet is None:
            # coord: (B, sH * sW, 2)
            coord = None
            out = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            return out

        coef = self.coeff
        freq = self.freqq

        # get input and output size
        [in_b, in_c, in_h, in_w] = feat.shape
        [out_c, out_h, out_w]  = out_size[-3:]

        [h_num_seq, w_num_seq] = coord_num_seqs
        h_max_num, w_max_num = int(h_num_seq.max()), int(w_num_seq.max())
        h_min_num, w_min_num = int(h_num_seq.min()), int(w_num_seq.min())

        # determine coordinate slice direction, row or column
        slice_by_w = True
        if slice_by_w:
            main_max_num, cross_max_num = h_max_num, w_max_num
            main_dim_index, cross_dim_index = 0, 1
        else:
            main_max_num, cross_max_num = w_max_num, h_max_num
            main_dim_index, cross_dim_index = 1, 0

        # coordinate shift relative to slice start border (top, left)
        coord_shifts = torch.ones(2, in_h, in_w).to(feat.device)

        # cell info, shape: ((area) + start coord + end coord, H, W)
        if self.coarse_cell_decode:
            cell_info = torch.ones(6, in_h, in_w).to(feat.device)
            cell_info[:2, :, :] = cell_areas.clone()

            # TO DO: supporting training with x4 only model
            if self.fixed_area and out_h > in_h * self.max_scale:
                cell_info[:1, :, :] = 2 / self.max_scale
            if self.fixed_area and out_w > in_w * self.max_scale:
                cell_info[1:2, :, :] = 2 / self.max_scale
        else:
            cell_info = torch.ones(4, in_h, in_w).to(feat.device)

        # determine slice mode
        if self.single_slice:
            main_borders = [[0, main_max_num]]
            cross_borders = [[0, cross_max_num]]
        else:
            main_borders = []
            for main_index in range(main_max_num):
                main_borders.append([main_index, main_index + 1])
            cross_borders = [[0, cross_max_num]]

        # merge slices based on the input size
        #batched_interval = 1 if self.single_slice or self.training else int(256 * 256 / (in_w * in_h)) # int(self.batch_unit / (in_w * in_h))
        batched_interval = 1
        batched_index = 0
        batched_coef, batched_freq, batched_slice_coords, batched_slice_info = [], [], [], []

        # get pixels by coordinate slice order
        preds = []
        for main_border in main_borders:
            for cross_border in cross_borders:

                if batched_interval > 1:
                    batched_index += 1

                if main_border[0] == main_border[1] or cross_border[0] == cross_border[1]:
                    continue

                slice_info = cell_info.clone()
                # slice start coordinate (top, left)
                coord_shifts[main_dim_index, :, :] = main_border[0]
                coord_shifts[cross_dim_index, :, :] = cross_border[0]
                slice_info[-4:-2, :, :] = start_borders + coord_shifts * cell_areas
                # slice end coordinate (bottom, right)
                coord_shifts[main_dim_index, :, :] = main_border[1] - 1
                coord_shifts[cross_dim_index, :, :] = cross_border[1] - 1
                slice_info[-2:, :, :] = start_borders + coord_shifts * cell_areas

                # fixed borders with scale 4
                if self.fixed_border:
                    slice_info[-4 + cross_dim_index, :, :] = -0.75
                    slice_info[-2 + cross_dim_index, :, :] = 0.75
                    if self.single_slice:
                        slice_info[-4 + main_dim_index, :, :] = -0.75
                        slice_info[-2 + main_dim_index, :, :] = 0.75

                # get slice coordinates
                slice_coords = []
                for main_index in range(main_border[0], main_border[1]):
                    coord_shifts[main_dim_index, :, :] = main_index
                    for cross_index in range(cross_border[0], cross_border[1]):
                        coord_shifts[cross_dim_index, :, :] = cross_index
                        slice_coords.append(start_borders + coord_shifts * cell_areas)
                        if self.fine_cell_decode:
                            this_cell_area = cell_areas.clone()

                            # TO DO: supporting training with x4 only model
                            if self.fixed_area and out_h > in_h * self.max_scale:
                                this_cell_area[:1, :, :] = 2 / self.max_scale
                            if self.fixed_area and out_w > in_w * self.max_scale:
                                this_cell_area[1:2, :, :] = 2 / self.max_scale
                            slice_coords.append(this_cell_area)
                slice_coords = torch.cat(slice_coords, dim=0)

                # repeat input by batch size
                slice_info = slice_info.unsqueeze(0).repeat(in_b, 1, 1, 1)
                slice_coords = slice_coords.unsqueeze(0).repeat(in_b, 1, 1, 1)
                slice_coords = slice_coords.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)

                if self.local_ensemble:
                    # reshape mlp input to (B * H * W, C')
                    #inp = feat.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)
                    slice_info = slice_info.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)

                    if batched_interval <= 1:
                        # use mlp to predict pixels, shape: (B * H * W, S * 3)
                        pred = self.imnet(coef, freq, slice_coords, slice_info)
                        preds.append(pred)
                    else:
                        batched_coef.append(coef)
                        batched_freq.append(freq)
                        batched_slice_coords.append(slice_coords)
                        batched_slice_info.append(slice_info)

                        if batched_index == batched_interval:
                            # use mlp to predict pixels, shape: (B * H * W, bi * S * 3)
                            pred = self.imnet(torch.cat(batched_coef, dim=0),
                                              torch.cat(batched_freq, dim=0),
                                              torch.cat(batched_slice_coords, dim=0),
                                              torch.cat(batched_slice_info, dim=0))
                            for pred_index in range(batched_interval):
                                preds.append(pred[pred_index * in_b * in_h * in_w : (pred_index + 1) * in_b * in_h * in_w])
                            batched_index = 0
                            batched_coef, batched_freq, batched_slice_coords, batched_slice_info = [], [], [], []
                else:
                    # concat feature map and slice info as mlp input, shape: (B, C', H, W)
                    #inp = torch.cat([feat, slice_info], dim=1)
                    # reshape mlp input to (B * H * W, C')
                    #inp = inp.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)

                    # use mlp to predict pixels, shape: (B * H * W, S * 3)
                    pred = self.imnet(coef, freq, slice_coords)
                    preds.append(pred)

        if len(batched_coef) > 0:
            pred = self.imnet(torch.cat(batched_coef, dim=0),
                              torch.cat(batched_freq, dim=0),
                              torch.cat(batched_slice_coords, dim=0),
                              torch.cat(batched_slice_info, dim=0))
            for pred_index in range(len(batched_coef)):
                preds.append(pred[pred_index * in_b * in_h * in_w : (pred_index + 1) * in_b * in_h * in_w])

        # reshape predicted pixels to (B. H, W, main axis coord num, cross axis coord num, channel)
        pred_rgbs = torch.cat(preds, dim=1).view(in_b, in_h, in_w, main_max_num, cross_max_num, out_c)

        # reshape predicted pixels to (B. C, H, h_num, W, w_num)
        if slice_by_w:
            pred_rgbs = pred_rgbs.permute(0, -1, 1, 3, 2, 4).contiguous()
        else:
            pred_rgbs = pred_rgbs.permute(0, -1, 1, 4, 2, 3).contiguous()
        # reshape predicted pixels to (B. C, H * h_num, W * w_num)
        pred_rgbs = pred_rgbs.view(in_b, out_c, in_h * h_max_num, in_w * w_max_num)

        if h_max_num > h_min_num or w_max_num > w_min_num:
            # remove unnecessary pixels in h axis
            pred_img_h = []
            for in_h_index in range(in_h):
                h_num = int(h_num_seq[in_h_index])
                h_start_index = in_h_index * h_max_num
                pred_img_h.append(pred_rgbs[:, :, h_start_index: h_start_index + h_num, :])
            pred_img_h = torch.cat(pred_img_h, dim=-2)

            # remove unnecessary pixels in w axis
            pred_img_w = []
            for in_w_index in range(in_w):
                w_num = int(w_num_seq[in_w_index])
                w_start_index = in_w_index * w_max_num
                pred_img_w.append(pred_img_h[:, :, :, w_start_index: w_start_index + w_num])
            pred_img = torch.cat(pred_img_w, dim=-1)
        else:
            pred_img = pred_rgbs

        return pred_img

    def forward(self, x, h = 0, w = 0):
        """ x: input image
            h: output height
            w: output width
        """

        if h == 0 or w == 0:
            out_shape = [x.shape[0], x.shape[1], x.shape[-2], x.shape[-1]]  # B, C, H, W
        else:
            out_shape = [x.shape[0], x.shape[1], h, w]  # B, C, H, W

        feat = self.encoder(x)
        coef = self.coef(feat)
        freq = self.freq(feat)

        cell_areas, start_borders, end_borders, coord_num_seqs = make_cell_info(
            out_shape[-2:], feat.shape[-2:], device=feat.device)

        if self.training:
            feat, coef, freq = self.unfold_feat(feat, coef, freq)
            self.coeff = coef
            self.freqq = freq

            out = self.query_rgb(feat, out_shape, cell_areas=cell_areas, start_borders=start_borders,
                                 end_borders=end_borders, coord_num_seqs=coord_num_seqs)
        else:
            # query rgb values by batches in w axis
            batch_num = math.ceil(feat.shape[-2] * feat.shape[-1] / self.batch_unit)
            out_batches = []
            for batch_index in range(batch_num):
                w_start = int(feat.shape[-1] * batch_index / batch_num)
                w_end = feat.shape[-1] if batch_index == batch_num - 1 else int(feat.shape[-1] * (batch_index + 1) / batch_num)
                out_w = sum(coord_num_seqs[1][w_start: w_end])

                feat_batch, coef_batch, freq_batch = self.unfold_feat(feat, coef, freq, [w_start, w_end])
                self.coeff = coef_batch
                self.freqq = freq_batch
                out_batch = self.query_rgb(feat_batch, [out_shape[0], out_shape[1], out_shape[-2], out_w],
                                           cell_areas=cell_areas[:, :, w_start: w_end],
                                           start_borders=start_borders[:, :, w_start: w_end],
                                           end_borders=end_borders[:, :, w_start: w_end],
                                           coord_num_seqs=[coord_num_seqs[0], coord_num_seqs[1][w_start: w_end]])
                out_batches.append(out_batch)
            out = torch.cat(out_batches, dim=-1)

        out += F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
        return out


class C2FMLP(nn.Module):
    """ Corse-to-fine MLP network for LTE """
    def __init__(self, in_dim, out_dim, c_dim, head_hidden_list, tail_hidden_list, unfold_num=3, ensemble_num=0, local_ensemble=False):
        super(C2FMLP, self).__init__()
        #self.useReshape = out_dim < in_dim
        self.c_dim = c_dim if c_dim is not None else 2
        self.unfold_num = unfold_num if unfold_num is not None else 3
        self.ensemble_num = ensemble_num if ensemble_num is not None else int(1 + self.unfold_num // 2)
        self.local_ensemble = local_ensemble if local_ensemble is not None else False

        #self.v_unit = 256 * 256
        self.coord_unit = 16 # 4 * 3

        self.phase = nn.Linear(2, in_dim // 2, bias=False)

        layers = []
        lastv = in_dim
        for hidden in head_hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        self.layers = nn.Sequential(*layers)

        tails = []
        lastv = lastv + self.c_dim
        for hidden in tail_hidden_list:
            tails.append(nn.Linear(lastv, hidden))
            tails.append(nn.ReLU())
            lastv = hidden
        tails.append(nn.Linear(lastv, out_dim))
        self.tail = nn.Sequential(*tails)

    def get_coarse_inp(self, coef, freq, slice_coord=None, vh=0, vw=0):
        this_slice_coord = slice_coord.clone()
        # start border
        this_slice_coord[:, -4] -= vh
        this_slice_coord[:, -3] -= vw
        # end border
        this_slice_coord[:, -2] -= vh
        this_slice_coord[:, -1] -= vw

        h_start = self.unfold_num // 2 + int(vh * 0.5 + 0.5) - self.ensemble_num // 2
        w_start = self.unfold_num // 2 + int(vw * 0.5 + 0.5) - self.ensemble_num // 2

        freq_ = freq[:, :, h_start:h_start + self.ensemble_num, w_start:w_start + self.ensemble_num, :, :]
        freq_ = freq_.reshape(freq_.shape[0], freq_.shape[1] * self.ensemble_num * self.ensemble_num, freq_.shape[-2], freq_.shape[-1]).permute(
            0, 2, 3, 1).contiguous()
        freq_ = freq_.view(freq_.shape[0], freq_.shape[1] * freq_.shape[2], -1)

        coef_ = coef[:, :, h_start:h_start + self.ensemble_num, w_start:w_start + self.ensemble_num, :, :]
        coef_ = coef_.reshape(coef_.shape[0], coef_.shape[1] * self.ensemble_num * self.ensemble_num, coef_.shape[-2], coef_.shape[-1]).permute(
            0, 2, 3, 1).contiguous()
        coef_ = coef_.view(coef_.shape[0], coef_.shape[1] * coef_.shape[2], -1)

        bs, q = coef_.shape[0], coef_.shape[1]
        this_slice_coord = this_slice_coord.view(bs, q, this_slice_coord.shape[-1])

        freq_ = torch.stack(torch.split(freq_, 2, dim=-1), dim=-1)
        freq_start = torch.mul(freq_, this_slice_coord[:, :, :2].unsqueeze(-1))
        freq_end = torch.mul(freq_, this_slice_coord[:, :, 2:4].unsqueeze(-1))
        freq_ = torch.sum(torch.cat([freq_start, freq_end], dim=-2), dim=-2)
        freq_ += self.phase(this_slice_coord[:, :, 4:]).view(bs, q, -1)
        freq_ = torch.cat((torch.cos(np.pi * freq_), torch.sin(np.pi * freq_)), dim=-1)

        return torch.mul(coef_, freq_).contiguous().view(bs * q, -1)

    def get_fine_inp(self, vs, coord, coord_index):
        # c_dim: coordinate (2 dim) + area (2 dim)
        this_coord = coord[:, coord_index * self.c_dim:(coord_index + 1) * self.c_dim]

        # get average rgb hidden vector using area sizes
        areas = []
        total_area = 0
        for vh in [-1, 1]:  # top, bottom
            for vw in [-1, 1]:  # left, right
                area = torch.abs((this_coord[:, 0] - vh) * (this_coord[:, 1] - vw))
                areas.append(area + 1e-9)
                total_area += area + 1e-9
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t

        v_avg = 0
        for this_v, this_area in zip(vs, areas):
            v_avg = v_avg + this_v * (this_area / total_area).unsqueeze(-1)
        return torch.cat([v_avg, this_coord], dim=-1)

    def query(self, coef, freq, coord, slice_coord=None):
        vs = []
        # get four rgb hidden vectors
        for vh in [-1, 1]:  # top, bottom
            for vw in [-1, 1]:  # left, right
                # get rgb hidden vector
                this_v = self.layers(self.get_coarse_inp(coef, freq, slice_coord, vh, vw))
                vs.append(this_v)

        coord_num = coord.shape[-1] // self.c_dim
        coord_unit_num = int(math.ceil(coord_num/self.coord_unit))
        outs = []
        for coord_unit_index in range(coord_unit_num):
            coord_start = coord_unit_index * self.coord_unit
            coord_end = min((coord_unit_index + 1) * self.coord_unit, coord_num)
            inps = []
            for coord_index in range(coord_start, coord_end):
                inps.append(self.get_fine_inp(vs, coord, coord_index))
            out = self.tail(torch.cat(inps, dim=0))
            outs.append(out)

        v_num = coef.shape[0] * coef.shape[-2] * coef.shape[-1]
        outs = torch.cat(outs, dim=0).view(coord_num, v_num, -1).permute(1, 0, 2).contiguous().view(v_num, -1)
        return outs

    def forward(self, coef, freq, coord, slice_coord=None):
        """
        :param coef: shape (b, c, u, u, h, w)
        :param freq: shape (b, c, u, u, h, w)
        :param coord: shape (v_num, coord_num * coord_dim)
        :param slice_coord: shape (v_num, coord_dim * 2 + area_dim)
        :return: shape (v_num, coord_num * out_dim)
        """

        if self.local_ensemble:
            if self.training:
                vs = []
                for vh in [-1, 1]:  # top, bottom
                    for vw in [-1, 1]:  # left, right
                        # get rgb hidden vector
                        this_v = self.layers(self.get_coarse_inp(coef, freq, slice_coord, vh, vw))
                        vs.append(this_v)

                coord_num = coord.shape[-1] // self.c_dim
                out = []
                for coord_index in range(coord_num):
                    p = self.tail(self.get_fine_inp(vs, coord, coord_index))
                    out.append(p)
                return torch.cat(out, dim=-1)
            else:
                return self.query(coef, freq, coord, slice_coord)
        else:
            x = None
            v = self.layers(x)

            # 2 dimension coordinate + (2 dimension area)
            coord_num = coord.shape[-1] // self.c_dim
            out = []
            for coord_index in range(coord_num):
                this_coord = coord[:, coord_index * self.c_dim:(coord_index + 1) * self.c_dim]
                p = self.tail(torch.cat([v, this_coord], dim=-1))
                out.append(p)
            return torch.cat(out, dim=-1)