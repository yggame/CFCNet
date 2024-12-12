import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import to_pixel_samples


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
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


def generate_meshgrid(height, width):
    """
    Generate a meshgrid of coordinates for a given image dimensions.
    Args:
        height (int): Height of the image.
        width (int): Width of the image.
    Returns:
        torch.Tensor: A tensor of shape [height * width, 2] containing the (x, y) coordinates for each pixel in the image.
    """
    # Generate all pixel coordinates for the given image dimensions
    y_coords, x_coords = torch.arange(0, height), torch.arange(0, width)
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(y_coords, x_coords)
    # Flatten and stack the coordinates to obtain a list of (x, y) pairs
    all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return all_coords


def fetching_features_from_tensor(image_tensor, input_coords):
    """
    Extracts pixel values from a tensor of images at specified coordinate locations.
    Args:
        image_tensor (torch.Tensor): A 4D tensor of shape [batch, channel, height, width] representing a batch of images.
        input_coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the (x, y) coordinates at which to extract pixel values.
    Returns:
        color_values (torch.Tensor): A 3D tensor of shape [batch, N, channel] containing the pixel values at the specified coordinates.
        coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the normalized coordinates in the range [-1, 1].
    """
    # Normalize pixel coordinates to [-1, 1] range
    input_coords = input_coords.to(image_tensor.device)
    coords = input_coords / torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]],
                                         device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # Fetching the colour of the pixels in each coordinates
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1)

    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device)

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]

    return color_values, coords


@register('gaussian-splatter')
class GaussianSplatter(nn.Module):
    """A module that applies 2D Gaussian splatting to input features."""
    def __init__(self, encoder_spec, fc_spec, kernel_size, hidden_dim=256,
                 unfold_row=7, unfold_column=7, num_points=100):
        """
        Initialize the 2D Gaussian Splatter module.
        Args:
            kernel_size (int): The size of the kernel to convert rasterization.
            unfold_row (int): The number of points in the row dimension of the Gaussian grid.
            unfold_column (int): The number of points in the column dimension of the Gaussian grid.
        """
        super(GaussianSplatter, self).__init__()
        self.encoder = models.make(encoder_spec)
        self.feat, self.logits = None, None  # LR feature and LR logits
        self.inp = None
        self.feat_coord = None
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)
        # Fully-connected layers
        self.fc = models.make(fc_spec, args={'in_dim': hidden_dim})

        # Key parameter in 2D Gaussian Splatter
        self.kernel_size = kernel_size
        self.row = unfold_row
        self.column = unfold_column
        self.num_points = num_points

        # Initialize Trainable Parameters
        sigma_x, sigma_y = torch.meshgrid(torch.linspace(0.2, 3.0, 10), torch.linspace(0.2, 3.0, 10))
        self.sigma_x = sigma_x.reshape(-1)
        self.sigma_y = sigma_y.reshape(-1)
        self.opacity = torch.sigmoid(torch.ones(self.num_points, 1, requires_grad=True))
        self.rho = torch.clamp(torch.zeros(self.num_points, 1, requires_grad=True), min=-1, max=1)
        self.sigma_x = nn.Parameter(self.sigma_x)  # Standard deviation in x-axis
        self.sigma_y = nn.Parameter(self.sigma_y)  # Standard deviation in y-axis
        self.opacity = nn.Parameter(self.opacity)  # Transparency of feature, shape=[num_points, 1]
        self.rho = nn.Parameter(self.rho)

    def weighted_gaussian_parameters(self, logits):
        """
        Computes weighted Gaussian parameters based on logits and the Gaussian kernel parameters (sigma_x, sigma_y, opacity).
        The logits tensor is used as a weight to compute a weighted sum of the Gaussian kernel parameters for each spatial
        location across the batch dimension. The resulting weighted parameters are then averaged across the batch dimension.
        Args:
            logits (torch.Tensor): Logits tensor of shape [batch, class, height, width].
        Returns:
            tuple: A tuple containing the weighted Gaussian parameters:
                - weighted_sigma_x (torch.Tensor): Tensor of shape [height * width] representing the weighted x-axis standard deviations.
                - weighted_sigma_y (torch.Tensor): Tensor of shape [height * width] representing the weighted y-axis standard deviations.
                - weighted_opacity (torch.Tensor): Tensor of shape [height * width] representing the weighted opacities.
        Description:
            This function computes weighted Gaussian parameters based on the input tensor, logits, and the provided Gaussian kernel
            parameters (sigma_x, sigma_y, and opacity). The logits tensor is used as a weight to compute a weighted sum of the Gaussian
            kernel parameters for each spatial location (height and width) across the batch dimension. The resulting weighted parameters
            are then averaged across the batch dimension, yielding tensors of shape [height * width] for the weighted sigma_x, sigma_y,
            and opacity.
        """
        batch_size, num_classes, height, width = logits.size()
        logits = logits.permute(0, 2, 3, 1)  # Reshape logits to [batch, height, width, class]

        # Compute weighted sum of Gaussian parameters across class dimension
        weighted_sigma_x = (logits * self.sigma_x.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_sigma_y = (logits * self.sigma_y.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_opacity = (logits * self.opacity[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_rho = (logits * self.rho[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        # Reshape and average across batch dimension
        weighted_sigma_x = weighted_sigma_x.reshape(batch_size, -1).mean(dim=0)
        weighted_sigma_y = weighted_sigma_y.reshape(batch_size, -1).mean(dim=0)
        weighted_opacity = weighted_opacity.reshape(batch_size, -1).mean(dim=0)
        weighted_rho = weighted_rho.reshape(batch_size, -1).mean(dim=0)

        return weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho

    def gen_feat(self, inp):
        """Generate feature and logits by encoder."""
        self.inp = inp
        self.feat, self.logits = self.encoder(inp)
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda().permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])
        return self.feat, self.logits

    def query_rgb(self, coord, scale, cell=None):
        """
        Continuous sampling through 2D Gaussian Splatting.
        Args:
            coord (torch.Tensor): [Batch, Sample_q, 2]. The normalized coordinates of HR space (of range [-1, 1]).
            cell (torch.Tensor): [Batch, Sample_q, 2]. The normalized cell size of HR space.
            scale (torch.Tensor): [Batch]. The magnification scale of super-resolution. (1, 4) during training.
        Returns:
            torch.Tensor: The output features after Gaussian splatting, of the same shape as the input.
        """
        # 1. Get LR feature and logits
        feat, lr_feat, logits = self.feat[:, :8, :, :], self.feat[:, 8:, :, :], self.logits  # channel decoupling
        feat_size, feat_device = feat.shape, feat.device

        # 2. Calculate the high-resolution image size
        scale = float(scale[0])
        hr_h = round(feat.shape[-2] * scale)  # shape: [batch size]
        hr_w = round(feat.shape[-1] * scale)

        # 3. Unfold the feature / logits to many small patches to avoid extreme GPU memory consumption
        num_kernels_row = math.ceil(feat_size[-2] / self.row)
        num_kernels_column = math.ceil(feat_size[-1] / self.column)
        upsampled_size = (num_kernels_row * self.row, num_kernels_column * self.column)
        upsampled_inp = F.interpolate(feat, size=upsampled_size, mode='bicubic', align_corners=False)
        upsampled_logits = F.interpolate(logits, size=upsampled_size, mode='bicubic', align_corners=False)
        unfold = nn.Unfold(kernel_size=(self.row, self.column), stride=(self.row, self.column))
        unfolded_feature = unfold(upsampled_inp)
        unfolded_logits = unfold(upsampled_logits)
        # Unfolded_feature dimension becomes [Batch, C*K*K, L], where L is the number of columns after unfolding
        L = unfolded_feature.shape[-1]
        unfolded_feature_reshaped = unfolded_feature.transpose(1, 2). \
            reshape(feat_size[0] * L, feat_size[1], self.row, self.column)
        unfold_feat = unfolded_feature_reshaped  # shape: [num of patch * batch, channel, self.row, self.column]
        unfolded_logits_reshaped = unfolded_logits.transpose(1, 2). \
            reshape(logits.shape[0] * L, logits.shape[1], self.row, self.column)
        unfold_logits = unfolded_logits_reshaped  # shape: [num of patch * batch, channel, self.row, self.column]

        # 4. Generate colors_(features) and coords_norm
        coords_ = generate_meshgrid(unfold_feat.shape[-2], unfold_feat.shape[-1])
        num_LR_points = unfold_feat.shape[-2] * unfold_feat.shape[-1]
        colors_, coords_norm = fetching_features_from_tensor(unfold_feat, coords_)

        # 5. Rasterization: Generating grid
        # 5.1. Spread Gaussian points over the whole feature map
        batch_size, channel, _, _ = unfold_feat.shape
        weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho = \
            self.weighted_gaussian_parameters(unfold_logits)
        sigma_x = weighted_sigma_x.view(num_LR_points, 1, 1)
        sigma_y = weighted_sigma_y.view(num_LR_points, 1, 1)
        rho = weighted_rho.view(num_LR_points, 1, 1)

        # 5.2. Gaussian expression
        covariance = torch.stack(
            [torch.stack([sigma_x ** 2 + 1e-5, rho * sigma_x * sigma_y], dim=-1),
             torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2 + 1e-5], dim=-1)], dim=-2
        )  # covariance matrix of Gaussian Distribution
        inv_covariance = torch.inverse(covariance).to(feat_device)

        # 5.3. Choosing a broad range for the distribution [-5,5] to avoid any clipping
        start = torch.tensor([-5.0], device=feat_device).view(-1, 1)
        end = torch.tensor([5.0], device=feat_device).view(-1, 1)
        base_linspace = torch.linspace(0, 1, steps=self.kernel_size, device=feat_device)
        ax_batch = start + (end - start) * base_linspace
        # Expanding dims for broadcasting
        ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, self.kernel_size, -1)

        # 5.4. Creating a batch-wise meshgrid using broadcasting
        xx, yy = ax_batch_expanded_x, ax_batch_expanded_y
        xy = torch.stack([xx, yy], dim=-1)
        z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
        kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=feat_device) *
                                 torch.sqrt(torch.det(covariance)).to(feat_device).view(num_LR_points, 1, 1))
        kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
        kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
        kernel_normalized = kernel / kernel_max_2
        kernel_reshaped = kernel_normalized.repeat(1, channel, 1).contiguous(). \
            view(num_LR_points * channel, self.kernel_size, self.kernel_size)
        kernel_color = kernel_reshaped.unsqueeze(0).reshape(num_LR_points, channel, self.kernel_size, self.kernel_size)

        # 5.5. Adding padding to make kernel size equal to the image size
        pad_h = round(unfold_feat.shape[-2] * scale) - self.kernel_size
        pad_w = round(unfold_feat.shape[-1] * scale) - self.kernel_size
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Kernel size should be smaller or equal to the image size.")
        padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
        kernel_color_padded = torch.nn.functional.pad(kernel_color, padding, "constant", 0)

        # 5.6. Create a batch of 2D affine matrices
        b, c, h, w = kernel_color_padded.shape  # num_LR_points, channel, hr_h, hr_w
        theta = torch.zeros(batch_size, b, 2, 3, dtype=torch.float32, device=feat_device)
        theta[:, :, 0, 0] = 1.0
        theta[:, :, 1, 1] = 1.0
        theta[:, :, :, 2] = coords_norm
        grid = F.affine_grid(theta.view(-1, 2, 3), size=[batch_size * b, c, h, w], align_corners=True).contiguous()
        kernel_color_padded_expanded = kernel_color_padded.repeat(batch_size, 1, 1, 1).contiguous()
        kernel_color_padded_translated = F.grid_sample(kernel_color_padded_expanded.contiguous(), grid.contiguous(),
                                                       align_corners=True)
        kernel_color_padded_translated = kernel_color_padded_translated.view(batch_size, b, c, h, w)

        # 6. Apply Gaussian splatting
        # colors_.shape = [batch, num_LR_points, channel], colors.shape = [batch, num_LR_points, channel]
        colors = colors_ * weighted_opacity.to(feat_device).unsqueeze(-1).expand(batch_size, -1, -1)
        color_values_reshaped = colors.unsqueeze(-1).unsqueeze(-1)
        final_image_layers = color_values_reshaped * kernel_color_padded_translated
        final_image = final_image_layers.sum(dim=1)
        final_image = torch.clamp(final_image, 0, 1)

        # 7. Fold the input back to the original size
        # Calculate the number of kernels needed to cover each dimension.
        kernel_h, kernel_w = round(self.row * scale), round(self.column * scale)
        fold = nn.Fold(output_size=(kernel_h * num_kernels_row, kernel_w * num_kernels_column),
                       kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        final_image = final_image.reshape(feat_size[0], L, feat_size[1] * kernel_h * kernel_w).transpose(1, 2)
        final_image = fold(final_image)
        final_image = F.interpolate(final_image, size=(hr_h, hr_w), mode='bicubic', align_corners=False)
        # Combine channel
        lr_feat = F.interpolate(lr_feat, size=(hr_h, hr_w), mode='bicubic', align_corners=False)
        final_image = torch.concat((final_image, lr_feat), dim=1)

        # 8. Fourier space augmentation
        coef = self.coef(final_image)
        freq = self.freq(final_image)
        feat_coord = self.feat_coord
        coord_ = coord.clone()
        q_coef = F.grid_sample(coef, coord_.flip(-1).unsqueeze(1), mode='nearest',
                               align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        q_freq = F.grid_sample(freq, coord_.flip(-1).unsqueeze(1), mode='nearest',
                               align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        # calculate relative distance
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        # calculate cell size
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]
        # basis generation
        bs, q = coord.shape[:2]
        q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
        q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
        q_freq = torch.sum(q_freq, dim=-2)
        q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
        q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)
        inp = torch.mul(q_coef, q_freq)
        pred = self.fc(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)

        return pred

    def forward(self, inp, coord, scale, cell=None):
        self.gen_feat(inp)
        return self.query_rgb(coord, scale, cell)


if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    import encoder
    import mlp
    # A simple example of the forward process of GaussianSR
    model = GaussianSplatter(encoder_spec={"name": "edsr-baseline", "args": {"no_upsampling": True}},
                             fc_spec={"name": "mlp", "args": {"out_dim": 3, "hidden_list": [256, 256, 256, 256]}},
                             kernel_size=3)
    input = torch.rand(1, 3, 64, 64)
    sr_scale = 2
    hr_coord, hr_rgb = to_pixel_samples(
        F.interpolate(input, size=(round(input.shape[-2] * sr_scale), round(input.shape[-1] * sr_scale)),
                      mode='bicubic', align_corners=False))
    v0_x, v1_x, v0_y, v1_y = -1, 1, -1, 1
    nx, ny = round(input.shape[-2] * sr_scale), round(input.shape[-1] * sr_scale)

    x = ((hr_coord[..., 0] - v0_x) / (v1_x - v0_x) * 2 * (nx - 1) / 2).round().long()
    y = ((hr_coord[..., 1] - v0_y) / (v1_y - v0_y) * 2 * (ny - 1) / 2).round().long()
    restored_coords = torch.stack([x, y], dim=-1)

    sample_lst = np.random.choice(len(hr_coord), 2304, replace=False)
    hr_coord = hr_coord[sample_lst]
    hr_rgb = hr_rgb[sample_lst]
    cell_ = torch.ones_like(hr_coord.unsqueeze(0))
    cell_[:, 0] *= 2 / nx
    cell_[:, 1] *= 2 / ny
    sr_scale = 2 * torch.ones(1)
    print(model(input, hr_coord.unsqueeze(0), sr_scale, cell_).shape)
