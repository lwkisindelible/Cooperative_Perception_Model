import random

import torch
import torch.nn as nn
import numpy as np
import math

from opencood.models.sub_modules.torch_transformation_utils import get_discretized_transformation_matrix, \
    get_transformation_matrix, warp_affine
from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, B):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
        """
        _, _, H, W = batch_confidence_maps[0].shape
        communication_masks = []
        communication_rates = []
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps
            L = communication_maps.shape[0]
            # training
            if self.training:
                # Official training proxy objective 随机选择k个最大值，将对应值记为1
                K = int(H * W * random.uniform(0, 1))
                communication_maps = communication_maps.reshape(L, H * W)  # 为什么要选k个？
                _, indices = torch.topk(communication_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
                # torch.scatter(input, dim, index, src),将src中的数据根据index中的索引按照dim的方向填入到input中
            elif self.threshold: # 如果有threshold
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
            else:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)

            communication_rate = communication_mask.sum() / (L * H * W)
            # Ego
            communication_mask[0] = 1
            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        return communication_masks, communication_rates


## STTF 利用变换矩阵转换特征，和原始的特征合并。
class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, spatial_correction_matrix):
        x = x.permute(0, 1, 4, 2, 3)
        dist_correction_matrix = get_discretized_transformation_matrix(  # 获取离散化变换矩阵
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(  # x就是整体数据，下面1：其实就是表示其他车辆，0就是自己车辆
            dist_correction_matrix[:, 1:, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, 1:, :, :, :].reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)
        x = torch.cat([x[:, 0, :, :, :].unsqueeze(1), cav_features], dim=1)
        x = x.permute(0, 1, 3, 4, 2)
        return x  # 返回的是自己提取的特征和已经接受到的其他车辆的特征。

class Channel_Request_Attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Channel_Request_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class Spatial_Request_Attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(Spatial_Request_Attention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)
        self.hidden_dim = feature_dim * 2
        self.conv_query = nn.Conv2d(
            feature_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv_key = nn.Conv2d(
            feature_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv_value = nn.Conv2d(
            feature_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv_temporal_key = nn.Conv1d(
            self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1)
        self.conv_temporal_value = nn.Conv1d(
            self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_feat = nn.Conv2d(
            self.hidden_dim, feature_dim, kernel_size=3, padding=1)

    def forward(self, x):  # 输入是整个时间段的特征。
        frame, C, H, W = x.shape
        ego = x[:1]  # 这是ego在当前时刻的特征，作为查询
        query = self.conv_query(ego)
        query = query.view(1, self.hidden_dim, -1).permute(2, 0, 1)

        key = self.conv_key(x)
        key_avg = key
        value = self.conv_value(x)
        val_avg = value
        key = key.view(frame, self.hidden_dim, -1).permute(2, 0, 1)
        value = value.view(frame, self.hidden_dim, -
        1).permute(2, 0, 1)

        key_avg = self.pool(key_avg).squeeze(-1).squeeze(-1)
        val_avg = self.pool(val_avg).squeeze(-1).squeeze(-1)
        key_avg = self.conv_temporal_key(
            key_avg.unsqueeze(0).permute(0, 2, 1))
        val_avg = self.conv_temporal_value(
            val_avg.unsqueeze(0).permute(0, 2, 1))
        key_avg = key_avg.permute(0, 2, 1)
        val_avg = val_avg.permute(0, 2, 1)
        key = key * key_avg
        value = value * val_avg

        x = self.att(query, key, value)
        x = x.permute(1, 2, 0).view(1, self.hidden_dim, H, W)
        out = self.conv_feat(x)

        return out

class RelTemporalEncoding(nn.Module):
    """
    Implement the Temporal Encoding (Sinusoid) function.
    """

    def __init__(self, n_hid, RTE_ratio, max_len=100, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(
            n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(
            n_hid)
        emb.requires_grad = False
        self.RTE_ratio = RTE_ratio
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        # When t has unit of 50ms, rte_ratio=1.
        # So we can train on 100ms but test on 50ms
        return x + self.lin(self.emb(t * self.RTE_ratio)).unsqueeze(
            0).unsqueeze(1)

# Delay-aware positional encoding
class RTE(nn.Module):
    def __init__(self, dim, RTE_ratio=2):
        super(RTE, self).__init__()
        self.RTE_ratio = RTE_ratio

        self.emb = RelTemporalEncoding(dim, RTE_ratio=self.RTE_ratio)

    def forward(self, x, dts):
        # x: (B,L,H,W,C)
        # dts: (B,L)
        rte_batch = []
        for b in range(x.shape[0]):
            rte_list = []
            for i in range(x.shape[1]):
                rte_list.append(
                    self.emb(x[b, i, :, :, :], dts[b, i]).unsqueeze(0))
            rte_batch.append(torch.cat(rte_list, dim=0).unsqueeze(0))
        return torch.cat(rte_batch, dim=0)
