import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

from opencood.models.fuse_modules.where2comm_fuse import AttentionFusion
from opencood.models.comm_modules.communication import Communication, STTF, RTE, CrossAttention
from opencood.models.fuse_modules.fuse_utils import regroup


## TODO: 你自己的模块
class MyFusion(nn.Module):
    def __init__(self, args):
        super(MyFusion, self).__init__()

        cav_att_config = args['cav_att_config']
        self.max_cav = 5
        # self.sttf = STTF(args['sttf'])
        # self.downsample_rate = args['sttf']['downsample_rate']
        # self.discrete_ratio = args['sttf']['voxel_size'][0]
        # self.use_roi_mask = args['use_roi_mask']
        # self.use_RTE = cav_att_config['use_RTE']
        # self.RTE_ratio = cav_att_config['RTE_ratio']
        self.naive_communication = Communication(args['myfusion']['communication'])
        self.multi_scale = args['multi_scale']
        # self.rtes = nn.ModuleList()
        if self.multi_scale:
            layer_nums = args['layer_nums']  # [ 3, 5, 8 ]
            num_filters = args['num_filters']  # [ 64, 128, 256 ]
            self.num_levels = len(layer_nums) # 3
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttentionFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttentionFusion(args['in_channels'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, psm_single, record_len, pairwise_t_matrix, backbone=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
        """
        # ([8, 256, 48, 176])
        _, C, H, W = x.shape
        # 2
        B = pairwise_t_matrix.shape[0]

        if self.multi_scale:
            ups = []
            for i in range(self.num_levels):
                x = eval(f"backbone.resnet.layer{i}")(x)
                # 1. Communication (mask the features)
                if i == 0:
                    # Prune (N,2,H,W) -> (K,L,2,H,W)
                    batch_confidence_maps = self.regroup(psm_single, record_len)
                    communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
                    if x.shape[-1] != communication_masks.shape[-1]:
                        communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                            mode='bilinear', align_corners=False)
                    x = x * communication_masks  # 这一步已经完成了 Z = M * F 这个公式了
                # 2. Split the features
                # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
                # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
                batch_node_features = self.regroup(x, record_len)

                # 3. Fusion
                x_fuse = []
                for b in range(B):
                    neighbor_feature = batch_node_features[b]
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)
                # 4. Deconv
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            # 1. Communication (mask the features)
            # Prune
            batch_confidence_maps = self.regroup(psm_single, record_len)
            communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
            x = x * communication_masks

            # 2. Split the features
            # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
            # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
            batch_node_features = self.regroup(x, record_len)

            # 3. Fusion
            x_fuse = []
            for b in range(B):
                neighbor_feature = batch_node_features[b]  # (L,C,H,W)
                x_fuse.append(self.fuse_modules(neighbor_feature))  # 利用注意力融合周围特征，
            x_fuse = torch.stack(x_fuse)
        return x_fuse, communication_rates


