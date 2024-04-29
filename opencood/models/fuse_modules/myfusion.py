import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

from opencood.models.fuse_modules.hmsa import HGTCavAttention
from opencood.models.fuse_modules.mswin import PyramidWindowAttention
from opencood.models.fuse_modules.where2comm_fuse import AttentionFusion
from opencood.models.comm_modules.communication import Communication, STTF
from opencood.models.sub_modules.base_transformer import CavAttention, PreNorm
from opencood.models.sub_modules.torch_transformation_utils import get_roi_and_cav_mask


## TODO: 你自己的模块
class MyFusion(nn.Module):
    def __init__(self, args, cav_att_config, pwindow_config):
        super(MyFusion, self).__init__()
        self.sttf = STTF(args['sttf'])
        self.fuse_modules = AttentionFusion(args['in_channels'])
        self.naive_communication = Communication(args['communication'])
        for _ in range(args['num_blocks']):
            att = HGTCavAttention(cav_att_config['dim'],
                                  heads=cav_att_config['heads'],
                                  dim_head=cav_att_config['dim_head'],
                                  dropout=cav_att_config['dropout']) if \
                cav_att_config['use_hetero'] else \
                CavAttention(cav_att_config['dim'],
                             heads=cav_att_config['heads'],
                             dim_head=cav_att_config['dim_head'],
                             dropout=cav_att_config['dropout'])
            self.layers.append(nn.ModuleList([
                PreNorm(cav_att_config['dim'], att),
                PreNorm(cav_att_config['dim'],
                        PyramidWindowAttention(pwindow_config['dim'],
                                               heads=pwindow_config['heads'],
                                               dim_heads=pwindow_config[
                                                   'dim_head'],
                                               drop_out=pwindow_config[
                                                   'dropout'],
                                               window_size=pwindow_config[
                                                   'window_size'],
                                               relative_pos_embedding=
                                               pwindow_config[
                                                   'relative_pos_embedding'],
                                               fuse_method=pwindow_config[
                                                   'fusion_method']))]))

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, mask, psm_single, record_len, pairwise_t_matrix, backbone=None):
        """
        Fusion forwarding.
        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).
        Returns:
            Fused feature.
        """
        _, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0]
        x = self.sttf(x, pairwise_t_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask else get_roi_and_cav_mask(x.shape,
                                                                  mask,
                                                                  pairwise_t_matrix,
                                                                  self.discrete_ratio,
                                                                  self.downsample_rate)
        # Communication (mask the features)
        batch_confidence_maps = self.regroup(psm_single, record_len)
        communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
        x = x * communication_masks
        for attn, ff in self.layers:
            x = attn(x, mask=com_mask)
            x = ff(x) + x
        return x, communication_rates
        # 2. Split the features
        # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
        # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
        # batch_node_features = self.regroup(x, record_len)
        # # 3. Fusion
        # x_fuse = []
        # for b in range(B):
        #     neighbor_feature = batch_node_features[b]
        #     x_fuse.append(self.fuse_modules(neighbor_feature))  # 利用注意力融合周围特征，
        # x_fuse = torch.stack(x_fuse)
        # return x_fuse, communication_rates


if __name__ == '__main__':
    myfusion = MyFusion(5)
    """
    x的输入是 N、C、W、H
    N：即同一时间戳下有多少辆车
    C：channel数
    W：宽
    H：高 
    这里假设4辆，一个二维矩阵其实就代表一个特征或图像了，这里总共3个channel
    """
    x = torch.tensor([
        [
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]]
        ],
        [
            [[1.2, 2.3], [3.5, 4.6]],
            [[5.7, 6.2], [7.1, 8.9]],
            [[9.1, 10.2], [11.6, 12.3]]
        ],
        [
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]]
        ],
        [
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]]
        ]
    ])
    record_len = torch.tensor([2, 2])
    for epoch in range(3):
        out = myfusion(x, record_len)
        print("epoch{}: ".format(epoch), out)
