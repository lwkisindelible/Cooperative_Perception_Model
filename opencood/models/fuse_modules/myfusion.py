import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

from opencood.models.fuse_modules.hmsa import HGTCavAttention
from opencood.models.fuse_modules.mswin import PyramidWindowAttention
from opencood.models.fuse_modules.where2comm_fuse import AttentionFusion
from opencood.models.comm_modules.communication import Communication, STTF, RTE
from opencood.models.sub_modules.base_transformer import CavAttention, PreNorm
from opencood.models.sub_modules.torch_transformation_utils import get_roi_and_cav_mask
from opencood.models.fuse_modules.fuse_utils import regroup

## TODO: 你自己的模块
class MyFusion(nn.Module):
    def __init__(self, args):
        super(MyFusion, self).__init__()

        cav_att_config = args['cav_att_config']
        pwindow_config = args['pwindow_config']
        self.max_cav = 5
        self.sttf = STTF(args['sttf'])
        self.downsample_rate = args['sttf']['downsample_rate']
        self.discrete_ratio = args['sttf']['voxel_size'][0]
        self.use_roi_mask = args['use_roi_mask']
        self.use_RTE = cav_att_config['use_RTE']
        self.RTE_ratio = cav_att_config['RTE_ratio']
        self.naive_communication = Communication(args['myfusion']['communication'])
        if self.use_RTE:
            self.rte = RTE(cav_att_config['dim'], self.RTE_ratio)
        self.layers = nn.ModuleList([])
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

    def forward(self, x, prior_encoding, psm_single, record_len, pairwise_t_matrix, spatial_correction_matrix, backbone=None):
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
        # Communication (mask the features)
        batch_confidence_maps = self.regroup(psm_single, record_len)
        communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
        # print(x.shape, communication_masks.shape)
        x = x * communication_masks
        # print("x.shape: ", x.shape)
        # B, L, C, H, W :([4, 5, 256, 48, 176])
        x, mask = regroup(x, record_len, self.max_cav)
        # print("regroup_feature.shape:", x.shape)
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               x.shape[3],
                                               x.shape[4])
        x = torch.cat([x, prior_encoding], dim=2)
        # b l c h w -> b l h w c
        x = x.permute(0, 1, 3, 4, 2)
        # velocity, time_delay, infra
        # (B,L,H,W,3)
        prior_encoding = x[..., -3:]
        # (B,L,H,W,C)
        x = x[..., :-3]
        if self.use_RTE:
            # dt: (B,L)
            dt = prior_encoding[:, :, 0, 0, 1].to(torch.int)
            x = self.rte(x, dt)
        x = self.sttf(x, spatial_correction_matrix)
        # mask(B,L) --> (B,H,W,1,L)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask else get_roi_and_cav_mask(x.shape,
                                                                  mask,
                                                                  spatial_correction_matrix,
                                                                  self.discrete_ratio,
                                                                  self.downsample_rate)
        for attn, ff in self.layers:
            x = attn(x, mask=com_mask, prior_encoding=prior_encoding)
            x = ff(x) + x
        return x[:,0], communication_rates
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
    x = [1,2,3]
    print(x[:1])

