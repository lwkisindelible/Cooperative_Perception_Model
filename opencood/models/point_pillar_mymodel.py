# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.res_bev_backbone import ResBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.myfusion import MyFusion
from opencood.models.comm_modules.defomavle_conv import DeformConv2d


class PointPillarMymodel(nn.Module):
    def __init__(self, args):
        super(PointPillarMymodel, self).__init__()
        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        ## TODO: 自己的融合模块
        self.fusion_net = MyFusion(args['myfusion'])
        # self.multi_scale = args['myfusion']['myfusion']['multi_scale']
        self.deform_head = DeformConv2d(args['head_dim'], args['anchor_number'])

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'],
                                  kernel_size=1)
        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        # 关于H，opv2v是50， v2xvit是48
        # ([8, 384, 100, 352]) --> ([8, 256, 50, 176]) # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)  # request map
        # psm_single = self.deform_head(spatial_features_2d)
        # # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # (B, max_cav, 3) --> ([4, 5, 3, 1, 1])
        prior_encoding = \
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)
        ## TODO: 你的模块在forward函数中的代码
        fused_feature, communication_rates = self.fusion_net(spatial_features_2d,
                                                             prior_encoding,
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix,
                                                             spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict


if __name__ == '__main__':
    """
            Args:
                batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
            """
    batch_confidence_maps = torch.randn(2, 3, 4)
    ori_communication_maps, haah = batch_confidence_maps.sigmoid().max(dim=1, keepdim=True)
    print(batch_confidence_maps.sigmoid())
    print(ori_communication_maps)
    print(haah)
