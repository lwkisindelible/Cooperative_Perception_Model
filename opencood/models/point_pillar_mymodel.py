# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.res_bev_backbone import ResBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.myfusion import MyFusion
from opencood.models.comm_modules.defomavle_conv import DeformConv2d
from opencood.models.comm_modules.communication import CBAM, generate_heatmap


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
        self.multi_scale = args['myfusion']['multi_scale']
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

        self.cbam = CBAM(256)

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

    def create_bev_mask(self, bev_map, x, y, z, h, w, l, yaw):
        """
        输入:
        - bev_map: tensor, 尺寸为 (C, H, W)
        - x, y, z: 坐标框的中心点 (以BEV的中心为原点)
        - h, w, l: 坐标框的高度、宽度、长度
        - yaw: 坐标框的旋转角度（绕z轴）

        输出:
        - mask: tensor, 与bev_map形状相同的掩膜，框内为1，框外为0
        """

        _, H, W = bev_map.shape

        # 将坐标框中心转换为BEV地图的索引
        center_x = int((x / bev_map.shape[2]) * W)
        center_y = int((y / bev_map.shape[1]) * H)

        # 计算框的半径
        half_w = int((w / bev_map.shape[2]) * W / 2)
        half_l = int((l / bev_map.shape[1]) * H / 2)

        # 生成初始掩膜
        mask = torch.zeros_like(bev_map)

        # 计算框的范围
        x_min = max(center_x - half_w, 0)
        x_max = min(center_x + half_w, W)
        y_min = max(center_y - half_l, 0)
        y_max = min(center_y + half_l, H)

        # 旋转框（如果需要）
        # 这里只处理矩形的旋转，如果需要更精确的旋转，请考虑使用仿射变换
        if yaw != 0:
            # 创建一个旋转矩阵
            rot_matrix = torch.tensor([
                [torch.cos(yaw), -torch.sin(yaw)],
                [torch.sin(yaw), torch.cos(yaw)]
            ])

            # 计算框的四个角点
            corners = torch.tensor([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ]) - torch.tensor([center_x, center_y])

            rotated_corners = torch.matmul(rot_matrix, corners.T).T + torch.tensor([center_x, center_y])

            # 创建旋转后的框
            x_min_rot, y_min_rot = torch.min(rotated_corners, dim=0)[0]
            x_max_rot, y_max_rot = torch.max(rotated_corners, dim=0)[0]

            # 填充掩膜
            mask[:, int(y_min_rot):int(y_max_rot), int(x_min_rot):int(x_max_rot)] = 1
        else:
            # 如果不旋转，直接填充掩膜
            mask[:, y_min:y_max, x_min:x_max] = 1

        return mask
    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']  # torch.Size([19235, 32, 4])
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        # print("object_bbx_center:", data_dict['object_bbx_center'][0][0])
        # print(data_dict['object_bbx_center'].size())
        # print("object_bbx_mask:", data_dict['object_bbx_mask'])
        # print(data_dict['object_bbx_mask'].size())

        # x, y, z, h, w, l, yaw
        object_bbx_centers = data_dict['object_bbx_center']  # [2,100,7]
        masks = data_dict['object_bbx_mask']  # [2,100]

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        print(batch_dict.keys())
        print("batch_dict['spatial_features']", batch_dict['spatial_features'].type())  # ([7, 64, 192, 704])
        # print("batch_dict['voxel_features']", batch_dict['voxel_features'].size())  # ([35436, 32, 4])
        batch_dict = self.backbone(batch_dict)
        # N, C, H', W': [N, 256, 48, 176]
        spatial_features_2d = batch_dict['spatial_features_2d']
        # 关于H，opv2v是50， v2xvit是48
        # ([8, 384, 96, 352]) --> ([8, 256, 48, 176]) # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # psm->([8, 2, 48, 176])
        psm_single = self.cls_head(spatial_features_2d)  # request map
        # psm_single = self.deform_head(spatial_features_2d)
        # # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        if self.multi_scale:
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            fused_feature, communication_rates = self.fusion_net(batch_dict['spatial_features'],
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix,
                                                                 self.backbone)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates = self.fusion_net(spatial_features_2d,
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix)
        # torch.Size([2, 256, 48, 176]) B, C, H, W
        fused_feature = self.cbam(fused_feature)
        # heatmaps = generate_heatmap(fused_feature)
        # for i in range(len(heatmaps)):
        #     plt.imshow(heatmaps[i])
        #     plt.title(f"Heatmap {i + 1}")
        #     plt.axis('off')
        #     plt.show()
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm, 'rm': rm, 'com': communication_rates}
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
