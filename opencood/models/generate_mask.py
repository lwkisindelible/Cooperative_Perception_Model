import torch
import numpy as np


def create_bev_mask(bev_map, x, y, z, h, w, l, yaw):
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


# 示例
C, H, W = 1, 256, 256  # BEV map尺寸
bev_map = torch.zeros((C, H, W))

x, y, z = 0.5, 0.5, 0.5  # 假设的坐标框中心（BEV的归一化坐标）
h, w, l = 2, 1, 3  # 坐标框的尺寸
yaw = torch.tensor(np.pi / 6)  # 坐标框的旋转角度

mask = create_bev_mask(bev_map, x, y, z, h, w, l, yaw)
