import os
import numpy as np
from tqdm import tqdm

from Viewer.viewer.viewer import Viewer


def box_to_corners(center, dimensions, yaw):
    """
    将框的中心坐标、尺寸和yaw角度转换为8个角点的坐标
    """
    x, y, z = center
    l, w, h = dimensions
    yaw_rad = np.radians(yaw)

    # 计算角点
    corners = np.array([
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2]
    ])

    # 旋转矩阵
    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    rotated_corners = np.dot(corners, R.T)
    rotated_corners += np.array(center)

    return rotated_corners


def compute_iou_3d(box1, box2):
    """
    计算两个三维框的 IoU (Intersection over Union)。
    """
    center1, dimensions1, yaw1 = box1[:3], box1[3:6], box1[6]
    center2, dimensions2, yaw2 = box2[:3], box2[3:6], box2[6]

    # 获取每个框的角点
    corners1 = box_to_corners(center1, dimensions1, yaw1)
    corners2 = box_to_corners(center2, dimensions2, yaw2)

    # 计算交集体积的简化方法
    def intersect_volume(corners1, corners2):
        """
        使用包围盒的简单方法计算体积交集
        """

        def overlap(a, b):
            """计算两个区间的重叠长度"""
            return max(0, min(a[1], b[1]) - max(a[0], b[0]))

        # 计算包围盒的边界
        def bounding_box(corners):
            min_corner = np.min(corners, axis=0)
            max_corner = np.max(corners, axis=0)
            return min_corner, max_corner

        min1, max1 = bounding_box(corners1)
        min2, max2 = bounding_box(corners2)

        # 计算交集体积
        dx = overlap([min1[0], max1[0]], [min2[0], max2[0]])
        dy = overlap([min1[1], max1[1]], [min2[1], max2[1]])
        dz = overlap([min1[2], max1[2]], [min2[2], max2[2]])

        return dx * dy * dz

    volume1 = np.prod(dimensions1)
    volume2 = np.prod(dimensions2)
    inter_volume = intersect_volume(corners1, corners2)
    union_volume = volume1 + volume2 - inter_volume

    return inter_volume / union_volume if union_volume > 0 else 0


def compute_precision_recall(gt_folder, pred_folder, pre_score_folder, iou_threshold=0.5, score_threshold=0.1):
    """
    计算文件夹中所有三维框的 Precision 和 Recall
    gt_folder: Ground Truth 框文件夹
    pred_folder: 预测框文件夹
    iou_threshold: IoU 阈值，决定 True Positive 的标准
    """
    tp, fp, fn = 0, 0, 0  # 初始化 True Positive, False Positive, False Negative

    # pre_box = sorted(os.listdir("E:\\OPV2V\\pre_box"))
    # 获取文件夹中所有的文件名
    gt_files = sorted(os.listdir(gt_folder))
    pred_files_ = os.listdir(pred_folder)
    pred_files = [s for s in pred_files_ if 'noise' not in s]
    pred_files = sorted(pred_files)

    pre_score_files = sorted(os.listdir(pre_score_folder))

    # 确保文件数量一致
    assert len(gt_files) == len(pred_files), "Ground Truth 和预测框文件数量不一致！"

    for gt_file, pred_file, pre_score_file in tqdm(zip(gt_files, pred_files, pre_score_files)):  # , pre_box
        # 加载 npy 文件中的 3D 框
        gt_boxes = np.load(os.path.join(gt_folder, gt_file))  # N x 7 的数组
        pred_boxes_ = np.load(os.path.join(pred_folder, pred_file))  # M x 7 的数组
        pre_scores = np.load(os.path.join(pre_score_folder, pre_score_file))

        assert len(pred_boxes_) == len(pre_scores), "框的数量 和 score 的数量不一致，"
        pred_boxes = []
        for pred_box_, pre_score in zip(pred_boxes_, pre_scores):
            if pre_score > score_threshold:
                pred_boxes.append(pred_box_)

        # 标记哪些 Ground Truth 框已经匹配上
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        # 遍历预测框，计算 IoU 并统计 True Positive, False Positive
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            # 找到 IoU 最大的 Ground Truth 框
            for i, gt_box in enumerate(gt_boxes):
                if matched_gt[i]:
                    continue
                current_iou = compute_iou_3d(pred_box, gt_box)
                # print(current_iou)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = i

            # 根据 IoU 阈值判断是 TP 还是 FP
            # print(best_iou)
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt[best_gt_idx] = True
            else:
                fp += 1

        # 统计 False Negative
        fn += len(gt_boxes) - np.sum(matched_gt)

    # 计算 Precision 和 Recall
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return precision, recall


if __name__ == '__main__':
    vi = Viewer()
    # 文件夹路径
    gt_folder = 'E:/OPV2V/pseduo_label_val/gt_box_test_full'  # Ground Truth 框文件夹路径
    pred_folder = "E:\\OPV2V\\pseduo_label_val\\pre_box_test_full"  # 预测框文件夹路径
    pre_score_folder = "E:\\OPV2V\\pseduo_label_val\\pre_score_test_full"

    # 计算 Precision 和 Recall
    precision, recall = compute_precision_recall(gt_folder, pred_folder, pre_score_folder, iou_threshold=0.3, score_threshold=0.9)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    exit()
    # multi_agent_point = np.load(f'E:\\OPV2V\\pseduo_label_val\\points\\origin_lidar_18.npy',
    #                             allow_pickle=True)
    # pre_score = np.load(f'E:\\OPV2V\\pseduo_label_val\\pre_score_test\\score_18.npy',
    #                     allow_pickle=True)
    # print(pre_score.shape)  # (115,)
    # vi.add_points(multi_agent_point[0][:, :3])
    # vi.show_3D()
