import numpy as np
import os
from scipy.spatial import ConvexHull


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


def process_scene(gt_boxes, pred_boxes, iou_threshold=0.1):
    # 保留IOU大于阈值的预测框
    kept_boxes = []
    for pred_box in pred_boxes:
        for gt_box in gt_boxes:
            iou = compute_iou_3d(gt_box, pred_box)
            if iou > iou_threshold:
                kept_boxes.append(pred_box)
                break
    return np.array(kept_boxes)


def process_folders(gt_folder, pred_folder, output_folder, iou_threshold=0.1):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件列表
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files)):
        gt_boxes = np.load(os.path.join(gt_folder, gt_file))
        pred_boxes = np.load(os.path.join(pred_folder, pred_file))

        kept_boxes = process_scene(gt_boxes, pred_boxes, iou_threshold)
        # if kept_boxes == np.array([]):
        #     vi.add_3D_boxes(gt_boxes, color='red')
        #     vi.add_3D_boxes(pred_boxes, color='blue')
        #     vi.show_3D()
        # 保存保留的框到npy文件
        np.save(os.path.join(output_folder, pred_file), kept_boxes)
        # print(f'Processed {pred_file}, kept {len(kept_boxes)} boxes.')

def read(pred_folder):
    pred_files = sorted(os.listdir(pred_folder))
    for pred_file in tqdm(pred_files):
        pred_boxes = np.load(os.path.join(pred_folder, pred_file))
        if pred_boxes.shape[1] != 7:
            print(pred_boxes.shape[1])

if __name__ == '__main__':
    # vi = Viewer()
    # 文件夹路径
    from Viewer.viewer.viewer import Viewer
    import numpy as np

    vi = Viewer()
    gt_folder = 'E:/OPV2V/gt_box'  # Ground Truth 框文件夹路径
    pred_folder = "E:/OPV2V/pre_box"  # 预测框文件夹路径
    out_folder = "E:/OPV2V/out_final_05"
    # 计算 Precision 和 Recall
    from tqdm import tqdm
    # read(out_folder)
    process_folders(gt_folder, pred_folder, out_folder, iou_threshold=0.5)

