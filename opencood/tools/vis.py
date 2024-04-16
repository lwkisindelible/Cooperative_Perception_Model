import open3d as o3d
import numpy as np

# 读取PCD文件
point_cloud = o3d.io.read_point_cloud(r"C:\Users\22592\Desktop\TryCode\OpenCOOD\opv2v_data_dumping\train\2021_08_21_22_21_37\2996\000073.pcd")

# 可视化点云
o3d.visualization.draw_geometries([point_cloud])