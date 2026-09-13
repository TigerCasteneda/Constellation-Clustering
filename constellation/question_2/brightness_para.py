#我疯了，main的调用复杂度太高了，全是重复的，而且逻辑我看不懂
#我将不用逐步筛选的方法，毕竟这个东西太主观了，这么筛没用，纯粹屎山

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from scipy.cluster.hierarchy import single
from sklearn.cluster import DBSCAN
import json


import math

from 提取tsv import extract_column_from_tsv

FILE_CONFIGS = {
    "stars": {"path": "asu.tsv", "column_spec": {'usecols': [0, 1, 2, 3, 18], 'names': ["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"]}, "sep": ";"},
    "famous_stars": {"path": "asu_names.tsv", "column_spec": {'usecols': [0, 1], 'names': ["HR", "Famous_Name"]}, "sep": ";"},
}
import numpy as np
import math
赤经 = extract_column_from_tsv('asu.tsv', '_RAJ2000')
赤纬 = extract_column_from_tsv('asu.tsv', '_DEJ2000')
编号 = extract_column_from_tsv('asu.tsv', 'HR')
亮度 = extract_column_from_tsv('asu.tsv', 'Vmag')

x=[]
y=[]
z=[]
for i in range(9096):
    if 赤纬[i]==None or 赤经 == None:
        del 赤经[i]
        del 赤纬[i]
        del 编号[i]
        del 亮度[i]

for i in range(9096):
    dec_rad = 赤纬[i]
    ra_rad = 赤经[i]
    x.append(np.cos(dec_rad) * np.cos(ra_rad))
    y.append(np.cos(dec_rad) * np.sin(ra_rad))
    z.append(np.sin(dec_rad))
# 数据处理结束可调用数据赤经赤纬亮度x y z
import numpy as np
#########################################################################################################################################
coordinates_list = [list(point) for point in zip(x, y, z)]    #数据处理完毕，调用格式coordinates【【】】，列表套列表
#########################################################################################################################################


# --- 2. 辅助函数 ---
def calculate_angular_distance(ra1, dec1, ra2, dec2):#从经纬度求球面距离
    ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
    ra2_rad, dec2_rad = np.radians(ra2), np.radians(dec2)
    delta_ra = np.abs(ra1_rad - ra2_rad)
    distance_rad = np.arccos(np.sin(dec1_rad) * np.sin(dec2_rad) + np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(delta_ra))
    return np.degrees(distance_rad)

def manual_euclidean_distance(p1, p2): return np.sqrt(np.sum((p1 - p2)**2))#输入两个点的坐标求两点间距离

def calculate_bounding_box_area(points):#基于投影出的二维平面使用，计算团簇的面积，输入格式为坐标数组
    if len(points) < 2: return 0.0
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    return (max_x - min_x) * (max_y - min_y)

#将要进行更改，输入为坐标数组，半径， 最小点数
#团簇聚类之后直接记录在list的位置
#下面定义能用的亮度评分（对于单颗星星）

import numpy as np

import numpy as np


def project_points_on_tangent_plane(spot, points:list, center=np.array([0, 0, 0])):
    """
    将多个三维点投影到以球面上某点为原点的切平面上，并返回二维坐标。

    参数:
    spot (np.ndarray): 球面上的点，形状为 (3,)，将作为切平面的原点和投影中心。
    points (np.ndarray): 待投影的三维点数组，形状为 (N, 3)，其中 N 是点的数量。
    center (np.ndarray, 可选): 球面的球心，默认值为原点 [0, 0, 0]。

    返回:
    np.ndarray: 投影后的二维坐标数组，形状为 (N, 2)。
                如果某个点与球心连线和切平面平行（无交点），则该点的坐标为 [np.nan, np.nan]。
    """
    # 确保输入是 numpy 数组
    spot = np.asarray(spot)
    points = np.asarray(points)
    center = np.asarray(center)

    # 1. 定义切平面
    # 切平面在 spot 点，其法向量 n 是从球心指向 spot 的向量
    n = spot - center
    # 如果 spot 和 center 重合，法向量无意义，直接返回全 NaN
    if np.linalg.norm(n) < 1e-10:
        return np.full((len(points), 2), np.nan)

    # 2. 计算平面方程的常数项 d: n · spot = d
    d = np.dot(n, spot)

    # 3. 为切平面定义一个二维坐标系 (u, v)
    # 寻找一个与 n 不共线的向量来构建 u 轴
    if abs(n[0]) < abs(n[1]) and abs(n[0]) < abs(n[2]):
        vec1 = np.array([1, 0, 0])
    elif abs(n[1]) < abs(n[2]):
        vec1 = np.array([0, 1, 0])
    else:
        vec1 = np.array([0, 0, 1])

    # u 轴是 vec1 在切平面上的投影（即与 n 垂直的分量）
    u_axis = np.cross(n, np.cross(vec1, n))
    u_axis = u_axis / np.linalg.norm(u_axis)  # 单位化

    # v 轴由 n 和 u 轴叉乘得到，天然垂直于两者且在平面内
    v_axis = np.cross(n, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)  # 单位化

    # 4. 对每个点进行投影计算
    projected_2d_points = []
    for P in points:
        # 定义从球心到 P 点的直线方向向量
        direction = P - center

        # 检查直线是否与平面平行 (方向向量与法向量点积为 0)
        denominator = np.dot(n, direction)
        if abs(denominator) < 1e-10:
            # 直线与平面平行或重合，无有效交点
            projected_2d_points.append([np.nan, np.nan])
            continue

        # 计算直线与平面交点 I 的参数 t
        # 直线方程: L(t) = center + t * direction
        # 代入平面方程: n · L(t) = d, 解得 t
        t = (d - np.dot(n, center)) / denominator

        # 计算交点 I 的三维坐标
        intersection_point = center + t * direction

        # 5. 将交点 I 变换到以 spot 为原点的坐标系
        vector_from_spot = intersection_point - spot

        # 6. 将三维向量投影到 (u, v) 二维坐标系
        u = np.dot(vector_from_spot, u_axis)
        v = np.dot(vector_from_spot, v_axis)

        projected_2d_points.append([u, v])

    return np.array(projected_2d_points)


def score_brightness_dynamic(Vmags, points, spot):#points是一个二维坐标的列表,Vmag对应其中每一个点的亮度
    area=calculate_bounding_box_area(points)
    single_light=[]
    star_sign_light=sum(Vmags)/area
    for i in range(len(Vmags)):
        single_light.append(Vmags[i]/star_sign_light)
    return star_sign_light, single_light
#输出星座整体的亮度与单星相对于当前星座的亮度
#聚类
def cluster_3d_points(point_set, eps=0.5, min_samples=5):
    """
    对三维点集进行DBSCAN聚类。

    参数:
        point_set (np.ndarray): 一个形状为 (n_samples, 3) 的NumPy数组，
                                其中每行代表一个点的(x, y, z)坐标。
        eps (float): DBSCAN算法中的邻域半径。
        min_samples (int): 形成一个簇所需的最小点数。

    返回:
        np.ndarray: 一个形状为 (n_samples,) 的整数数组，
                   其中每个元素是对应点的聚类标签。-1表示噪声点。
    """
    # 初始化DBSCAN，使用欧氏距离
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')

    # 执行聚类并获取标签
    labels = dbscan.fit_predict(point_set)

    # 仅返回标签
    return labels
#开始主体与调用
eps=3
minsamples=5
lables=cluster_3d_points(coordinates_list, eps, minsamples)
print(lables)

n_clusters = len(np.unique(lables))

# def convert_clusters_to_indices(labels):
#     cluster_dict = {}
#
#     for idx, label in enumerate(labels):
#         # 跳过噪声点
#         if label == -1 or label == 0:
#             continue
#
#         # 如果簇编号不在字典中，就为它创建一个新列表
#         if label not in cluster_dict:
#             cluster_dict[label] = []
#
#         # 将当前数据点的索引添加到对应的簇列表中
#         cluster_dict[label].append(idx)
#
#     # 对簇编号进行排序
#     sorted_cluster_ids = sorted(cluster_dict.keys())
#
#     # 创建只有两层的字典结构
#     result = {}
#     for cid in sorted_cluster_ids:
#         cluster_list = cluster_dict[cid]
#         # 只添加非空的簇
#         if cluster_list:  # 检查列表是否非空
#             result[cid] = cluster_list
#
#     return result
#
# clusters1 = convert_clusters_to_indices(lables)
#########################################################
# for j in range (1 , n_clusters):
#     cluster_coords = [coordinates_list[i] for i, label in enumerate(lables) if label == i]
#     spot=
#     two_dimention_cordinates=project_points_on_tangent_plane(1, cluster_coords, center=np.array([0, 0, 0]))

