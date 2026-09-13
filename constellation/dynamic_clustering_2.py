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