import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, Delaunay
import json
import math
from scipy.spatial import distance

# --- 1. Configuration Parameters ---
FILE_CONFIGS = {
    "stars": {"path": "asu.tsv", "column_spec": {'usecols': [0, 1, 2, 3, 18], 'names': ["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"]}, "sep": ";"},
    "famous_stars": {"path": "asu_names.tsv", "column_spec": {'usecols': [0, 1], 'names': ["HR", "Famous_Name"]}, "sep": ";"},
}

# DBSCAN parameters (now in degrees)
EPS = 5.0  # Clustering radius in degrees
MIN_PTS = 3  # Minimum number of points for a cluster

# Star selection parameters
CORE_STAR_VMAG_LIMIT = 4.5
ALL_STAR_VMAG_LIMIT = 6.5

# Boundary algorithm parameters
ALPHA_SHAPE_ALPHA = 2.0  # Alpha parameter for alpha shapes
DOUGLAS_PEUCKER_EPSILON = 0.5  # Epsilon parameter for Douglas-Peucker simplification

# Output files
RESULTS_JSON_FILE = 'constellation_division_results_deg.json'
CONSTELLATIONS_TSV_FILE = 'new_constellations_final_hulls_deg.tsv'
CONSTELLATION_NAME_PREFIX = "NewCst_"

# --- 2. Helper Functions (Degree-based) ---

def calculate_angular_distance_deg(point1, point2):
    ra1, dec1 = point1
    ra2, dec2 = point2

    # 将度转换为弧度
    ra1_rad = math.radians(ra1)
    dec1_rad = math.radians(dec1)
    ra2_rad = math.radians(ra2)
    dec2_rad = math.radians(dec2)

    # 计算赤经差和赤纬差
    delta_ra = abs(ra1_rad - ra2_rad)
    delta_dec = abs(dec1_rad - dec2_rad)

    # Vincenty公式（球面模型）
    sin_delta_dec_half = math.sin(delta_dec / 2)
    sin_delta_ra_half = math.sin(delta_ra / 2)

    a = sin_delta_dec_half **2 + \
        math.cos(dec1_rad) * math.cos(dec2_rad) * sin_delta_ra_half** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 弧度转换为度
    return math.degrees(c)

def is_point_in_convex_hull_deg(point, hull):
    """Check if a point (RA, Dec in degrees) lies inside a convex hull."""
    point = np.array(point)
    for eq in hull.equations:
        # eq: ax + by + c <= 0
        if np.dot(eq[:-1], point) + eq[-1] > 1e-10:
            return False
    return True

def find_closest_convex_hull_deg(point, hulls):
    """Find the convex hull closest to a given point (RA, Dec in degrees)."""
    min_distance = float('inf')
    closest_hull_idx = -1
    for i, hull in enumerate(hulls):
        # Calculate distance from point to all vertices in the hull and take the minimum
        distances = np.array([calculate_angular_distance_deg(point, p) for p in hull.points])
        min_dist_to_hull = np.min(distances)
        if min_dist_to_hull < min_distance:
            min_distance = min_dist_to_hull
            closest_hull_idx = i
    return closest_hull_idx

def calculate_spherical_center_deg(ra_series, dec_series):
    """Calculate the geometric center of a set of spherical coordinates (input in degrees)."""
    ra_rad = np.radians(ra_series)
    dec_rad = np.radians(dec_series)
    
    # Convert to 3D Cartesian coordinates on the unit sphere
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    # Find the mean of the 3D points
    mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)
    
    # Convert back to spherical coordinates
    center_dec_rad = np.arcsin(mean_z)
    center_ra_rad = np.arctan2(mean_y, mean_x)
    
    # Convert back to degrees and normalize RA to [0, 360)
    return np.degrees(center_ra_rad) % 360.0, np.degrees(center_dec_rad)

# --- 3. Advanced Boundary Algorithms ---

def douglas_peucker(points, epsilon):
    """
    Douglas-Peucker 算法简化多边形
    """
    if len(points) <= 2:
        return points
    
    # 找到距离首尾连线最远的点
    dmax = 0
    index = 0
    start, end = points[0], points[-1]
    
    for i in range(1, len(points)-1):
        d = perpendicular_distance(points[i], start, end)
        if d > dmax:
            index = i
            dmax = d
    
    # 如果最大距离大于阈值，递归简化
    if dmax > epsilon:
        results1 = douglas_peucker(points[:index+1], epsilon)
        results2 = douglas_peucker(points[index:], epsilon)
        return results1[:-1] + results2
    else:
        return [start, end]

def perpendicular_distance(point, line_start, line_end):
    """
    计算点到直线的垂直距离
    """
    if np.all(line_start == line_end):
        return distance.euclidean(point, line_start)
    
    return np.abs(np.cross(line_end-line_start, line_start-point)) / np.linalg.norm(line_end-line_start)

def alpha_shape(points, alpha):
    """
    Alpha Shapes 算法 - 生成更自然的星座边界
    """
    if len(points) < 4:
        return set()
    
    try:
        tri = Delaunay(points)
        edges = set()
        
        # 遍历所有三角形
        for simplex in tri.simplices:
            # 三角形的三个边
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                
                p1, p2, p3 = points[simplex[i]], points[simplex[j]], points[simplex[k]]
                
                # 计算外接圆半径
                radius = circumradius(p1, p2, p3)
                
                # 如果半径小于 alpha，保留这条边
                if radius < alpha:
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        
        return edges
    except Exception as e:
        print(f"Alpha Shapes 计算失败: {e}")
        return set()

def circumradius(p1, p2, p3):
    """
    计算三角形外接圆半径
    """
    a = distance.euclidean(p1, p2)
    b = distance.euclidean(p2, p3)
    c = distance.euclidean(p3, p1)
    
    # 使用公式计算外接圆半径
    s = (a + b + c) / 2
    area = math.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))  # 避免负数
    
    if area == 0:
        return float('inf')
    
    return (a * b * c) / (4 * area)

def reconstruct_boundary(edges, points):
    """
    从边集合重建边界多边形
    """
    if not edges:
        return []
    
    # 构建邻接表
    graph = {}
    for edge in edges:
        u, v = edge
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
    
    # 找到边界循环 - 使用更稳健的方法
    if not graph:
        return []
    
    # 找到所有度数为1的节点（边界起点）
    boundary_starts = [node for node, neighbors in graph.items() if len(neighbors) == 1]
    
    if not boundary_starts:
        # 如果没有度数为1的节点，尝试找到一个循环
        start_node = next(iter(graph.keys()))
        visited = set()
        path = []
        
        def dfs(node, parent):
            if node in visited:
                return path if node == path[0] and len(path) > 2 else None
            
            visited.add(node)
            path.append(node)
            
            for neighbor in graph[node]:
                if neighbor == parent:
                    continue
                result = dfs(neighbor, node)
                if result:
                    return result
            
            path.pop()
            return None
        
        cycle = dfs(start_node, -1)
        if cycle:
            return [points[i] for i in cycle]
        return []
    
    # 从边界起点开始追踪边界
    def trace_boundary(start):
        path = [start]
        current = start
        prev = -1
        
        while True:
            neighbors = [n for n in graph[current] if n != prev]
            if not neighbors:
                break
            next_node = neighbors[0]
            if next_node in path:
                break
            path.append(next_node)
            prev = current
            current = next_node
        
        return path
    
    # 追踪所有边界并选择最长的
    boundaries = [trace_boundary(start) for start in boundary_starts]
    if not boundaries:
        return []
    
    longest_boundary = max(boundaries, key=len)
    return [points[i] for i in longest_boundary]

def sort_points_clockwise(points):
    """
    将点按顺时针顺序排序
    """
    if len(points) < 3:
        return points
    
    points = np.array(points)
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]

def calculate_boundary_area(points):
    """
    计算多边形面积（用于评估边界质量）
    """
    if len(points) < 3:
        return 0
    
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# --- 4. Improved Boundary Calculation ---

def calculate_optimal_boundary(points):
    """
    计算最优边界，使用多种算法并选择最佳结果
    """
    if len(points) < 3:
        return None, None
    
    best_boundary = None
    best_method = "none"
    best_area = 0
    
    # 方法1: Alpha Shapes
    # try:
    #     alpha_edges = alpha_shape(points, ALPHA_SHAPE_ALPHA)
    #     if alpha_edges:
    #         boundary_points = reconstruct_boundary(alpha_edges, points)
    #         if len(boundary_points) >= 3:
    #             boundary_points = sort_points_clockwise(boundary_points)
    #             area = calculate_boundary_area(boundary_points)
    #             if area > best_area:
    #                 best_boundary = boundary_points
    #                 best_method = "alpha_shape"
    #                 best_area = area
    # except Exception as e:
    #     print(f"Alpha Shapes 失败: {e}")
    
    # # 方法2: 凸包 + 简化
    # try:
    #     hull = ConvexHull(points)
    #     hull_points = points[hull.vertices]
        
    #     # 简化凸包
    #     simplified_points = douglas_peucker(hull_points, DOUGLAS_PEUCKER_EPSILON)
    #     if len(simplified_points) >= 3:
    #         simplified_points = sort_points_clockwise(np.array(simplified_points))
    #         area = calculate_boundary_area(simplified_points)
    #         if area > best_area or best_boundary is None:
    #             best_boundary = simplified_points
    #             best_method = "convex_hull_simplified"
    #             best_area = area
    # except Exception as e:
    #     print(f"凸包简化失败: {e}")
    
    # # 方法3: 原始凸包
    try:
        if best_boundary is None:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            best_boundary = sort_points_clockwise(hull_points)
            best_method = "convex_hull"
    except Exception as e:
        print(f"原始凸包失败: {e}")
    
    return best_boundary, best_method

# --- 5. Modified Save Function ---

def save_constellations_as_tsv(constellation_data, filename):
    """
    Save the FINAL constellation boundaries in TSV format.
    Format: ra_deg\tdec_deg\tcst_name
    """
    print("\n--- Preparing to save FINAL CONSTELLATION BOUNDARIES ---")
    tsv_data = []
    
    for constellation in constellation_data:
        cluster_id = constellation["cluster_id"]
        constellation_name = f"{CONSTELLATION_NAME_PREFIX}{cluster_id:02d}"
        
        # 使用优化后的边界点
        if "optimal_boundary" in constellation and constellation["optimal_boundary"] is not None:
            boundary_points = constellation["optimal_boundary"]
            
            # 转换坐标并添加到数据列表
            for ra_deg, dec_deg in boundary_points:
                tsv_data.append({
                    'ra_deg': ra_deg,
                    'dec_deg': dec_deg,
                    'cst': constellation_name
                })
            
            print(f"Constellation {cluster_id}: Saved {len(boundary_points)} boundary points using {constellation['boundary_method']}")
        else:
            print(f"Warning: Constellation {cluster_id} has no valid boundary")
    
    if tsv_data:
        df = pd.DataFrame(tsv_data)
        df = df.sort_values(['cst', 'ra_deg'])
        df[['ra_deg', 'dec_deg', 'cst']].to_csv(
            filename, 
            sep='\t', 
            index=False, 
            header=False, 
            float_format='%.6f'
        )
        print(f"Successfully saved FINAL constellation boundaries to: {filename}")
    else:
        print("Warning: No boundary data to save")

# --- 6. Main Logic ---

def main():
    print("--- Step 1: Data Loading and Preprocessing ---")
    try:
        stars_df = pd.read_csv(FILE_CONFIGS["stars"]["path"], sep=FILE_CONFIGS["stars"]["sep"], header=None,** FILE_CONFIGS["stars"]["column_spec"])
        famous_stars_df = pd.read_csv(FILE_CONFIGS["famous_stars"]["path"], sep=FILE_CONFIGS["famous_stars"]["sep"], header=None, **FILE_CONFIGS["famous_stars"]["column_spec"])
        
        # 数据清洗和类型转换
        for col in ["_RAJ2000", "_DEJ2000", "HR", "Vmag"]:
            stars_df[col] = pd.to_numeric(stars_df[col], errors="coerce")
        stars_df = stars_df.dropna(subset=["_RAJ2000", "_DEJ2000", "HR", "Vmag"])
        
        famous_stars_df["HR"] = pd.to_numeric(famous_stars_df["HR"], errors="coerce")
        stars_df = pd.merge(stars_df, famous_stars_df, on="HR", how="left")
        stars_df["Display_Name"] = stars_df["Famous_Name"].fillna(stars_df["Name"]).fillna(f"HR {stars_df['HR'].astype(int)}")
        
        # 将赤经从小时转换为度
        stars_df["ra_deg"] = stars_df["_RAJ2000"]
        stars_df["dec_deg"] = stars_df["_DEJ2000"]
        
        print(f"Total stars after preprocessing: {len(stars_df)}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found -> {e.filename}")
        exit()
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()

    print("\n--- Step 2: Select Core Stars and Perform DBSCAN Clustering ---")
    core_stars_df = stars_df[(stars_df["Vmag"] < CORE_STAR_VMAG_LIMIT)].copy()
    print(f"Number of core stars (Vmag < {CORE_STAR_VMAG_LIMIT}): {len(core_stars_df)}")
    
    if len(core_stars_df) < MIN_PTS:
        print(f"Error: Insufficient core stars ({len(core_stars_df)} < {MIN_PTS}).")
        exit()
    
    # 准备DBSCAN输入 (RA, Dec in degrees)
    core_coords_deg = core_stars_df[["ra_deg", "dec_deg"]].values
    
    print(f"Performing DBSCAN clustering (EPS={EPS} degrees, MIN_PTS={MIN_PTS})")
    # 使用自定义的距离度量
    db = DBSCAN(eps=EPS, min_samples=MIN_PTS, metric=calculate_angular_distance_deg).fit(core_coords_deg)
    labels = db.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Discovered {n_clusters} initial constellations")
    
    core_stars_df["cluster_label"] = labels
    clustered_core_stars = core_stars_df[core_stars_df["cluster_label"] != -1]
    
    if n_clusters == 0:
        print("Error: No constellations found. Please adjust DBSCAN parameters.")
        exit()

    
    print("\n--- Step 3: Calculate Convex Hull for Each Constellation ---")
    constellation_data = []
    hulls_for_assignment = []
    
    for cluster_id in range(n_clusters):
        cluster_df = clustered_core_stars[clustered_core_stars["cluster_label"] == cluster_id].copy()
        cluster_coords_deg = cluster_df[["ra_deg", "dec_deg"]].values
        
        if len(cluster_coords_deg) >= 3:
            try:
                hull = ConvexHull(cluster_coords_deg)
                hulls_for_assignment.append(hull)
                
                constellation_data.append({
                    "cluster_id": cluster_id,
                    "core_stars": cluster_df,
                    "all_stars": cluster_df.copy(),
                    "hull": hull,
                    "center_ra": None,
                    "center_dec": None
                })
                print(f"Constellation {cluster_id}: {len(cluster_df)} core stars, convex hull calculated.")
                
            except Exception as e:
                print(f"Warning: Cannot calculate convex hull for constellation {cluster_id}. Error: {e}")
        else:
            print(f"Warning: Insufficient points for convex hull for constellation {cluster_id}")
    
    if not constellation_data:
        print("Error: No valid convex hulls calculated.")
        exit()

    print("\n--- Step 4: Assign Stars to Constellations ---")
    all_stars_df = stars_df[(stars_df["Vmag"] < ALL_STAR_VMAG_LIMIT)].copy()
    all_stars_df["constellation_id"] = -1
    
    print("Assigning stars inside convex hulls...")
    for idx, constellation in enumerate(constellation_data):
        hull = constellation["hull"]
        cluster_id = constellation["cluster_id"]
        
        core_hr = set(constellation["core_stars"]["HR"].values)
        mask = all_stars_df["HR"].isin(core_hr)
        all_stars_df.loc[mask, "constellation_id"] = cluster_id
        
        unassigned_stars = all_stars_df[all_stars_df["constellation_id"] == -1]
        if len(unassigned_stars) == 0: continue
            
        in_hull_mask = unassigned_stars.apply(
            lambda row: is_point_in_convex_hull_deg([row['ra_deg'], row['dec_deg']], hull),
            axis=1
        )
        
        if in_hull_mask.any():
            mask = unassigned_stars.index[in_hull_mask]
            all_stars_df.loc[mask, "constellation_id"] = cluster_id
            print(f"Constellation {cluster_id}: Assigned {in_hull_mask.sum()} stars from inside the hull.")

    # Update all stars for each constellation
    for constellation in constellation_data:
        cluster_id = constellation["cluster_id"]
        constellation_stars = all_stars_df[all_stars_df["constellation_id"] == cluster_id]
        constellation["all_stars"] = constellation_stars
        
        # Calculate constellation center
        center_ra, center_dec = calculate_spherical_center_deg(constellation_stars["ra_deg"], constellation_stars["dec_deg"])
        constellation["center_ra"] = center_ra
        constellation["center_dec"] = center_dec

    print("\n--- Step 5: Calculate Optimal Boundaries (including all stars) ---")
    for constellation in constellation_data:
        all_coords = constellation["all_stars"][["ra_deg", "dec_deg"]].values
        
        if len(all_coords) >= 3:
            try:
                # 使用优化的边界计算算法
                optimal_boundary, boundary_method = calculate_optimal_boundary(all_coords)
                
                constellation["optimal_boundary"] = optimal_boundary
                constellation["boundary_method"] = boundary_method
                
                if optimal_boundary is not None:
                    print(f"Constellation {constellation['cluster_id']}: {len(optimal_boundary)} boundary points using {boundary_method}")
                else:
                    print(f"Constellation {constellation['cluster_id']}: Failed to calculate boundary")
                    
            except Exception as e:
                print(f"计算星座 {constellation['cluster_id']} 的边界时出错: {e}")
                constellation["optimal_boundary"] = None
                constellation["boundary_method"] = "error"
        else:
            print(f"Warning: Not enough points for constellation {constellation['cluster_id']}")
            constellation["optimal_boundary"] = None
            constellation["boundary_method"] = "insufficient_points"
        
        print(f"Constellation {constellation['cluster_id']}: Total {len(constellation['all_stars'])} stars.")

    print("\n--- Step 6: Save Results ---")
    save_constellations_as_tsv(constellation_data, CONSTELLATIONS_TSV_FILE)

    results = {}
    for constellation in constellation_data:
        cluster_id = constellation["cluster_id"]
        results[f"Constellation_{cluster_id}"] = {
            "number_of_core_stars": len(constellation["core_stars"]),
            "total_number_of_stars": len(constellation["all_stars"]),
            "center_right_ascension_deg": constellation["center_ra"],
            "center_declination_deg": constellation["center_dec"],
            "boundary_method": constellation.get("boundary_method", "unknown"),
            "boundary_points_count": len(constellation["optimal_boundary"]) if constellation.get("optimal_boundary") is not None else 0,
            "core_star_HR_numbers": constellation["core_stars"]["HR"].tolist(),
            "all_star_HR_numbers": constellation["all_stars"]["HR"].tolist()
        }
    
    with open(RESULTS_JSON_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n--- Processing Complete ---")
    print("Summary of boundary methods used:")
    method_counts = {}
    for constellation in constellation_data:
        method = constellation.get("boundary_method", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1
    
    for method, count in method_counts.items():
        print(f"  {method}: {count} constellations")

if __name__ == "__main__":
    main()