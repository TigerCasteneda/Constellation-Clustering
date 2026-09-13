import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# --- 1. 配置参数 ---

FILE_CONFIGS = {
    "stars": {"path": "asu.tsv", "column_spec": {'usecols': [0, 1, 2, 3, 18], 'names': ["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"]}, "sep": ";"},
    "famous_stars": {"path": "asu_names.tsv", "column_spec": {'usecols': [0, 1], 'names': ["HR", "Famous_Name"]}, "sep": ";"},
}

# DBSCAN 聚类参数 (放宽标准)
EPS = 15.0
MIN_PTS = 4

# 评分权重
BRIGHTNESS_WEIGHT = 0.4
MORPHOLOGY_WEIGHT = 0.3
SEPARATION_WEIGHT = 0.3

# 动态调整参数
TARGET_TOTAL = 30
MAX_STARS_PER_CLUSTER = 10  # 新增：每个星座的最大恒星数

# 亮度相关
INITIAL_VMAG_MAX = 1.5
FINAL_VMAG_MAX = 4.5
STEP_VMAG = 0.5

# 亮度梯度相关
INITIAL_G_MAX = 1.5
FINAL_G_MAX = 2.5
STEP_G = 0.25

# 亮度均匀性相关
INITIAL_R_UNI_MAX = 3.5
FINAL_R_UNI_MAX = 5.5
STEP_R_UNI = 0.5

# 形态相关
INITIAL_COMPACTNESS_MIN = 0.25
FINAL_COMPACTNESS_MIN = 0.15
STEP_COMPACTNESS = -0.05

INITIAL_DISPERSION_MIN = 0.35
FINAL_DISPERSION_MIN = 0.25
STEP_DISPERSION = -0.05

# 分离性参数
SEPARATION_MIN = 15.0

# 绘图参数
FIG_SIZE = (12, 8)
MAX_DPI = 300
TARGET_SIZE_MB = 8

# --- 2. 辅助函数 ---
def calculate_angular_distance(ra1, dec1, ra2, dec2):
    ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
    ra2_rad, dec2_rad = np.radians(ra2), np.radians(dec2)
    delta_ra = np.abs(ra1_rad - ra2_rad)
    distance_rad = np.arccos(np.sin(dec1_rad) * np.sin(dec2_rad) + np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(delta_ra))
    return np.degrees(distance_rad)

def manual_euclidean_distance(p1, p2): return np.sqrt(np.sum((p1 - p2)**2))

def calculate_compactness(points):
    if len(points) < 2: return 0.0
    dist_matrix = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(len(points)): dist_matrix[i, j] = manual_euclidean_distance(points[i], points[j])
    max_dist = np.max(dist_matrix)
    avg_dist = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
    return avg_dist / max_dist if max_dist > 0 else 0.0

def calculate_bounding_box_area(points):
    if len(points) < 2: return 0.0
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    return (max_x - min_x) * (max_y - min_y)

def calculate_dispersion(points):
    if len(points) < 3: return 0.0
    bbox_area = calculate_bounding_box_area(points)
    if bbox_area == 0: return 0.0
    min_radius = float('inf')
    for i in range(len(points)):
        distances = np.array([manual_euclidean_distance(p, points[i]) for p in points])
        radius = np.max(distances)
        if radius < min_radius: min_radius = radius
    circle_area = np.pi * min_radius**2
    return bbox_area / circle_area if circle_area > 0 else 0.0

def dbscan_manual(points, eps, min_samples):
    n_points, labels, cluster_id = len(points), np.full(len(points), -1), 0
    for i in range(n_points):
        if labels[i] != -1: continue
        distances = np.array([manual_euclidean_distance(p, points[i]) for p in points])
        neighbors = np.where(distances <= eps)[0]
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = list(neighbors); queue.remove(i)
            while queue:
                j = queue.pop(0)
                if labels[j] == -1:
                    labels[j] = cluster_id
                    j_distances = np.array([manual_euclidean_distance(p, points[j]) for p in points])
                    j_neighbors = np.where(j_distances <= eps)[0]
                    if len(j_neighbors) >= min_samples: queue.extend(j_neighbors)
                elif labels[j] == 0: labels[j] = cluster_id
            cluster_id += 1
    return labels

# --- 3. 动态评分函数 ---
def score_brightness_dynamic(cluster_df, vmag_min_global, current_params):
    sorted_mag = cluster_df["Vmag"].sort_values().values
    if len(sorted_mag) < 2: return 0.0, False
    if sorted_mag[-1] > current_params['vmag_max']: return 0.0, False
    norm_mag = (sorted_mag[-1] - vmag_min_global) / (current_params['vmag_max'] - vmag_min_global)
    max_mag_score = 1.0 - np.clip(norm_mag, 0.0, 1.0)
    gradients = np.diff(sorted_mag)
    if np.any(gradients > current_params['g_max']): return 0.0, False
    avg_gradient = np.mean(gradients)
    gradient_score = 1.0 - np.clip(avg_gradient / current_params['g_max'], 0.0, 1.0)
    e1, e2 = 10**(-0.4 * sorted_mag[0]), 10**(-0.4 * sorted_mag[1])
    ratio = e1 / e2
    if ratio > current_params['r_uni_max']: return 0.0, False
    uni_score = 1.0 - np.clip((ratio - 1) / (current_params['r_uni_max'] - 1), 0.0, 1.0)
    return (max_mag_score + gradient_score + uni_score) / 3.0, True

def score_morphology_dynamic(points, current_params):
    if len(points) < 2: return 0.0, False
    compactness = calculate_compactness(points)
    if compactness < current_params['compactness_min']: return 0.0, False
    compact_score = np.clip(compactness / current_params['compactness_min'], 0.0, 1.0)
    if len(points) == 2: return compact_score, True
    dispersion = calculate_dispersion(points)
    if dispersion < current_params['dispersion_min']: return 0.0, False
    disp_score = np.clip(dispersion / current_params['dispersion_min'], 0.0, 1.0)
    return (compact_score + disp_score) / 2.0, True

# --- 4. 自适应图像保存函数 ---
def save_safe_image(filename):
    fig = plt.gcf()
    w_inch, h_inch = fig.get_size_inches()
    max_pixels_for_size = TARGET_SIZE_MB * 1024 * 1024 // 3
    max_dim_for_size = int(np.sqrt(max_pixels_for_size))
    dpi_based_on_size = int(max_dim_for_size / max(w_inch, h_inch))
    dpi_based_on_pil_limit = int(65535 / max(w_inch, h_inch)) - 100
    final_dpi = max(100, min(MAX_DPI, dpi_based_on_size, dpi_based_on_pil_limit))
    plt.savefig(filename, dpi=final_dpi, bbox_inches='tight', pil_kwargs={"optimize": True, "quality": 95})
    w_px, h_px = int(w_inch * final_dpi), int(h_inch * final_dpi)
    print(f"Image saved successfully as '{filename}' (Resolution: {w_px}x{h_px} pixels @ DPI: {final_dpi}).")

# --- 5. 主逻辑 ---
def main():
    print("--- Step 1: Data Loading and Preprocessing ---")
    try:
        stars_df = pd.read_csv(FILE_CONFIGS["stars"]["path"], sep=FILE_CONFIGS["stars"]["sep"], header=None,** FILE_CONFIGS["stars"]["column_spec"])
        famous_stars_df = pd.read_csv(FILE_CONFIGS["famous_stars"]["path"], sep=FILE_CONFIGS["famous_stars"]["sep"], header=None, **FILE_CONFIGS["famous_stars"]["column_spec"])
    except FileNotFoundError as e:
        print(f"Error: File not found -> {e.filename}")
        exit()
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()

    for col in ["_RAJ2000", "_DEJ2000", "HR", "Vmag"]: stars_df[col] = pd.to_numeric(stars_df[col], errors="coerce")
    stars_df = stars_df.dropna(subset=["_RAJ2000", "_DEJ2000", "HR", "Vmag"])
    famous_stars_df["HR"] = pd.to_numeric(famous_stars_df["HR"], errors="coerce")
    famous_stars_df = famous_stars_df.dropna(subset=["HR"])
    stars_df = pd.merge(stars_df, famous_stars_df, on="HR", how="left")
    stars_df["Display_Name"] = stars_df["Famous_Name"].fillna(stars_df["Name"]).fillna(f"HR {stars_df['HR'].astype(int)}")
    vmag_min_global = stars_df["Vmag"].min()

    print(f"Total stars loaded: {len(stars_df)}")

    print("\n--- Step 2: Initial DBSCAN Clustering ---")
    initial_bright_stars_df = stars_df[stars_df["Vmag"] <= FINAL_VMAG_MAX].copy()
    initial_bright_stars_df["ra_deg"] = initial_bright_stars_df["_RAJ2000"] * 15.0
    initial_bright_stars_df["dec_deg"] = initial_bright_stars_df["_DEJ2000"]
    print(f"Stars considered for initial DBSCAN (brighter than Vmag {FINAL_VMAG_MAX}): {len(initial_bright_stars_df)}")

    coords = initial_bright_stars_df[["ra_deg", "dec_deg"]].values
    print(f"Running manual DBSCAN clustering... (EPS={EPS}, MIN_PTS={MIN_PTS})")
    labels = dbscan_manual(coords, EPS, MIN_PTS)
    initial_bright_stars_df["cluster_label"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"DBSCAN found {n_clusters} initial clusters.")

    all_dbscan_clusters = []
    for label in set(labels):
        if label == -1: continue
        cluster_df = initial_bright_stars_df[initial_bright_stars_df["cluster_label"] == label].copy()
        # 新增：过滤掉恒星数超过上限的聚类
        if 2 <= len(cluster_df) <= MAX_STARS_PER_CLUSTER:
            all_dbscan_clusters.append(cluster_df)

    print(f"Filtered to {len(all_dbscan_clusters)} clusters with 2~{MAX_STARS_PER_CLUSTER} stars.")

    # --- 动态筛选循环 ---
    print("\n--- Step 3: Dynamic Filtering Loop ---")
    all_qualified_clusters = []
    current_vmag_max = INITIAL_VMAG_MAX
    current_g_max = INITIAL_G_MAX
    current_r_uni_max = INITIAL_R_UNI_MAX
    current_compactness_min = INITIAL_COMPACTNESS_MIN
    current_dispersion_min = INITIAL_DISPERSION_MIN

    while len(all_qualified_clusters) < TARGET_TOTAL:
        print(f"\n--- Trying with criteria: Vmag<={current_vmag_max}, G<={current_g_max}, R_uni<={current_r_uni_max}, C>={current_compactness_min:.2f}, D>={current_dispersion_min:.2f} ---")
        
        current_params = {
            'vmag_max': current_vmag_max,
            'g_max': current_g_max,
            'r_uni_max': current_r_uni_max,
            'compactness_min': current_compactness_min,
            'dispersion_min': current_dispersion_min
        }
        
        newly_qualified = []
        for cluster_df in all_dbscan_clusters:
            if any(id(c) == id(cluster_df) for c, *_ in all_qualified_clusters):
                continue
                
            # 再次确认恒星数在范围内（避免后续处理中被修改）
            if not (2 <= len(cluster_df) <= MAX_STARS_PER_CLUSTER):
                continue
                
            bright_score, bright_passed = score_brightness_dynamic(cluster_df, vmag_min_global, current_params)
            if not bright_passed: continue
            
            points = cluster_df[["ra_deg", "dec_deg"]].values
            morph_score, morph_passed = score_morphology_dynamic(points, current_params)
            if not morph_passed: continue
            
            newly_qualified.append( (cluster_df, bright_score, morph_score) )

        if newly_qualified:
            print(f"Found {len(newly_qualified)} new clusters with current criteria.")
            all_qualified_clusters.extend(newly_qualified)
            print(f"Total qualified clusters now: {len(all_qualified_clusters)}")
        else:
            print("No new clusters found with current criteria.")

        standard_relaxed = False
        if current_vmag_max < FINAL_VMAG_MAX:
            current_vmag_max = min(current_vmag_max + STEP_VMAG, FINAL_VMAG_MAX)
            standard_relaxed = True
        elif current_g_max < FINAL_G_MAX:
            current_g_max = min(current_g_max + STEP_G, FINAL_G_MAX)
            standard_relaxed = True
        elif current_r_uni_max < FINAL_R_UNI_MAX:
            current_r_uni_max = min(current_r_uni_max + STEP_R_UNI, FINAL_R_UNI_MAX)
            standard_relaxed = True
        elif current_compactness_min > FINAL_COMPACTNESS_MIN:
            current_compactness_min = max(current_compactness_min + STEP_COMPACTNESS, FINAL_COMPACTNESS_MIN)
            standard_relaxed = True
        elif current_dispersion_min > FINAL_DISPERSION_MIN:
            current_dispersion_min = max(current_dispersion_min + STEP_DISPERSION, FINAL_DISPERSION_MIN)
            standard_relaxed = True
        
        if not standard_relaxed:
            print("\n!!! Warning: Reached final relaxed criteria and still haven't found enough clusters. !!!")
            break

    # --- 最终排序和选择 ---
    print("\n--- Step 4: Final Ranking and Selection ---")
    if not all_qualified_clusters:
        print("Error: No clusters qualified even with the most relaxed criteria.")
        exit()

    final_candidates_with_scores = []
    for cluster_df, bright_score, morph_score in all_qualified_clusters:
        # 最终筛选：确保恒星数不超过上限
        if not (2 <= len(cluster_df) <= MAX_STARS_PER_CLUSTER):
            continue
            
        center_ra = cluster_df["ra_deg"].mean()
        center_dec = cluster_df["dec_deg"].mean()
        
        min_sep_dist = float('inf')
        for c, _ in final_candidates_with_scores:
            c_center_ra = c["ra_deg"].mean()
            c_center_dec = c["dec_deg"].mean()
            dist = calculate_angular_distance(center_ra, center_dec, c_center_ra, c_center_dec)
            if dist < min_sep_dist: min_sep_dist = dist
        
        sep_score = np.clip(min_sep_dist / SEPARATION_MIN, 0.0, 1.0) if min_sep_dist != float('inf') else 1.0
        total_score = (BRIGHTNESS_WEIGHT * bright_score + MORPHOLOGY_WEIGHT * morph_score + SEPARATION_WEIGHT * sep_score)
        final_candidates_with_scores.append( (cluster_df, total_score) )

    final_candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_clusters = final_candidates_with_scores[:TARGET_TOTAL]

    print(f"Selected Top-{TARGET_TOTAL} clusters from {len(final_candidates_with_scores)} qualified clusters.")

    # --- 6. 可视化 (2D) ---
    print("\n--- Step 5: 2D Visualization ---")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIG_SIZE, gridspec_kw={'height_ratios': [1, 2]})
    background_stars = stars_df[stars_df["Vmag"] <= FINAL_VMAG_MAX * 1.1]

    H, _, _ = np.histogram2d(background_stars["_DEJ2000"], background_stars["_RAJ2000"] * 15.0, bins=[np.arange(-90, 91, 3), np.arange(0, 361, 3)])
    ax1.imshow(H.T, extent=[0, 360, -90, 90], origin='lower', cmap='YlOrRd', aspect='auto')
    ax1.set_title(f'Overall Star Density Map (Vmag < {FINAL_VMAG_MAX * 1.1})', fontsize=12); ax1.set_xlabel('RA (deg)'); ax1.set_ylabel('Dec (deg)')

    ax2.scatter(initial_bright_stars_df["ra_deg"], initial_bright_stars_df["dec_deg"], s=1, c='lightgray', alpha=0.3)
    ax2.set_title(f'DBSCAN Results ({n_clusters} initial clusters)', fontsize=12); ax2.set_xlabel('RA (deg)'); ax2.set_ylabel('Dec (deg)')

    for ax, title_suffix, ra_lim in [(ax3, ' (Full Sky)', [0, 360]), (ax4, ' (Zoomed)', [150, 250])]:
        ax.scatter(background_stars["_RAJ2000"] * 15.0, background_stars["_DEJ2000"], s=1, c='gray', alpha=0.1)
        clusters_to_plot = top_clusters[:10] if title_suffix == ' (Zoomed)' else top_clusters
        for i, (cluster_df, score) in enumerate(clusters_to_plot):
            color = plt.cm.Spectral(i / len(top_clusters))
            ax.scatter(cluster_df["ra_deg"], cluster_df["dec_deg"], s=100, c=[color], edgecolors='k', alpha=0.8, zorder=3)
            if i < 15:
                center_ra = cluster_df["ra_deg"].mean()
                center_dec = cluster_df["dec_deg"].mean()
                rank = next(idx + 1 for idx, (c, _) in enumerate(top_clusters) if id(c) == id(cluster_df))
                ax.text(center_ra, center_dec + 2, f'#{rank}', color='white', fontsize=9, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8), zorder=4)
        ax.set_title(f'Final Top-{TARGET_TOTAL} New Constellations {title_suffix}', fontsize=12)
        ax.set_xlabel('Right Ascension (degrees)'); ax.set_ylabel('Declination (degrees)')
        ax.set_xlim(ra_lim); ax.set_ylim([-60, 90])
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_safe_image(f'top_{TARGET_TOTAL}_new_constellations_2d.png')
    plt.show()

    # --- 保存结果到 JSON 文件 ---
    print("\n--- Step 6: Saving Results for 3D Visualization ---")
    new_constellations_data = {}
    for i, (cluster_df, score) in enumerate(top_clusters):
        constellation_name = f"New_Const_{i+1}"
        hr_numbers = cluster_df['HR'].tolist()
        avg_ra = cluster_df['ra_deg'].mean()
        avg_dec = cluster_df['dec_deg'].mean()
        
        new_constellations_data[constellation_name] = {
            "hr_numbers": hr_numbers,
            "avg_ra_deg": float(avg_ra),
            "avg_dec_deg": float(avg_dec),
            "score": float(score),
            "num_stars": len(cluster_df)  # 新增：记录恒星数量
        }

    output_filename = "new_constellations_results.json"
    with open(output_filename, 'w') as f:
        json.dump(new_constellations_data, f, indent=4)

    print(f"Top-{TARGET_TOTAL} new constellations have been saved to '{output_filename}'.")
    for name, data in new_constellations_data.items():
        print(f"- {name} (Score: {data['score']:.2f}, Stars: {data['num_stars']}): HRs {data['hr_numbers']}")

if __name__ == "__main__":
    main()