import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import json
import math

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

# Output files
RESULTS_JSON_FILE = 'constellation_division_results_deg.json'
CONSTELLATIONS_TSV_FILE = 'new_constellations_final_hulls_deg.tsv'
CONSTELLATION_NAME_PREFIX = "NewCst_"

# --- 2. Helper Functions (Degree-based) ---

def calculate_polygon_centroid(points):
    """
    计算二维多边形的质心 (Centroid / Geometric Center)。
    该多边形由一系列按顺序排列的顶点坐标定义。
    """
    n = len(points)
    if n < 3:
        return np.mean(points, axis=0)

    centroid = np.zeros(2)
    area = 0.0

    for i in range(n):
        x_i, y_i = points[i]
        x_j, y_j = points[(i + 1) % n]
        cross = (x_i * y_j) - (x_j * y_i)
        area += cross
        centroid[0] += (x_i + x_j) * cross
        centroid[1] += (y_i + y_j) * cross

    if abs(area) < 1e-10:
        return np.mean(points, axis=0)

    return centroid / (3 * area)

def calculate_angular_distance_deg(point1, point2):
    """
    使用Vincenty公式计算球面上两点间的角距离（适用于所有纬度，包括极点）。
    """
    ra1, dec1 = point1
    ra2, dec2 = point2
    ra1_rad, dec1_rad = math.radians(ra1), math.radians(dec1)
    ra2_rad, dec2_rad = math.radians(ra2), math.radians(dec2)
    delta_ra, delta_dec = abs(ra1_rad - ra2_rad), abs(dec1_rad - dec2_rad)
    sin_delta_dec_half, sin_delta_ra_half = math.sin(delta_dec / 2), math.sin(delta_ra / 2)
    a = sin_delta_dec_half** 2 + math.cos(dec1_rad) * math.cos(dec2_rad) * sin_delta_ra_half**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return math.degrees(c)

def is_point_in_convex_hull_deg(point, hull):
    """Check if a point (RA, Dec in degrees) lies inside a convex hull."""
    point = np.array(point)
    for eq in hull.equations:
        if np.dot(eq[:-1], point) + eq[-1] > 1e-10:
            return False
    return True

def find_closest_convex_hull_deg(point, hulls):
    """Find the convex hull closest to a given point (RA, Dec in degrees)."""
    min_distance = float('inf')
    closest_hull_idx = -1
    for i, hull in enumerate(hulls):
        distances = np.array([calculate_angular_distance_deg(point, p) for p in hull.points])
        min_dist_to_hull = np.min(distances)
        if min_dist_to_hull < min_distance:
            min_distance, closest_hull_idx = min_dist_to_hull, i
    return closest_hull_idx

def save_constellations_as_tsv(constellation_data, filename):
    """
    Save the FINAL convex hull vertices (after assigning faint stars) in TSV format.
    """
    print("\n--- Preparing to save FINAL CONSTELLATION HULLS ---")
    tsv_data = []
    for constellation in constellation_data:
        cluster_id = constellation["cluster_id"]
        final_hull = constellation["final_hull"]
        constellation_name = f"{CONSTELLATION_NAME_PREFIX}{cluster_id:02d}"
        ordered_vertices = constellation.get("ordered_vertices_indices", final_hull.vertices)
        vertices = final_hull.points[ordered_vertices]
        for ra_deg, dec_deg in vertices:
            tsv_data.append({'ra_deg': ra_deg, 'dec_deg': dec_deg, 'cst': constellation_name})
    
    df = pd.DataFrame(tsv_data)
    df = df.sort_values(['cst', 'ra_deg'])
    df[['ra_deg', 'dec_deg', 'cst']].to_csv(
        filename, sep='\t', index=False, header=False, float_format='%.6f'
    )
    print(f"Successfully saved FINAL constellation hulls to: {filename}")

def calculate_spherical_center_deg(ra_series, dec_series):
    """Calculate the geometric center of a set of spherical coordinates (input in degrees)."""
    ra_rad, dec_rad = np.radians(ra_series), np.radians(dec_series)
    x, y, z = np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)
    mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)
    center_dec_rad, center_ra_rad = np.arcsin(mean_z), np.arctan2(mean_y, mean_x)
    return np.degrees(center_ra_rad) % 360.0, np.degrees(center_dec_rad)

# --- 3. Main Logic ---
def main():
    print("--- Step 1: Data Loading and Preprocessing ---")
    try:
        stars_df = pd.read_csv(FILE_CONFIGS["stars"]["path"], sep=FILE_CONFIGS["stars"]["sep"], header=None,** FILE_CONFIGS["stars"]["column_spec"])
        famous_stars_df = pd.read_csv(FILE_CONFIGS["famous_stars"]["path"], sep=FILE_CONFIGS["famous_stars"]["sep"], header=None, **FILE_CONFIGS["famous_stars"]["column_spec"])
        
        for col in ["_RAJ2000", "_DEJ2000", "HR", "Vmag"]:
            stars_df[col] = pd.to_numeric(stars_df[col], errors="coerce")
        stars_df = stars_df.dropna(subset=["_RAJ2000", "_DEJ2000", "HR", "Vmag"])
        
        famous_stars_df["HR"] = pd.to_numeric(famous_stars_df["HR"], errors="coerce")
        stars_df = pd.merge(stars_df, famous_stars_df, on="HR", how="left")
        stars_df["Display_Name"] = stars_df["Famous_Name"].fillna(stars_df["Name"]).fillna(f"HR {stars_df['HR'].astype(int)}")
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
    
    core_coords_deg = core_stars_df[["ra_deg", "dec_deg"]].values
    print(f"Performing DBSCAN clustering (EPS={EPS} degrees, MIN_PTS={MIN_PTS})")
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
    constellation_data, hulls_for_assignment = [], []
    
    for cluster_id in range(n_clusters):
        cluster_df = clustered_core_stars[clustered_core_stars["cluster_label"] == cluster_id].copy()
        cluster_coords_deg = cluster_df[["ra_deg", "dec_deg"]].values
        
        if len(cluster_coords_deg) >= 3:
            try:
                hull = ConvexHull(cluster_coords_deg)
                hulls_for_assignment.append(hull)
                constellation_data.append({
                    "cluster_id": cluster_id, "core_stars": cluster_df, "all_stars": cluster_df.copy(),
                    "hull": hull, "center_ra": None, "center_dec": None
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
        hull, cluster_id = constellation["hull"], constellation["cluster_id"]
        core_hr = set(constellation["core_stars"]["HR"].values)
        mask = all_stars_df["HR"].isin(core_hr)
        all_stars_df.loc[mask, "constellation_id"] = cluster_id
        
        unassigned_stars = all_stars_df[all_stars_df["constellation_id"] == -1]
        if len(unassigned_stars) == 0: continue
            
        in_hull_mask = unassigned_stars.apply(
            lambda row: is_point_in_convex_hull_deg([row['ra_deg'], row['dec_deg']], hull), axis=1
        )
        
        if in_hull_mask.any():
            mask = unassigned_stars.index[in_hull_mask]
            all_stars_df.loc[mask, "constellation_id"] = cluster_id
            print(f"Constellation {cluster_id}: Assigned {in_hull_mask.sum()} stars from inside the hull.")

    for constellation in constellation_data:
        cluster_id = constellation["cluster_id"]
        constellation_stars = all_stars_df[all_stars_df["constellation_id"] == cluster_id]
        constellation["all_stars"] = constellation_stars
        center_ra, center_dec = calculate_spherical_center_deg(constellation_stars["ra_deg"], constellation_stars["dec_deg"])
        constellation["center_ra"], constellation["center_dec"] = center_ra, center_dec

    print("\n--- Step 5: Recalculate FINAL Convex Hulls (including all stars) ---")
    for constellation in constellation_data:
        cluster_id = constellation["cluster_id"]
        all_coords = constellation["all_stars"][["ra_deg", "dec_deg"]].values
        
        if len(all_coords) >= 3:
            try:
                new_hull = ConvexHull(all_coords)
                hull_points = all_coords[new_hull.vertices]

                # --- MODIFICATION: Calculate Centroid instead of Mean Center ---
                centroid = calculate_polygon_centroid(hull_points)
                
                angles = np.arctan2(hull_points[:, 1] - centroid[1], hull_points[:, 0] - centroid[0])
                sorted_indices = np.argsort(angles)
                constellation["ordered_vertices_indices"] = new_hull.vertices[sorted_indices]
                # --- END MODIFICATION ---
                
                constellation["final_hull"] = new_hull
            except Exception as e:
                print(f"Warning: Cannot recalculate FINAL convex hull for constellation {cluster_id}. Error: {e}")
                constellation["final_hull"] = constellation["hull"]
                if "hull" in constellation:
                    constellation["ordered_vertices_indices"] = constellation["hull"].vertices
        else:
            print(f"Warning: Not enough points for FINAL convex hull for constellation {cluster_id}")
            constellation["final_hull"] = constellation["hull"]
            if "hull" in constellation:
                constellation["ordered_vertices_indices"] = constellation["hull"].vertices        
        
        print(f"Constellation {cluster_id}: Total {len(constellation['all_stars'])} stars.")

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
            "core_star_HR_numbers": constellation["core_stars"]["HR"].tolist(),
            "all_star_HR_numbers": constellation["all_stars"]["HR"].tolist()
        }
    
    with open(RESULTS_JSON_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n--- Processing Complete ---")

if __name__ == "__main__":
    main()