import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import json

# --- 1. Configuration Parameters ---
FILE_CONFIGS = {
    "stars": {"path": "asu.tsv", "column_spec": {'usecols': [0, 1, 2, 3, 18], 'names': ["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"]}, "sep": ";"},
    "famous_stars": {"path": "asu_names.tsv", "column_spec": {'usecols': [0, 1], 'names': ["HR", "Famous_Name"]}, "sep": ";"},
}

# DBSCAN parameters
EPS = 5.0  # Clustering radius in degrees
MIN_PTS = 3  # Minimum number of points for a cluster

# Star selection parameters
CORE_STAR_VMAG_LIMIT = 4.5 # Brightness limit for core stars
ALL_STAR_VMAG_LIMIT = 6.5   # Brightness limit for all considered stars

# Plotting parameters
FIG_SIZE = (16, 10)
MAX_DPI = 300
STAR_SIZE = 5
CORE_STAR_SIZE = 15
CONVEX_HULL_ALPHA = 0.3

# --- 2. Helper Functions ---

def calculate_angular_distance(ra1, dec1, ra2, dec2):
    """Calculate the angular distance between two celestial objects (in degrees)."""
    ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
    ra2_rad, dec2_rad = np.radians(ra2), np.radians(dec2)
    delta_ra = np.abs(ra1_rad - ra2_rad)
    
    cos_dist = (np.sin(dec1_rad) * np.sin(dec2_rad) + 
                np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(delta_ra))
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_dist))

def is_point_in_convex_hull(point, hull):
    """Check if a point lies inside a convex hull."""
    point = np.array(point)
    
    # For each face of the hull, check if the point is on the inner side
    for eq in hull.equations:
        if np.dot(eq[:-1], point) + eq[-1] > 1e-10:  # Consider numerical errors
            return False
    return True

def find_closest_convex_hull(point, hulls):
    """Find the convex hull closest to a given point."""
    min_distance = float('inf')
    closest_hull_idx = -1
    
    for i, hull in enumerate(hulls):
        # Calculate the minimum distance from the point to any point on the hull
        distances = np.sqrt(np.sum((hull.points - point)**2, axis=1))
        min_dist_to_hull = np.min(distances)
        
        if min_dist_to_hull < min_distance:
            min_distance = min_dist_to_hull
            closest_hull_idx = i
    
    return closest_hull_idx

def calculate_spherical_center(ra_series, dec_series):
    """Calculate the geometric center of a set of spherical coordinates."""
    ra_rad = np.radians(ra_series)
    dec_rad = np.radians(dec_series)
    
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)
    
    center_dec_rad = np.arcsin(mean_z)
    center_ra_rad = np.arctan2(mean_y, mean_x)
    
    center_ra_deg = np.degrees(center_ra_rad) % 360.0
    center_dec_deg = np.degrees(center_dec_rad)
    
    return center_ra_deg, center_dec_deg

# --- 3. Main Logic ---

def main():
    print("--- Step 1: Data Loading and Preprocessing ---")
    try:
        # Load data
        stars_df = pd.read_csv(FILE_CONFIGS["stars"]["path"], sep=FILE_CONFIGS["stars"]["sep"], header=None,** FILE_CONFIGS["stars"]["column_spec"])
        famous_stars_df = pd.read_csv(FILE_CONFIGS["famous_stars"]["path"], sep=FILE_CONFIGS["famous_stars"]["sep"], header=None, **FILE_CONFIGS["famous_stars"]["column_spec"])
        
        # Data type conversion and cleaning
        for col in ["_RAJ2000", "_DEJ2000", "HR", "Vmag"]:
            stars_df[col] = pd.to_numeric(stars_df[col], errors="coerce")
        stars_df = stars_df.dropna(subset=["_RAJ2000", "_DEJ2000", "HR", "Vmag"])
        
        # Merge with famous star names
        famous_stars_df["HR"] = pd.to_numeric(famous_stars_df["HR"], errors="coerce")
        famous_stars_df = famous_stars_df.dropna(subset=["HR"])
        stars_df = pd.merge(stars_df, famous_stars_df, on="HR", how="left")
        stars_df["Display_Name"] = stars_df["Famous_Name"].fillna(stars_df["Name"]).fillna(f"HR {stars_df['HR'].astype(int)}")
        
        # Convert coordinates to degrees
        stars_df["ra_deg"] = stars_df["_RAJ2000"] * 15.0
        stars_df["dec_deg"] = stars_df["_DEJ2000"]
        
        print(f"Total number of stars: {len(stars_df)}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found -> {e.filename}")
        exit()
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()

    print("\n--- Step 2: Select Core Stars and Perform DBSCAN Clustering ---")
    # Select core stars (Vmag < 3)
    core_stars_df = stars_df[(stars_df["Vmag"] < CORE_STAR_VMAG_LIMIT)].copy()
    print(f"Number of core stars (Vmag < {CORE_STAR_VMAG_LIMIT}): {len(core_stars_df)}")
    
    if len(core_stars_df) < MIN_PTS:
        print(f"Error: Insufficient core stars ({len(core_stars_df)} < {MIN_PTS}), cannot perform DBSCAN clustering.")
        exit()
    
    # Prepare DBSCAN input
    core_coords = core_stars_df[["ra_deg", "dec_deg"]].values
    
    print(f"Performing DBSCAN clustering (EPS={EPS} degrees, MIN_PTS={MIN_PTS})")
    
    # Custom DBSCAN implementation with angular distance
    def dbscan_custom(points, eps, min_samples):
        n_points = len(points)
        labels = np.full(n_points, -1)
        cluster_id = 0
        
        for i in range(n_points):
            if labels[i] != -1:
                continue
                
            # Find all neighbors
            neighbors = []
            for j in range(n_points):
                dist = calculate_angular_distance(points[i][0], points[i][1], points[j][0], points[j][1])
                if dist <= eps:
                    neighbors.append(j)
            
            if len(neighbors) < min_samples:
                labels[i] = -1  # Noise point
            else:
                # Start a new cluster
                labels[i] = cluster_id
                queue = neighbors.copy()
                queue.remove(i)
                
                while queue:
                    j = queue.pop(0)
                    if labels[j] == -1:
                        labels[j] = cluster_id
                        
                        # Find neighbors of j
                        j_neighbors = []
                        for k in range(n_points):
                            dist = calculate_angular_distance(points[j][0], points[j][1], points[k][0], points[k][1])
                            if dist <= eps:
                                j_neighbors.append(k)
                        
                        if len(j_neighbors) >= min_samples:
                            queue.extend(j_neighbors)
                
                cluster_id += 1
        
        return labels
    
    # Run DBSCAN
    labels = dbscan_custom(core_coords, EPS, MIN_PTS)
    core_stars_df["cluster_label"] = labels
    
    # Statistics of clustering results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Discovered {n_clusters} initial constellations")
    
    # Filter out noise points
    clustered_core_stars = core_stars_df[core_stars_df["cluster_label"] != -1]
    print(f"Number of core stars after clustering: {len(clustered_core_stars)}")
    
    if n_clusters == 0:
        print("Error: No constellations found. Please adjust DBSCAN parameters.")
        exit()

    print("\n--- Step 3: Calculate Convex Hull for Each Constellation ---")
    # Calculate convex hull for each cluster
    constellation_data = []
    hulls = []
    
    for cluster_id in range(n_clusters):
        cluster_df = clustered_core_stars[clustered_core_stars["cluster_label"] == cluster_id].copy()
        cluster_coords = cluster_df[["ra_deg", "dec_deg"]].values
        
        # Calculate convex hull (requires at least 3 points)
        if len(cluster_coords) >= 3:
            try:
                hull = ConvexHull(cluster_coords)
                hulls.append(hull)
                
                constellation_data.append({
                    "cluster_id": cluster_id,
                    "core_stars": cluster_df,
                    "all_stars": cluster_df.copy(),
                    "hull": hull,
                    "center_ra": None,
                    "center_dec": None
                })
                
                print(f"Constellation {cluster_id}: {len(cluster_df)} core stars")
                
            except:
                print(f"Warning: Cannot calculate convex hull for constellation {cluster_id}")
        else:
            print(f"Warning: Insufficient points for constellation {cluster_id} to calculate convex hull")
    
    if not constellation_data:
        print("Error: No convex hulls calculated successfully")
        exit()
    
    n_valid_clusters = len(constellation_data)
    print(f"Number of constellations with successful convex hull calculation: {n_valid_clusters}")

    print("\n--- Step 4: Assign Stars to Constellations ---")
    # Select all stars to be considered
    all_stars_df = stars_df[(stars_df["Vmag"] < ALL_STAR_VMAG_LIMIT)].copy()
    all_stars_df["constellation_id"] = -1  # -1 indicates unassigned
    
    # First, assign stars inside convex hulls
    print("Assigning stars inside convex hulls...")
    
    for idx, constellation in enumerate(constellation_data):
        hull = constellation["hull"]
        cluster_id = constellation["cluster_id"]
        
        # Mark core stars
        core_hr = set(constellation["core_stars"]["HR"].values)
        mask = all_stars_df["HR"].isin(core_hr)
        all_stars_df.loc[mask, "constellation_id"] = cluster_id
        
        # Find other stars inside the convex hull
        unassigned_stars = all_stars_df[all_stars_df["constellation_id"] == -1]
        
        if len(unassigned_stars) == 0:
            continue
            
        in_hull_mask = []
        for _, star in unassigned_stars.iterrows():
            point = [star["ra_deg"], star["dec_deg"]]
            in_hull = is_point_in_convex_hull(point, hull)
            in_hull_mask.append(in_hull)
        
        if np.any(in_hull_mask):
            mask = unassigned_stars.index[in_hull_mask]
            all_stars_df.loc[mask, "constellation_id"] = cluster_id
            print(f"Constellation {cluster_id}: Assigned {np.sum(in_hull_mask)} stars from inside the convex hull")
    
    # Then, assign stars not in any convex hull to the nearest constellation
    print("\nAssigning stars outside convex hulls...")
    
    unassigned_stars = all_stars_df[all_stars_df["constellation_id"] == -1]
    
    if len(unassigned_stars) > 0:
        print(f"There are {len(unassigned_stars)} unassigned stars, will assign to the nearest constellation")
        
        for idx, (_, star) in enumerate(unassigned_stars.iterrows()):
            point = [star["ra_deg"], star["dec_deg"]]
            closest_hull_idx = find_closest_convex_hull(point, hulls)
            
            if closest_hull_idx != -1:
                all_stars_df.loc[star.name, "constellation_id"] = constellation_data[closest_hull_idx]["cluster_id"]
                
                # Print progress every 100 stars
                if (idx + 1) % 100 == 0:
                    print(f"Assigned {idx + 1}/{len(unassigned_stars)} stars")
    
    # Update all stars for each constellation
    for constellation in constellation_data:
        cluster_id = constellation["cluster_id"]
        constellation_stars = all_stars_df[all_stars_df["constellation_id"] == cluster_id]
        constellation["all_stars"] = constellation_stars
        
        # Calculate constellation center
        center_ra, center_dec = calculate_spherical_center(constellation_stars["ra_deg"], constellation_stars["dec_deg"])
        constellation["center_ra"] = center_ra
        constellation["center_dec"] = center_dec
        
        print(f"Constellation {cluster_id}: Total {len(constellation_stars)} stars")

    print("\n--- Step 5: Recalculate Convex Hulls with All Assigned Stars ---")
    # Recalculate convex hulls using all assigned stars
    final_hulls = []
    
    for constellation in constellation_data:
        all_coords = constellation["all_stars"][["ra_deg", "dec_deg"]].values
        
        if len(all_coords) >= 3:
            try:
                new_hull = ConvexHull(all_coords)
                constellation["final_hull"] = new_hull
                final_hulls.append(new_hull)
            except:
                print(f"Warning: Cannot recalculate convex hull for constellation {constellation['cluster_id']}")
                constellation["final_hull"] = constellation["hull"]
                final_hulls.append(constellation["hull"])
        else:
            constellation["final_hull"] = constellation["hull"]
            final_hulls.append(constellation["hull"])

    print("\n--- Step 6: Visualization ---")
    # Create figure and axes
    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE, squeeze=False)
    ax1, ax2 = axes[0, 0], axes[1, 0]
    
    # Define colors
    colors = plt.cm.get_cmap('rainbow', n_valid_clusters)
    
    # First subplot: Constellation outlines
    ax1.set_title('Constellation Division Results - Outlines', fontsize=16)
    ax1.set_xlabel('Right Ascension (degrees)', fontsize=12)
    ax1.set_ylabel('Declination (degrees)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(-90, 90)
    
    # Plot all stars
    all_coords = all_stars_df[["ra_deg", "dec_deg"]].values
    ax1.scatter(all_coords[:, 0], all_coords[:, 1], s=STAR_SIZE, alpha=0.5, c='lightgray', label='Other Stars')
    
    # Plot each constellation
    for idx, constellation in enumerate(constellation_data):
        color = colors(idx)
        cluster_id = constellation["cluster_id"]
        
        # Plot stars in this constellation
        constellation_coords = constellation["all_stars"][["ra_deg", "dec_deg"]].values
        ax1.scatter(constellation_coords[:, 0], constellation_coords[:, 1], s=STAR_SIZE+2, alpha=0.7, c=[color], label=f'Constellation {cluster_id}')
        
        # Plot core stars
        core_coords = constellation["core_stars"][["ra_deg", "dec_deg"]].values
        ax1.scatter(core_coords[:, 0], core_coords[:, 1], s=CORE_STAR_SIZE, alpha=0.9, c=[color], marker='*', edgecolors='black')
        
        # Plot convex hull
        hull = constellation["final_hull"]
        polygon = Polygon(hull.points[hull.vertices], facecolor=color, alpha=CONVEX_HULL_ALPHA, edgecolor=color, linewidth=2)
        ax1.add_patch(polygon)
        
        # Label constellation center and name
        ax1.text(constellation["center_ra"], constellation["center_dec"], f'Constellation {cluster_id}', 
                fontsize=10, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_aspect('equal')
    
    # Second subplot: Celestial sphere projection
    ax2.set_title('Distribution of Constellations on the Celestial Sphere', fontsize=16)
    ax2.set_xlabel('Right Ascension (degrees)', fontsize=12)
    ax2.set_ylabel('Declination (degrees)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-90, 90)
    
    # Plot all stars
    ax2.scatter(all_stars_df["ra_deg"], all_stars_df["dec_deg"], s=STAR_SIZE, alpha=0.3, c='lightgray')
    
    # Plot each constellation
    for idx, constellation in enumerate(constellation_data):
        color = colors(idx)
        
        # Plot stars in this constellation
        constellation_stars = constellation["all_stars"]
        ax2.scatter(constellation_stars["ra_deg"], constellation_stars["dec_deg"], s=STAR_SIZE+3, alpha=0.8, c=[color])
        
        # Plot convex hull
        hull = constellation["final_hull"]
        polygon = Polygon(hull.points[hull.vertices], facecolor=color, alpha=CONVEX_HULL_ALPHA, edgecolor=color, linewidth=2)
        ax2.add_patch(polygon)
    
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('constellation_division_results.png', dpi=MAX_DPI, bbox_inches='tight')
    plt.show()
    
    print("\n--- Step 7: Save Results ---")
    # Save constellation information
    results = {}
    
    for constellation in constellation_data:
        cluster_id = constellation["cluster_id"]
        results[f"Constellation_{cluster_id}"] = {
            "number_of_core_stars": len(constellation["core_stars"]),
            "total_number_of_stars": len(constellation["all_stars"]),
            "center_right_ascension": constellation["center_ra"],
            "center_declination": constellation["center_dec"],
            "core_star_HR_numbers": constellation["core_stars"]["HR"].tolist(),
            "all_star_HR_numbers": constellation["all_stars"]["HR"].tolist()
        }
    
    # Save as JSON
    with open('constellation_division_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Results saved to 'constellation_division_results.json'")
    print("\n--- Processing Complete ---")

if __name__ == "__main__":
    main()