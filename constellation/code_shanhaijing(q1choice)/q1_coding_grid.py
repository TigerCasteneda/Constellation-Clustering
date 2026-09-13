import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

# --- 1. Configuration ---

# Get current script directory and parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

FILE_CONFIGS = {
    "stars": {"path": os.path.join(parent_dir, "asu.tsv"), "column_spec": {'usecols': [0, 1, 2, 3, 18], 'names': ["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"]}, "sep": ";"},
    "famous_stars": {"path": os.path.join(parent_dir, "asu_names.tsv"), "column_spec": {'usecols': [0, 1], 'names': ["HR", "Famous_Name"]}, "sep": ";"},
    "constellations": {"path": os.path.join(parent_dir, "asu_constellations.tsv"), "column_spec": {'usecols': [0, 1, 4], 'names': ["_RAJ2000", "_DEJ2000", "cst"]}, "sep": ";"}
}

TARGET_CONSTELLATION = "UMA"
# HR number list for the Big Dipper (for filtering)
BIG_DIPPER_HR = [4295, 4301, 4554, 4660, 4905, 5054, 5191]
# Correct connection order of the Big Dipper (from the end of the handle to the bowl)
# Alkaid -> Mizar -> Alioth -> Megrez -> Phecda -> Merak -> Dubhe
BIG_DIPPER_ORDER = [5191,5054,4905,4660,4554,4295,4301]
POLARIS_HR = 424

# --- Color Configuration ---
COLORSCALE = "Plasma_r"  

# --- Grid and Label Configuration ---
GRID_ALPHA = 0.6
GRID_COLOR = 'rgb(180, 180, 180)'
GRID_WIDTH = 1.5
LABEL_FONT_SIZE = 10
LABEL_COLOR = 'rgb(200, 200, 200)'

# --- 2. Data Loading and Preprocessing ---
data_frames = {}
print("Starting to read files in batch...")
for key, config in FILE_CONFIGS.items():
    try:
        df = pd.read_csv(config["path"], sep=config["sep"], header=None, **config["column_spec"])
        data_frames[key] = df
        print(f"Successfully read file: {config['path']}, Rows: {len(df)}")
    except FileNotFoundError:
        print(f"Error: File not found -> {config['path']}")
        exit()
    except Exception as e:
        print(f"Error reading file {config['path']}: {e}")
        exit()

stars_df = data_frames["stars"]
famous_stars_df = data_frames["famous_stars"]
constellation_df = data_frames["constellations"]

print("\nFile reading completed, starting data cleaning and preprocessing...")
for col in ["_RAJ2000", "_DEJ2000", "HR", "Vmag"]:
    stars_df[col] = pd.to_numeric(stars_df[col], errors="coerce")
stars_df = stars_df.dropna(subset=["_RAJ2000", "_DEJ2000", "HR", "Vmag"])
print(f"Star data cleaning completed, remaining rows: {len(stars_df)}")

famous_stars_df["HR"] = pd.to_numeric(famous_stars_df["HR"], errors="coerce")
famous_stars_df = famous_stars_df.dropna(subset=["HR"])
print(f"Famous star data cleaning completed, remaining rows: {len(famous_stars_df)}")

stars_df = pd.merge(stars_df, famous_stars_df, on="HR", how="left")
stars_df["Display_Name"] = stars_df["Famous_Name"].fillna(stars_df["Name"]).fillna(f"HR{stars_df['HR']}")

constellation_df["cst"] = list(map(str.rstrip,constellation_df["cst"]))
target_constellation_df = constellation_df[constellation_df["cst"] == TARGET_CONSTELLATION].copy()
if not target_constellation_df.empty:
    for col in ["_RAJ2000", "_DEJ2000"]:
        target_constellation_df[col] = pd.to_numeric(target_constellation_df[col], errors="coerce")
    target_constellation_df = target_constellation_df.dropna(subset=["_RAJ2000", "_DEJ2000"])
    print(f"'{TARGET_CONSTELLATION}' constellation boundary data cleaning completed, remaining points: {len(target_constellation_df)}")
else:
    print(f"Warning: No boundary data found for constellation {TARGET_CONSTELLATION}")

print("\nAll data preprocessing completed.")

# --- 3. Coordinate Conversion (Fixed Version) ---
def convert_to_cartesian(ra_series, dec_series):
    """
    Convert Right Ascension (RA) and Declination (Dec) to Cartesian coordinates
    Assumes input units are degrees
    """
    try:
        # Create SkyCoord object, assuming input is in degrees
        coords = SkyCoord(ra=ra_series.values * u.degree, dec=dec_series.values * u.degree, frame='icrs')
        xyz = coords.cartesian.xyz.value
        return xyz[0], xyz[1], xyz[2]
    except Exception as e:
        print(f"Error in coordinate conversion: {e}")
        # Return empty arrays if conversion fails
        return np.array([]), np.array([]), np.array([])

# Ensure RA and Dec data are complete
print(f"Filtering stars with complete RA/Dec data...")
stars_df = stars_df.dropna(subset=["_RAJ2000", "_DEJ2000"])
print(f"Remaining stars after RA/Dec filter: {len(stars_df)}")

# Convert coordinates for all stars
stars_df["x"], stars_df["y"], stars_df["z"] = convert_to_cartesian(stars_df["_RAJ2000"], stars_df["_DEJ2000"])

# Filter and sort Big Dipper data in correct order
big_dipper_df = stars_df[stars_df["HR"].isin(BIG_DIPPER_HR)]
if not big_dipper_df.empty:
    # Use .set_index and .loc to reorder data according to specified HR order
    big_dipper_df = big_dipper_df.set_index("HR").loc[BIG_DIPPER_ORDER].reset_index()
    bd_x, bd_y, bd_z = big_dipper_df["x"].values, big_dipper_df["y"].values, big_dipper_df["z"].values
    print(f"Found all {len(big_dipper_df)} Big Dipper stars")
    # Print Big Dipper star information
    print("Big Dipper stars information:")
    for idx, row in big_dipper_df.iterrows():
        print(f"{row['Display_Name']} (HR{row['HR']}): Vmag={row['Vmag']:.2f}")
else:
    bd_x, bd_y, bd_z = [], [], []
    print("Warning: Big Dipper stars not found in dataset")

# Polaris (Fixed logic: wrap single value in a list)
polaris_df = stars_df[stars_df["HR"] == POLARIS_HR]
if not polaris_df.empty:
    p_x, p_y, p_z = [polaris_df["x"].values[0]], [polaris_df["y"].values[0]], [polaris_df["z"].values[0]]
    print(f"Found Polaris (HR{POLARIS_HR}): Vmag={polaris_df['Vmag'].values[0]:.2f}")
else:
    p_x, p_y, p_z = [], [], []
    print("Warning: Polaris not found in dataset")

# Constellation boundaries
if not target_constellation_df.empty:
    const_x, const_y, const_z = convert_to_cartesian(target_constellation_df["_RAJ2000"], target_constellation_df["_DEJ2000"])
else:
    const_x, const_y, const_z = [], [], []

# Close the boundary polygon by appending the first point at the end
const_x = np.append(const_x,const_x[0])
const_y = np.append(const_y,const_y[0])
const_z = np.append(const_z,const_z[0])

# --- 4. Grid and Label Generation for Matplotlib ---
def create_sky_grid_mpl(ax):
    # Generate Declination circles
    dec_values = np.arange(-60, 61, 30)  # From -60° to +60°, every 30°
    for dec in dec_values:
        ra_points = np.linspace(0, 360, 100)  # RA from 0° to 360°
        dec_points = np.full_like(ra_points, dec)
        
        # Convert to Cartesian coordinates
        ra_series = pd.Series(ra_points)
        dec_series = pd.Series(dec_points)
        x, y, z = convert_to_cartesian(ra_series, dec_series)
        
        ax.plot(x, y, z, color='gray', alpha=GRID_ALPHA, linewidth=GRID_WIDTH)
    
    # Generate Right Ascension lines
    ra_values = np.arange(0, 360, 60)  # Every 60° (equivalent to 4 hours)
    for ra in ra_values:
        dec_points = np.linspace(-60, 60, 50)  # Dec from -60° to +60°
        ra_points = np.full_like(dec_points, ra)
        
        # Convert to Cartesian coordinates
        ra_series = pd.Series(ra_points)
        dec_series = pd.Series(dec_points)
        x, y, z = convert_to_cartesian(ra_series, dec_series)
        
        ax.plot(x, y, z, color='gray', alpha=GRID_ALPHA, linewidth=GRID_WIDTH)
        
        # Add RA labels (converted to hours)
        ra_hours = ra / 15  # Convert degrees to hours
        label_dec = 65  # Label position
        x_label, y_label, z_label = convert_to_cartesian(pd.Series([ra]), pd.Series([label_dec]))
        if len(x_label) > 0:
            scale = 1.15
            ax.text(x_label[0] * scale, y_label[0] * scale, z_label[0] * scale, 
                   f'{ra_hours:.0f}h', fontsize=LABEL_FONT_SIZE, color='white', 
                   ha='center', va='center')
    
    # Add Dec labels
    dec_labels = [-60, -30, 0, 30, 60]
    ra_for_labels = 0  # Add labels at 0° RA
    for dec in dec_labels:
        x_label, y_label, z_label = convert_to_cartesian(pd.Series([ra_for_labels]), pd.Series([dec]))
        if len(x_label) > 0:
            scale = 1.15
            ax.text(x_label[0] * scale, y_label[0] * scale, z_label[0] * scale, 
                   f'{dec}°', fontsize=LABEL_FONT_SIZE, color='white', 
                   ha='center', va='center')

# --- 5. Visualization Plotting with Matplotlib ---
print("Plotting background stars and other elements using Matplotlib...")

# Create figure and 3D axes
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Set black background
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Hide coordinate axes
ax.set_axis_off()

# Plot background stars
# For performance, sample star data if too large
sample_size = min(5000, len(stars_df))  # Show maximum 5000 stars
if len(stars_df) > sample_size:
    stars_sample_df = stars_df.sample(sample_size, random_state=42)
else:
    stars_sample_df = stars_df.copy()

# Map colors using magnitude
norm = Normalize(vmin=stars_df['Vmag'].min(), vmax=stars_df['Vmag'].max())
cmap = plt.cm.plasma_r  # Reverse plasma colormap
colors = cmap(norm(stars_sample_df['Vmag']))

# Set transparency
colors[:, 3] = np.clip(1 - stars_sample_df['Vmag']/10, 0.3, 0.9)

# Calculate point sizes
point_sizes = np.clip(6 - stars_sample_df['Vmag'], 2, 10)

# Plot stars
scatter = ax.scatter(stars_sample_df['x'], stars_sample_df['y'], stars_sample_df['z'], 
                    s=point_sizes, c=colors, marker='o', alpha=0.7)

# Add colorbar
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.1)
cbar.set_label('Apparent Magnitude (Vmag)', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.yaxis.set_label_position('right')
txt = cbar.ax.yaxis.get_label()
txt.set_va('bottom')
txt.set_ha('left')
for label in cbar.ax.get_yticklabels():
    label.set_color('white')

# Plot constellation boundary
if len(const_x) > 0:
    ax.plot(const_x, const_y, const_z, color='white', alpha=0.7, linewidth=2, 
           label=f'{TARGET_CONSTELLATION} Boundary')

# Plot famous stars
famous_stars_visual_df = stars_df[stars_df["Famous_Name"].notna()]
if not famous_stars_visual_df.empty:
    # Special marking for Dubhe and Merak (pointer stars) in the Big Dipper
    dubhe_merak_df = famous_stars_visual_df[famous_stars_visual_df["HR"].isin([4301, 4295])]
    if not dubhe_merak_df.empty:
        ax.scatter(dubhe_merak_df['x'], dubhe_merak_df['y'], dubhe_merak_df['z'], 
                  s=100, c='g', marker='D', alpha=1.0, label='Pointer Stars (Dubhe & Merak)')
        # Add labels
        for _, row in dubhe_merak_df.iterrows():
            ax.text(row['x'], row['y'], row['z'], f"{row['Famous_Name']}", 
                   fontsize=8, color='g', ha='right', va='bottom')
    
    # Plot other famous stars
    other_famous_df = famous_stars_visual_df[~famous_stars_visual_df["HR"].isin([4301, 4295])]
    if not other_famous_df.empty:
        ax.scatter(other_famous_df['x'], other_famous_df['y'], other_famous_df['z'], 
                  s=50, c='r', marker='s', alpha=0.9, label='Other Famous Stars')

# Plot Big Dipper connecting lines (in correct order)
if len(bd_x) > 0:
    ax.plot(bd_x, bd_y, bd_z, color='y', linewidth=4, label='Big Dipper Connecting Line')
    
    # Add connecting line from Big Dipper pointer stars to Polaris to help users find Polaris
    if len(p_x) > 0 and len(bd_x) >= 2:
        # Dubhe (HR=4301) is the last element in BIG_DIPPER_ORDER
        # Merak (HR=4295) is the second last element in BIG_DIPPER_ORDER
        dubhe_idx = BIG_DIPPER_ORDER.index(4301)
        merak_idx = BIG_DIPPER_ORDER.index(4295)
        
        # Calculate midpoint between Dubhe and Merak
        mid_x = (bd_x[dubhe_idx] + bd_x[merak_idx]) / 2
        mid_y = (bd_y[dubhe_idx] + bd_y[merak_idx]) / 2
        mid_z = (bd_z[dubhe_idx] + bd_z[merak_idx]) / 2
        
        # Calculate vector from Dubhe to Merak
        vec_x = bd_x[merak_idx] - bd_x[dubhe_idx]
        vec_y = bd_y[merak_idx] - bd_y[dubhe_idx]
        vec_z = bd_z[merak_idx] - bd_z[dubhe_idx]
        
        # Extended point outward from the bowl
        extend_factor = 5  # Extension multiple
        extend_x = mid_x + vec_x * extend_factor
        extend_y = mid_y + vec_y * extend_factor
        extend_z = mid_z + vec_z * extend_factor
        
        # Plot guide line from pointer stars to Polaris
        guide_x = [mid_x, extend_x, p_x[0]]
        guide_y = [mid_y, extend_y, p_y[0]]
        guide_z = [mid_z, extend_z, p_z[0]]
        
        ax.plot(guide_x, guide_y, guide_z, color='m', alpha=0.7, 
               linewidth=2, linestyle='--', label='Find Polaris Guide Line')
    
    # Plot Big Dipper stars
    ax.scatter(bd_x, bd_y, bd_z, s=50, c='y', marker='D', label='Big Dipper Stars')
    # Add Big Dipper labels
    for idx, row in big_dipper_df.iterrows():
        ax.text(row['x'], row['y'], row['z'], f"{row['Display_Name']}", 
               fontsize=6, color='y', ha='left', va='top')

# Plot Polaris
if len(p_x) > 0:
    ax.scatter(p_x, p_y, p_z, s=70, c='c', marker='o', label='Polaris')
    ax.text(p_x[0], p_y[0], p_z[0], "Polaris (North Star)", 
           fontsize=7, color='c', ha='right', va='bottom')

# Add celestial grid and labels
print("Generating and plotting sky grid and coordinate labels...")
create_sky_grid_mpl(ax)

# Set viewing angle
ax.view_init(elev=30, azim=45)  # Set initial viewing angle

# Set coordinate range to make celestial sphere appear spherical
max_range = max([stars_df['x'].max(), stars_df['y'].max(), stars_df['z'].max()])
ax.set_xlim(-max_range*1.2, max_range*1.2)
ax.set_ylim(-max_range*1.2, max_range*1.2)
ax.set_zlim(-max_range*1.2, max_range*1.2)

# Add title
plt.title("Celestial Sphere Visualization: Finding Polaris with the Big Dipper", 
         color='white', fontsize=16, pad=20)

# Add legend
ax.legend(loc='upper left', fontsize=8, framealpha=0.5, facecolor='black', edgecolor='white')
for text in ax.get_legend().get_texts():
    text.set_color('white')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()