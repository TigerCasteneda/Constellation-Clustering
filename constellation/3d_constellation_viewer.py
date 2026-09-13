import pandas as pd
import plotly.graph_objects as go
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import json
import random # 新增：用于生成随机颜色

# --- 1. Configuration ---
FILE_CONFIGS = {
    "stars": {"path": "asu.tsv", "column_spec": {'usecols': [0, 1, 2, 3, 18], 'names': ["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"]}, "sep": ";"},
    "famous_stars": {"path": "asu_names.tsv", "column_spec": {'usecols': [0, 1], 'names': ["HR", "Famous_Name"]}, "sep": ";"},
    "constellations": {"path": "asu_constellations.tsv", "column_spec": {'usecols': [0, 1, 4], 'names': ["_RAJ2000", "_DEJ2000", "cst"]}, "sep": ";"}
}

# 新增：新星座结果文件路径
NEW_CONSTELLATIONS_FILE = "new_constellations_results.json"

TARGET_CONSTELLATION = "UMa"
BIG_DIPPER_HR = [4295, 4301, 4554, 4660, 4905, 5054, 5191]
BIG_DIPPER_ORDER = [5191,5054,4905,4660,4554,4295,4301]
POLARIS_HR = 424

# --- Color Configuration ---
COLORSCALE = "Plasma_r"
# NEW_CONSTELLATIONS_COLOR = "rgb(0, 255, 255)" # 不再使用单一颜色

# --- Grid and Label Configuration ---
GRID_ALPHA = 0.6
GRID_COLOR = 'rgb(180, 180, 180)'
GRID_WIDTH = 1.5
LABEL_FONT_SIZE = 10
LABEL_COLOR = 'rgb(200, 200, 200)'
# NEW_CONSTELLATION_LABEL_COLOR = "rgb(0, 255, 255)" # 不再使用单一颜色

# --- 新增：生成随机RGB颜色的函数 ---
def get_random_color():
    """生成一个随机的、明亮的RGB颜色字符串。"""
    r = random.randint(100, 255)
    g = random.randint(100, 255)
    b = random.randint(100, 255)
    return f"rgb({r}, {g}, {b})"

# --- 2. Data Loading and Preprocessing ---
# (这部分代码与之前完全相同)
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

target_constellation_df = constellation_df[constellation_df["cst"] == TARGET_CONSTELLATION].copy()
if not target_constellation_df.empty:
    for col in ["_RAJ2000", "_DEJ2000"]:
        target_constellation_df[col] = pd.to_numeric(target_constellation_df[col], errors="coerce")
    target_constellation_df = target_constellation_df.dropna(subset=["_RAJ2000", "_DEJ2000"])
    print(f"'{TARGET_CONSTELLATION}' constellation boundary data cleaning completed, remaining points: {len(target_constellation_df)}")
else:
    print(f"Warning: No boundary data found for constellation {TARGET_CONSTELLATION}")

# --- 新增：加载新星座数据 ---
new_constellations_data = None
try:
    with open(NEW_CONSTELLATIONS_FILE, 'r') as f:
        new_constellations_data = json.load(f)
    print(f"\nSuccessfully loaded new constellations data from '{NEW_CONSTELLATIONS_FILE}'. Found {len(new_constellations_data)} constellations.")
except FileNotFoundError:
    print(f"\nWarning: New constellations file '{NEW_CONSTELLATIONS_FILE}' not found. No new constellations will be displayed.")
except Exception as e:
    print(f"\nError loading new constellations file: {e}. No new constellations will be displayed.")

print("\nAll data preprocessing completed.")

# --- 3. Coordinate Conversion ---
def convert_to_cartesian(ra_series, dec_series):
    coords = SkyCoord(ra=ra_series.values * u.hourangle, dec=dec_series.values * u.degree, frame="icrs")
    return coords.cartesian.xyz.value

stars_df["x"], stars_df["y"], stars_df["z"] = convert_to_cartesian(stars_df["_RAJ2000"], stars_df["_DEJ2000"])

# 筛选并按正确顺序排序北斗七星数据
big_dipper_df = stars_df[stars_df["HR"].isin(BIG_DIPPER_HR)]
if not big_dipper_df.empty:
    big_dipper_df = big_dipper_df.set_index("HR").loc[BIG_DIPPER_ORDER].reset_index()
    bd_coords = convert_to_cartesian(big_dipper_df["_RAJ2000"], big_dipper_df["_DEJ2000"])
else:
    bd_coords = ([], [], [])
    
bd_x, bd_y, bd_z = bd_coords

polaris_df = stars_df[stars_df["HR"] == POLARIS_HR]
p_coords = convert_to_cartesian(polaris_df["_RAJ2000"], polaris_df["_DEJ2000"]) if not polaris_df.empty else ([], [], [])
p_x, p_y, p_z = p_coords

const_coords = convert_to_cartesian(target_constellation_df["_RAJ2000"], target_constellation_df["_DEJ2000"]) if not target_constellation_df.empty else ([], [], [])
const_x, const_y, const_z = const_coords

# --- 新增：为新星座准备3D坐标 (修改后) ---
new_constellations_traces = []
if new_constellations_data:
    print("\nPreparing 3D traces for new constellations...")
    for name, data in new_constellations_data.items():
        # --- 核心修改点 ---
        # 为每个星座生成一个随机颜色
        constellation_color = get_random_color()
        
        hr_list = data["hr_numbers"]
        # 提取恒星并按HR列表顺序排列
        constellation_stars_df = stars_df[stars_df["HR"].isin(hr_list)]
        if not constellation_stars_df.empty:
            constellation_stars_df = constellation_stars_df.set_index("HR").loc[hr_list].reset_index()
            
            x, y, z = constellation_stars_df["x"].tolist(), constellation_stars_df["y"].tolist(), constellation_stars_df["z"].tolist()
            
            # 创建星座轮廓线 (使用随机颜色)
            line_trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(color=constellation_color, width=3), # 使用随机颜色
                name=f"{name} (New)",
                hoverinfo='none'
            )
            new_constellations_traces.append(line_trace)
            
            # 在星座中心添加标签 (使用与线条相同的颜色)
            avg_ra, avg_dec = data["avg_ra_deg"], data["avg_dec_deg"]
            label_x, label_y, label_z = convert_to_cartesian(pd.Series([avg_ra/15.0]), pd.Series([avg_dec]))
            
            scale = 1.05
            label_trace = go.Scatter3d(
                x=[label_x[0] * scale], y=[label_y[0] * scale], z=[label_z[0] * scale],
                mode="text",
                text=[name],
                textfont=dict(size=LABEL_FONT_SIZE+1, color=constellation_color, weight="bold"), # 使用随机颜色
                showlegend=False,
                hoverinfo='none'
            )
            new_constellations_traces.append(label_trace)
    print("Preparation of new constellation traces completed.")

# --- 4. Grid and Label Generation ---
# (这部分代码与之前完全相同)
def create_sky_grid_with_labels():
    all_traces = []
    dec_values = np.arange(-90, 91, 30)
    for dec in dec_values:
        ra_points = np.linspace(0, 24, 200)
        dec_points = np.full_like(ra_points, dec)
        x, y, z = convert_to_cartesian(pd.Series(ra_points), pd.Series(dec_points))
        all_traces.append(go.Scatter3d(x=x[:-1], y=y[:-1], z=z[:-1], mode='lines', line=dict(color=GRID_COLOR, width=GRID_WIDTH), opacity=GRID_ALPHA, showlegend=False, hoverinfo='none'))

    ra_values = np.arange(0, 24, 4)
    label_dec_positions = [15, -15]
    for i, ra in enumerate(ra_values):
        dec_points = np.linspace(-90, 90, 100)
        ra_points = np.full_like(dec_points, ra)
        x, y, z = convert_to_cartesian(pd.Series(ra_points), pd.Series(dec_points))
        all_traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=GRID_COLOR, width=GRID_WIDTH), opacity=GRID_ALPHA, showlegend=False, hoverinfo='none'))
        
        label_dec = label_dec_positions[i % len(label_dec_positions)]
        x_label, y_label, z_label = convert_to_cartesian(pd.Series([ra]), pd.Series([label_dec]))
        scale = 1.1
        all_traces.append(go.Scatter3d(x=[x_label[0] * scale], y=[y_label[0] * scale], z=[z_label[0] * scale], mode='text', text=[f'{ra}h'], textfont=dict(size=LABEL_FONT_SIZE, color=LABEL_COLOR), showlegend=False, hoverinfo='none'))

    ra_for_dec_labels = 2
    dec_labels = [-60, -30, 0, 30, 60]
    for dec in dec_labels:
        x_label, y_label, z_label = convert_to_cartesian(pd.Series([ra_for_dec_labels]), pd.Series([dec]))
        scale = 1.1
        all_traces.append(go.Scatter3d(x=[x_label[0] * scale], y=[y_label[0] * scale], z=[z_label[0] * scale], mode='text', text=[f'{dec}°'], textfont=dict(size=LABEL_FONT_SIZE, color=LABEL_COLOR), showlegend=False, hoverinfo='none'))
    return all_traces

# --- 5. Visualization Plotting ---
fig = go.Figure()
print("\nPlotting background stars and other elements...")

# Plot background stars with custom color scale
fig.add_trace(go.Scatter3d(
    x=stars_df["x"], y=stars_df["y"], z=stars_df["z"], mode="markers",
    marker=dict(
        size=2 - stars_df["Vmag"] / 6,
        color=stars_df["Vmag"],
        colorscale=COLORSCALE,
        opacity=0.7,
        showscale=True,
        colorbar=dict(title="Apparent Magnitude (Vmag)", x=1.1)
    ),
    text=stars_df["Display_Name"] + "<br>HR: " + stars_df["HR"].astype(str) + "<br>Vmag: " + stars_df["Vmag"].round(2).astype(str),
    hoverinfo="text",
    name="Background Stars"
))

# Plot constellation boundary
if len(const_x) > 0:
    fig.add_trace(go.Scatter3d(x=const_x, y=const_y, z=const_z, mode="lines", line=dict(color="rgba(255, 255, 255, 0.7)", width=2), name=f"{TARGET_CONSTELLATION} Boundary"))

# --- 新增：绘制新星座 ---
if new_constellations_traces:
    print("Adding new constellations to the plot...")
    for trace in new_constellations_traces:
        fig.add_trace(trace)

# Add grid and labels
print("Generating and plotting sky grid and coordinate labels...")
grid_and_label_traces = create_sky_grid_with_labels()
for trace in grid_and_label_traces:
    fig.add_trace(trace)

# --- 6. Layout Settings ---
fig.update_layout(
    title=dict(text="Celestial Sphere: New Constellations Discovery (Colorful)", font=dict(size=20, color="white"), x=0.5, xanchor="center"),
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="rgb(0, 0, 0)", camera=dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0.7, y=0.5, z=0.5))),
    paper_bgcolor="rgb(0, 0, 0)",
    legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.5)", x=0.01, y=0.99, xanchor="left", yanchor="top"),
    margin=dict(l=50, r=50, t=50, b=50)
)

fig.show()