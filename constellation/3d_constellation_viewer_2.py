import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import json

# --- 1. Configuration ---
FILE_CONFIGS = {
    "stars": {"path": "asu.tsv", "column_spec": {'usecols': [0, 1, 2, 3, 18], 'names': ["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"]}, "sep": ";"},
    "famous_stars": {"path": "asu_names.tsv", "column_spec": {'usecols': [0, 1], 'names': ["HR", "Famous_Name"]}, "sep": ";"},
}

# 星座数据文件
CONSTELLATIONS_3D_FILE = "constellation_division_results.json"

# --- Color Configuration ---
COLORSCALE = "Plasma_r"
CORE_STARS_CONNECTION_COLOR = "rgb(255, 255, 0)" # 黄色连线
BOUNDARY_STARS_COLOR = "rgba(0, 255, 255, 0.5)" # 半透明青色
CORE_STARS_MARKER_COLOR = "rgb(255, 255, 0)"     # 黄色标记
CONSTELLATION_LABEL_COLOR = "rgb(255, 255, 255)"

# --- Grid and Label Configuration ---
GRID_ALPHA = 0.3
GRID_COLOR = 'rgb(100, 100, 100)'
GRID_WIDTH = 1
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

print("\nFile reading completed, starting data cleaning and preprocessing...")
for col in ["_RAJ2000", "_DEJ2000", "HR", "Vmag"]:
    stars_df[col] = pd.to_numeric(stars_df[col], errors="coerce")
stars_df = stars_df.dropna(subset=["_RAJ2000", "_DEJ2000", "HR", "Vmag"])

famous_stars_df["HR"] = pd.to_numeric(famous_stars_df["HR"], errors="coerce")
famous_stars_df = famous_stars_df.dropna(subset=["HR"])
stars_df = pd.merge(stars_df, famous_stars_df, on="HR", how="left")
stars_df["Display_Name"] = stars_df["Famous_Name"].fillna(stars_df["Name"]).fillna(f"HR{stars_df['HR']}")

# --- Load Constellation Data ---
constellation_data = None
try:
    with open(CONSTELLATIONS_3D_FILE, 'r') as f:
        constellation_data = json.load(f)
    print(f"\nSuccessfully loaded constellation data from '{CONSTELLATIONS_3D_FILE}'. Found {len(constellation_data)} constellations.")
except FileNotFoundError:
    print(f"\nError: Constellation file '{CONSTELLATIONS_3D_FILE}' not found.")
    exit()
except Exception as e:
    print(f"\nError loading constellation file: {e}")
    exit()

print("\nAll data preprocessing completed.")

# --- 3. Coordinate Conversion ---
def convert_to_cartesian(ra_series, dec_series):
    coords = SkyCoord(ra=ra_series.values * u.hourangle, dec=dec_series.values * u.degree, frame="icrs")
    return coords.cartesian.xyz.value

stars_df["x"], stars_df["y"], stars_df["z"] = convert_to_cartesian(stars_df["_RAJ2000"], stars_df["_DEJ2000"])

# --- 4. Prepare Constellation Traces ---
print("\nPreparing 3D traces for constellations...")
constellation_traces = []

# 创建一个颜色映射，为每个星座分配一个独特的颜色
num_constellations = len(constellation_data)
colors = plt.cm.get_cmap('hsv', num_constellations) # 使用HSV颜色空间以获得鲜明对比

for i, (name, data) in enumerate(constellation_data.items()):
    print(i,name,data)
    # --- Plot Boundary Stars (all stars in the constellation) ---
    all_hr_list = data["all_hr_numbers"]
    boundary_df = stars_df[stars_df["HR"].isin(all_hr_list)]
    
    if not boundary_df.empty:
        # 使用独特的颜色绘制边界恒星
        boundary_color = f'rgba({int(colors(i)[0]*255)}, {int(colors(i)[1]*255)}, {int(colors(i)[2]*255)}, 0.4)'
        
        boundary_trace = go.Scatter3d(
            x=boundary_df["x"], y=boundary_df["y"], z=boundary_df["z"],
            mode="markers",
            marker=dict(size=3, color=boundary_color, symbol="circle"),
            name=f"{name} (Boundary)",
            hoverinfo='text',
            text=boundary_df["Display_Name"] + "<br>HR: " + boundary_df["HR"].astype(str) + "<br>Vmag: " + boundary_df["Vmag"].round(2).astype(str),
            showlegend=False # 在图例中隐藏边界恒星，避免过于冗长
        )
        constellation_traces.append(boundary_trace)

    # --- Plot Core Stars and Connections ---
    core_hr_list = data["core_hr_numbers"]
    core_df = stars_df[stars_df["HR"].isin(core_hr_list)]
    
    if not core_df.empty:
        # 按HR列表顺序排列核心恒星（这会影响连线顺序）
        core_df = core_df.set_index("HR").loc[core_hr_list].reset_index()
        
        x_core, y_core, z_core = core_df["x"], core_df["y"], core_df["z"]
        
        # 绘制连接线
        connection_trace = go.Scatter3d(
            x=x_core, y=y_core, z=z_core,
            mode="lines",
            line=dict(color=CORE_STARS_CONNECTION_COLOR, width=3),
            name=f"{name}", # 图例中只显示星座名称
            hoverinfo='none'
        )
        constellation_traces.append(connection_trace)
        
        # 绘制核心恒星标记
        core_marker_trace = go.Scatter3d(
            x=x_core, y=y_core, z=z_core,
            mode="markers+text",
            marker=dict(size=6, color=CORE_STARS_MARKER_COLOR, symbol="diamond"),
            text=core_df["Display_Name"],
            textfont=dict(size=8, color=CONSTELLATION_LABEL_COLOR),
            textposition="top center",
            hoverinfo='text',
            # text=core_df["Display_Name"] + "<br>(Core Star)",
            showlegend=False
        )
        constellation_traces.append(core_marker_trace)
        
        # 在星座中心添加标签
        avg_ra_deg, avg_dec_deg = data["avg_ra_deg"], data["avg_dec_deg"]
        label_x, label_y, label_z = convert_to_cartesian(pd.Series([avg_ra_deg/15.0]), pd.Series([avg_dec_deg]))
        scale = 1.1
        label_trace = go.Scatter3d(
            x=[label_x[0] * scale], y=[label_y[0] * scale], z=[label_z[0] * scale],
            mode="text",
            text=[name],
            textfont=dict(size=LABEL_FONT_SIZE+2, color=CONSTELLATION_LABEL_COLOR, weight="bold"),
            showlegend=False,
            hoverinfo='none'
        )
        constellation_traces.append(label_trace)

print("Preparation of constellation traces completed.")

# --- 5. Grid and Label Generation ---
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

# --- 6. Visualization Plotting ---
fig = go.Figure()
print("\nAssembling the 3D plot...")

# Add constellation traces
for trace in constellation_traces:
    fig.add_trace(trace)

# Add grid and labels
print("Generating and plotting sky grid and coordinate labels...")
grid_and_label_traces = create_sky_grid_with_labels()
for trace in grid_and_label_traces:
    fig.add_trace(trace)

# --- 7. Layout Settings ---
fig.update_layout(
    title=dict(text="3D Constellation Map: Core Stars & Boundaries", font=dict(size=20, color="white"), x=0.5, xanchor="center"),
    scene=dict(
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False), 
        zaxis=dict(visible=False), 
        bgcolor="rgb(0, 0, 0)", 
        camera=dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0.7, y=0.5, z=0.5))
    ),
    paper_bgcolor="rgb(0, 0, 0)",
    legend=dict(
        font=dict(color="white"), 
        bgcolor="rgba(0,0,0,0.7)", 
        x=0.01, y=0.99, xanchor="left", yanchor="top",
        title=dict(text="Constellations (Core Connections)", font=dict(color="white"))
    ),
    margin=dict(l=50, r=50, t=50, b=50)
)

fig.show()