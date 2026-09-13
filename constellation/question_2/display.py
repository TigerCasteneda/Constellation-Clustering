import pandas as pd
import plotly.graph_objects as go
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import json

# --- 1. Configuration ---
FILE_CONFIGS = {
    "stars": {"path": "asu.tsv", "column_spec": {'usecols': [0, 1, 2, 3, 18], 'names': ["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"]}, "sep": ";"},
    "famous_stars": {"path": "asu_names.tsv", "column_spec": {'usecols': [0, 1], 'names': ["HR", "Famous_Name"]}, "sep": ";"},
}

# 星座数据文件 (由之前的DBSCAN代码生成)
CONSTELLATIONS_3D_FILE = "constellation_division_results.json"

# --- Color Configuration ---
BOUNDARY_STAR_BASE_COLOR = "rgb(173, 216, 230)"  # 淡蓝色
CORE_STARS_CONNECTION_COLOR = "rgb(255, 255, 0)"  # 黄色连线
CONSTELLATION_LABEL_COLOR = "rgb(255, 255, 255)"  # 白色标签

# --- Grid and Label Configuration ---
GRID_ALPHA = 0.2
GRID_COLOR = 'rgb(100, 100, 100)'
GRID_WIDTH = 1
LABEL_FONT_SIZE = 10
LABEL_COLOR = 'rgb(200, 200, 200)'

# --- Marker Size Configuration ---
MIN_STAR_SIZE = 1.5
MAX_STAR_SIZE = 8
CORE_STAR_SIZE = 6

# --- 2. Data Loading and Preprocessing ---
print("--- Step 1: Loading and Preprocessing Star Data ---")
data_frames = {}
try:
    for key, config in FILE_CONFIGS.items():
        df = pd.read_csv(config["path"], sep=config["sep"], header=None,** FILE_CONFIGS[key]["column_spec"])
        data_frames[key] = df
        print(f"Successfully read file: {config['path']}, Rows: {len(df)}")
except FileNotFoundError as e:
    print(f"Error: File not found -> {e.filename}")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

stars_df = data_frames["stars"]
famous_stars_df = data_frames["famous_stars"]

# 清洗恒星数据
for col in ["_RAJ2000", "_DEJ2000", "HR", "Vmag"]:
    stars_df[col] = pd.to_numeric(stars_df[col], errors="coerce")
stars_df = stars_df.dropna(subset=["_RAJ2000", "_DEJ2000", "HR", "Vmag"])

# 合并著名恒星名称
famous_stars_df["HR"] = pd.to_numeric(famous_stars_df["HR"], errors="coerce")
famous_stars_df = famous_stars_df.dropna(subset=["HR"])
stars_df = pd.merge(stars_df, famous_stars_df, on="HR", how="left")
stars_df["Display_Name"] = stars_df["Famous_Name"].fillna(stars_df["Name"]).fillna(f"HR {stars_df['HR'].astype(int)}")

print(f"Total stars after preprocessing: {len(stars_df)}")

# --- Load Constellation Data ---
print("\n--- Step 2: Loading Constellation Data ---")
constellation_data = None
try:
    with open(CONSTELLATIONS_3D_FILE, 'r') as f:
        constellation_data = json.load(f)
    print(f"Successfully loaded constellation data from '{CONSTELLATIONS_3D_FILE}'. Found {len(constellation_data)} constellations.")
except FileNotFoundError:
    print(f"\nError: Constellation file '{CONSTELLATIONS_3D_FILE}' not found. Please run the DBSCAN clustering code first.")
    exit()
except Exception as e:
    print(f"\nError loading constellation file: {e}")
    exit()

# --- 3. Coordinate Conversion ---
print("\n--- Step 3: Converting to Cartesian Coordinates ---")
def convert_to_cartesian(ra_hours, dec_degrees):
    """将赤经(小时)和赤纬(度)转换为笛卡尔坐标"""
    coords = SkyCoord(ra=ra_hours.values * u.hourangle, dec=dec_degrees.values * u.degree, frame="icrs")
    return coords.cartesian.xyz.value

# 为所有恒星计算三维坐标
stars_df["x"], stars_df["y"], stars_df["z"] = convert_to_cartesian(stars_df["_RAJ2000"], stars_df["_DEJ2000"])
stars_df["ra_deg"] = stars_df["_RAJ2000"] * 15.0
stars_df["dec_deg"] = stars_df["_DEJ2000"]

# --- 4. Prepare Constellation Traces for Plotly ---
print("\n--- Step 4: Preparing 3D Traces for Constellations ---")
constellation_traces = []
all_boundary_stars = []

for i, (constellation_name, data) in enumerate(constellation_data.items()):
    all_hr_list = data["all_star_HR_numbers"]
    core_hr_list = data["core_star_HR_numbers"]
    
    boundary_df = stars_df[stars_df["HR"].isin(all_hr_list)]
    if not boundary_df.empty:
        all_boundary_stars.append(boundary_df)
        
        core_df = stars_df[stars_df["HR"].isin(core_hr_list)]
        if not core_df.empty:
            center_ra = data["center_right_ascension"]
            center_dec = data["center_declination"]
            
            core_df['angle'] = np.arctan2(
                core_df['dec_deg'] - center_dec,
                core_df['ra_deg'] - center_ra
            )
            core_df = core_df.sort_values('angle')
            
            x_core, y_core, z_core = core_df["x"], core_df["y"], core_df["z"]
            
            # 绘制连接线
            connection_trace = go.Scatter3d(
                x=x_core, y=y_core, z=z_core,
                mode="lines",
                line=dict(color=CORE_STARS_CONNECTION_COLOR, width=3),
                name=f"{constellation_name}",
                hoverinfo='none',
                showlegend=True
            )
            constellation_traces.append(connection_trace)
            
            # 绘制核心恒星标记
            core_marker_trace = go.Scatter3d(
                x=x_core, y=y_core, z=z_core,
                mode="markers",
                marker=dict(size=CORE_STAR_SIZE, color=CORE_STARS_CONNECTION_COLOR, symbol="diamond", line=dict(width=1, color='black')),
                name=f"{constellation_name} (Core)",
                hoverinfo='text',
                text=core_df["Display_Name"] + "<br>(Core Star)",
                showlegend=False
            )
            constellation_traces.append(core_marker_trace)
            
            # 在星座中心添加标签
            label_ra_hours = center_ra / 15.0
            label_x, label_y, label_z = convert_to_cartesian(pd.Series([label_ra_hours]), pd.Series([center_dec]))
            
            scale = 1.15
            label_trace = go.Scatter3d(
                x=[label_x[0] * scale], y=[label_y[0] * scale], z=[label_z[0] * scale],
                mode="text",
                text=[constellation_name.replace("Constellation_", "Const. ")],
                textfont=dict(size=LABEL_FONT_SIZE+2, color=CONSTELLATION_LABEL_COLOR, weight="bold"),
                showlegend=False,
                hoverinfo='none'
            )
            constellation_traces.append(label_trace)

# --- 创建统一的边界恒星轨迹 (应用统一的颜色方案) ---
if all_boundary_stars:
    combined_boundary_df = pd.concat(all_boundary_stars)
    
    # 根据星等计算大小和透明度
    sizes = np.clip(8 - combined_boundary_df["Vmag"], MIN_STAR_SIZE, MAX_STAR_SIZE)
    opacities = np.clip(1.2 - (combined_boundary_df["Vmag"] / 10), 0.3, 0.9)
    
    # 将基础颜色和透明度结合成rgba格式
    # 从 rgb 字符串中提取 r, g, b 值
    r, g, b = map(int, BOUNDARY_STAR_BASE_COLOR.replace('rgb(', '').replace(')', '').split(','))
    # 为每个点创建一个 rgba 颜色字符串
    colors = [f'rgba({r}, {g}, {b}, {opacity:.2f})' for opacity in opacities]
    
    boundary_trace = go.Scatter3d(
        x=combined_boundary_df["x"], y=combined_boundary_df["y"], z=combined_boundary_df["z"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,  # 使用带有透明度的颜色列表
            showscale=True,
            colorbar=dict(
                title="Apparent Magnitude (Vmag)",
                title_font=dict(color="white", size=12),
                tickfont=dict(color="white", size=10),
                x=1.05,
                y=0.5,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1,
            )
        ),
        name="Stars",
        hoverinfo='text',
        text=combined_boundary_df["Display_Name"] + "<br>HR: " + combined_boundary_df["HR"].astype(str) + "<br>Vmag: " + combined_boundary_df["Vmag"].round(2).astype(str),
        showlegend=True
    )
    constellation_traces.insert(0, boundary_trace)

print("Preparation of constellation traces completed.")

# --- 5. Grid and Label Generation ---
def create_sky_grid_with_labels():
    print("\n--- Step 5: Generating Sky Grid and Labels ---")
    all_traces = []
    
    dec_values = np.arange(-90, 91, 30)
    for dec in dec_values:
        ra_points = np.linspace(0, 24, 200)
        dec_points = np.full_like(ra_points, dec)
        x, y, z = convert_to_cartesian(pd.Series(ra_points), pd.Series(dec_points))
        all_traces.append(go.Scatter3d(
            x=x, y=y, z=z, 
            mode='lines', 
            line=dict(color=GRID_COLOR, width=GRID_WIDTH), 
            opacity=GRID_ALPHA, 
            showlegend=False, 
            hoverinfo='none'
        ))

    ra_values = np.arange(0, 24, 4)
    label_dec_positions = [15, -15]
    for i, ra in enumerate(ra_values):
        dec_points = np.linspace(-90, 90, 100)
        ra_points = np.full_like(dec_points, ra)
        x, y, z = convert_to_cartesian(pd.Series(ra_points), pd.Series(dec_points))
        all_traces.append(go.Scatter3d(
            x=x, y=y, z=z, 
            mode='lines', 
            line=dict(color=GRID_COLOR, width=GRID_WIDTH), 
            opacity=GRID_ALPHA, 
            showlegend=False, 
            hoverinfo='none'
        ))
        
        label_dec = label_dec_positions[i % len(label_dec_positions)]
        x_label, y_label, z_label = convert_to_cartesian(pd.Series([ra]), pd.Series([label_dec]))
        scale = 1.1
        all_traces.append(go.Scatter3d(
            x=[x_label[0] * scale], y=[y_label[0] * scale], z=[z_label[0] * scale],
            mode='text', 
            text=[f'{ra}h'], 
            textfont=dict(size=LABEL_FONT_SIZE, color=LABEL_COLOR), 
            showlegend=False, 
            hoverinfo='none'
        ))

    ra_for_dec_labels = 2
    dec_labels = [-60, -30, 0, 30, 60]
    for dec in dec_labels:
        x_label, y_label, z_label = convert_to_cartesian(pd.Series([ra_for_dec_labels]), pd.Series([dec]))
        scale = 1.1
        all_traces.append(go.Scatter3d(
            x=[x_label[0] * scale], y=[y_label[0] * scale], z=[z_label[0] * scale],
            mode='text', 
            text=[f'{dec}°'], 
            textfont=dict(size=LABEL_FONT_SIZE, color=LABEL_COLOR), 
            showlegend=False, 
            hoverinfo='none'
        ))
        
    return all_traces

# --- 6. Assemble and Show the Plot ---
print("\n--- Step 6: Assembling the 3D Plot ---")
fig = go.Figure()

for trace in constellation_traces:
    fig.add_trace(trace)

grid_and_label_traces = create_sky_grid_with_labels()
for trace in grid_and_label_traces:
    fig.add_trace(trace)

# --- 7. Layout Settings ---
fig.update_layout(
    title=dict(
        text="3D Visualization of Newly Divided Constellations", 
        font=dict(size=20, color="white"), 
        x=0.5, 
        xanchor="center"
    ),
    scene=dict(
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False), 
        zaxis=dict(visible=False), 
        bgcolor="rgb(0, 0, 0)", 
        camera=dict(
            up=dict(x=0, y=1, z=0), 
            center=dict(x=0, y=0, z=0), 
            eye=dict(x=0.7, y=0.5, z=0.5)
        )
    ),
    paper_bgcolor="rgb(0, 0, 0)",
    legend=dict(
        font=dict(color="white"), 
        bgcolor="rgba(0,0,0,0.7)", 
        x=0.01, 
        y=0.99, 
        xanchor="left", 
        yanchor="top",
        title=dict(text="Constellations", font=dict(color="white"))
    ),
    margin=dict(l=20, r=80, t=50, b=20)
)

print("\n--- Step 7: Displaying the Plot ---")
fig.show()