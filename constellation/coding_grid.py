import pandas as pd
import plotly.graph_objects as go
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# --- 1. Configuration ---
FILE_CONFIGS = {
    "stars": {"path": "asu.tsv", "column_spec": {'usecols': [0, 1, 2, 3, 18], 'names': ["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"]}, "sep": ";"},
    "famous_stars": {"path": "asu_names.tsv", "column_spec": {'usecols': [0, 1], 'names': ["HR", "Famous_Name"]}, "sep": ";"},
    "constellations": {"path": "asu_constellations.tsv", "column_spec": {'usecols': [0, 1, 4], 'names': ["_RAJ2000", "_DEJ2000", "cst"]}, "sep": ";"}
}

TARGET_CONSTELLATION = "UMA"
# 北斗七星的HR编号列表 (用于筛选)
BIG_DIPPER_HR = [4295, 4301, 4554, 4660, 4905, 5054, 5191]
# 北斗七星正确的连线顺序 (从斗柄末端到斗口)
# 摇光(Alkaid) -> 开阳(Mizar) -> 玉衡(Alioth) -> 天权(Megrez) -> 天玑(Phecda) -> 天璇(Merak) -> 天枢(Dubhe)
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

# --- 3. Coordinate Conversion (修正版) ---
def convert_to_cartesian(ra_series, dec_series):
    """
    将赤经(RA)和赤纬(Dec)转换为笛卡尔坐标
    假设输入的单位是度(degrees)
    """
    try:
        # 创建SkyCoord对象，假设输入是度数
        coords = SkyCoord(ra=ra_series.values * u.degree, dec=dec_series.values * u.degree, frame='icrs')
        xyz = coords.cartesian.xyz.value
        return xyz[0], xyz[1], xyz[2]
    except Exception as e:
        print(f"Error in coordinate conversion: {e}")
        # 如果转换失败，返回空数组
        return np.array([]), np.array([]), np.array([])

# 转换所有星星的坐标
stars_df["x"], stars_df["y"], stars_df["z"] = convert_to_cartesian(stars_df["_RAJ2000"], stars_df["_DEJ2000"])

# 筛选并按正确顺序排序北斗七星数据
big_dipper_df = stars_df[stars_df["HR"].isin(BIG_DIPPER_HR)]
if not big_dipper_df.empty:
    # 使用 .set_index 和 .loc 来按照指定的 HR 顺序重新排列数据
    big_dipper_df = big_dipper_df.set_index("HR").loc[BIG_DIPPER_ORDER].reset_index()
    bd_x, bd_y, bd_z = big_dipper_df["x"].values, big_dipper_df["y"].values, big_dipper_df["z"].values
else:
    bd_x, bd_y, bd_z = [], [], []
    print("Warning: Big Dipper stars not found in dataset")

# 北极星（修正判断逻辑：将单个数值包装为列表）
polaris_df = stars_df[stars_df["HR"] == POLARIS_HR]
if not polaris_df.empty:
    p_x, p_y, p_z = [polaris_df["x"].values[0]], [polaris_df["y"].values[0]], [polaris_df["z"].values[0]]
else:
    p_x, p_y, p_z = [], [], []
    print("Warning: Polaris not found in dataset")

# 星座边界
if not target_constellation_df.empty:
    const_x, const_y, const_z = convert_to_cartesian(target_constellation_df["_RAJ2000"], target_constellation_df["_DEJ2000"])
else:
    const_x, const_y, const_z = [], [], []

const_x = np.append(const_x,const_x[0])
const_y = np.append(const_y,const_y[0])
const_z = np.append(const_z,const_z[0])

# --- 4. Grid and Label Generation (修正版) ---
def create_sky_grid_with_labels():
    all_traces = []
    
    # 生成赤纬圈 (Dec circles)
    dec_values = np.arange(-60, 61, 30)  # 从-60°到+60°，每30°一条
    for dec in dec_values:
        ra_points = np.linspace(0, 360, 100)  # 赤经从0°到360°
        dec_points = np.full_like(ra_points, dec)
        
        # 转换为笛卡尔坐标
        ra_series = pd.Series(ra_points)
        dec_series = pd.Series(dec_points)
        x, y, z = convert_to_cartesian(ra_series, dec_series)
        
        all_traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode='lines', 
            line=dict(color=GRID_COLOR, width=GRID_WIDTH), 
            opacity=GRID_ALPHA, showlegend=False, hoverinfo='none'
        ))
    
    # 生成赤经线 (RA lines)
    ra_values = np.arange(0, 360, 60)  # 每60°一条赤经线 (相当于4小时)
    for ra in ra_values:
        dec_points = np.linspace(-60, 60, 50)  # 赤纬从-60°到+60°
        ra_points = np.full_like(dec_points, ra)
        
        # 转换为笛卡尔坐标
        ra_series = pd.Series(ra_points)
        dec_series = pd.Series(dec_points)
        x, y, z = convert_to_cartesian(ra_series, dec_series)
        
        all_traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode='lines', 
            line=dict(color=GRID_COLOR, width=GRID_WIDTH), 
            opacity=GRID_ALPHA, showlegend=False, hoverinfo='none'
        ))
        
        # 添加赤经标签 (转换为小时表示)
        ra_hours = ra / 15  # 将度转换为小时
        label_dec = 65  # 标签位置
        x_label, y_label, z_label = convert_to_cartesian(pd.Series([ra]), pd.Series([label_dec]))
        if len(x_label) > 0:
            scale = 1.15
            all_traces.append(go.Scatter3d(
                x=[x_label[0] * scale], y=[y_label[0] * scale], z=[z_label[0] * scale], 
                mode='text', text=[f'{ra_hours:.0f}h'], 
                textfont=dict(size=LABEL_FONT_SIZE, color=LABEL_COLOR), 
                showlegend=False, hoverinfo='none'
            ))
    
    # 添加赤纬标签
    dec_labels = [-60, -30, 0, 30, 60]
    ra_for_labels = 0  # 在0°赤经处添加标签
    for dec in dec_labels:
        x_label, y_label, z_label = convert_to_cartesian(pd.Series([ra_for_labels]), pd.Series([dec]))
        if len(x_label) > 0:
            scale = 1.15
            all_traces.append(go.Scatter3d(
                x=[x_label[0] * scale], y=[y_label[0] * scale], z=[z_label[0] * scale], 
                mode='text', text=[f'{dec}°'], 
                textfont=dict(size=LABEL_FONT_SIZE, color=LABEL_COLOR), 
                showlegend=False, hoverinfo='none'
            ))
    
    return all_traces

# --- 5. Visualization Plotting ---
fig = go.Figure()
print("Plotting background stars and other elements...")

# 绘制背景星星
fig.add_trace(go.Scatter3d(
    x=stars_df["x"], y=stars_df["y"], z=stars_df["z"], mode="markers",
    marker=dict(
        size=np.clip(6 - stars_df["Vmag"], 2, 10),  # 星等越小，星星越大越亮
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

# 绘制星座边界
if len(const_x) > 0:
    fig.add_trace(go.Scatter3d(
        x=const_x, y=const_y, z=const_z, 
        mode="lines", 
        line=dict(color="rgba(255, 255, 255, 0.7)", width=2), 
        name=f"{TARGET_CONSTELLATION} Boundary"
    ))

# 绘制著名星星
famous_stars_visual_df = stars_df[stars_df["Famous_Name"].notna()]
if not famous_stars_visual_df.empty:
    fig.add_trace(go.Scatter3d(
        x=famous_stars_visual_df["x"], y=famous_stars_visual_df["y"], z=famous_stars_visual_df["z"], 
        mode="markers", 
        marker=dict(size=5, color="red", symbol="square", opacity=0.9), 
        text=famous_stars_visual_df["Famous_Name"] + " (Famous Star)", 
        hoverinfo="text", 
        name="Famous Stars"
    ))

# 绘制北斗七星连线 (按正确顺序)
if len(bd_x) > 0:
    fig.add_trace(go.Scatter3d(
        x=bd_x, y=bd_y, z=bd_z, 
        mode="lines", 
        line=dict(color="rgb(255, 255, 0)", width=4), 
        name="Big Dipper Connecting Line"
    ))
    fig.add_trace(go.Scatter3d(
        x=bd_x, y=bd_y, z=bd_z, 
        mode="markers+text", 
        marker=dict(size=5, color="rgb(255, 255, 0)", symbol="diamond"), 
        text=big_dipper_df["Display_Name"], 
        textfont=dict(size=5, color="rgb(255, 255, 0)"), 
        textposition="top center", 
        hoverinfo="text", 
        name="Big Dipper Stars"
    ))

# # 绘制北极星（使用列表包装单个数值，确保len()判断有效）
# if len(p_x) > 0:
#     fig.add_trace(go.Scatter3d(
#         x=p_x, y=p_y, z=p_z, 
#         mode="markers+text", 
#         marker=dict(size=7, color="rgb(0, 255, 255)", symbol="circle"), 
#         text=["Polaris (North Star)"], 
#         textfont=dict(size=7, color="rgb(0, 255, 255)", weight="bold"), 
#         textposition="top center", 
#         hoverinfo="text", 
#         name="Polaris"
#     ))

# 添加天球网格和标签
print("Generating and plotting sky grid and coordinate labels...")
grid_and_label_traces = create_sky_grid_with_labels()
for trace in grid_and_label_traces:
    fig.add_trace(trace)

# --- 6. Layout Settings ---
fig.update_layout(
    title=dict(
        text="Celestial Sphere Visualization: Finding Polaris with the Big Dipper", 
        font=dict(size=20, color="white"), 
        x=0.5, xanchor="center"
    ),
    scene=dict(
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False), 
        zaxis=dict(visible=False), 
        bgcolor="rgb(0, 0, 0)", 
        camera=dict(
            up=dict(x=0, y=1, z=0), 
            center=dict(x=0, y=0, z=0), 
            eye=dict(x=1.5, y=1.0, z=1.0)  # 调整视角以获得更好的观察角度
        )
    ),
    paper_bgcolor="rgb(0, 0, 0)",
    legend=dict(
        font=dict(color="white"), 
        bgcolor="rgba(0,0,0,0.5)", 
        x=0.01, y=0.99, xanchor="left", yanchor="top"
    ),
    margin=dict(l=50, r=50, t=50, b=50)
)

fig.show()
