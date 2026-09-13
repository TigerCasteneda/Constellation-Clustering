import pandas as pd
import plotly.graph_objects as go
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import json

#
# --- 1. Configuration ---
# 文件路径
STARS_FILE = "asu.tsv"
FAMOUS_STARS_FILE = "asu_names.tsv"
CONSTELLATIONS_RESULTS_FILE = "constellation_division_results_deg.json"  # 使用度单位的结果文件
CONSTELLATIONS_HULLS_FILE = "new_constellations_final_hulls_deg.tsv"      # 使用度单位的边界文件

# 颜色配置
STAR_COLOR = "rgb(173, 216, 230)"        # 淡蓝色恒星
CORE_STAR_COLOR = "rgb(255, 204, 0)"     # 金黄色核心恒星
CONSTELLATION_BOUNDARY_COLOR = "rgb(255, 255, 0)"  # 黄色边界
CONSTELLATION_LABEL_COLOR = "rgb(255, 255, 255)"  # 白色标签
GRID_COLOR = 'rgb(100, 100, 100)'        # 灰色网格

# 大小和透明度配置
MIN_STAR_SIZE = 1.5
MAX_STAR_SIZE = 8
CORE_STAR_SIZE_MULTIPLIER = 2.0  # 核心恒星大小的放大倍数
GRID_ALPHA = 0.2

# 字体配置
LABEL_FONT_SIZE = 10
TITLE_FONT_SIZE = 20

# 要显示的星座数量
NUM_CONSTELLATIONS_TO_DISPLAY = 86

# --- 2. Helper Functions ---

def convert_to_cartesian(ra_degrees, dec_degrees):
    """将赤经(小时)和赤纬(度)转换为笛卡尔坐标"""
    coords = SkyCoord(ra=ra_degrees.values * u.degree, dec=dec_degrees.values * u.degree, frame="icrs")
    return coords.cartesian.xyz.value

def create_sky_grid_with_labels():
    """创建天球网格和坐标标签"""
    all_traces = []
    
    # 绘制赤纬线 (Declination)
    dec_values = np.arange(-90, 91, 30)
    for dec in dec_values:
        ra_points = np.linspace(0, 24, 200)
        dec_points = np.full_like(ra_points, dec)
        x, y, z = convert_to_cartesian(pd.Series(ra_points), pd.Series(dec_points))
        all_traces.append(go.Scatter3d(
            x=x, y=y, z=z, 
            mode='lines', 
            line=dict(color=GRID_COLOR, width=1), 
            opacity=GRID_ALPHA, 
            showlegend=False, 
            hoverinfo='none'
        ))

    # 绘制赤经线 (Right Ascension) 和标签
    ra_values = np.arange(0, 361, 90)
    label_dec_positions = [15, -15]
    for i, ra in enumerate(ra_values):
        dec_points = np.linspace(-90, 90, 100)
        ra_points = np.full_like(dec_points, ra)
        x, y, z = convert_to_cartesian(pd.Series(ra_points), pd.Series(dec_points))
        all_traces.append(go.Scatter3d(
            x=x, y=y, z=z, 
            mode='lines', 
            line=dict(color=GRID_COLOR, width=1), 
            opacity=GRID_ALPHA, 
            showlegend=False, 
            hoverinfo='none'
        ))
        
        # 添加赤经标签
        label_dec = label_dec_positions[i % len(label_dec_positions)]
        x_label, y_label, z_label = convert_to_cartesian(pd.Series([ra]), pd.Series([label_dec]))
        scale = 1.1
        all_traces.append(go.Scatter3d(
            x=[x_label[0] * scale], y=[y_label[0] * scale], z=[z_label[0] * scale],
            mode='text', 
            text=[f'{ra}°'], 
            textfont=dict(size=LABEL_FONT_SIZE, color=GRID_COLOR), 
            showlegend=False, 
            hoverinfo='none'
        ))

    # 添加赤纬标签
    ra_for_dec_labels = 2
    dec_labels = [-60, -30, 0, 30, 60]
    for dec in dec_labels:
        x_label, y_label, z_label = convert_to_cartesian(pd.Series([ra_for_dec_labels]), pd.Series([dec]))
        scale = 1.1
        all_traces.append(go.Scatter3d(
            x=[x_label[0] * scale], y=[y_label[0] * scale], z=[z_label[0] * scale],
            mode='text', 
            text=[f'{dec}°'], 
            textfont=dict(size=LABEL_FONT_SIZE, color=GRID_COLOR), 
            showlegend=False, 
            hoverinfo='none'
        ))
        
    return all_traces

# --- 3. Data Loading and Preprocessing ---
print("--- Step 1: Loading Data ---")

# 加载恒星数据
try:
    stars_df = pd.read_csv(STARS_FILE, sep=";", header=None, 
                          usecols=[0, 1, 2, 3, 18], 
                          names=["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"])
    
    famous_stars_df = pd.read_csv(FAMOUS_STARS_FILE, sep=";", header=None,
                                 usecols=[0, 1], names=["HR", "Famous_Name"])
    
    # 数据清洗
    for col in ["_RAJ2000", "_DEJ2000", "HR", "Vmag"]:
        stars_df[col] = pd.to_numeric(stars_df[col], errors="coerce")
    stars_df = stars_df.dropna(subset=["_RAJ2000", "_DEJ2000", "HR", "Vmag"])
    
    famous_stars_df["HR"] = pd.to_numeric(famous_stars_df["HR"], errors="coerce")
    stars_df = pd.merge(stars_df, famous_stars_df, on="HR", how="left")
    stars_df["Display_Name"] = stars_df["Famous_Name"].fillna(stars_df["Name"]).fillna(f"HR {stars_df['HR'].astype(int)}")
    
    # 计算笛卡尔坐标
    stars_df["x"], stars_df["y"], stars_df["z"] = convert_to_cartesian(stars_df["_RAJ2000"], stars_df["_DEJ2000"])
    
    print(f"Loaded {len(stars_df)} stars")
    
except FileNotFoundError as e:
    print(f"Error: Star data file not found -> {e.filename}")
    exit()
except Exception as e:
    print(f"Error loading star data: {e}")
    exit()

# 加载星座结果数据
try:
    with open(CONSTELLATIONS_RESULTS_FILE, 'r') as f:
        constellation_results = json.load(f)
    print(f"Loaded {len(constellation_results)} constellations from JSON")
except FileNotFoundError:
    print(f"Error: Constellation results file '{CONSTELLATIONS_RESULTS_FILE}' not found")
    exit()
except Exception as e:
    print(f"Error loading constellation results: {e}")
    exit()

# 加载星座边界数据
try:
    hulls_df = pd.read_csv(CONSTELLATIONS_HULLS_FILE, sep="\t", header=None,
                          names=["ra_deg", "dec_deg", "cst_name"])
    print(f"Loaded {len(hulls_df)} boundary points from TSV")
except FileNotFoundError:
    print(f"Error: Constellation hulls file '{CONSTELLATIONS_HULLS_FILE}' not found")
    exit()
except Exception as e:
    print(f"Error loading constellation hulls: {e}")
    exit()

# --- 4. Prepare Visualization Data ---
print("\n--- Step 2: Preparing Visualization Data ---")

# 获取所有星座名称并选择前5个
all_constellation_names = hulls_df["cst_name"].unique()
selected_constellation_names = all_constellation_names[:NUM_CONSTELLATIONS_TO_DISPLAY]
print(f"Selected constellations to display: {list(selected_constellation_names)}")

# 准备星座边界轨迹、恒星和核心恒星
constellation_traces = []
selected_hr_numbers = set()
core_hr_numbers = set() # 用于存储所有核心恒星的HR编号

for cst_name in selected_constellation_names:
    # 获取该星座的边界点
    cst_hull_df = hulls_df[hulls_df["cst_name"] == cst_name].copy()
    
    # 转换为笛卡尔坐标
    x, y, z = convert_to_cartesian(cst_hull_df["ra_deg"], cst_hull_df["dec_deg"])
    
    # 添加边界线（闭合）
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    z = np.append(z, z[0])
    
    boundary_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=CONSTELLATION_BOUNDARY_COLOR, width=2),
        name=cst_name,
        hoverinfo='text',
        text=f"{cst_name} Boundary",
        showlegend=True
    )
    constellation_traces.append(boundary_trace)
    
    # 从JSON结果中获取对应星座的恒星HR编号
    cluster_id = int(cst_name.replace("NewCst_", ""))
    json_cst_name = f"Constellation_{cluster_id}"
    
    if json_cst_name in constellation_results:
        cst_data = constellation_results[json_cst_name]
        cst_hr_list = cst_data["all_star_HR_numbers"]
        selected_hr_numbers.update(cst_hr_list)
        
        # 收集核心恒星的HR编号
        core_hr_list = cst_data["core_star_HR_numbers"]
        core_hr_numbers.update(core_hr_list)
        
        # 添加星座中心标签
        # 注意：JSON文件中的中心坐标是度，需要转换为小时
        center_ra_deg = cst_data["center_right_ascension_deg"]
        center_dec_deg = cst_data["center_declination_deg"]
        
        label_x, label_y, label_z = convert_to_cartesian(pd.Series([center_ra_deg]), pd.Series([center_dec_deg]))
        
        # 添加标签
        scale = 1.15
        label_trace = go.Scatter3d(
            x=[label_x[0] * scale], y=[label_y[0] * scale], z=[label_z[0] * scale],
            mode="text",
            text=[cst_name.replace("NewCst_", "Cst. ")],
            textfont=dict(size=LABEL_FONT_SIZE+2, color=CONSTELLATION_LABEL_COLOR, weight="bold"),
            showlegend=False,
            hoverinfo='none'
        )
        constellation_traces.append(label_trace)

# 准备选中的恒星数据
if selected_hr_numbers:
    selected_stars_df = stars_df[stars_df["HR"].isin(selected_hr_numbers)].copy()
    
    # --- 核心修改点 ---
    # 根据星等计算基础大小
    selected_stars_df["base_size"] = np.clip(8 - selected_stars_df["Vmag"], MIN_STAR_SIZE, MAX_STAR_SIZE)
    
    # 创建一个掩码，标识哪些是核心恒星
    is_core_star = selected_stars_df["HR"].isin(core_hr_numbers)
    
    # 为核心恒星应用更大的尺寸，并设置不同的颜色
    selected_stars_df["final_size"] = np.where(is_core_star, 
                                              selected_stars_df["base_size"] * CORE_STAR_SIZE_MULTIPLIER, 
                                              selected_stars_df["base_size"])
    selected_stars_df["final_color"] = np.where(is_core_star, CORE_STAR_COLOR, STAR_COLOR)
    
    # 根据星等计算透明度
    opacities = np.clip(1.2 - (selected_stars_df["Vmag"] / 10), 0.3, 0.9)
    
    # 为了在Plotly中使用不同颜色，我们需要分别创建轨迹或使用颜色列表
    # 这里我们创建两个轨迹：一个用于普通恒星，一个用于核心恒星
    
    # 1. 普通恒星轨迹
    normal_stars_df = selected_stars_df[~is_core_star]
    if not normal_stars_df.empty:
        normal_colors = [f'rgba({int(c.split(",")[0].strip()[4:])}, {int(c.split(",")[1].strip())}, {int(c.split(",")[2].strip()[:-1])}, {o:.2f})' 
                         for c, o in zip(normal_stars_df["final_color"], opacities[~is_core_star])]
        
        normal_stars_trace = go.Scatter3d(
            x=normal_stars_df["x"], y=normal_stars_df["y"], z=normal_stars_df["z"],
            mode="markers",
            marker=dict(
                size=normal_stars_df["final_size"],
                color=normal_colors,
            ),
            name="Stars",
            hoverinfo='text',
            text=normal_stars_df["Display_Name"] + "<br>HR: " + normal_stars_df["HR"].astype(str) + 
                 "<br>Vmag: " + normal_stars_df["Vmag"].round(2).astype(str),
            showlegend=True
        )
        constellation_traces.insert(0, normal_stars_trace)

    # 2. 核心恒星轨迹
    core_stars_df = selected_stars_df[is_core_star]
    if not core_stars_df.empty:
        core_colors = [f'rgba({int(c.split(",")[0].strip()[4:])}, {int(c.split(",")[1].strip())}, {int(c.split(",")[2].strip()[:-1])}, {o:.2f})' 
                       for c, o in zip(core_stars_df["final_color"], opacities[is_core_star])]
        
        core_stars_trace = go.Scatter3d(
            x=core_stars_df["x"], y=core_stars_df["y"], z=core_stars_df["z"],
            mode="markers",
            marker=dict(
                size=core_stars_df["final_size"],
                color=core_colors,
                symbol='diamond' # 可选：使用不同的形状来区分
            ),
            name="Core Stars", # 在图例中显示
            hoverinfo='text',
            text=core_stars_df["Display_Name"] + "<br>HR: " + core_stars_df["HR"].astype(str) + 
                 "<br>Vmag: " + core_stars_df["Vmag"].round(2).astype(str) + "<br>(Core Star)",
            showlegend=True
        )
        constellation_traces.insert(0, core_stars_trace) # 插入到前面，使其在图例中靠前

    # 单独创建颜色条轨迹 (只基于普通恒星的星等)
    if not normal_stars_df.empty:
        colorbar_trace = go.Scatter3d(
            x=[None], y=[None], z=[None], # 不显示任何点
            mode='markers',
            marker=dict(
                size=[0],
                color=normal_stars_df["Vmag"],
                colorscale='Viridis',
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
            showlegend=False
        )
        constellation_traces.append(colorbar_trace)

# --- 5. Create and Show the Plot ---
print("\n--- Step 3: Creating 3D Visualization ---")

fig = go.Figure()

# 添加所有轨迹
for trace in constellation_traces:
    fig.add_trace(trace)

# 添加天球网格
grid_traces = create_sky_grid_with_labels()
for trace in grid_traces:
    fig.add_trace(trace)

# 布局设置
fig.update_layout(
    title=dict(
        text=f"3D Visualization of First {NUM_CONSTELLATIONS_TO_DISPLAY} New Constellations", 
        font=dict(size=TITLE_FONT_SIZE, color="white"), 
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
        title=dict(text="Legend", font=dict(color="white"))
    ),
    margin=dict(l=20, r=80, t=50, b=20)
)

print("\n--- Step 4: Displaying the Plot ---")
fig.show()

print("\n--- Visualization Complete ---")