import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. 配置区域 ---
INPUT_DATA_FILE = "constellation_hull_vertices.csv"  # 你的星座数据CSV文件
OUTPUT_IMAGE_DIR = "constellation_maps"     # 输出星图的文件夹
CANVAS_SIZE = (10, 10)                      # 固定画布大小 (宽, 高) 英寸
DPI = 300                                   # 图片分辨率
BACKGROUND_COLOR = "black"                  # 画布背景色（星空感）
STAR_COLOR = "white"                        # 星星颜色
STAR_SIZE = 50                              # 固定的星点大小
LINE_COLOR = "white"                        # 连接线颜色
LINE_WIDTH = 1.5                            # 连接线宽度

# --- 2. 辅助函数 ---

def create_star_map(constellation_name, x_coords, y_coords, output_path):
    """
    为单个星座创建并保存一张星图，并按vertex_index顺序连接各点。
    """
    # 创建一个新的图形
    plt.figure(figsize=CANVAS_SIZE)
    
    # 设置背景色
    ax = plt.gca()
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # 绘制散点图（星图）
    scatter = plt.scatter(
        x=x_coords, 
        y=y_coords, 
        s=STAR_SIZE,
        c=STAR_COLOR, 
        alpha=1,
        edgecolors='none',
        zorder=2  # 确保星星在连线之上
    )
    
    # 绘制连接线
    if len(x_coords) > 1:
        plt.plot(
            x_coords, 
            y_coords, 
            color=LINE_COLOR, 
            linewidth=LINE_WIDTH, 
            linestyle='-',
            zorder=1  # 确保连线在星星之下
        )
    
    # 设置标题
    plt.title(
        constellation_name, 
        color=STAR_COLOR, 
        fontsize=20, 
        pad=20
    )
    
    # 隐藏坐标轴
    plt.axis('off')
    
    # 优化坐标轴范围，留出适当边距
    x_margin = (x_coords.max() - x_coords.min()) * 0.1
    y_margin = (y_coords.max() - y_coords.min()) * 0.1
    if x_margin == 0: x_margin = 1.0
    if y_margin == 0: y_margin = 1.0
    
    plt.xlim(x_coords.min() - x_margin, x_coords.max() + x_margin)
    plt.ylim(y_coords.min() - y_margin, y_coords.max() + y_margin)
    
    # 保存图片
    plt.tight_layout(pad=0)
    plt.savefig(
        output_path, 
        dpi=DPI, 
        bbox_inches='tight', 
        pad_inches=0,
        facecolor=BACKGROUND_COLOR
    )
    plt.close()

# --- 3. 主执行函数 ---

def main():
    """
    主函数，读取CSV数据并为每个星座生成带连接线的星图。
    """
    if not os.path.exists(INPUT_DATA_FILE):
        print(f"错误: 输入文件 '{INPUT_DATA_FILE}' 不存在。")
        return

    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)
        print(f"创建输出文件夹: {OUTPUT_IMAGE_DIR}")

    print(f"正在读取数据文件: {INPUT_DATA_FILE}")
    try:
        df = pd.read_csv(INPUT_DATA_FILE)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    required_columns = ["constellation_id", "constellation_name", "ra_deg", "dec_deg", "vertex_index"]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"错误: 数据文件必须包含以下必需列: {', '.join(required_columns)}")
        print(f"缺少的列: {', '.join(missing)}")
        return

    unique_constellations = df[["constellation_id", "constellation_name"]].drop_duplicates().sort_values(by="constellation_id")
    print(f"共找到 {len(unique_constellations)} 个独特的星座，开始绘制...")

    for _, row in unique_constellations.iterrows():
        cst_id = row["constellation_id"]
        cst_name = row["constellation_name"]
        
        # 筛选当前星座的数据并按 vertex_index 排序
        constellation_data = df[df["constellation_id"] == cst_id].sort_values(by="vertex_index")
        
        x = constellation_data["ra_deg"]
        y = constellation_data["dec_deg"]
        
        output_file = os.path.join(OUTPUT_IMAGE_DIR, f"{cst_id}_{cst_name}.png")
        
        try:
            create_star_map(cst_name, x, y, output_file)
            print(f"成功生成: {os.path.basename(output_file)}")
        except Exception as e:
            print(f"生成星座 '{cst_name}' (ID: {cst_id}) 的星图时出错: {e}")

    print("\n所有星图绘制完成！")

# --- 4. 脚本入口 ---
if __name__ == "__main__":
    main()