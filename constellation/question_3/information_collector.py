import pandas as pd
import json

def extract_hull_vertices_to_csv(json_file, output_csv):
    """
    从星座划分结果JSON文件中提取凸包顶点坐标并保存为CSV
    
    参数:
    json_file: 星座划分结果JSON文件路径
    output_csv: 输出CSV文件路径
    """
    # 读取JSON文件
    with open(json_file, 'r') as f:
        constellation_data = json.load(f)
    
    # 准备存储凸包顶点的数据列表
    hull_vertices = []
    
    # 读取边界点数据（从TSV文件，因为JSON中没有存储具体坐标）
    # 这里假设边界点TSV文件已经存在
    boundary_df = pd.read_csv('new_constellations_final_hulls_deg.tsv', 
                             sep='\t', 
                             header=None, 
                             names=['ra_deg', 'dec_deg', 'cst_name'])
    
    # 整理数据
    for idx, row in boundary_df.iterrows():
        # 提取星座ID
        cst_id = int(row['cst_name'].replace('NewCst_', ''))
        
        hull_vertices.append({
            'constellation_id': cst_id,
            'constellation_name': row['cst_name'],
            'ra_deg': row['ra_deg'],
            'dec_deg': row['dec_deg'],
            'vertex_index': idx  # 顶点在边界中的索引
        })
    
    # 转换为DataFrame并保存为CSV
    df = pd.DataFrame(hull_vertices)
    df.to_csv(output_csv, index=False)
    print(f"已成功将凸包顶点坐标保存到 {output_csv}")

# 在主函数的最后调用此函数（在保存结果之后）
# 在main()函数的"--- Step 6: Save Results ---"部分添加：
# extract_hull_vertices_to_csv(RESULTS_JSON_FILE, 'constellation_hull_vertices.csv')

# 如果要单独运行此功能，可以使用以下代码：
if __name__ == "__main__":
    extract_hull_vertices_to_csv('constellation_division_results_deg.json', 
                                'constellation_hull_vertices.csv')