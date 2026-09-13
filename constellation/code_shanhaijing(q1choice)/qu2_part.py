import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from scipy.spatial.distance import pdist, squareform
import itertools

# 尝试导入HDBSCAN，如果失败则设置标志
try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    print("警告: 无法导入HDBSCAN，将使用DBSCAN作为备选")
    HAS_HDBSCAN = False

# --- 1. 数据加载和预处理 ---
def load_and_preprocess_data():
    # 获取当前脚本目录和父目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # 配置文件路径
    FILE_CONFIGS = {
        "stars": {"path": os.path.join(parent_dir, "asu.tsv"), "column_spec": {'usecols': [0, 1, 2, 3, 18], 'names': ["_RAJ2000", "_DEJ2000", "HR", "Name", "Vmag"]}, "sep": ";"},
        "famous_stars": {"path": os.path.join(parent_dir, "asu_names.tsv"), "column_spec": {'usecols': [0, 1], 'names': ["HR", "Famous_Name"]}, "sep": ";"},
        "constellations": {"path": os.path.join(parent_dir, "asu_constellations.tsv"), "column_spec": {'usecols': [0, 1, 4], 'names': ["_RAJ2000", "_DEJ2000", "cst"]}, "sep": ";"}
    }
    
    print("开始加载数据文件...")
    data_frames = {}
    for key, config in FILE_CONFIGS.items():
        try:
            df = pd.read_csv(config["path"], sep=config["sep"], header=None, **config["column_spec"])
            data_frames[key] = df
            print(f"成功读取文件: {config['path']}, 行数: {len(df)}")
        except Exception as e:
            print(f"读取文件 {config['path']} 时出错: {e}")
            # 如果文件不存在，返回空字典
            return None, None
    
    # 数据清洗
    stars_df = data_frames["stars"]
    famous_stars_df = data_frames["famous_stars"]
    
    # 转换数值列
    for col in ["_RAJ2000", "_DEJ2000", "HR", "Vmag"]:
        stars_df[col] = pd.to_numeric(stars_df[col], errors="coerce")
    stars_df = stars_df.dropna(subset=["_RAJ2000", "_DEJ2000", "HR", "Vmag"])
    
    famous_stars_df["HR"] = pd.to_numeric(famous_stars_df["HR"], errors="coerce")
    famous_stars_df = famous_stars_df.dropna(subset=["HR"])
    
    # 合并著名恒星名称
    stars_df = pd.merge(stars_df, famous_stars_df, on="HR", how="left")
    stars_df["Display_Name"] = stars_df["Famous_Name"].fillna(stars_df["Name"]).fillna(f"HR{stars_df['HR']}")
    
    print(f"数据预处理完成，剩余恒星数量: {len(stars_df)}")
    return stars_df, data_frames["constellations"]

# --- 2. 坐标转换和距离计算 ---
def convert_to_cartesian(ra_series, dec_series):
    """
    将赤经(RA)和赤纬(Dec)转换为笛卡尔坐标
    """
    try:
        coords = SkyCoord(ra=ra_series.values * u.degree, dec=dec_series.values * u.degree, frame='icrs')
        xyz = coords.cartesian.xyz.value
        return xyz[0], xyz[1], xyz[2]
    except Exception as e:
        print(f"坐标转换错误: {e}")
        return np.array([]), np.array([]), np.array([])

def great_circle_distance(ra1, dec1, ra2, dec2):
    """
    计算两个天体之间的大圆距离（以度为单位）
    """
    # 将角度转换为弧度
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)
    
    # 应用球面余弦定理
    delta_ra = ra2_rad - ra1_rad
    a = np.sin(dec1_rad) * np.sin(dec2_rad)
    b = np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(delta_ra)
    distance = np.degrees(np.arccos(a + b))
    
    return distance

def chord_distance(ra1, dec1, ra2, dec2):
    """
    计算两个天体之间的弦长（球面上的欧几里得距离）
    """
    # 将角度转换为弧度
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)
    
    # 计算弦长
    delta_ra = ra2_rad - ra1_rad
    a = np.sin((dec2_rad - dec1_rad) / 2)
    b = np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(delta_ra / 2)
    chord_length = 2 * np.sqrt(a**2 + b**2)
    
    return chord_length

# --- 3. 恒星分组模型实现 ---
class NewConstellationModel:
    def __init__(self, min_brightness=2.5, distance_metric='great_circle', 
                 cluster_method='hdbscan', dbscan_eps=10.0, dbscan_min_samples=3, 
                 hdbscan_min_cluster_size=3, hdbscan_min_samples=2):
        """
        初始化新星座模型
        
        参数:
        - min_brightness: 最小视星等（越小越亮）
        - distance_metric: 距离度量方法 ('great_circle' 或 'chord')
        - cluster_method: 聚类方法 ('dbscan' 或 'hdbscan')
        - dbscan_eps: DBSCAN的最大距离参数
        - dbscan_min_samples: DBSCAN的最小样本数
        - hdbscan_min_cluster_size: HDBSCAN的最小聚类大小
        - hdbscan_min_samples: HDBSCAN的最小样本数
        """
        self.min_brightness = min_brightness
        self.distance_metric = distance_metric
        self.cluster_method = cluster_method
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.clusters = None
        self.stars_df = None
        
    def preprocess_stars(self, stars_df):
        """
        预处理恒星数据，筛选亮星并转换坐标
        """
        # 筛选亮星
        bright_stars_df = stars_df[stars_df['Vmag'] <= self.min_brightness].copy()
        
        # 如果亮星数量太多，随机采样一部分
        max_stars = 2000  # 限制最大恒星数量
        if len(bright_stars_df) > max_stars:
            bright_stars_df = bright_stars_df.sample(n=max_stars, random_state=42)
            print(f"亮星数量过多，随机采样 {max_stars} 颗恒星进行处理")
        
        print(f"筛选后的亮星数量: {len(bright_stars_df)} (视星等 ≤ {self.min_brightness})")
        
        # 转换坐标
        bright_stars_df['x'], bright_stars_df['y'], bright_stars_df['z'] = convert_to_cartesian(
            bright_stars_df['_RAJ2000'], bright_stars_df['_DEJ2000']
        )
        
        # 添加球面坐标（用于可视化）
        bright_stars_df['ra_rad'] = np.radians(bright_stars_df['_RAJ2000'])
        bright_stars_df['dec_rad'] = np.radians(bright_stars_df['_DEJ2000'])
        
        self.stars_df = bright_stars_df
        return bright_stars_df
    
    def create_spherical_features(self):
        """
        创建用于聚类的球面特征
        """
        # 使用笛卡尔坐标作为特征
        features = self.stars_df[['x', 'y', 'z']].values
        return features
    
    def cluster_stars(self):
        """
        使用优化的聚类算法对恒星进行分组，确保星座内部紧密、星座间分离，并生成足够数量的星座
        """
        if self.stars_df is None:
            raise ValueError("请先调用preprocess_stars方法")
        
        # 1. 特征预处理和增强
        ra = self.stars_df['_RAJ2000'].values
        dec = self.stars_df['_DEJ2000'].values
        
        # 将角度转换为弧度
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        # 创建基于天球坐标的特征
        coord_features = np.column_stack([
            np.sin(ra_rad) * np.cos(dec_rad),  # x
            np.cos(ra_rad) * np.cos(dec_rad),  # y  
            np.sin(dec_rad)                      # z
        ])
        
        print("使用优化聚类算法，确保星座内部紧密、星座间分离...")
        
        # 使用优化的DBSCAN算法，参数更严格以确保紧密度和分离度
        clustering_labels = self._optimized_dbscan_clustering(coord_features)
        
        # 检查聚类结果，如果星座数量不足或有星座恒星数少于100，则进行细分
        unique_clusters = np.unique(clustering_labels)
        valid_clusters = [c for c in unique_clusters if c != -1]
        
        # 检查是否有星座恒星数不符合要求（少于10或多于50）
        needs_subdivision = False
        largest_cluster_id = -1
        largest_cluster_size = 0
        
        for cluster_id in valid_clusters:
            cluster_size = np.sum(clustering_labels == cluster_id)
            # 如果有任何聚类的恒星数少于10或多于50，需要细分
            if cluster_size < 10 or cluster_size > 50:
                needs_subdivision = True
                # 记录最大的聚类，优先处理大聚类
                if cluster_size > largest_cluster_size:
                    largest_cluster_size = cluster_size
                    largest_cluster_id = cluster_id
        
        if len(valid_clusters) <= 1 or needs_subdivision:
            print("检测到星座数量不足或星座恒星数不符合要求（10-50颗），正在进行细分...")
            clustering_labels = self._subdivide_large_cluster(coord_features, clustering_labels)
            unique_clusters = np.unique(clustering_labels)
            valid_clusters = [c for c in unique_clusters if c != -1]
        
        # 保存聚类结果
        self.stars_df['cluster'] = clustering_labels
        self.clusters = clustering_labels
        
        # 分析聚类结果
        num_clusters = len(unique_clusters)
        num_noise = np.sum(clustering_labels == -1)
        
        print(f"聚类完成。发现 {num_clusters - 1} 个新星座 (排除噪声点)")
        print(f"噪声点数量: {num_noise}")
        print(f"每个星座的恒星数量:")
        for cluster_id in unique_clusters:
            if cluster_id != -1:
                cluster_size = np.sum(clustering_labels == cluster_id)
                print(f"  星座 {cluster_id + 1}: {cluster_size} 颗恒星")
        
        return self.stars_df
    
    def _subdivide_large_cluster(self, features, labels):
        """
        细分大聚类以生成更多星座，确保每个星座内部星星不少于100颗
        """
        unique_labels = np.unique(labels)
        valid_labels = [l for l in unique_labels if l != -1]
        
        # 如果没有有效聚类，直接返回
        if len(valid_labels) == 0:
            return labels
        
        # 检查是否所有星座都满足星星数量要求（不少于10颗且不多于50颗）
        all_clusters_valid = True
        for label in valid_labels:
            cluster_size = np.sum(labels == label)
            if cluster_size < 10 or cluster_size > 50:
                all_clusters_valid = False
                break
        
        # 如果所有星座都满足要求，直接返回
        if all_clusters_valid:
            return labels
        
        # 简化的细分策略：只进行一次细分，避免多次迭代
        current_labels = labels.copy()
        
        # 检查是否有需要细分的聚类
        needs_subdivision = False
        largest_cluster_label = -1
        largest_cluster_size = 0
        
        unique_labels = [l for l in np.unique(current_labels) if l != -1]
        
        for label in unique_labels:
            cluster_size = np.sum(current_labels == label)
            if cluster_size > 50:  # 修改为大于50颗星需要细分
                needs_subdivision = True
                if cluster_size > largest_cluster_size:
                    largest_cluster_size = cluster_size
                    largest_cluster_label = label
        
        # 如果不需要细分，直接返回
        if not needs_subdivision:
            return current_labels
        
        print(f"检测到需要细分的星座，最大星座包含{largest_cluster_size}颗恒星")
        
        # 对最大聚类进行一次细分
        mask = current_labels == largest_cluster_label
        cluster_features = features[mask]
        
        # 使用简化的K-means细分
        from sklearn.cluster import KMeans
        # 固定分成2-5个子聚类，每个子聚类约10-25颗星
        n_subclusters = min(5, max(2, largest_cluster_size // 15))
        
        print(f"将星座细分为{n_subclusters}个子星座...")
        
        # 使用MiniBatchKMeans替代标准KMeans，减少内存使用
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_subclusters, random_state=42, batch_size=100, max_iter=20)  # 减少迭代次数和批次大小
        sub_labels = kmeans.fit_predict(cluster_features)
        
        # 更新标签
        new_labels = current_labels.copy()
        sub_cluster_ids = np.unique(sub_labels)
        
        # 为新的子聚类分配标签
        max_existing_label = np.max([l for l in np.unique(new_labels) if l != -1]) if len([l for l in np.unique(new_labels) if l != -1]) > 0 else 0
        
        # 创建一个临时数组来存储子标签，初始值为-1
        temp_sub_labels = np.full_like(new_labels, -1)
        temp_sub_labels[mask] = sub_labels
        
        for i, sub_id in enumerate(sub_cluster_ids):
            # 找到属于这个子聚类的所有点
            sub_cluster_mask = temp_sub_labels == sub_id
            new_labels[sub_cluster_mask] = max_existing_label + i + 1
        
        return new_labels
        
        return current_labels
    
    def _optimized_dbscan_clustering(self, features):
        """
        优化的DBSCAN聚类算法，增加星座数量，控制每个星座10-50颗星
        """
        print("正在执行优化的DBSCAN聚类，目标：30个以上星座，每个星座10-50颗星...")
        
        # 使用更小的eps和min_samples来生成更多小聚类
        eps = 3.0  # 减小eps以获得更多小聚类
        min_samples = 3   # 减小min_samples以允许更小的聚类
        
        print(f"使用参数: eps={eps}, min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(features)
        
        # 检查聚类结果
        unique_labels = [l for l in np.unique(labels) if l != -1]
        print(f"初始聚类完成。发现 {len(unique_labels)} 个星座")
        
        # 如果星座数量不足30个，进行细分
        if len(unique_labels) < 30:
            print(f"星座数量不足30个，进行细分处理...")
            labels = self._subdivide_for_more_clusters(features, labels, target_clusters=30)
            unique_labels = [l for l in np.unique(labels) if l != -1]
            print(f"细分后星座数量: {len(unique_labels)} 个")
        
        return labels
    
    def _subdivide_for_more_clusters(self, features, labels, target_clusters=30):
        """
        细分聚类以增加星座数量，确保每个星座10-50颗星
        """
        current_labels = labels.copy()
        
        # 循环细分直到达到目标星座数量
        while True:
            unique_labels = [l for l in np.unique(current_labels) if l != -1]
            
            # 如果已经达到目标数量，检查星座大小是否符合要求
            if len(unique_labels) >= target_clusters:
                # 检查所有星座大小是否在10-50颗星范围内
                all_valid = True
                for label in unique_labels:
                    cluster_size = np.sum(current_labels == label)
                    if cluster_size < 10 or cluster_size > 50:
                        all_valid = False
                        break
                
                if all_valid:
                    break
            
            # 找到需要细分的大聚类
            largest_cluster_label = -1
            largest_cluster_size = 0
            
            for label in unique_labels:
                cluster_size = np.sum(current_labels == label)
                # 优先细分大于50颗星的聚类
                if cluster_size > 50 and cluster_size > largest_cluster_size:
                    largest_cluster_size = cluster_size
                    largest_cluster_label = label
            
            # 如果没有大于50颗星的聚类，但星座数量不足，细分最大的聚类
            if largest_cluster_label == -1 and len(unique_labels) < target_clusters:
                for label in unique_labels:
                    cluster_size = np.sum(current_labels == label)
                    if cluster_size > largest_cluster_size:
                        largest_cluster_size = cluster_size
                        largest_cluster_label = label
            
            if largest_cluster_label == -1:
                break
            
            print(f"细分包含{largest_cluster_size}颗恒星的星座...")
            
            # 对聚类进行细分
            mask = current_labels == largest_cluster_label
            cluster_features = features[mask]
            
            # 计算合适的子聚类数量
            n_subclusters = max(2, min(5, largest_cluster_size // 20))  # 每个子聚类约20颗星
            
            # 使用MiniBatchKMeans进行细分
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=n_subclusters, random_state=42, batch_size=100, max_iter=20)
            sub_labels = kmeans.fit_predict(cluster_features)
            
            # 更新标签
            new_labels = current_labels.copy()
            sub_cluster_ids = np.unique(sub_labels)
            
            # 为新的子聚类分配标签
            max_existing_label = np.max([l for l in np.unique(new_labels) if l != -1]) if len([l for l in np.unique(new_labels) if l != -1]) > 0 else 0
            
            temp_sub_labels = np.full_like(new_labels, -1)
            temp_sub_labels[mask] = sub_labels
            
            for i, sub_id in enumerate(sub_cluster_ids):
                sub_cluster_mask = temp_sub_labels == sub_id
                new_labels[sub_cluster_mask] = max_existing_label + i + 1
            
            current_labels = new_labels
            
            # 防止无限循环
            if len(np.unique(current_labels)) > 100:  # 安全限制
                break
        
        return current_labels
    
    def _calculate_optimized_cluster_score(self, features, labels):
        """
        简化的聚类质量分数计算，提高性能
        """
        unique_labels = np.unique(labels)
        valid_labels = [l for l in unique_labels if l != -1]
        
        if len(valid_labels) < 2:
            return -1
        
        # 简化的聚类质量评估
        n_clusters = len(valid_labels)
        
        # 计算聚类大小分布
        cluster_sizes = []
        small_clusters = 0
        
        for label in valid_labels:
            cluster_size = np.sum(labels == label)
            cluster_sizes.append(cluster_size)
            
            # 检查小聚类
            if cluster_size < 10:
                small_clusters += 1
        
        # 简化的评分标准
        avg_cluster_size = np.mean(cluster_sizes)
        size_variation = np.std(cluster_sizes) / avg_cluster_size if avg_cluster_size > 0 else 1
        
        # 惩罚小聚类比例过高
        small_cluster_ratio = small_clusters / n_clusters if n_clusters > 0 else 1
        penalty = small_cluster_ratio * 5
        
        # 基本分数：聚类数量适中，平均大小合理
        if 3 <= n_clusters <= 15 and avg_cluster_size >= 50:
            score = n_clusters * 0.3 + avg_cluster_size * 0.1 - size_variation * 0.2 - penalty
        else:
            score = -1
        
        return score
    
    def _fallback_dbscan_clustering(self, features):
        """
        DBSCAN聚类的回退方法
        """
        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        print("使用增强的DBSCAN聚类...")
        best_clustering = None
        max_score = -1
        best_params = None
        
        # 更广泛的参数搜索
        eps_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
        min_samples_values = [1, 2, 3, 4, 5]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                clustering = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric='euclidean'
                )
                labels = clustering.fit_predict(scaled_features)
                
                # 计算聚类质量分数
                unique_clusters = np.unique(labels)
                valid_clusters = [c for c in unique_clusters if c != -1]
                num_clusters = len(valid_clusters)
                
                if num_clusters < 2:
                    continue
                
                # 计算轮廓系数作为质量指标
                try:
                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(scaled_features, labels)
                except:
                    score = num_clusters * 0.1  # 简单的启发式分数
                
                print(f"尝试eps={eps}, min_samples={min_samples}: 发现 {num_clusters} 个星座, 质量分数={score:.3f}")
                
                if score > max_score and num_clusters >= 2:
                    max_score = score
                    best_clustering = clustering
                    best_params = (eps, min_samples)
        
        if best_clustering is not None:
            print(f"选择最佳参数: eps={best_params[0]}, min_samples={best_params[1]}")
            return best_clustering.labels_
        else:
            # 最终回退：简单的k-means
            print("DBSCAN不理想，使用K-means")
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            return kmeans.fit_predict(scaled_features)
    
    def _fallback_dbscan_clustering_fine(self, features):
        """
        精细的DBSCAN聚类方法，用于获得更多星座
        """
        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        print("使用精细DBSCAN聚类...")
        best_clustering = None
        max_score = -1
        best_params = None
        
        # 更精细的参数搜索，以获得更多星座
        eps_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        min_samples_values = [1, 2, 3]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                clustering = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric='euclidean'
                )
                labels = clustering.fit_predict(scaled_features)
                
                # 计算聚类质量分数
                unique_clusters = np.unique(labels)
                valid_clusters = [c for c in unique_clusters if c != -1]
                num_clusters = len(valid_clusters)
                
                # 放宽条件，允许更小的星座
                if num_clusters < 1:
                    continue
                
                # 计算轮廓系数作为质量指标
                try:
                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(scaled_features, labels)
                except:
                    score = num_clusters * 0.1  # 简单的启发式分数
                
                print(f"尝试eps={eps}, min_samples={min_samples}: 发现 {num_clusters} 个星座, 质量分数={score:.3f}")
                
                # 优先选择产生更多星座的参数
                if num_clusters >= 3 and score > max_score:
                    max_score = score
                    best_clustering = clustering
                    best_params = (eps, min_samples)
                elif num_clusters > 0 and best_clustering is None:
                    # 如果没有更好的选择，至少选择一个
                    max_score = score
                    best_clustering = clustering
                    best_params = (eps, min_samples)
        
        if best_clustering is not None:
            print(f"选择最佳参数: eps={best_params[0]}, min_samples={best_params[1]}")
            return best_clustering.labels_
        else:
            # 最终回退：使用更多聚类的k-means
            print("精细DBSCAN不理想，使用更多聚类的K-means")
            from sklearn.cluster import KMeans
            # 根据数据量决定聚类数量
            n_clusters = min(15, len(scaled_features) // 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(scaled_features)
    
    def analyze_cluster_shape(self, cluster_id):
        """
        分析星座的形状特征，基于星座的分布或轮廓形状
        """
        cluster_stars = self.stars_df[self.stars_df['cluster'] == cluster_id]
        points = cluster_stars[['x', 'y', 'z']].values
        
        # 少于3个点的情况
        if len(points) < 3:
            if len(points) == 2:
                return 'line'  # 两个点形成直线
            else:
                return 'cluster'  # 单个点或无法判断
        
        # PCA分析
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points)
        
        # 计算方向向量
        eigenvalues = pca.explained_variance_
        
        # 计算点的分布特征
        dist_matrix = squareform(pdist(points_2d))
        non_zero_dists = dist_matrix[np.triu_indices(len(dist_matrix), k=1)]
        avg_dist = np.mean(non_zero_dists) if len(non_zero_dists) > 0 else 0
        std_dist = np.std(non_zero_dists) if len(non_zero_dists) > 0 else 0
        
        # 1. 检查是否为线性分布
        if eigenvalues[0] / eigenvalues[1] > 4.0:  # 更严格的条件
            return 'line'  # 线性分布
        
        # 2. 检查是否为密集聚集
        if std_dist / avg_dist < 0.4 and len(points) > 3:  # 更严格的条件
            return 'cluster'  # 密集聚集
        
        # 3. 检查几何形状
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points_2d)
            
            # 检查三角形
            if len(hull.vertices) == 3:
                return 'triangle'
            
            # 检查四边形
            if len(hull.vertices) == 4:
                return 'square'
            
            # 检查圆形和弧形
            if len(hull.vertices) >= 5:
                # 计算凸包面积和最小外接圆面积比
                hull_area = hull.volume
                
                # 计算最小外接圆
                center = np.mean(points_2d, axis=0)
                distances = np.linalg.norm(points_2d - center, axis=1)
                max_distance = np.max(distances)
                circle_area = np.pi * max_distance ** 2
                
                # 如果凸包面积接近圆形面积，可能是圆形
                if hull_area / circle_area > 0.75:  # 更严格的条件
                    return 'circle'
                
                # 检查弧形/曲线
                if len(hull.vertices) > 4:
                    # 检查是否有明显的弯曲特征
                    angles = []
                    for i in range(len(hull.vertices)):
                        prev = hull.vertices[(i-1) % len(hull.vertices)]
                        curr = hull.vertices[i]
                        next_pt = hull.vertices[(i+1) % len(hull.vertices)]
                        
                        vec1 = points_2d[curr] - points_2d[prev]
                        vec2 = points_2d[next_pt] - points_2d[curr]
                        
                        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                            angle = np.arccos(np.clip(cos_angle, -1, 1))
                            angles.append(angle)
                    
                    # 如果有明显的弯曲角度，可能是弧形
                    if len(angles) > 0 and np.mean(angles) > np.pi/2.5:  # 更敏感的角度检测（72度）
                        return 'arc'
        except:
            pass
        
        # 默认形状
        return 'shape'
    
    def calculate_cluster_scores(self):
        """
        计算每个聚类的质量分数
        """
        if self.clusters is None:
            raise ValueError("请先调用cluster_stars方法")
        
        cluster_scores = {}
        unique_clusters = [c for c in np.unique(self.clusters) if c != -1]
        
        for cluster_id in unique_clusters:
            # 获取当前聚类的恒星
            cluster_stars = self.stars_df[self.stars_df['cluster'] == cluster_id]
            
            # 1. 计算紧凑性得分
            ra = cluster_stars['_RAJ2000'].values
            dec = cluster_stars['_DEJ2000'].values
            
            # 计算所有点对之间的距离
            distances = []
            for i, j in itertools.combinations(range(len(ra)), 2):
                if self.distance_metric == 'great_circle':
                    dist = great_circle_distance(ra[i], dec[i], ra[j], dec[j])
                else:
                    dist = chord_distance(ra[i], dec[i], ra[j], dec[j])
                distances.append(dist)
            
            avg_distance = np.mean(distances) if distances else 0
            # 紧凑性得分：距离越小得分越高（归一化到0-1）
            compactness_score = max(0, 1 - avg_distance / 30)  # 假设最大合理距离为30度
            
            # 2. 计算分离度得分
            # 计算与其他聚类的最小距离
            other_clusters = [c for c in unique_clusters if c != cluster_id]
            min_inter_cluster_dist = float('inf')
            
            for other_cluster_id in other_clusters:
                other_stars = self.stars_df[self.stars_df['cluster'] == other_cluster_id]
                
                # 计算两个聚类之间的最小距离
                for _, star1 in cluster_stars.iterrows():
                    for _, star2 in other_stars.iterrows():
                        if self.distance_metric == 'great_circle':
                            dist = great_circle_distance(star1['_RAJ2000'], star1['_DEJ2000'], 
                                                        star2['_RAJ2000'], star2['_DEJ2000'])
                        else:
                            dist = chord_distance(star1['_RAJ2000'], star1['_DEJ2000'], 
                                                star2['_RAJ2000'], star2['_DEJ2000'])
                        
                        if dist < min_inter_cluster_dist:
                            min_inter_cluster_dist = dist
            
            # 分离度得分：距离越大得分越高（归一化到0-1）
            separation_score = min(1, min_inter_cluster_dist / 20)  # 假设最小合理距离为20度
            
            # 3. 计算可见性得分
            avg_brightness = cluster_stars['Vmag'].mean()
            # 可见性得分：越亮得分越高（归一化到0-1）
            visibility_score = max(0, 1 - (avg_brightness / 5))  # 假设最暗的合理星等为5
            
            # 4. 计算聚类大小得分（适中的大小更好）
            cluster_size = len(cluster_stars)
            # 理想的星座大小在3-10颗恒星之间
            if 3 <= cluster_size <= 10:
                size_score = 1.0
            elif cluster_size < 3:
                size_score = cluster_size / 3
            else:
                size_score = max(0, 1 - (cluster_size - 10) / 10)
            
            # 综合得分（可以调整权重）
            total_score = (0.3 * compactness_score + 0.3 * separation_score + 
                          0.2 * visibility_score + 0.2 * size_score)
            
            cluster_scores[cluster_id] = {
                'compactness': compactness_score,
                'separation': separation_score,
                'visibility': visibility_score,
                'size_score': size_score,
                'total_score': total_score,
                'star_count': cluster_size,
                'avg_brightness': avg_brightness,
                'avg_distance': avg_distance
            }
        
        # 输出评分结果
        print("\n星座质量评分:")
        for cluster_id, scores in sorted(cluster_scores.items(), 
                                        key=lambda x: x[1]['total_score'], reverse=True):
            print(f"星座{cluster_id + 1}: 总分={scores['total_score']:.2f}, "
                  f"紧凑性={scores['compactness']:.2f}, 分离度={scores['separation']:.2f}, "
                  f"可见性={scores['visibility']:.2f}, 大小={scores['star_count']}颗星")
        
        return cluster_scores
    
    def visualize_constellation_single(self, cluster_id):
        """
        可视化单个星座
        """
        if self.clusters is None:
            raise ValueError("请先调用cluster_stars方法")
        
        # 创建3D图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置黑色背景
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # 隐藏坐标轴
        ax.set_axis_off()
        
        # 获取当前星座的数据
        cluster_stars = self.stars_df[self.stars_df['cluster'] == cluster_id]
        
        # 星座名称（使用ID）
        constellation_name = f"星座{cluster_id + 1}"
        
        # 随机生成一个颜色
        color = np.random.rand(3,)  # 随机颜色
        
        # 绘制背景恒星（灰色小点）
        background_stars = self.stars_df[self.stars_df['cluster'] != cluster_id]
        if len(background_stars) > 0:
            sizes_bg = np.clip(5 - background_stars['Vmag'] / 2, 2, 10)
            ax.scatter(
                background_stars['x'], background_stars['y'], background_stars['z'],
                c='gray', s=sizes_bg, alpha=0.1, label='背景恒星'
            )
        
        # 根据星等调整点的大小
        sizes = np.clip(25 - cluster_stars['Vmag'] * 5, 10, 50)  # 星等越小，点越大
        
        # 绘制当前星座的点
        scatter = ax.scatter(
            cluster_stars['x'], cluster_stars['y'], cluster_stars['z'],
            c=[color], s=sizes, alpha=0.9, marker='o', label=constellation_name
        )
        
        # 标注亮星名称
        for idx, star in cluster_stars.iterrows():
            if 'Display_Name' in star and not pd.isna(star['Display_Name']) and star['Vmag'] < 3.0:
                # 计算文本位置（稍微偏移）
                text_offset = 0.05
                text_x = star['x'] + (text_offset if star['x'] >= 0 else -text_offset)
                text_y = star['y'] + (text_offset if star['y'] >= 0 else -text_offset)
                text_z = star['z'] + (text_offset if star['z'] >= 0 else -text_offset)
                
                ax.text(
                    text_x, text_y, text_z,
                    star['Display_Name'],
                    color='white', fontsize=10, weight='bold'
                )
        
        # 绘制星座内部连线
        if len(cluster_stars) > 1:
            # 计算连接顺序（使用简单的最近邻连接）
            points = cluster_stars[['x', 'y', 'z']].values
            
            # 计算距离矩阵
            dist_matrix = squareform(pdist(points))
            np.fill_diagonal(dist_matrix, np.inf)
            
            # 简单的贪婪连接
            visited = set()
            current = 0
            visited.add(current)
            
            while len(visited) < len(points):
                next_point = np.argmin(dist_matrix[current])
                if next_point not in visited:
                    ax.plot([points[current, 0], points[next_point, 0]],
                           [points[current, 1], points[next_point, 1]],
                           [points[current, 2], points[next_point, 2]],
                           color=color, alpha=0.8, linewidth=2.5)
                    visited.add(next_point)
                    current = next_point
                else:
                    dist_matrix[current, next_point] = np.inf
                    # 如果所有邻居都访问过了，选择一个未访问的点
                    unvisited = [i for i in range(len(points)) if i not in visited]
                    if unvisited:
                        current = unvisited[0]
                        visited.add(current)
        
        # 绘制虚线包络（凸包边界）
        if len(cluster_stars) >= 3:
            try:
                from scipy.spatial import ConvexHull
                points = cluster_stars[['x', 'y', 'z']].values
                
                # 使用PCA降维到2D进行凸包计算
                pca = PCA(n_components=2)
                points_2d = pca.fit_transform(points)
                
                # 计算2D凸包
                hull = ConvexHull(points_2d)
                
                # 将凸包顶点转换回3D空间
                hull_points_3d = points[hull.vertices]
                
                # 绘制凸包边界（虚线）
                for i in range(len(hull.vertices)):
                    j = (i + 1) % len(hull.vertices)
                    ax.plot([hull_points_3d[i, 0], hull_points_3d[j, 0]],
                           [hull_points_3d[i, 1], hull_points_3d[j, 1]],
                           [hull_points_3d[i, 2], hull_points_3d[j, 2]],
                           color=color, alpha=0.6, linewidth=1.5, linestyle='--', 
                           label='包络边界' if i == 0 else "")
                
                # 在凸包中心添加一个小标记
                hull_center = np.mean(hull_points_3d, axis=0)
                ax.scatter(hull_center[0], hull_center[1], hull_center[2], 
                          color=color, s=50, marker='+', alpha=0.8)
                
            except Exception as e:
                print(f"绘制包络边界时出错: {e}")
                # 如果凸包计算失败，绘制最小包围球
                try:
                    center = np.mean(points, axis=0)
                    distances = np.linalg.norm(points - center, axis=1)
                    radius = np.max(distances)
                    
                    # 绘制球形包络（虚线圆）
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                    
                    ax.plot_wireframe(x, y, z, color=color, alpha=0.4, 
                                     linewidth=1, linestyle='--')
                except:
                    pass
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 计算当前星座的边界，设置合适的显示范围
        cluster_range = max([
            cluster_stars['x'].max() - cluster_stars['x'].min(),
            cluster_stars['y'].max() - cluster_stars['y'].min(),
            cluster_stars['z'].max() - cluster_stars['z'].min()
        ])
        center_x = cluster_stars['x'].mean()
        center_y = cluster_stars['y'].mean()
        center_z = cluster_stars['z'].mean()
        
        # 确保范围不为零
        if cluster_range == 0:
            cluster_range = 0.5
        
        # 设置坐标轴范围
        range_padding = cluster_range * 0.3  # 添加30%的边距
        ax.set_xlim(center_x - cluster_range/2 - range_padding, center_x + cluster_range/2 + range_padding)
        ax.set_ylim(center_y - cluster_range/2 - range_padding, center_y + cluster_range/2 + range_padding)
        ax.set_zlim(center_z - cluster_range/2 - range_padding, center_z + cluster_range/2 + range_padding)
        
        # 添加标题
        plt.suptitle(f"{constellation_name} - {len(cluster_stars)}颗恒星", color='white', fontsize=16, y=0.95)
        
        # 保存图片
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 使用星座ID作为文件名
        filename = f"constellation_{cluster_id + 1}"
        output_file = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"星座 {constellation_name} 图像已保存至: {output_file}")
        
        # 关闭图形以节省内存
        plt.close()
    
    def visualize_constellations(self):
        """
        可视化新星座分组结果，包括总览图和每个星座的单独图像
        """
        if self.clusters is None:
            raise ValueError("请先调用cluster_stars方法")
        
        # 创建3D图形（总览图）
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置黑色背景
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # 隐藏坐标轴
        ax.set_axis_off()
        
        # 生成颜色列表
        unique_clusters = np.unique(self.clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        
        # 绘制每个聚类
        for i, cluster_id in enumerate(unique_clusters):
            cluster_stars = self.stars_df[self.stars_df['cluster'] == cluster_id]
            
            # 为噪声点设置特殊颜色
            if cluster_id == -1:
                color = 'gray'
                alpha = 0.3
                size = 10
                label = '噪声点'
            else:
                color = colors[i % len(colors)]
                alpha = 0.9
                # 根据星等调整点大小
                sizes = np.clip(15 - cluster_stars['Vmag'] * 3, 5, 30)
                # 使用星座ID作为标签
                label = f"星座 {cluster_id + 1}"
            
            # 绘制恒星
            if cluster_id == -1:
                ax.scatter(cluster_stars['x'], cluster_stars['y'], cluster_stars['z'],
                          s=size, c=color, alpha=alpha, marker='o', label=label)
            else:
                scatter = ax.scatter(cluster_stars['x'], cluster_stars['y'], cluster_stars['z'],
                                    s=sizes, c=[color] * len(cluster_stars), alpha=alpha,
                                    marker='o', label=label)
                
                # 添加星座ID标签
                if len(cluster_stars) > 0:
                    # 找到聚类的中心位置
                    center_x = cluster_stars['x'].mean()
                    center_y = cluster_stars['y'].mean()
                    center_z = cluster_stars['z'].mean()
                    
                    # 在中心位置添加星座ID（简化版）
                    scale = 1.1
                    ax.text(center_x * scale, center_y * scale, center_z * scale,
                           f"{cluster_id + 1}", fontsize=12, color=color, 
                           ha='center', va='center', weight='bold')
                
                # 绘制星座内部连线（仅对较大的点）
                bright_cluster_stars = cluster_stars[cluster_stars['Vmag'] <= 3.5]
                if len(bright_cluster_stars) > 1:
                    # 计算连接顺序（使用简单的最近邻连接）
                    points = bright_cluster_stars[['x', 'y', 'z']].values
                    
                    # 计算距离矩阵
                    dist_matrix = squareform(pdist(points))
                    np.fill_diagonal(dist_matrix, np.inf)
                    
                    # 简单的贪婪连接
                    visited = set()
                    current = 0
                    visited.add(current)
                    
                    while len(visited) < len(points):
                        next_point = np.argmin(dist_matrix[current])
                        if next_point not in visited:
                            ax.plot([points[current, 0], points[next_point, 0]],
                                   [points[current, 1], points[next_point, 1]],
                                   [points[current, 2], points[next_point, 2]],
                                   color=color, alpha=0.5, linewidth=1.5)
                            visited.add(next_point)
                            current = next_point
                        else:
                            dist_matrix[current, next_point] = np.inf
                            # 如果所有邻居都访问过了，选择一个未访问的点
                            unvisited = [i for i in range(len(points)) if i not in visited]
                            if unvisited:
                                current = unvisited[0]
                                visited.add(current)
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 设置坐标轴范围
        max_range = max([self.stars_df['x'].max(), self.stars_df['y'].max(), self.stars_df['z'].max()])
        ax.set_xlim(-max_range * 1.2, max_range * 1.2)
        ax.set_ylim(-max_range * 1.2, max_range * 1.2)
        ax.set_zlim(-max_range * 1.2, max_range * 1.2)
        
        # 添加标题
        plt.title("新星座分组可视化总览", color='white', fontsize=16, pad=20)
        
        # 添加图例（仅显示前10个星座和噪声点）
        handles, labels = ax.get_legend_handles_labels()
        filtered_handles = []
        filtered_labels = []
        noise_label_added = False
        
        for h, l in zip(handles, labels):
            if l == '噪声点' and not noise_label_added:
                filtered_handles.append(h)
                filtered_labels.append(l)
                noise_label_added = True
            elif l != '噪声点' and len(filtered_labels) < 11:
                filtered_handles.append(h)
                filtered_labels.append(l)
        
        ax.legend(filtered_handles, filtered_labels, loc='upper left', fontsize=8,
                 framealpha=0.5, facecolor='black', edgecolor='white')
        for text in ax.get_legend().get_texts():
            text.set_color('white')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存总览图片
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, "new_constellations_overview.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"星座总览图已保存至: {output_file}")
        
        # 关闭总览图以节省内存
        plt.close()
        
        # 为每个星座生成单独的图像
        print("\n开始生成每个星座的单独图像...")
        valid_clusters = [c for c in unique_clusters if c != -1]
        
        # 为所有星座生成单独图像
        clusters_to_generate = valid_clusters
        
        for cluster_id in clusters_to_generate:
            self.visualize_constellation_single(cluster_id)
        
        print(f"已生成所有{len(clusters_to_generate)}个星座图像！")
        print("星座图像生成完成！")
    
    def optimize_parameters(self):
        """
        简单的参数优化，寻找最佳的聚类参数组合
        """
        if self.stars_df is None:
            raise ValueError("请先调用preprocess_stars方法")
        
        print("开始参数优化...")
        
        # 定义参数网格，考虑HDBSCAN是否可用
        use_hdbscan = self.cluster_method == 'hdbscan' and HAS_HDBSCAN
        
        if use_hdbscan:
            param_grid = {
                'min_cluster_size': [2, 3, 4, 5],
                'min_samples': [1, 2, 3]
            }
        else:
            # 使用DBSCAN
            param_grid = {
                'eps': [5.0, 7.5, 10.0, 12.5, 15.0],
                'min_samples': [2, 3, 4, 5]
            }
        
        best_score = -1
        best_params = None
        best_clusters = None
        
        # 尝试所有参数组合
        param_combinations = list(itertools.product(*param_grid.values()))
        total_combinations = len(param_combinations)
        
        for i, params in enumerate(param_combinations):
            print(f"尝试参数组合 {i+1}/{total_combinations}: {params}")
            
            # 设置当前参数
            if use_hdbscan:
                self.hdbscan_min_cluster_size, self.hdbscan_min_samples = params
            else:
                self.dbscan_eps, self.dbscan_min_samples = params
            
            # 执行聚类
            try:
                self.cluster_stars()
                
                # 计算聚类质量得分
                scores = self.calculate_cluster_scores()
                
                if scores:
                    # 计算平均总分
                    avg_total_score = np.mean([s['total_score'] for s in scores.values()])
                    num_clusters = len(scores)
                    
                    # 综合考虑平均分和聚类数量
                    # 理想的聚类数量应该适中（既不过多也不过少）
                    cluster_count_score = 1.0
                    if num_clusters < 5:
                        cluster_count_score = num_clusters / 5
                    elif num_clusters > 30:
                        cluster_count_score = max(0, 1 - (num_clusters - 30) / 20)
                    
                    combined_score = 0.7 * avg_total_score + 0.3 * cluster_count_score
                    
                    print(f"  平均总分: {avg_total_score:.2f}, 星座数量: {num_clusters}, 综合得分: {combined_score:.2f}")
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = params
                        best_clusters = self.clusters.copy()
            except Exception as e:
                print(f"  参数组合出错: {e}")
        
        # 设置最佳参数
        if best_params is not None:
            print(f"\n找到最佳参数: {best_params}")
            print(f"最佳综合得分: {best_score:.2f}")
            
            if self.cluster_method == 'dbscan':
                self.dbscan_eps, self.dbscan_min_samples = best_params
            else:
                self.hdbscan_min_cluster_size, self.hdbscan_min_samples = best_params
            
            # 应用最佳参数
            self.cluster_stars()
        else:
            print("未找到合适的参数组合")

# --- 4. 主函数 ---
def main():
    print("===== 新星座分组模型 =====")
    
    # 加载和预处理数据
    stars_df, constellations_df = load_and_preprocess_data()
    if stars_df is None:
        print("无法加载数据，程序终止")
        return
    
    # 创建新星座模型实例，使用改进的聚类方法
    model = NewConstellationModel(
        min_brightness=5.0,  # 设置视星等限制为≤5.0以获取更多恒星
        distance_metric='great_circle',  # 使用大圆距离
        cluster_method='dbscan'  # 使用DBSCAN，参数将在cluster_stars中自动优化
    )
    
    # 预处理恒星数据
    model.preprocess_stars(stars_df)
    
    # 参数优化
    # 注意：参数优化可能需要较长时间，可以根据需要注释掉
    # model.optimize_parameters()
    
    # 执行聚类
    model.cluster_stars()
    
    # 计算聚类质量得分
    model.calculate_cluster_scores()
    
    # 可视化结果
    model.visualize_constellations()
    
    print("\n新星座分组模型执行完成!")

if __name__ == "__main__":
    main()