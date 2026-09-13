import os
import numpy as np
from PIL import Image
from skimage import measure, filters
from skimage.color import rgb2gray
import pandas as pd
import requests
import json
import time
import hmac
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. 配置区域 ---
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_SECRET_KEY = os.getenv("DOUBAO_SECRET_KEY", "")
IMAGE_DIR = "D:\code\constellation_maps"  # 存放星座图片的文件夹路径
OUTPUT_EXCEL = "naming_system.xlsx"  # 输出结果文件名
MAX_WORKERS = 5  # 并发处理数量，根据你的网络和CPU调整

# --- 2. 核心功能函数 ---

def generate_doubao_token(api_key, secret_key, expire=3600):
    """生成豆包API的访问令牌"""
    current_time = int(time.time())
    to_sign = f"{api_key}{current_time}{expire}"
    signature = hmac.new(
        secret_key.encode("utf-8"),
        to_sign.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return f"{api_key}:{signature}:{current_time}:{expire}"

def extract_shape_features(image_path):
    """
    使用Pillow和scikit-image提取图片中星座的形态特征。
    返回一个包含特征描述的字符串。
    """
    try:
        # 1. 使用Pillow打开图片并转换为灰度图
        with Image.open(image_path) as img:
            # 如果是彩色图，转换为灰度图
            if img.mode != 'L':
                img = img.convert('L')
            # 转换为numpy数组
            img_array = np.array(img)

        # 2. 二值化处理 (模拟 cv2.threshold)
        # 使用Otsu's方法自动寻找最佳阈值，适合前景和背景对比明显的图像
        thresh_value = filters.threshold_otsu(img_array)
        # 对图像进行二值化，大于阈值的为1（前景），小于的为0（背景）
        # 注意：我们反转了阈值，因为通常星图是白色星星在黑色背景上
        binary_mask = img_array < thresh_value

        # 3. 查找轮廓 (模拟 cv2.findContours)
        # measure.find_contours返回的是轮廓的坐标点列表
        contours = measure.find_contours(binary_mask, level=0.5)

        # 4. 分析轮廓
        shape_types = []
        star_count = 0
        line_segments = 0

        for contour in contours:
            # contour是一个Nx2的数组，包含轮廓上点的坐标

            # 计算轮廓的面积 (模拟 cv2.contourArea)
            area = measure.area(contour)

            # 计算轮廓的周长 (模拟 cv2.arcLength)
            perimeter = measure.perimeter(contour)

            # 简单分类：小面积的视为星点，大面积的视为连线或形状
            if area < 50:  # 星点
                star_count += 1
            else:  # 连线或形状
                if perimeter > 100:  # 较长的轮廓视为连线
                    line_segments += 1
                else:  # 较短的轮廓尝试识别形状
                    # 计算轮廓的多边形逼近 (模拟 cv2.approxPolyDP)
                    # 使用轮廓的凸包来简化形状
                    hull = measure.convex_hull_object(contour)
                    # 计算凸包的顶点数
                    num_vertices = len(hull.vertices)

                    if num_vertices == 3:
                        shape_types.append("三角形")
                    elif num_vertices == 4:
                        shape_types.append("四边形")
                    elif num_vertices > 4:
                        shape_types.append("多边形")

        # 5. 生成特征描述文本
        unique_shapes = list(set(shape_types))
        shape_desc = f"包含{', '.join(unique_shapes)}" if unique_shapes else "无明显几何形状"
        
        features = (
            f"这是一个自定义星座的星图。{shape_desc}，"
            f"可辨识的星点约{star_count}个，主要连线片段约{line_segments}条。"
        )
        return features

    except Exception as e:
        return f"特征提取失败: {e}"

def call_doubao_api(prompt, token):
    """调用豆包API生成星座名称"""
    url = "https://aquasearch.volcengineapi.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "doubao-pro",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 50
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"API调用失败: {e}"

def process_single_image(image_path, token):
    """处理单张图片的完整流程：特征提取 -> API调用 -> 返回结果"""
    print(f"正在处理: {os.path.basename(image_path)}...")
    
    # 提取特征
    features = extract_shape_features(image_path)
    if "失败" in features:
        return {
            "image_path": image_path,
            "features": features,
            "constellation_name": "提取失败"
        }

    # 构建提示词
    prompt = f"""
    根据以下对一个全新、自定义划分的星座的形态描述，为它起一个独特的中文名称（2-4字）。
    这个名称应该富有想象力、诗意或神话色彩，并且能够反映出它的形状特征。
    不要使用现有88星座的名称。

    星座形态描述:
    {features}

    请只返回星座名称。
    """
    
    # 调用API
    name = call_doubao_api(prompt, token)
    
    return {
        "image_path": image_path,
        "features": features,
        "constellation_name": name
    }

# --- 3. 主执行函数 ---

def main():
    """主函数，执行批量处理流程"""
    print("--- 开始批量处理星座命名任务 ---")
    
    # 检查图片目录是否存在
    if not os.path.isdir(IMAGE_DIR):
        print(f"错误: 图片目录 '{IMAGE_DIR}' 不存在。")
        return

    # 获取所有图片文件路径
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(image_extensions)]
    
    if not image_paths:
        print(f"错误: 在目录 '{IMAGE_DIR}' 中未找到任何图片。")
        return

    print(f"共发现 {len(image_paths)} 张图片，准备开始处理...")
    
    # 生成一次Token供所有请求使用
    print("正在获取API访问令牌...")
    try:
        token = generate_doubao_token(DOUBAO_API_KEY, DOUBAO_SECRET_KEY)
    except Exception as e:
        print(f"获取Token失败: {e}")
        return
        
    # 使用线程池并发处理图片
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_image, path, token): path for path in image_paths}
        
        # 收集结果
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"处理图片 {futures[future]} 时发生严重错误: {e}")

    # 将结果保存到Excel
    print("\n处理完成，正在将结果写入Excel文件...")
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')
    print(f"任务成功结束！结果已保存至 {OUTPUT_EXCEL}")

# --- 4. 脚本入口 ---
if __name__ == "__main__":
    main()
