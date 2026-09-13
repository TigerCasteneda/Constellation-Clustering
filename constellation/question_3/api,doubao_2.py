import pandas as pd
import os
import base64
import requests

# ==============================================================================
# 核心配置
# ==============================================================================
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_API_URL = "https://ark-api.volcengine.com/api/v1/chat/completions"
DOUBAO_MODEL = "Doubao-1.5-vision-pro-32k"  # 多模态模型


# ==============================================================================
# 工具函数 (这些函数在使用本地文件时不会被调用，但保留以备不时之需)
# ==============================================================================
def image_to_base64(image_path):
    """将图片转换为Base64格式"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"图片转换失败：{image_path}，错误：{str(e)}")
        return None


def call_doubao_vision(image_base64, prompt):
    """调用新版豆包多模态API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DOUBAO_API_KEY}"
    }

    data = {
        "model": DOUBAO_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.3
    }

    try:
        response = requests.post(DOUBAO_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        print("API调用超时，请检查网络")
        return None
    except requests.exceptions.RequestException as e:
        print(f"API调用失败：{str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容：{e.response.text}")
        return None


# ==============================================================================
# 第一步：批量识别星群形状与意象 (使用本地文件时，此步骤会被跳过)
# ==============================================================================
def batch_recognize_clusters(img_dir="."):
    """批量识别目录下的星群图片，生成AI识别结果CSV"""
    # 此函数内容与原版相同，无需修改
    shape_results = []
    # ... (此处省略了与原版完全相同的代码，以保持简洁)
    # 注意：由于默认使用本地文件，这段代码在常规运行时不会执行
    print("警告：程序正在执行API识别流程，这可能会产生费用。")
    # (如果需要测试API功能，可以将主函数中的 use_existing_recognition_file 设置为 False)
    # 以下是函数的剩余部分，为保持完整性而保留
    for filename in os.listdir(img_dir):
        if filename.startswith("星群") and filename.endswith(".png"):
            try:
                cluster_id = int(filename.replace("星群", "").replace(".png", ""))
            except ValueError:
                print(f"跳过无效文件名：{filename}（无法提取星群编号）")
                continue

            img_path = os.path.join(img_dir, filename)
            img_base64 = image_to_base64(img_path)
            if img_base64 is None:
                continue

            prompt = """
            这是星群的平面点图（黑色圆点为恒星），严格按以下要求输出，仅返回结果无多余文字：
            1. 形状类型：从【三角形、四边形、五边形、折线、弧形、圆形、十字形、不规则形】中选一个；
            2. 意象类比：给该形状类比1个2字以内的中国传统事物（如山尖、灯枢、云弧、鼎、箭）；
            输出格式：形状类型+中文逗号+意象类比（例：三角形，山巅）
            """

            print(f"正在识别星群{cluster_id}...")
            result = call_doubao_vision(img_base64, prompt)
            if result is None:
                shape_type = "不规则形"
                analogy = "玄曜"
                print(f"星群{cluster_id}识别失败，已赋默认值")
            else:
                try:
                    shape_type, analogy = result.split("，")
                    valid_shapes = ["三角形", "四边形", "五边形", "折线", "弧形", "圆形", "十字形", "不规则形"]
                    if shape_type not in valid_shapes:
                        shape_type = "不规则形"
                    analogy = analogy[:2] if len(analogy) > 2 else analogy
                    print(f"星群{cluster_id}识别成功：形状={shape_type}，类比={analogy}")
                except Exception as e:
                    shape_type = "不规则形"
                    analogy = "玄曜"
                    print(f"星群{cluster_id}结果解析失败：{str(e)}，已赋默认值")

            shape_results.append({
                "星群编号": cluster_id,
                "形状类型": shape_type,
                "类比事物": analogy
            })

    recognition_csv = "星群形状识别结果_豆包.csv"
    results_df = pd.DataFrame(shape_results)
    results_df.to_csv(recognition_csv, index=False, encoding="utf-8-sig")
    print(f"\n所有星群识别完成！结果已保存至：{recognition_csv}")
    return recognition_csv


# ==============================================================================
# 第二步：批量生成星座名 (此函数已增强，以检查必要的列)
# ==============================================================================
def generate_constellation_names(recognition_csv):
    """基于AI识别结果，生成最终星座名"""
    try:
        ai_results = pd.read_csv(recognition_csv, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"错误：找不到识别结果文件：{recognition_csv}")
        return None

    # 检查必要的列是否存在
    required_column = "类比事物"
    if required_column not in ai_results.columns:
        print(f"错误：文件 '{recognition_csv}' 中缺少必要的列：'{required_column}'")
        print("请确保你的CSV文件包含此列，或重新运行API识别流程生成正确的文件。")
        return None

    name_map = {
        "灯座": "灯枢", "弓箭": "箭羽", "云纹": "云弧", "方鼎": "鼎枢", "山尖": "山巅",
        "未知": "玄曜", "圆点": "玉盘", "直线": "长庚", "曲线": "云舒", "十字": "玄针",
        "五角": "星枢", "方块": "方隅", "月牙": "月弧"
    }

    traditional_constellations = [
        # ... (此处省略了完整的传统星座列表，与原版相同)
        "白羊座", "金牛座", "双子座", "巨蟹座", "狮子座", "处女座", "天秤座", "天蝎座", "射手座", "摩羯座", "水瓶座",
        "双鱼座", "大熊座", "小熊座", "天龙座", "仙后座", "仙王座", "鹿豹座", "御夫座", "猎犬座", "狐狸座", "天鹅座",
        "天琴座", "天鹰座", "海豚座", "小马座", "飞马座", "三角座", "仙女座", "英仙座", "武仙座", "蝎虎座", "天猫座",
        "小狮座", "巨爵座", "长蛇座", "麒麟座", "猎户座", "鲸鱼座", "天坛座", "绘架座", "苍蝇座", "山案座", "时钟座",
        "杜鹃座", "圆规座"
    ]

    def generate_name(row):
        analogy = row["类比事物"]
        core_word = name_map.get(analogy, analogy)
        base_name = f"{core_word}座"

        if base_name in traditional_constellations:
            base_name = f"{core_word}曜座"

        if len(base_name) < 2:
            base_name = f"{core_word}枢座"[:3]
        elif len(base_name) > 3:
            base_name = base_name[:2] + "座"

        return base_name

    ai_results["最终星座名"] = ai_results.apply(generate_name, axis=1)

    final_csv = "final_naming.csv"
    ai_results.to_csv(final_csv, index=False, encoding="utf-8-sig")
    print(f"\n星座命名完成！最终结果已保存至：{final_csv}")

    print("\n最终星座名示例：")
    print(ai_results[["星群编号", "形状类型", "类比事物", "最终星座名"]].head(15).to_string(index=False))

    return final_csv


# ==============================================================================
# 主函数
# ==============================================================================
if __name__ == "__main__":
    print("=== 开始星群识别与星座命名流程 ===")

    # --- 配置 ---
    # 设置为 True 以使用已有的识别结果文件，跳过API调用
    # 设置为 False 以重新运行API识别
    use_existing_recognition_file = True  # 默认使用本地文件
    # 指定已有的识别结果文件路径
    existing_recognition_file_path = "constellation_hull_vertices.csv"  # 你的本地文件

    recognition_csv = None

    if use_existing_recognition_file and os.path.exists(existing_recognition_file_path):
        print(f"发现本地识别结果文件: '{existing_recognition_file_path}'，将直接使用它进行命名。")
        recognition_csv = existing_recognition_file_path
    else:
        if use_existing_recognition_file:
            print(f"警告: 指定的本地文件 '{existing_recognition_file_path}' 不存在。")
        print("将启动API识别流程来生成所需文件...")
        # 执行API识别流程
        recognition_csv = batch_recognize_clusters(img_dir=".")

    # 执行命名流程
    if recognition_csv:
        generate_constellation_names(recognition_csv)

    print("\n=== 全流程执行完成！ ===")
