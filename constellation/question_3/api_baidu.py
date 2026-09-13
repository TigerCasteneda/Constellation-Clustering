import pandas as pd
import os
import base64
from aip import AipImageClassify  # 百度AI的Python库

APP_ID = os.getenv("BAIDU_APP_ID", "")
API_KEY = os.getenv("BAIDU_API_KEY", "")
SECRET_KEY = os.getenv("BAIDU_SECRET_KEY", "")
# 连接百度AI
client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)


def get_image_bytes(image_path):
    with open(image_path, 'rb') as f:
        return f.read()


# ---------------------- 3. 批量识别所有星群图片
shape_results = []
# 遍历所有星群图片（之前画的“星群X.png”）
for filename in os.listdir("."):
    if filename.startswith("星群") and filename.endswith(".png"):
        # 提取星群编号（比如“星群0.png”→ 0）
        cluster_id = int(filename.replace("星群", "").replace(".png", ""))
        # 图片转成百度API需要的格式

        image_bytes = get_image_bytes(filename)
        print(type(image_bytes))
        # ---------------------- 4. 调用百度AI识别形状（指令直白，AI能懂）
        response = client.advancedGeneral(image_bytes)

        # ---------------------- 5. 提取AI识别结果（整理成我们需要的格式）
        if "result" in response:
            # 百度AI会返回多个标签，比如["三角形", "点列", "几何图形"]
            labels = [item["keyword"] for item in response["result"]]
            print(labels)
            # 筛选出形状相关的标签（比如“三角形”“四边形”“折线”）
            shape_labels = [label for label in labels if
                            label in ["三角形", "四边形", "五边形", "折线", "弧形", "圆形", "十字形"]]
            # 如果没识别出具体形状，默认“不规则形”
            shape_type = shape_labels[0] if shape_labels else "不规则形"

            # 让AI再类比一个简单事物（用百度的“文本生成”API，免费）
            # 调用百度文心一言的简易接口（不用单独申请，和图像识别共用密钥）
            # text_response = client.textSummary({
            #     "text": f"请给{shape_type}类比一个2字以内的常见事物，只返回事物名称，不要多余文字",
            #     "max_summary_len": 2
            # })
            # analogy = text_response.get("summary", "未知") if "summary" in text_response else "未知"
            analogy = "未知"
            # 保存结果
            shape_results.append({
                "星群编号": cluster_id,
                "形状类型": shape_type,
                "类比事物": analogy
            })
            print(f"星群{cluster_id}识别完成：形状={shape_type}，类比={analogy}")

# ---------------------- 6. 保存结果（后续命名要用）
results_df = pd.DataFrame(shape_results)
results_df.to_csv("星群形状识别结果_国内AI.csv", index=False, encoding="utf-8")
print("所有星群识别完成！结果已保存到CSV文件。")

# 加载AI识别结果（之前保存的CSV）
ai_results = pd.read_csv("星群形状识别结果_国内AI.csv")

# 定义命名映射（处理重复或需要调整的关键词）
name_map = {
    "灯座": "灯枢",  # 避免“灯座座”重复
    "弓箭": "箭羽",  # 更简洁
    "云纹": "云弧",  # 贴合弧形形状
    "方鼎": "鼎枢",  # 突出中心亮星
    "山尖": "山巅"   # 更有文化感
}

# 批量生成星座名
def generate_constellation_name(shape_type, analogy):
    # 优先用映射表调整类比词，没有则直接用
    core_word = name_map.get(analogy, analogy)
    # 组合成完整星座名（核心词+座）
    return f"{core_word}座"

# 应用到所有星群
ai_results["最终星座名"] = ai_results.apply(
    lambda x: generate_constellation_name(x["形状类型"], x["类比事物"]),
    axis=1
)

# 保存最终命名结果
ai_results.to_csv("星群最终命名结果.csv", index=False, encoding="utf-8")

# 打印前10个结果看看
print("最终星座名示例：")
print(ai_results[["星群编号", "形状类型", "类比事物", "最终星座名"]].head(10))
