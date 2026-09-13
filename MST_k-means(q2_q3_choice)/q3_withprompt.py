import os
import base64
import json
import re
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import threading
from pathlib import Path




prompt_text = """
    你是一位富有想象力的天文学家和动物学家。请观察这张图片中的星点连线图案。

    任务要求：
    1. **视觉联想 (Pareidolia)**：忽略这是星星的事实，单纯看线条和形状。它看起来像什么？（动物、工具、神话生物、人物等）。
    2. **命名**：
       - 给它起一个拉丁文学名（格式如 Ursa Major）。
       - 给它起一个中文名，匹配动物名称。
    3. **评分 (Visual Coherence Score)**：给这个形状的“具象程度”打分（1-10分）。
       - 10分：形状非常清晰，一眼就能认出是什么（如清晰的三角形、十字、长蛇）。
       - 1分：形状杂乱无章，像一团乱麻或随机的线条。

    必须仅返回以下 JSON 格式，不要包含其他文字：
    {
        "latin_name": "String",
        "chinese_name": "String",
        "visual_description": "String",
        "coherence_score": Integer
    }

    可以参考山海经神话生物的命名。

"""

# =================配置区域=================
# 输入图片文件夹路径
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT
IMAGE_DIR = os.path.join(f"{DATA_ROOT}", "constellation_images")
OUTPUT_FILE = os.path.join(f"{DATA_ROOT}", "constellation_data.json")
CONCURRENCY = int(os.getenv("Q3_CONCURRENCY", "10"))
MAX_RETRY = int(os.getenv("Q3_RENAME_RETRIES", "10"))

# 初始化客户端
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# =========================================

USED_LATIN = set()
USED_CHINESE = set()
NAME_LOCK = threading.Lock()

def normalize_name(name):
    return (name or "").strip().lower()

def encode_image_to_base64(image_path):
    """读取本地图片并转换为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_json_from_text(text):
    """清洗 AI 返回的文本，提取 JSON 部分"""
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def analyze_single_image(image_path):
    global prompt_text
    filename = os.path.basename(image_path)
    print(f"正在处理: {filename} ...")

    base64_image = encode_image_to_base64(image_path)

    attempt = 0
    extra_req = "请确保拉丁名与中文名在本批次内唯一，若重复请重新命名。"
    last_data = None

    while attempt < MAX_RETRY:
        try:
            completion = client.chat.completions.create(
                model="qwen3-vl-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                            {"type": "text", "text": prompt_text + "\n" + extra_req},
                        ],
                    },
                ],
                extra_body={'enable_thinking': True, "thinking_budget": 1024}
            )

            answer_content = completion.choices[0].message.content
            clean_json = extract_json_from_text(answer_content)
            data = json.loads(clean_json)
            last_data = data

            latin = normalize_name(data.get('latin_name', ''))
            chinese = normalize_name(data.get('chinese_name', ''))

            with NAME_LOCK:
                latin_dup = latin in USED_LATIN if latin else False
                chinese_dup = chinese in USED_CHINESE if chinese else False

                if not latin_dup and not chinese_dup:
                    if latin:
                        USED_LATIN.add(latin)
                    if chinese:
                        USED_CHINESE.add(chinese)
                    data['filename'] = filename
                    data['file_path'] = image_path
                    print(f"  -> 成功: {data.get('chinese_name','')} (评分: {data.get('coherence_score','?')})")
                    return data

            attempt += 1
            print(f"  -> 检测到命名重复，正在重试({attempt}/{MAX_RETRY}) {filename}")
            with NAME_LOCK:
                current_latin = sorted(USED_LATIN)
                current_chinese = sorted(USED_CHINESE)
            extra_req = (
                "避免以下已占用名称。Latin: " + ", ".join(current_latin[:50]) +
                "；中文: " + ", ".join(current_chinese[:50])
            )

        except Exception as e:
            print(f"  -> 处理失败 {filename}: {e}")
            return None

    try:
        base_latin = (last_data.get('latin_name') if last_data else '')
        base_chinese = (last_data.get('chinese_name') if last_data else '')
        latin_candidate = base_latin
        chinese_candidate = base_chinese
        idx = 2
        with NAME_LOCK:
            while normalize_name(latin_candidate) in USED_LATIN or normalize_name(chinese_candidate) in USED_CHINESE:
                latin_candidate = f"{base_latin}-{idx}" if base_latin else f"unnamed-{idx}"
                chinese_candidate = f"{base_chinese}{idx}" if base_chinese else f"未命名{idx}"
                idx += 1
            if latin_candidate:
                USED_LATIN.add(normalize_name(latin_candidate))
            if chinese_candidate:
                USED_CHINESE.add(normalize_name(chinese_candidate))
        data = last_data or {}
        data['latin_name'] = latin_candidate
        data['chinese_name'] = chinese_candidate
        data['filename'] = filename
        data['file_path'] = image_path
        print(f"  -> 重命名成功: {data.get('chinese_name','')} (评分: {data.get('coherence_score','?')})")
        return data
    except Exception as e:
        print(f"  -> 最终重命名失败 {filename}: {e}")
        return None

def main():
    # 1. 检查目录是否存在
    if not os.path.exists(IMAGE_DIR):
        print(f"错误: 找不到文件夹 {IMAGE_DIR}")
        return

    # 2. 获取所有 png 图片
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    
    if not image_files:
        print(f"在 {IMAGE_DIR} 中没有找到 PNG 图片。")
        return

    print(f"共发现 {len(image_files)} 张图片，开始批量生成数据...\n")
    
    all_results = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(analyze_single_image, img_path): img_path for img_path in image_files}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception:
                pass

    # 4. 保存结果到 JSON 文件
    if all_results:
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            print(f"\nSUCCESS! 所有数据已保存至: {OUTPUT_FILE}")
            
            # 计算平均分
            avg_score = sum(item['coherence_score'] for item in all_results) / len(all_results)
            print(f"本次生成星座的平均具象得分: {avg_score:.2f}")
            
        except IOError as e:
            print(f"保存文件失败: {e}")
    else:
        print("没有生成任何有效数据。")

if __name__ == "__main__":
    main()
