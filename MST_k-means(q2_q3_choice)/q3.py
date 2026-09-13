import os
import base64
import json
import re
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI




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

    附录：《山海经》怪兽特性描述库
    # 线性/长条形怪兽特性
    '烛龙': '人面蛇身，赤色，身长千里，睁眼为昼，闭眼为夜',
    '相柳': '九头蛇身，食于九土，所到之处皆为泽国',
    '巴蛇': '巨蛇，能吞象，三年后方吐出象骨',
    '长蛇': '身长百丈，声如鼓柝，见则大旱',
    '修蛇': '巨蛇，能吞鹿，居于洞庭湖',
    '延维': '人首蛇身，长如车辕，见之者霸',
    '应龙': '有翼之龙，曾助黄帝战蚩尤，能呼风唤雨',
    '螭龙': '无角之龙，性好水，常居于深渊',
    '腾蛇': '能兴云雾而游其中，无足而飞',
    '虺': '小蛇，五百年化为蛟，蛟千年化为龙',
    
    # 三角形/鸟类怪兽特性
    '凤凰': '五彩之鸟，首文曰德，翼文曰义，见则天下安宁',
    '毕方': '独足之鹤，衔火精在人家作怪，见则邑有讹火',
    '青鸾': '西王母信使，常伴西王母左右',
    '比翼鸟': '一目一翼，不比不飞，见则吉良',
    '鹓鶵': '似凤，非梧桐不止，非练实不食，非醴泉不饮',
    '九凤': '九首人面鸟身，司天之九部',
    '玄鸟': '燕也，春分而来，秋分而去',
    '重明鸟': '双睛，状如鸡，鸣似凤，能逐猛兽',
    '青鸟': '三足神鸟，为西王母取食',
    '三足乌': '日中之精，载日运行',
    
    # 四边形/巨型走兽怪兽特性
    '麒麟': '仁兽，麋身牛尾一角，不履生虫，不折生草',
    '饕餮': '羊身人面，目在腋下，虎齿人爪，贪食无厌',
    '穷奇': '状如虎，有翼，食人从首始',
    '梼杌': '状如虎而犬毛，人面虎足猪口牙',
    '混沌': '状如犬，长毛四足，有目而不见',
    '白泽': '能言，知万物之情，通晓鬼神之事',
    '狻猊': '形如狮，好烟火，常立于香炉',
    '驳': '状如白马，锯牙食虎豹',
    '兕': '状如牛，苍黑一角',
    '罴': '人熊，黄白文，能立，疾走',
    
    # 弧形/猫科狼形怪兽特性
    '白虎': '西方之神，主杀伐，性威猛',
    '陆吾': '人面虎身九尾，司天之九部及帝之囿时',
    '驺吾': '大若虎，五彩毕具，尾长于身，乘之日行千里',
    '天狗': '状如狸而白首，其音如榴榴，可以御凶',
    '蛊雕': '状如雕而有角，其音如婴儿之音，是食人',
    '猰貐': '龙首，居弱水中，食人',
    '狰': '状如赤豹，五尾一角，其音如击石',
    '狡': '状如犬而豹文，其角如牛，其音如吠犬',
    '獙獙': '状如狐而有翼，其音如鸿雁',
    
    # 密集聚集/多足虫形怪兽特性
    '九婴': '九头水火之怪，能喷水吐火',
    '开明兽': '身大类虎而九首，皆人面，东向立昆仑上',
    '九尾狐': '状如狐而九尾，其音如婴儿，能食人',
    '九头鸟': '色赤，似鸭，九头皆鸣',
    '化蛇': '人面豺身，鸟翼蛇行，其音如叱呼',
    '旋龟': '鸟首虺尾，佩之不聋，可以为底',
    '灌灌': '状如鸠，其音若呵，佩之不惑',
    '赤鱬': '人面鱼身，其音如鸳鸯，食之不疥',
    '夫诸': '状如白鹿而四角，见则其邑大水',
    
    # 其他形状/通用奇兽特性
    '狌狌': '状如禺而白耳，伏行人走，知人名',
    '鹿蜀': '状如马而白首，其文如虎而赤尾，其音如谣',
    '狸力': '状如豚，有距，其音如狗吠，见则其县多土功',
    '鯥': '状如牛，陵居，蛇尾有翼，其羽在魼下',
    '颙': '状如枭，人面四目而有耳，见则天下大旱',
    '狸狌': '善伏，能捕鼠',
    '猾褢': '状如人而彘鬣，穴居而冬蛰，其音如斫木',
    '彘': '状如虎而牛尾，其音如吠犬，是食人',
    '何罗鱼': '一首而十身，其音如吠犬，食之已痈',
    '当康': '状如豚而有牙，其鸣自叫，见则天下大穰',
    
    # 特殊形状/圆形多足怪兽特性
    '玄武': '龟蛇合体，北方之神，主风雨',
    '商羊': '一足鸟，能招大雨',
    '酸与': '状如蛇，四翼六目三足，其鸣自詨',
    '跂踵': '状如鸮，一足彘尾，见则其国大疫',
    '讙': '状如狸，一目而三尾，其音如夺百声',
    '孟极': '状如豹而文题白身',
    '乘黄': '状如狐，其背有角，乘之寿二千岁',
    '英招': '马身人面，虎文鸟翼，司帝之平圃'
    

"""

# =================配置区域=================
# 输入图片文件夹路径
IMAGE_DIR = os.path.join("code", "constellation_images")
OUTPUT_FILE = os.path.join("code", "constellation_data.json")
CONCURRENCY = int(os.getenv("Q3_CONCURRENCY", "10"))

# 初始化客户端
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# =========================================

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
    """调用 AI 分析单张图片"""
    global prompt_text
    filename = os.path.basename(image_path)
    print(f"正在处理: {filename} ...")

    base64_image = encode_image_to_base64(image_path)

    try:
        completion = client.chat.completions.create(
            model="qwen3-vl-flash", # 使用阿里最新的 Qwen3-VL 模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ],
            extra_body={'enable_thinking': True, "thinking_budget": 1024} # 开启思考模式
        )

        # 获取回复内容
        answer_content = completion.choices[0].message.content
        
        # 解析 JSON
        clean_json = extract_json_from_text(answer_content)
        data = json.loads(clean_json)
        
        # 将文件名加入数据中，方便对应
        data['filename'] = filename
        data['file_path'] = image_path
        
        print(f"  -> 成功: {data['chinese_name']} (评分: {data['coherence_score']})")
        return data

    except Exception as e:
        print(f"  -> 处理失败 {filename}: {e}")
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
