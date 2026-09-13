import os
import json
import re
import glob
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# =================Configuration Area=================
# Input image folder path
IMAGE_DIR = os.path.join("code", "constellation_images")
OUTPUT_FILE = os.path.join("code", "constellation_data.json")
CONCURRENCY = int(os.getenv("Q3_CONCURRENCY", "5"))  

# AI Image Recognition API Configuration
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_ENDPOINT = "https://vision.aliyuncs.com/api/v1/image/recognize-shape"  # Shape recognition interface

# "Classic of Mountains and Seas" Monster Naming Library (Chinese names preserved)
ANIMAL_NAMES_BY_SHAPE = {
    'line': ['烛龙', '相柳', '巴蛇', '长蛇', '修蛇', '延维', '应龙', '螭龙', '腾蛇', '虺'],
    'triangle': ['凤凰', '毕方', '青鸾', '比翼鸟', '鹓鶵', '九凤', '玄鸟', '重明鸟', '青鸟', '三足乌'],
    'square': ['麒麟', '饕餮', '穷奇', '梼杌', '混沌', '白泽', '狻猊', '驳', '兕', '罴'],
    'arc': ['白虎', '陆吾', '驺吾', '天狗', '蛊雕', '猰貐', '梼杌', '狰', '狡', '獙獙'],
    'cluster': ['九婴', '开明兽', '九尾狐', '九头鸟', '相柳', '化蛇', '旋龟', '灌灌', '赤鱬', '夫诸'],
    'shape': ['狌狌', '鹿蜀', '狸力', '鯥', '颙', '狸狌', '猾褢', '彘', '何罗鱼', '当康'],
    'circle': ['玄武', '毕方', '旋龟', '商羊', '酸与', '跂踵', '讙', '孟极', '乘黄', '英招']
}

# "Classic of Mountains and Seas" Monster Characteristic Description Library (Chinese names preserved)
ANIMAL_DESCRIPTIONS = {
            # Linear/Long-shaped Monster Characteristics
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
            
            # Triangle/Bird-shaped Monster Characteristics
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
            
            # Quadrilateral/Giant Beast Monster Characteristics
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
            
            # Arc/Feline Wolf-shaped Monster Characteristics
            '白虎': '西方之神，主杀伐，性威猛',
            '陆吾': '人面虎身九尾，司天之九部及帝之囿时',
            '驺吾': '大若虎，五彩毕具，尾长于身，乘之日行千里',
            '天狗': '状如狸而白首，其音如榴榴，可以御凶',
            '蛊雕': '状如雕而有角，其音如婴儿之音，是食人',
            '猰貐': '龙首，居弱水中，食人',
            '狰': '状如赤豹，五尾一角，其音如击石',
            '狡': '状如犬而豹文，其角如牛，其音如吠犬',
            '獙獙': '状如狐而有翼，其音如鸿雁',
            
            # Dense Cluster/Multi-legged Insect-shaped Monster Characteristics
            '九婴': '九头水火之怪，能喷水吐火',
            '开明兽': '身大类虎而九首，皆人面，东向立昆仑上',
            '九尾狐': '状如狐而九尾，其音如婴儿，能食人',
            '九头鸟': '色赤，似鸭，九头皆鸣',
            '化蛇': '人面豺身，鸟翼蛇行，其音如叱呼',
            '旋龟': '鸟首虺尾，佩之不聋，可以为底',
            '灌灌': '状如鸠，其音若呵，佩之不惑',
            '赤鱬': '人面鱼身，其音如鸳鸯，食之不疥',
            '夫诸': '状如白鹿而四角，见则其邑大水',
            
            # Other Shape/General Monster Characteristics
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
            
            # Special Shape/Circular Multi-legged Monster Characteristics
            '玄武': '龟蛇合体，北方之神，主风雨',
            '商羊': '一足鸟，能招大雨',
            '酸与': '状如蛇，四翼六目三足，其鸣自詨',
            '跂踵': '状如鸮，一足彘尾，见则其国大疫',
            '讙': '状如狸，一目而三尾，其音如夺百声',
            '孟极': '状如豹而文题白身',
            '乘黄': '状如狐，其背有角，乘之寿二千岁',
            '英招': '马身人面，虎文鸟翼，司帝之平圃'          
}
    


# Mapping from AI returned shapes to our defined shapes (needs adjustment based on actual API returns)
AI_SHAPE_MAPPING = {
    'triangle': 'triangle',
    'triangle_shape': 'triangle',
    'line': 'line',
    'line_segment': 'line',
    'square': 'square',
    'rectangle': 'square',
    'circle': 'circle',
    'circular': 'circle',
    'arc': 'arc',
    'curved_line': 'arc',
    'cluster': 'cluster',
    'group': 'cluster',
    'other': 'shape'
}


name_counters = {shape_type: 0 for shape_type in ANIMAL_NAMES_BY_SHAPE.keys()}
used_names = set()
from threading import Lock
name_lock = Lock()  # Thread lock to ensure thread safety for name generation

# =========================================

def call_ai_shape_recognition(image_path: str) -> Optional[str]:
    """Call the AI image recognition API to get the main shape type in the image"""
    if not AI_API_KEY or AI_API_KEY == "your_api_key":
        print(f"Warning: AI API key not configured, skipping AI recognition for {os.path.basename(image_path)}")
        return None

    try:
        # Read image file
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Call API
        headers = {
            "Authorization": f"Bearer {AI_API_KEY}",
            "Content-Type": "application/octet-stream"
        }
        response = requests.post(AI_ENDPOINT, headers=headers, data=image_data, timeout=10)
        response.raise_for_status()  # Raise exception if request fails

        # Parse API response (assuming return format is {"shape": "triangle"})
        result = response.json()
        ai_shape = result.get("shape", "").lower()

        # Map to our defined shape type
        return AI_SHAPE_MAPPING.get(ai_shape, "shape")

    except Exception as e:
        print(f"AI recognition failed for {os.path.basename(image_path)}: {str(e)}")
        return None

def analyze_shape(image_path: str) -> str:
    """Analyze the shape type of the image (prioritize AI recognition, fall back to filename recognition)"""
    # 1. Prioritize calling AI recognition
    ai_shape = call_ai_shape_recognition(image_path)
    if ai_shape:
        print(f"AI recognized shape for {os.path.basename(image_path)}: {ai_shape}")
        return ai_shape

    # 2. Fallback: Analyze shape from filename
    filename = os.path.basename(image_path)
    for shape_type in ANIMAL_NAMES_BY_SHAPE.keys():
        if shape_type in filename.lower():
            print(f"Recognized shape from filename for {filename}: {shape_type}")
            return shape_type

    # 3. Finally, randomly select a shape type
    random_shape = random.choice(list(ANIMAL_NAMES_BY_SHAPE.keys()))
    print(f"Randomly assigned shape for {filename}: {random_shape}")
    return random_shape

def generate_constellation_name(shape_type: str) -> str:
    """Generate a unique constellation name based on shape type (thread-safe)"""
    global name_counters, used_names

    with name_lock:  # Add lock to avoid concurrent modification of shared variables
        # Get the list of available names for this shape type
        available_names = [name for name in ANIMAL_NAMES_BY_SHAPE[shape_type] 
                          if name not in used_names]

        if available_names:
            # Select the first available name
            name = available_names[0]
        else:
            # If names for this shape type are exhausted, select from other types
            for other_shape in ANIMAL_NAMES_BY_SHAPE.keys():
                if other_shape != shape_type:
                    available = [name for name in ANIMAL_NAMES_BY_SHAPE[other_shape] 
                               if name not in used_names]
                    if available:
                        name = available[0]
                        break
            else:
                # If all names are exhausted, use default naming (Chinese)
                name = f"星兽{len(used_names) + 1}"

        # Record the used name
        used_names.add(name)
        # Update counter
        name_counters[shape_type] += 1

    return name

def analyze_single_image(image_path: str):
    """Analyze a single image and generate constellation data"""
    filename = os.path.basename(image_path)
    print(f"Processing: {filename} ...")

    try:
        # 1. Analyze shape type (AI + fallback)
        shape_type = analyze_shape(image_path)
        
        # 2. Generate name (Chinese monster name)
        chinese_name = generate_constellation_name(shape_type)
        
        # 3. Get description (preserve original Chinese)
        visual_description = ANIMAL_DESCRIPTIONS.get(chinese_name)
        
        # 4. Generate concreteness score (set default based on shape type)
        shape_scores = {
            'line': 8, 'triangle': 7, 'square': 6,
            'arc': 7, 'cluster': 5, 'circle': 6, 'shape': 6
        }
        coherence_score = shape_scores.get(shape_type, 6)
        
        # Construct result data (keep Chinese names in output)
        data = {
            'filename': filename,
            'file_path': image_path,
            'chinese_name': chinese_name,
            'visual_description': visual_description,
            'coherence_score': coherence_score,
            'shape_type': shape_type,
            'recognition_method': 'AI' if call_ai_shape_recognition(image_path) else 'filename/random'
        }
        
        print(f"  -> Success: {chinese_name} - Shape: {shape_type}, Score: {coherence_score}")
        return data

    except Exception as e:
        print(f"  -> Failed to process {filename}: {str(e)}")
        return None

def main():
    # 1. Check if directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Folder {IMAGE_DIR} not found")
        return

    # 2. Get all png images
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    
    if not image_files:
        print(f"No PNG images found in {IMAGE_DIR}.")
        return

    print(f"Found {len(image_files)} images, starting batch data generation...\n")
    
    all_results = []

    # 3. Process images concurrently
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(analyze_single_image, img_path): img_path for img_path in image_files}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"Error processing future task: {str(e)}")

    # 4. Save results to JSON file (preserve Chinese characters)
    if all_results:
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            print(f"\nSUCCESS! All data saved to: {OUTPUT_FILE}")
            
            # Calculate average score
            avg_score = sum(item['coherence_score'] for item in all_results) / len(all_results)
            print(f"Average concreteness score of generated constellations: {avg_score:.2f}")
            
            # Output statistical information
            shape_counts = {}
            recognition_counts = {}
            for result in all_results:
                shape_type = result['shape_type']
                shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
                
                method = result['recognition_method']
                recognition_counts[method] = recognition_counts.get(method, 0) + 1
            
            print("\nShape distribution statistics:")
            for shape, count in shape_counts.items():
                print(f"  {shape}: {count} constellations")
            
            print("\nRecognition method statistics:")
            for method, count in recognition_counts.items():
                print(f"  {method}: {count} constellations")
                
        except IOError as e:
            print(f"Failed to save file: {str(e)}")
    else:
        print("No valid data generated.")

if __name__ == "__main__":
    main()
