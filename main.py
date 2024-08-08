import pygame
import sys
import pymunk.pygame_util
import numpy as np
from scipy.interpolate import splprep, splev
from openai import OpenAI
import requests
import base64
import numpy as np
import re
import ast
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from itertools import permutations
from sklearn.cluster import DBSCAN
import math

def ƒ_space(gravity):
    """
    build a pymunk space with gravity
    """
    space = pymunk.Space()
    space.gravity = gravity
    return space

def add_ball(space, position, radius, mass, velocity):
    """
    add a ball to the space
    """
    inertia = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, inertia)
    body.position = position
    body.velocity = velocity
    shape = pymunk.Circle(body, radius)
    shape.friction = 0.7
    space.add(body, shape)

def generate_spline(points, num_points=100):
    """
    use spline interpolation to generate a smooth curve
    """
    points = np.array(points)
    tck, _ = splprep(points.T, s=0)
    u = np.linspace(0, 1, num_points)
    spline_points = splev(u, tck)
    return list(zip(spline_points[0], spline_points[1]))

def add_curved_floor(space, points, thickness, frictions):
    """
    add a curved floor to the space
    """
    spline_points = generate_spline(points)
    for i in range(len(spline_points) - 1):
        segment = pymunk.Segment(space.static_body, spline_points[i], spline_points[i + 1], thickness)
        segment.friction = frictions
        space.add(segment)

def add_polygon(space, position, vertices, mass, velocity):
    """
    add a polygon to the space
    """
    inertia = pymunk.moment_for_poly(mass, vertices)
    body = pymunk.Body(mass, inertia)
    body.position = position
    body.velocity = velocity
    shape = pymunk.Poly(body, vertices)
    shape.friction = 0.7
    space.add(body, shape)

def draw_space(space, screen):
    """
    build a pymunk space with gravity
    """
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    space.debug_draw(draw_options)

def parse_functions(response):
    # use regular expression to split the response into functions
    pattern = r"(?=add_\w+\()"
    functions = re.split(pattern, response)
    # remove leading and trailing whitespaces
    functions = [func.strip() for func in functions if func.strip()]
    return functions

def find_similar_and_average(arr, eps = 0.5, min_samples = 2):
    filtered_arr = [x for x in arr if x is not None]
    
    if not filtered_arr:
        return None
    
    filtered_arr = np.array(filtered_arr).reshape(-1, 1)
    
    # standardize the data
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(filtered_arr)
    
    # use DBSCAN to cluster the data
    dbscan = DBSCAN(eps = eps, min_samples = min_samples)
    clusters = dbscan.fit_predict(arr_scaled)
    
    # if all points are noise points, return the mean of the original array
    if np.all(clusters == -1):
        return np.mean(filtered_arr)
    
    # 
    unique_clusters, cluster_sizes = np.unique(clusters[clusters != -1], return_counts = True)
    largest_cluster = unique_clusters[np.argmax(cluster_sizes)]
    
    # 返回最大聚类的平均值
    cluster_points = filtered_arr[clusters == largest_cluster]
    return np.mean(cluster_points)
    
def extract_parameters(function_str_list):
    processed_params = {}
    function_counts = {}
    
    for function_str in function_str_list:
        # 提取完整的函数名（
        function_name_match = re.match(r"(add_\w+)", function_str)
        if function_name_match:
            function_name = function_name_match.group(1)
        else:
            function_name = "unknown"
        
        # 计算函数名出现的次数
        if function_name in function_counts:
            function_counts[function_name] += 1
        else:
            function_counts[function_name] = 1
        
        # 如果函数名出现超过一次，添加计数后缀
        if function_counts[function_name] > 1:
            function_name_with_suffix = f"{function_name}{function_counts[function_name]}"
        else:
            function_name_with_suffix = function_name
        
        # 移除 "add_xxx(space, " 并保留剩余的部分
        params_str = re.sub(r"add_\w+\(space,\s*", "", function_str)
        # 移除最后的右括号（如果存在）
        params_str = params_str.rstrip(")")
        
        # 将参数字符串添加到字典中
        processed_params[function_name_with_suffix] = params_str.strip()
    
    return processed_params

def extract_number(parameters):
    result = {}
    
    # 获取所有可能的键
    all_keys = set()
    for param_dict in parameters:
        all_keys.update(param_dict.keys())
        
    for key in all_keys:
        # 从每个字典中提取当前键的值，如果不存在则用空字符串替代
        values = [param_dict.get(key, '') for param_dict in parameters]
        
        # 使用正则表达式提取每个字符串的所有数字
        numbers = [re.findall(r'-?\d+(?:\.\d+)?', value) for value in values]
        
        # 确定最长的数字列表的长度
        max_length = max(len(num_list) for num_list in numbers)
        
        # 对每个位置进行迭代
        key_result = []
        for i in range(max_length):
            # 提取每个字符串中相同位置的数字（如果有的话）
            position_numbers = [float(num_list[i]) if i < len(num_list) else None for num_list in numbers]
            key_result.append(position_numbers)
        
        result[key] = key_result
    
    # 将结果转换为 list[dict] 格式
    return [{k: v for k, v in result.items()}]

def split_function_prototype(prototype_str):
    # 定义正则表达式模式来匹配 "float" 和括号、方括号、花括号、逗号
    pattern = r'float|\(|\)|\[|\]|\{|\}|,|\s+|[^,()\[\]\{\}\s]+'
    
    # 使用 re.findall 找到所有匹配的部分
    parts = re.findall(pattern, prototype_str)
    
    # 处理匹配结果，移除空字符串，并将逗号替换为", "
    processed_parts = []
    for part in parts:
        if part.strip():
            if part == ',':
                processed_parts.append(', ')
            else:
                processed_parts.append(part)
    
    return processed_parts

def add_function_name(parameters, function_prototype):
    def replace_floats_str(template_str, values):
        str_list = split_function_prototype(template_str)
        #print("str_list: ", str_list)
        value_index = 0
        
        for i, s in enumerate(str_list):
            if s == "float":
                
                str_list[i] = str(values[value_index])
                value_index += 1
        return ''.join(str_list)
    
    
    result = ""
    
    for key, values in parameters.items():
        # Remove the numbers at the end of the key to find the base function name
        base_key = re.sub(r'\d*$', '', key)
        
        # Fetch the prototype for the base function name
        prototype = function_prototype.get(base_key, "")
        
        if prototype:
            print("prototype: ", prototype)
            result += replace_floats_str(prototype, values) + "\n"
    
    return result.strip()  # Remove the trailing newline


def get_base_function_name(func_name):
    # 提取函数名中的字母部分作为基函数名
    match = re.match(r"([a-zA-Z_]+)", func_name)
    return match.group(1) if match else func_name

def find_similar_functions(func_dict):
    # 创建一个字典，基函数名作为键，相似函数名列表作为值
    similar_functions = defaultdict(list)

    for func_name in func_dict.keys():
        base_name = get_base_function_name(func_name)
        similar_functions[base_name].append(func_name)
    
    # 过滤掉只出现一次的基函数名，并构建返回结果
    result = [(base_name, group) for base_name, group in similar_functions.items() if len(group) > 1]
    
    return result

def update_dict_keys(param, keys_to_change):
    # 创建一个键映射字典
    key_mapping = dict(keys_to_change)
    
    # 创建一个新的字典来存储更新后的键值对
    updated_param = {}
    
    for old_key, value in param.items():
        # 如果旧键在映射中，使用新键；否则保持原样
        new_key = key_mapping.get(old_key, old_key)
        updated_param[new_key] = value
    
    # 返回更新后的字典
    return updated_param


def find_most_similar_functions(parameters):
    def get_base_key(key):
        return re.sub(r'\d+$', '', key)

    max_similar_funcs = 0
    max_similar_funcs_index = 0

    for i, param in enumerate(parameters):
        base_keys = defaultdict(list)
        for key in param:
            base_key = get_base_key(key)
            base_keys[base_key].append(key)
        
        similar_funcs_count = sum(len(keys) for keys in base_keys.values() if len(keys) > 1)
        
        if similar_funcs_count > max_similar_funcs:
            max_similar_funcs = similar_funcs_count
            max_similar_funcs_index = i

    return max_similar_funcs_index

def extract_numbers_from_string(s):
    # 使用正则表达式匹配所有数字，包括整数和小数
    numbers = re.findall(r'\d+', s)
    # 将匹配到的数字字符串转换为整数
    numbers = [int(num) for num in numbers]
    return numbers

def adjustment_function_name(parameters):
    def extract_numbers(s):
        return [float(x) for x in re.findall(r'-?\d+\.?\d*', s)]

    
    def cosine_similarity(list1, list2):
        min_length = min(len(list1), len(list2))
        
        list1 = list1[:min_length]
        list2 = list2[:min_length]
        dot_product = sum(a * b for a, b in zip(list1, list2))
        magnitude1 = math.sqrt(sum(a ** 2 for a in list1))
        magnitude2 = math.sqrt(sum(b ** 2 for b in list2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0  

        similarity = dot_product / (magnitude1 * magnitude2)
        return round(similarity, 8)

    def get_base_key(key):
        return key.rstrip('0123456789')
    
    max_similar_funcs = defaultdict(list)
    
    for i, param in enumerate(parameters):
        base_keys = defaultdict(list)
        for key in param:
            base_key = get_base_key(key)
            base_keys[base_key].append(key)
    max_similar_funcs_index = find_most_similar_functions(parameters)

    parameters[0], parameters[max_similar_funcs_index] = parameters[max_similar_funcs_index], parameters[0]
    base_name, similar_functions = find_similar_functions(parameters[0])[0]
    

    #print(similar_functions)
    def processfunction(param):
        similarity_value = []
        for key, value in param.items():
            if re.sub(r'\d+$', '', key) == base_name:
                similarity_dict = {}
                for key in similar_functions:
                    similarity_dict[key] = 0
                for function in similar_functions:
                    similarity_dict[function] = cosine_similarity(extract_numbers_from_string(value), extract_numbers_from_string(parameters[0][function]))
                similarity_value.append(similarity_dict)

        #print(similarity_value)
        keys_to_change = []
        i = 0

        for key in param.keys():
            if re.sub(r'\d+$', '', key) == base_name:
                function_name = max(similarity_value[i], key=similarity_value[i].get)
                keys_to_change.append((key, function_name))
                i += 1

        updated_param = update_dict_keys(param, keys_to_change)
        return updated_param     
    
    new_parameters = []
    
    for i, param in enumerate(parameters):   
        new_parameters.append(processfunction(param))
    return new_parameters


def main():
    # OpenAI API Key
    #api_key = "<insert your api_key>"

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Path to your image
    image_path = "/Users/dong/Desktop/video/code/frame_00001.jpg"

    # Getting the base64 string
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    def call_chatgpt(prompt):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        content = response.json().get("choices")[0].get("message").get("content")
        print("the response of the chatgpt: ")
        print(content)
        return content
    
    initial_prompt = (
        "Recognize the object parameters in the uploaded image, and then fill the parameters in the following functions(make sure recognize the right number of parameters for each function): "
        "add_ball(space, position, radius, mass, velocity) "
        "add_curved_floor(space, points, thickness, frictions) "
        "add_polygon(space, position, vertices, mass, velocity). The output should only be function, no other information. "
        "In terms of position: The image is 1280x720. Establish a coordinate system with the top-left corner of the image as the origin (0,0). The positive x-axis extends to the right, and the positive y-axis extends downward. The unit of measurement for the coordinate points is two pixels."
        "(For example: add_curved_floor(space, [(50, 500), (150, 400), (250, 350), (350, 450), (450, 500)], 5, 0.5)\nadd_ball(space, (100, 50), 25, 1, (200, 200))\nadd_polygon(space, (200, 300), [(0, 0), (50, 0), (50, 50), (0, 50)], 10, (0, 0)))\n"
    )
    
    def newprompt(last_prompt, index):
        function_prototype = {"add_ball": "add_ball(space, (float, float), float, float, (float, float))",
                              "add_curved_floor": "add_curved_floor(space, [(float, float), (float, float), (float, float), (float, float), (float, float)], float, float)",
                              "add_polygon": "add_polygon(space, (float, float), [(float, float), (float, float), (float, float), (float, float)], float, (float, float))"}
        responses = [call_chatgpt(last_prompt) for _ in range(3)] # 一次生成50个
        parameters = []
        for response in responses:
            parse_function = parse_functions(response)
            extract_parameter = extract_parameters(parse_function)
            parameters.append(extract_parameter)
        parameters = adjustment_function_name(parameters)
        processed_parameters = extract_number(parameters)
        potentialanswer = []
        m = 0


        for key, value_list in processed_parameters[0].items():
            for i, sublist in enumerate(value_list):
                value = find_similar_and_average(sublist)
                processed_parameters[0][key][i] = value
            
        potentialanswer = processed_parameters
        new_parameters = add_function_name(potentialanswer[0], function_prototype)
    
        if "(Hint)Their parameters maybe:" in last_prompt:
            new_prompt = re.sub(r"(Hint)Their parameters maybe:.*", f"(Hint)Their parameters maybe:\n{new_parameters}", last_prompt, flags = re.DOTALL)
        else:
            new_prompt = last_prompt + f"\n (Hint)Their parameters maybe: \n{new_parameters}"
        
        return new_prompt

    last_prompt = initial_prompt
    nums = 3
    for _ in range(nums):
        last_prompt = newprompt(last_prompt, _)
        print(f"Iteration {_+1}:")
        print(last_prompt)
        print("-" * 50)

    latest_prompt = last_prompt
    print(latest_prompt)
    
    # Initialize the pygame
    pygame.init()
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((1000, 1000))
    
    space = ƒ_space((0,0))
    
    start_index = last_prompt.find("maybe:") + len("maybe:")
    response_by_llm = last_prompt[start_index:].strip()
    
    """
    the content of response_by_llm:
    add_curved_floor(space, [(50, 500), (150, 400), (250, 350), (350, 450), (450, 500)], 5, 0.5)
    add_ball(space, (100, 50), 25, 1, (200, 200))
    add_polygon(space, (200, 300), [(0, 0), (50, 0), (50, 50), (0, 50)], 10, (0, 0))
    add_ball(space, (100, 50), 25, 1, (200, 200))
    """
    exec(response_by_llm)
    
    # 主循环
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((255, 255, 255))  # 填充背景颜色
        space.step(1 / 50.0)  # 模拟一个时间步长
        draw_space(space, screen)  # 绘制物理空间

        pygame.display.flip()
        clock.tick(50)  # 控制帧率

if __name__ == '__main__':
    main()
