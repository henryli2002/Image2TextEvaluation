import json

# 读取JSON文件的函数
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 读取两个JSON文件
json1 = read_json('res.json')
json2 = read_json('res_add.json')

# 存储组合后的结果
combined_values = {}

# 遍历第一个JSON文件的键
for key in json1:
    if key in json2:
        # 将两个文件中相同键的值组合在一起
        combined_values[key] = json1[key] + json2[key]

# 保存组合后的结果到新的JSON文件
with open('combined_input.json', 'w', encoding='utf-8') as file:
    json.dump(combined_values, file, ensure_ascii=False, indent=2)

print("Combined JSON saved as combined_output.json")