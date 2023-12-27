import requests
import json
import os
import datetime

API_KEY = 'jIHlGsMYHp4j1MrMZGNmYbCL'
SECRET_KEY = 'GzG1o4HC6G0qDVxPKcn9Zl4pv20j7CGA'

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def QianFan(url, inputs):
    responses = {}  # 创建一个字典来存储输入和相应的响应

    for key, user_input in inputs.items():  # inputs是一个字典
        request = {
            "messages": [
            {
                "role": "user",
                "content": "I will give you a sentence, where the first part is a summary of the background and the second part is information about a person's outfit. Please focus on the background information from the first part and provide an overall background description. "
            },
            {
                "role": "assistant",
                "content": "Of course, please provide the sentence, and I will only output a sentence like: 'The background is'. "
            },
            {
                "role": "user",
                "content": "a people in front of a bed. a pair of jeans"
            },
            {
                "role": "assistant",
                "content": "The backgroud is a homely bedroom."
            },
            {
                "role": "user",
                "content": "a people in front of a mirror. a man in a blue shorts and a white shirt"
            },
            {
                "role": "assistant",
                "content": "The backgroud is a mirror."
            },
            ]
        }#添加一些前置词
        request["messages"].append({"role": "user", "content": f"Describe the setting of the following scene, focusing solely on the background without including any details about the person:{user_input}"})


        try:
            response = requests.request("POST", url, headers=headers, data=json.dumps(request))
            text = response.text
            data = json.loads(text)
            model_response = data['result']
            print("\n回答：\n", model_response, '\n')
            # 根据句号分割文本
            sentences = model_response.split(". ")

            # 获取第一个句子
            first_sentence = sentences[0] + "."
            responses[key] = first_sentence  # 将响应存储在字典中

        except Exception as e:
            print(f"QianFan 接口调用出错: {e}")

    # 保存或处理responses字典
    return responses

def main():
    if not os.path.exists('./Amadeus/history'):
        os.makedirs('./Amadeus/history')

    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=" + get_access_token()

    inputs = read_json('res_new.json') 
    responses = QianFan(url, inputs)

    # 可以选择保存responses字典
    with open('./Amadeus/history/responses.json', 'w', encoding='utf-8') as file:
        json.dump(responses, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
