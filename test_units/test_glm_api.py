#"https://blog.csdn.net/l782060902/article/details/146409519" 

from zhipuai import ZhipuAI

# 初始化客户端
client = ZhipuAI(api_key="cbd4676a6366e635ae13e192d0a3cd02.9mqyd68GGiF2p5p8")  # 替换为实际API Key

# 发送请求
response = client.chat.completions.create(
    model="glm-4.5-flash",  # 模型名称
    messages=[
        {
            "role": "user",
            "content": "你好！请问你能帮我做什么？"
        }
    ]
)

# 正确获取JSON数据
data = response.model_dump_json()
# 解析JSON字符串为Python字典
import json
response_dict = json.loads(data)

# 访问choices字段
choices = response_dict["choices"]
first_choice = choices[0] if choices else None

output = first_choice["message"]["content"] if first_choice else "No response"

print(output)  # 输出结果
