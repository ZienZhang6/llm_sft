import requests
import json

def chat_with_qwen(message, model="qwen3:8b", host="http://localhost:11434"):
    """
    与Ollama部署的qwen3-8b模型进行对话
    
    参数:
        message: 发送给模型的消息
        model: 模型名称，默认为"qwen3:8b"
        host: Ollama服务地址，默认为本地地址
    
    返回:
        模型的回复内容
    """
    url = f"{host}/api/chat"
    
    # 构建请求数据
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": False  # 非流式响应
    }
    
    try:
        # 发送POST请求
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        
        # 检查响应状态
        if response.status_code == 200:
            # 解析响应内容
            result = response.json()
            return result["message"]["content"]
        else:
            return f"请求失败，状态码: {response.status_code}"
    
    except Exception as e:
        return f"发生错误: {str(e)}"

def multi_round_chat(initial_question, follow_up_question, model="qwen3:8b", host="http://localhost:11434"):
    """
    进行多轮对话
    
    参数:
        initial_question: 初始问题
        follow_up_question: 后续问题
        model: 模型名称，默认为"qwen3:8b"
        host: Ollama服务地址，默认为本地地址
    
    返回:
        模型对后续问题的回复内容
    """
    messages = [
        {"role": "user", "content": initial_question},
        {"role": "assistant", "content": chat_with_qwen(initial_question, model, host)},
        {"role": "user", "content": follow_up_question}
    ]
    
    url = f"{host}/api/chat"
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        if response.status_code == 200:
            result = response.json()
            return result["message"]["content"]
        else:
            return f"请求失败，状态码: {response.status_code}"
    except Exception as e:
        return f"发生错误: {str(e)}"


# 示例用法
if __name__ == "__main__":
    # 简单对话示例
    user_input = "你好，请介绍一下你自己"
    response = chat_with_qwen(user_input)
    print(f"用户: {user_input}")
    print(f"Qwen3-8B: {response}")
    
    
    # 多轮对话示例
    print("\n多轮对话:")
    initial_question = "什么是人工智能？"
    follow_up_question = "它有哪些应用领域？"
    multi_round_response = multi_round_chat(initial_question, follow_up_question)
    print(f"用户: {follow_up_question}")
    print(f"Qwen3-8B: {multi_round_response}")
