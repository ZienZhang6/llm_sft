import json
import re
import threading
import requests
import argparse
import os
import shutil
import logging
from dataclasses import dataclass, field
from typing import Dict
from tqdm import tqdm

def setup_logging(log_file: str):
    """
    配置日志记录,同时将日志输出到控制台和文件。
    :param log_file: 日志文件路径
    """
    # 创建一个 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

    # 创建一个文件处理器,用于将日志写入文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别

    # 创建一个控制台处理器,用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 设置控制台处理器的日志级别

    # 创建一个日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


@dataclass
class Config:
    model_name: str = "qwen3-0.6b"
    url: str = "http://localhost:11434/api/chat"
    headers: Dict[str, str] = field(
        default_factory=lambda: {
            "Content-Type": "application/json"
        }
    )

    input_path: str = "/mnt/metis/official_metis_data/sft_data/TelecomQA/data/train.jsonl"
    output_gen_path: str = "output/deepspeed70_gen_telecom.jsonl"
    output_judge_path: str = "output/deepspeed70_gen_judge_telecom.jsonl"

class PromptUtils:
    @staticmethod
    def get_prompt(prompt_type: str) -> str:
        prompts = {
            "generate_cot": "Please reason step by step, put your final answer within boxed {},The final answer to the question must follow this format:\n\n Therefore, the final answer is \\boxed{}.\n\n**Final Answer**\n\\boxed{}.\n\n",
            "judge": "You're a strict teacher, and then you'll receive a question, a reference answer, and a student's answer. You need to judge the correctness of the answer. You don't need to consider the realistic correctness of the reference answer. You only need to compare the consistency between the reference answer and the answer answered by the student. If they are consistent, you only need to answer YES. Otherwise, you need to answer NO and do not need to analyze it."
        }
        return prompts.get(prompt_type, "")

def get_message(data) -> Dict:
    idx = data.get('idx')
    meta_info = data["meta_info"]

    for msgItem in data.get('messages', []):
        if msgItem.get('role') == 'assistant':
            answer = msgItem.get('content')
        elif msgItem.get('role') == 'user':
            question = msgItem.get('content')
    return {"idx": idx, "meta_info": meta_info, "answer": answer, "question": question}

def processed_from_model(type: str, data: Dict):
    if type == 'generator':
        messages = [
            {"role": "user", "content": PromptUtils.get_prompt('generate_cot') + data['question']}
        ]
    elif type == 'judger':
        content = f"Question: {data['question']}\nGolden Answer: {data['meta_info']['ground_truth']}\nModel Answer: {data['answer']}"
        messages = [
            {"role": "system", "content": PromptUtils.get_prompt('judge')},
            {"role": "user", "content": content}
        ]
        # config = Config()
    requestBody = {
        "model": config.model_name,
        "messages": messages,
        "stream": False  # 非流式响应

        # "max_tokens": 32768,
        # "temperature": 0.6,
        # "repetition_penalty": 1.05
    }
    response = requests.post(config.url, headers=config.headers, json=requestBody)
    if response.status_code == 200:
       
        # for line in response.text.splitlines():    # 多线程结果分割  尝试
        #     if line.strip():  # 跳过空行
        #         data = json.loads(line)
                
        response_data = response.json()
        model_answer = response_data['message']['content']  # 单线程
        # model_answer = response_data['choices'][0]['message']['content']  # 这个['choices'][0] 像是多线程的结果
        return model_answer
    logger.error(f'Idx: {data["idx"]}, Error: {response.status_code},{response.text}')
    return None

def split_jsonl_to_subfile(input_file, output_prefix, num_files):
    """分割JSONL文件并返回生成的文件列表"""
    # 确保临时目录存在
    os.makedirs("./tmp")  # 如有必要, exist_ok=True
    with open(input_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    base, remainder = divmod(total_lines, num_files)
    
    file_list = []
    current_line = 0
    for i in range(num_files):
        # 计算每个文件包含的行数
        num_lines = base + (1 if i < remainder else 0)

        # 生成输出文件名
        output_file = f"tmp/{output_prefix}_{i + 1}.jsonl"
        file_list.append(output_file)
        # 写入分割后的文件
        with open(output_file, 'w') as f_out:
            end_line = current_line + num_lines
            f_out.writelines(lines[current_line:end_line])
        current_line = end_line
    return file_list


def filter(idx, answer):
    if not answer:
        logger.warning(f"Idx: {idx} failed, is None!")
        return False, None
    contentList = answer.split("</think>")
    if len(contentList) < 2:
        logger.warning(f"Idx: {idx} failed, haven't </think>!")
        return False, None
    elif "\\boxed{" not in contentList[0] and "\\boxed{" not in contentList[1]:
        logger.warning(f"Idx: {idx} failed, haven't boxed!")
        return False, None
    elif not re.search("answer", contentList[0], re.IGNORECASE) and not re.search("answer", contentList[1], re.IGNORECASE):
        logger.warning(f"Idx: {idx} failed, haven't answer!")
        return False, None
    
    final_answer = answer.replace("<think>", "<|begin_of_thought|>").replace("</think>", "<|end_of_thought|>\n\n<|begin_of_solution|>") + "\n<|end_of_solution|>"
    return True, final_answer


def save_to_jsonl(idx, question, final_answer, meta_info):
    dictData = {
        "idx": idx,
        "messages": [
            {
                "content": "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing,and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section,detail your reasoning process using the specified format: <|begin_of_thought|>{thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detailnecessary steps needed to reach the conclusion, formatted as follows:<|begin_of_solution|> {final formatted, precise, and clear solution}<|end_of_solution|> Now, try to solve the following question through the above guidelines:",
                "role": "system"
            },
            {
                "content": question,
                "role": "user"
            },
            {
                "content": final_answer,
                "role": "assistant"
            }
        ],
        "meta_info": meta_info
    }
    
    with open(config.output_gen_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(dictData, ensure_ascii=False) + "\n")

def gen_cot(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, data in tqdm(enumerate(file, start=1)):
            try:
                data = get_message(json.loads(data.strip()))
                answer = processed_from_model("generator", data)

                if answer is not None:
                    isValid, answer = filter(data['idx'], answer)
                    if isValid is True:
                        # print(f"Idx {data['idx']}: Answer:{answer}")
                        save_to_jsonl(data['idx'], data['question'], answer, data['meta_info'])

            except json.JSONDecodeError as e:
                logger.error(f'parse {line_number} err: {e}')


def multi_thread_gen_cot(config):
    split_tmp_folder = "tmp"
    
    if not os.path.exists(split_tmp_folder):
        logger.info(f'Start split files...')
        subfile_list = split_jsonl_to_subfile(config.input_path, "output", 64)
        logger.info(f'File split to: {subfile_list}')
    
    logger.info(f'Start generate cot...')
    threads = []
    for root, dirs, files in os.walk(split_tmp_folder):
        for file in files:
            file_path = os.path.join(root, file)
            logger.info(f'Start generate cot for file: {file_path}')
            gen_cot(file_path)
            logger.info(f'Finished generate cot for file: {file_path}')
    #         thread = threading.Thread(target=gen_cot, args=(file_path,))
    #         thread.start()
    #         threads.append(thread)

    # # 等待所有线程完成
    # for thread in threads:
    #     thread.join()

    logger.info(f"Finished generate cot!")

    # 删除子文件
    shutil.rmtree(split_tmp_folder)
    logger.info(f"Deleted split files!")


def judge_with_ground_truth(config):
    wrong_answer_idx = []
    with open(config.output_gen_path, 'r', encoding='utf-8') as infile, open(config.output_judge_path, 'w', encoding='utf-8') as outfile:
        for line_number, line in enumerate(infile, start=1):
            try:
                raw = json.loads(line.strip())
                data = get_message(raw)
                model_answer = processed_from_model("judger", data)
                if model_answer is not None and model_answer == 'YES':
                    outfile.write(json.dumps(raw, ensure_ascii=False) + '\n')
                else:
                    wrong_answer_idx.append(data['idx'])
                    error_detail = {
                        "Idx": data['idx'],
                        "judge": model_answer,
                        "ground_truth": data['meta_info']['ground_truth'],
                        "model_answer": data['answer']
                    }
                    logger.warning(f"Wrong answer: {error_detail}")
            except json.JSONDecodeError as e:
                logger.error(f"Parse {line_number} err: {e}")
    logger.info("Done!")
    return wrong_answer_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str,default='generator', help='generator or judger')
    args = parser.parse_args()
    config = Config()
    logger = setup_logging("error.log")
    if args.type == 'generator':
        config.model_name = "qwen3:0.6b"
        config.url = "http://localhost:11434/api/chat"  #
        config.input_path = "E:/learning/llm_sft/Telecom-QA-MultipleChoice/data/train.jsonl"  # 输入文件路径
        config.output_gen_path = "E:/learning/llm_sft/Telecom-QA-MultipleChoice/data/output/qwen3_0.6b_gen_telecom.jsonl"
        config.output_judge_path = "E:/learning/llm_sft/Telecom-QA-MultipleChoice/data/output/qwen3_0.6b_gen_judge_telecom.jsonl"
        
        # 如果不存在输出目录，则创建
        if not os.path.exists(os.path.dirname(config.output_gen_path)):
            os.makedirs(os.path.dirname(config.output_gen_path))

        multi_thread_gen_cot(config)
        
    if args.type == 'judger':
        config.model_name = "qwen2.5:7b"
        config.url = "http://xxx.xxx.xxx.xxx:xxx/v1/chat/completions"  # mindie服务ip:端口号
        wrong_answer_idx = judge_with_ground_truth(config)
        logger.warning(f"Wrong Answer Idx: {wrong_answer_idx}")