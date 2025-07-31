import os
import pandas as pd
from tqdm import tqdm


# 读取目录中的所有 Parquet 文件
def read_multiple_parquet_files(file_path):
    # 获取目录下指定的 Parquet 文件
    df = pd.read_parquet(file_path)
    return df


def save_to_jsonl(df, file_path):
    df.to_json(file_path, orient='records', lines=True)
    print(f"数据已保存为{file_path}")


def build_dataset_format(row):
    new_message = []
    choices = ['A', 'B', 'C', 'D', 'E']
    question = row['question'] + '\n'
    for choice in choices:
        if not row[choice]:
            break
        question = question + choice + ':' + row[choice] + '\n'
    new_message.append({'role': 'user', 'content': question})
    answer = row['explanation'] + '\nSo the correct answer is ' + row['answer'] + '\n'
    new_message.append({'role': 'assistant', 'content': answer})
    row['messages'] = new_message
    row['meta_info'] = {
        "domain": "instruction",
        "source": "Telecom-QA-MultipleChoice/" + row["category"],
        "task": "chat",
        "ground_truth": row['answer']
    }
    return row


def build_test_format(row):
    choices = ['A', 'B', 'C', 'D', 'E']
    question = row['question'] + '\n'
    for choice in choices:
        if not row[choice]:
            break
        # 修正缩进，将该行移到循环内部
        question = question + choice + ': ' + row[choice] + '\n'
    row['source'] = "Telecom-QA-MultipleChoice/" + row["category"]
    row['question'] = question
    return row  


def jsonl_to_json(jsonl_file, json_file):
    import json
    with open(jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f]
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    file_path = 'E:/learning/llm_sft/Telecom-QA-MultipleChoice/data/test-00000-of-00001.parquet'
    # 设置存储路径
    save_path = 'E:/learning/llm_sft/Telecom-QA-MultipleChoice/data/test.jsonl'
    df = read_multiple_parquet_files(file_path)
    
    # 显示前几行数据
    print("原始数据:")
    print(df.head())
    
    # 应用函数到每一行并添加进度条
    # df = df.apply(lambda x: build_dataset_format(x), axis=1, result_type='expand')
    # df = df.progress_apply(lambda x: build_dataset_format(x), axis=1)
    # 若上面的 progress_apply 不可用，可使用以下方式
    # from tqdm import tqdm
    tqdm.pandas()
    df = df.progress_apply(build_dataset_format, axis=1)
    
    # 删除无用列
    df.drop(
        columns=[
            'question_id', 'question', 'A', 'B', 'C', 'D', 'E',
            'answer', 'explanation', 'category', 'tag'
        ],
        inplace=True
    )
    df["idx"] = df.index
    df.insert(0, 'idx', df.pop('idx'))
   
    # 显示抽取的数据
    print("抽取后的数据:")
    print(df.head())
    
    # 将抽取的数据保存为新的JSONL文件
    save_to_jsonl(df, save_path)