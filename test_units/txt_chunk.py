import re
import os
from tqdm import tqdm


def read_text_file(file_path: str) -> list:
    """
    读取文本文件

    参数:
    file_path (str): 文件路径

    返回:
    str: 文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def save_chunks_to_files(chunks: list, output_dir: str) -> None:
    """
    将分割后的文本块保存到文件

    参数:
    chunks (list): 文本块列表
    output_dir (str): 输出目录路径
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将每个文本块保存为单独的文件
    for i, chunk in tqdm(enumerate(chunks)):
        chunk_file_path = os.path.join(output_dir, f"chunk_{i + 1}.txt")
        with open(chunk_file_path, 'w', encoding='utf-8') as file:
            file.write(chunk)
        print(f"已保存第 {i + 1} 个文本块到 {chunk_file_path}")


def get_chapter_indices(text_list: list) -> list:
    """
    获取章节索引

    参数:
    text_list (list): 文本行列表

    返回:
    list: 章节索引列表
    """
    chapter_indices = []
    for i, line in enumerate(text_list):
        if re.match(r'第(.*?)章\s*(.*)', line.strip()):
            chapter_indices.append(i)
    return chapter_indices

def clean_text(text_list: list) -> list:
    """
    清理文本，去除特定字符

    参数:
    text_list (list): 文本行列表

    返回:
    list: 清理后的文本行列表
    """
    return [line.replace('\u3000\u3000', '').replace('\n', '') for line in text_list]

def split_text_into_chunks(cleaned_text_list: list, chapter_indices: list) -> list:
    """
    将文本按照章节分割为多个块，确保最后一个章节不会漏掉。

    参数:
    cleaned_text_list (list): 清理后的文本列表，每个元素为一段文本。
    chapter_indices (list): 章节索引列表，每个元素为章节标题在cleaned_text_list中的索引。

    返回:
    list: 分割后的文本块列表。
    """
    chunk_list = []
    for i in range(len(chapter_indices)):
        if i != len(chapter_indices) - 1:
            # 章节标题 + \n + 内容
            chunk_list.append(''.join(cleaned_text_list[chapter_indices[i]]) + '\n' + ''.join(cleaned_text_list[chapter_indices[i] + 1:chapter_indices[i + 1]]))
        else:
            chunk_list.append(''.join(cleaned_text_list[chapter_indices[i]]) + '\n' + ''.join(cleaned_text_list[chapter_indices[i] + 1:]))
    
    return chunk_list

if __name__ == '__main__':

    # 设置输入和输出路径
    input_file_path = r'我的美女老师.txt'  # 替换为你的长文本文件路径
    output_dir = './saveChunk/'  # 替换为你希望保存文本块的目录路径

    # 读取长文本
    long_text = read_text_file(input_file_path)

    # 对列表中的每一项进行字符剔除操作
    cleaned_text_list = clean_text(long_text)

    # 获取章节索引 
    chapter_indices = get_chapter_indices(cleaned_text_list)

    # 打印章节索引和章节标题
    print("章节索引和标题:")
    for i in range(len(chapter_indices)):
        print(f"第 {i+1} 章: {cleaned_text_list[chapter_indices[i]].strip()}")

    # 每章节存为一个chunk 确保最后一个章节不会漏掉
    chunk_list = split_text_into_chunks(cleaned_text_list, chapter_indices)

    # 保存分割后的文本块到指定目录
    save_chunks_to_files(chunk_list, output_dir)

    print("分割完成！")

    