import random

# 主函数
def main():
    # 设置样本数量
    num_samples = 1000
    # 数据集文件路径
    file_path = "xxx/xxx/deepspeed70_gen_judge_telecom.jsonl"  # COT 数据集
    # 输出文件路径
    output_file = f"xxx/xxx/deepspeed70_gen_judge_telecom_{num_samples}.jsonl"  # 保存新地址

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 随机抽取样本
    sampled_elements = random.sample(lines, num_samples)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in sampled_elements:
            outfile.write(line)

    # 打印完成信息
    print(f"已从{file_path}随机抽取{num_samples}条数据，并保存到{output_file}.")

# 程序入口
if __name__ == "__main__":
    main()