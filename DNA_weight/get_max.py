# import os
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# # 初始化染色体最大结束位置字典
# chromosome_basepairs = {
#     "1": 249_000_000,
#     "2": 243_000_000,
#     "3": 198_260_000,
#     "4": 191_000_000,
#     "5": 182_000_000,
#     "6": 171_000_000,
#     "7": 160_000_000,
#     "8": 146_000_000,
#     "9": 141_000_000,
#     "10": 136_000_000,
#     "11": 135_090_000,
#     "12": 134_000_000,
#     "13": 115_000_000,
#     "14": 107_000_000,
#     "15": 102_000_000,
#     "16": 90_250_000,
#     "17": 84_000_000,
#     "18": 80_290_000,
#     "19": 59_000_000,
#     "20": 64_350_000,
#     "21": 47_000_000,
#     "22": 51_000_000,
#     "X": 156_050_000,
#     "Y": 57_000_000
# }

# # 计算文件中染色体的最大结束位置
# def process_file(file_path):
#     print(file_path)
#     chrom_max_end = chromosome_basepairs
#     data = pd.read_csv(file_path, sep='\t', header=None, names=['chromosome', 'start', 'end', 'weight'])
#     for key in chrom_max_end.keys():
#         chr1_data = data[data['chromosome'].str[3:]== key]
#         max_end = chr1_data['end'].max()
#         print(key,': ',max_end,' ',file_path)
#         if max_end > chrom_max_end[key]:
#             chrom_max_end[key]=max_end
#             print('update',key,': ',max_end)

#     return chrom_max_end

# # 合并多个文件的最大结束位置
# def merge_max_end_positions(results):
#     final_max_end = chromosome_basepairs
    
#     for result in results:
#         for chrom, max_end in result.items():
#             final_max_end[chrom] = max(final_max_end[chrom], max_end)
    
#     return final_max_end


# def get_max_end_positions_parallel(file_dir, files, num_threads=30):
#     # 使用 tqdm 显示进度条
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         # 通过线程池并行处理文件
#         file_paths = [os.path.join(file_dir, file) for file in files]
        
#         # 使用 tqdm 包装 executor.map，以显示进度条
#         results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths), desc="Processing Files"))
    
#     # 合并所有文件的结果
#     return merge_max_end_positions(results)

# # 使用示例
# file_dir = '/mnt/data/baishuhang/DNA/output'  # 你的文件目录路径
# files = os.listdir(file_dir)  # 获取文件夹中的所有文件
# max_end_positions = get_max_end_positions_parallel(file_dir, files, num_threads=8)

# # 输出最大结束位置
# for chrom, max_end in max_end_positions.items():
#     print(f"Chromosome {chrom}: max end position = {max_end}")











import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 初始化染色体最大结束位置字典
chromosome_basepairs = {
    "1": 249_000_000,
    "2": 243_000_000,
    "3": 198_260_000,
    "4": 191_000_000,
    "5": 182_000_000,
    "6": 171_000_000,
    "7": 160_000_000,
    "8": 146_000_000,
    "9": 141_000_000,
    "10": 136_000_000,
    "11": 135_090_000,
    "12": 134_000_000,
    "13": 115_000_000,
    "14": 107_000_000,
    "15": 102_000_000,
    "16": 90_250_000,
    "17": 84_000_000,
    "18": 80_290_000,
    "19": 59_000_000,
    "20": 64_350_000,
    "21": 47_000_000,
    "22": 51_000_000,
    "X": 156_050_000,
    "Y": 57_000_000
}

# 计算文件中染色体的最大结束位置
def process_file(file_path):
    chrom_max_end = chromosome_basepairs.copy()
    data = pd.read_csv(file_path, sep='\t', header=None, names=['chromosome', 'start', 'end', 'weight'])
    for key in chrom_max_end.keys():
        chr_data = data[data['chromosome'].str[3:] == key]
        max_end = chr_data['end'].max()
        if max_end > chrom_max_end[key]:
            chrom_max_end[key] = max_end
    save_to_json(f"./result/{os.path.basename(file_path)}.json", chrom_max_end)
    return file_path, chrom_max_end


# 合并多个文件的最大结束位置
def merge_max_end_positions(results):
    final_max_end = chromosome_basepairs.copy()
    for _, result in results:
        for chrom, max_end in result.items():
            final_max_end[chrom] = max(final_max_end[chrom], max_end)
    return final_max_end


def get_processed_files(log_file):
    """从日志文件中加载已处理文件"""
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f)


def save_processed_file(log_file, file_path):
    """将处理完成的文件名保存到日志文件"""
    with open(log_file, 'a') as f:
        f.write(file_path + '\n')

import json
def save_to_json(file_path, data):
    """将数据保存到 JSON 文件中"""
    with open(file_path, "a") as f:  # 'a' 模式追加写入
        json.dump(data, f)
        f.write("\n")  # 每个 JSON 对象单独一行
# import json
# with open("results.json", "r") as f:
#     data = [json.loads(line) for line in f]
# print(data)



def get_max_end_positions_parallel(file_dir, files, log_file, num_workers=8):
    # 获取已处理文件
    processed_files = get_processed_files(log_file)

    # 筛选未处理的文件
    unprocessed_files = [f for f in files if f not in processed_files]
    if not unprocessed_files:
        print("No unprocessed files found.")
        return chromosome_basepairs

    # 使用多进程处理未处理的文件
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_file, os.path.join(file_dir, file))
            for file in unprocessed_files
        ]

        # 按照任务完成顺序处理结果
        results = []
        for future in tqdm(futures, total=len(futures), desc="Processing files"):
            try:
                result = future.result()  # 阻塞直到任务完成
                results.append(result)
                print('-----------------',results)
                save_to_json("results.json", result)
                save_processed_file(log_file, result[0])  # 保存已处理文件名
            except Exception as e:
                print(f"Error processing file: {e}")

    # 合并结果
    return merge_max_end_positions(results)


# 使用示例
file_dir = '/mnt/data/baishuhang/DNA/output'  # 你的文件目录路径
log_file = 'processed_files.log'  # 保存已处理文件名的日志文件

# 获取文件列表
files = os.listdir(file_dir)

# 动态设置最大线程数
num_workers = min(15, os.cpu_count() - 3 or 1)

# 获取最大结束位置
max_end_positions = get_max_end_positions_parallel(file_dir, files, log_file, num_workers=num_workers)

# 输出结果
for chrom, max_end in max_end_positions.items():
    print(f"Chromosome {chrom}: max end position = {max_end}")
