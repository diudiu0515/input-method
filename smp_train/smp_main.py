import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pretraining import train_from_corpus 
from smp_pretraining import train_smp_only    
from src.model import viterbi_predict       

def load_py_map(pinyin_dict_path):
    py_map = {}
    with open(pinyin_dict_path, 'r', encoding='gbk') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            py_map[parts[0]] = parts[1:]
    return py_map

def run_experiment(mode):
    start_all = time.time()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)

    char_path = os.path.join(base_dir, "data/一二级汉字表.txt")
    py_dict_path = os.path.join(base_dir, "data/拼音汉字表.txt")
    input_path = os.path.join(base_dir, "data/input.txt")
    sina_corpus = os.path.join(base_dir, "corpus/sina_news_gbk")
    smp_corpus = os.path.join(base_dir, "corpus/SMP2020/usual_train_new.txt")
    
    with open(char_path, 'r', encoding='gbk') as f:
        common_chars = set(f.read().strip())

    py_map = load_py_map(py_dict_path)

    train_start = time.time()
    if mode == "sina":
        u, b = train_from_corpus(sina_corpus, common_chars)
    elif mode == "smp":
        u, b = train_smp_only(smp_corpus, common_chars)
    else: 
        u, b = train_from_corpus(sina_corpus, common_chars)
        u_smp, b_smp = train_smp_only(smp_corpus, common_chars)
        for k, v in u_smp.items(): u[k] += v
        for k, v in b_smp.items(): b[k] += v
    train_end = time.time()
    train_duration = train_end - train_start
    
    predict_start = time.time()
    total_count = sum(u.values()) if u else 1
    results = []
    with open(input_path, 'r', encoding='gbk') as f:
        for line in f:
            pys = line.strip().split()
            if pys:
                results.append(viterbi_predict(pys, u, b,py_map, total_count))
    
    out_file = f"output_{mode}.txt"
    predict_end = time.time()
    predict_duration = predict_end - predict_start
    print(f"1. 训练耗时: {train_duration:.2f} 秒")
    print(f"2. 预测耗时: {predict_duration:.2f} 秒 (共 {len(results)} 行)")
    print(f"3. 总计运行: {time.time() - start_all:.2f} 秒")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f"模式 {mode} 测试完成，结果已保存至 {out_file}")

if __name__ == "__main__":
    run_experiment("sina")
    run_experiment("smp")
    run_experiment("mixed")