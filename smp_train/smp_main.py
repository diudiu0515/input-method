import json
import re
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pretraining import train_from_corpus 
from smp_pretraining import train_smp_only    
from src.model import viterbi_predict       

def train_wiki_only(wiki_root_dir, common_chars):
    from collections import Counter
    unigram = Counter()
    bigram = Counter()
    regex_hanzi = re.compile(r'[^\u4e00-\u9fa5]')
    
    print(f"正在扫描 Wiki 语料库: {wiki_root_dir}...", file=sys.stderr)
    
    for root, dirs, files in os.walk(wiki_root_dir):
        for file in files:
            if not file.startswith('wiki_'): continue
            
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = regex_hanzi.sub(' ', data.get('text', ''))
                        clean_res = [c if c in common_chars else ' ' for c in text]
                        segments = "".join(clean_res).split()
                        for seg in segments:
                            for i in range(len(seg)):
                                unigram[seg[i]] += 1
                                if i < len(seg) - 1:
                                    bigram[seg[i] + seg[i+1]] += 1
                    except: continue
    return unigram, bigram
def load_py_map(pinyin_dict_path):
    py_map = {}
    with open(pinyin_dict_path, 'r', encoding='gbk') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            py_map[parts[0]] = parts[1:]
    return py_map

def load_test_data(input_path):
    test_data = []
    with open(input_path, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            pys = line.strip().split()
            if pys:
                test_data.append(pys)
    return test_data

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
    test_data = load_test_data(input_path)

    train_start = time.time()
    if mode == "sina":
        u, b = train_from_corpus(sina_corpus, common_chars)
    elif mode == "smp":
        u, b = train_smp_only(smp_corpus, common_chars)
        print(f"SMP 统计完成！一元组: {len(u)} 种, 二元组: {len(b)} 种", file=sys.stderr)
    elif mode == "wiki":
        wiki_corpus_dir = os.path.join(base_dir, "corpus/wiki_zh")
        u, b = train_wiki_only(wiki_corpus_dir, common_chars)
        print(f"SMP 统计完成！一元组: {len(u)} 种, 二元组: {len(b)} 种", file=sys.stderr)
    else: 
        u, b = train_from_corpus(sina_corpus, common_chars)
        u_smp, b_smp = train_wiki_only(smp_corpus, common_chars)
        for k, v in u_smp.items(): u[k] += v
        for k, v in b_smp.items(): b[k] += v
    train_end = time.time()
    train_duration = train_end - train_start
    
    predict_start = time.time()
    total_count = sum(u.values()) if u else 1
    results = []
    for idx, pys in enumerate(test_data): 
        if idx % 10 == 0:
            print(f"正在处理第 {idx} 条数据: {' '.join(pys)}")
        try:
            res = viterbi_predict(pys, u, b, py_map, total_count)
            results.append(res)
        except Exception as e:
            print(f"ERROR at line {idx} with pinyin {pys}: {e}")
            results.append("") 
        
    out_file = f"output_{mode}.txt"
    with open(out_file, 'w', encoding='utf-8', errors='replace') as f:
        f.write("\n".join(results))
    predict_end = time.time()
    predict_duration = predict_end - predict_start
    print(f"1. 训练耗时: {train_duration:.2f} 秒",file=sys.stderr)
    print(f"2. 预测耗时: {predict_duration:.2f} 秒 (共 {len(results)} 行)",file=sys.stderr)
    print(f"3. 总计运行: {time.time() - start_all:.2f} 秒",file=sys.stderr)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f"模式 {mode} 测试完成，结果已保存至 {out_file}",file=sys.stderr)

if __name__ == "__main__":
    run_experiment("sina")
    run_experiment("smp")
    run_experiment("wiki")
    run_experiment("mixed")