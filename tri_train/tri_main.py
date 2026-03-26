import sys
import os
import time

sys.path.append('./src')
from tri_pretraining import load_common_chars, train_from_corpus, load_pinyin_map
from model import viterbi_predict_tri

def main():
    char_table = "./data/一二级汉字表.txt"
    py_table = "./data/拼音汉字表.txt"
    corpus_dir = "./corpus/sina_news_gbk"

    start_train = time.time()
    common_chars = load_common_chars(char_table)
    unigram, bigram, trigram, total_count = train_from_corpus(corpus_dir, common_chars)
    
    py_map = load_pinyin_map(py_table, common_chars)
    
    end_train = time.time()
    print(f"三元模型训练耗时: {end_train - start_train:.2f}秒", file=sys.stderr)
    print(f"三元组数量: {len(trigram)}", file=sys.stderr) 

    start_predict = time.time()
    
    for line in sys.stdin:
        pinyins = line.strip().split()
        if pinyins:
            result = viterbi_predict_tri(pinyins, unigram, bigram, trigram, py_map, total_count)
            print(result)
            
    end_predict = time.time()
    print(f"三元预测总耗时: {end_predict - start_predict:.2f}秒", file=sys.stderr)

if __name__ == "__main__":
    main()