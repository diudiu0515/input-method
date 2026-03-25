import sys
import os
import time
sys.path.append('./src')
from pretraining import load_common_chars, train_from_corpus, load_pinyin_map
from model import viterbi_predict

def main():
    char_table = "./data/一二级汉字表.txt"
    py_table = "./data/拼音汉字表.txt"
    corpus_dir = "./corpus/sina_news_gbk"

    start_train = time.time()
    common_chars = load_common_chars(char_table)
    py_map = load_pinyin_map(py_table, common_chars)
    unigram, bigram = train_from_corpus(corpus_dir, common_chars)
    total_uni_count = sum(unigram.values()) if unigram else 1
    end_train = time.time()
    print(f"训练耗时: {end_train - start_train:.2f}秒", file=sys.stderr)
    
    start_predict = time.time()

    for line in sys.stdin:
        pinyins = line.strip().split()
        if pinyins:
            print(viterbi_predict(pinyins, unigram, bigram, py_map, total_uni_count))
    end_predict = time.time()
    print(f"预测耗时: {end_predict - start_predict:.2f}秒", file=sys.stderr)

if __name__ == "__main__":
    main()