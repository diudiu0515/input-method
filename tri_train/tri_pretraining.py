import os
import collections
import sys

def load_common_chars(filepath="./data/一二级汉字表.txt"):
    common_chars = set()
    try:
        f = open(filepath, 'r', encoding='gbk')
        content = f.read()
    except UnicodeDecodeError:
        f = open(filepath, 'r', encoding='utf-8')
        content = f.read()
    
    for char in content:
        if '\u4e00' <= char <= '\u9fa5':
            common_chars.add(char)
    f.close()
    return common_chars

def train_from_corpus(corpus_path, common_chars):
    unigram = collections.defaultdict(int)
    bigram = collections.defaultdict(int)
    trigram = collections.defaultdict(int) #新增的三元字典
    total_chars = 0
    for root, _, files in os.walk(corpus_path):
        for file in files:
            full_path = os.path.join(root, file)
            with open(full_path, 'r', encoding='gbk', errors='ignore') as f:
                content = f.read()
                clean_text_list = []
                for char in content:
                    if char in common_chars:
                        clean_text_list.append(char)
                    else:
                        clean_text_list.append(' ')
                
                segments = "".join(clean_text_list).split()
        
                for seg in segments:
                    for char in seg:
                        unigram[char] += 1
                        total_chars += 1
                    for i in range(len(seg) - 1):
                        bi = seg[i:i+2]
                        bigram[bi] += 1
                    for i in range(len(seg) - 2):#统计三元
                        tri = seg[i:i+3] 
                        trigram[tri] += 1

    print(f"二元组数量: {len(bigram)}", file=sys.stderr)
    print(f"三元组数量: {len(trigram)}",file=sys.stderr)
    return unigram, bigram, trigram ,total_chars
    

def load_pinyin_map(filepath="./data/拼音汉字表.txt", common_chars=None):
    py_map = {}
    with open(filepath, 'r', encoding='gbk') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                pinyin = parts[0]
                chars = [c for c in parts[1:] if c in common_chars]
                if chars:
                    py_map[pinyin] = chars
    return py_map