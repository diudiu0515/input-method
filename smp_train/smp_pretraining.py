import json
import re
from collections import defaultdict

def train_smp_only(file_path, common_chars, unigram=None, bigram=None):
    if unigram is None: unigram = defaultdict(int)
    if bigram is None: bigram = defaultdict(int)
    regex_noise = re.compile(r'http[s]?://\S+|@\S+|#.*?#|\[.*?\]')
    try:
        f = open(file_path, 'r', encoding='utf-8')
        test_line = f.readline()
        f.seek(0)
    except UnicodeDecodeError:
        f = open(file_path, 'r', encoding='gbk', errors='ignore')
    
    with f:
        for line in f:
            try:
                data = json.loads(line.strip())
                content = data.get('content', '')
                
                text = regex_noise.sub('', content)
                clean_res = [c if c in common_chars else ' ' for c in text]
                
                segments = "".join(clean_res).split()
                for seg in segments:
                    for i in range(len(seg)):
                        unigram[seg[i]] += 1
                        if i < len(seg) - 1:
                            bigram[seg[i:i+2]] += 1
            except:
                continue
                
    return unigram, bigram