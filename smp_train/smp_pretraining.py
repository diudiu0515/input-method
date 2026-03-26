import json
import re
from collections import defaultdict, Counter

def train_smp_only(file_path, common_chars):
    unigram = Counter()
    bigram = Counter()
    
    regex_hanzi = re.compile(r'[^\u4e00-\u9fa5]')
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            content = ""
            if line.startswith('{'):
                try:
                    data = json.loads(line)
                    content = data.get('content', '')
                except:
                    content = line
            else:
                content = line

            text = regex_hanzi.sub(' ', content)
            
            clean_res = [c if c in common_chars else ' ' for c in text]
            segments = "".join(clean_res).split()

            for seg in segments:
                if len(seg) < 1: continue
                for i in range(len(seg)):
                    unigram[seg[i]] += 1
                    if i < len(seg) - 1:
                        bigram[seg[i:i+2]] += 1
                        
    return unigram, bigram