import math

def viterbi_predict(pinyin_list, unigram, bigram, py_map, total_count):
    n = len(pinyin_list)
    if n == 0: return ""
    dp = [{} for _ in range(n)]
    LAMBDA = 0.99
    MIN_PROB = -1e18

    first_py = pinyin_list[0]
    candidates = py_map.get(first_py, [])
    if not candidates: return ""

    for w in candidates:
        prob = math.log(unigram.get(w, 1) / total_count)
        dp[0][w] = (prob, None)

    for i in range(1, n):
        curr_candidates = py_map.get(pinyin_list[i], [])
        for cur_w in curr_candidates:
            max_log_p, best_prev = MIN_PROB, None
            p_uni = unigram.get(cur_w, 1) / total_count
            for prev_w, (prev_log_p, _) in dp[i-1].items():
                c_bi = bigram.get(prev_w + cur_w, 0)
                c_prev = unigram.get(prev_w, total_count)
                p_trans = LAMBDA * (c_bi / c_prev) + (1 - LAMBDA) * p_uni
                score = prev_log_p + math.log(p_trans)
                if score > max_log_p:
                    max_log_p, best_prev = score, prev_w

            if best_prev is None:#没见过这个二元组就退回
                best_prev = max(dp[i-1], key=lambda x: dp[i-1][x][0])
                max_log_p = dp[i-1][best_prev][0] + math.log(p_uni)
        
            dp[i][cur_w] = (max_log_p, best_prev)


    if not dp[n-1]: return ""
    res, curr_w = [], max(dp[n-1], key=lambda x: dp[n-1][x][0])
    for i in range(n-1, -1, -1):
        res.append(curr_w)
        curr_w = dp[i][curr_w][1]
    return "".join(reversed(res))

def viterbi_predict_tri(pinyin_list, unigram, bigram, trigram, py_map, total_count):
    n = len(pinyin_list)
    if n == 0: return ""
    
    dp = [{} for _ in range(n)]
    
    L1, L2, L3 = 0.7, 0.2, 0.1

    for w in py_map.get(pinyin_list[0], []):
        prob = (unigram.get(w, 0) + 1) / (total_count + len(unigram))
        dp[0][(None, w)] = (math.log(prob), None)

    if n > 1:
        for cur_w in py_map.get(pinyin_list[1], []):
            for (prev_none, prev_w), (prev_log_p, _) in dp[0].items():
                c_bi = bigram.get(prev_w + cur_w, 0)
                c_uni = unigram.get(prev_w, 0)
                p_bi = (c_bi + 0.1) / (c_uni + 1) 
                
                score = prev_log_p + math.log(p_bi)
                state = (prev_w, cur_w)
                if state not in dp[1] or score > dp[1][state][0]:
                    dp[1][state] = (score, (None, prev_w))

    for i in range(2, n):
        curr_candidates = py_map.get(pinyin_list[i], [])
        for cur_w in curr_candidates: # w3
            for (w1, w2), (prev_log_p, _) in dp[i-1].items():
                c_tri = trigram.get(w1 + w2 + cur_w, 0)
                c_bi_12 = bigram.get(w1 + w2, 0)
                p_tri = c_tri / c_bi_12 if c_bi_12 > 0 else 0
                
                c_bi_23 = bigram.get(w2 + cur_w, 0)
                c_uni_2 = unigram.get(w2, 0)
                p_bi = c_bi_23 / c_uni_2 if c_uni_2 > 0 else 0
                
                p_uni = unigram.get(cur_w, 0) / total_count
                
                p_final = L1 * p_tri + L2 * p_bi + L3 * p_uni
                if p_final <= 0: p_final = 1e-20 
                
                score = prev_log_p + math.log(p_final)
                state = (w2, cur_w)
                
                if state not in dp[i] or score > dp[i][state][0]:
                    dp[i][state] = (score, (w1, w2))

    if not dp[n-1]: return ""
    curr_state = max(dp[n-1], key=lambda x: dp[n-1][x][0])
    
    res = []
    for i in range(n-1, -1, -1):
        res.append(curr_state[1])
        curr_state = dp[i][curr_state][1]
        
    return "".join(reversed(res))