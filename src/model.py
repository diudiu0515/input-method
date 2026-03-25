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
            dp[i][cur_w] = (max_log_p, best_prev)

    if not dp[n-1]: return ""
    res, curr_w = [], max(dp[n-1], key=lambda x: dp[n-1][x][0])
    for i in range(n-1, -1, -1):
        res.append(curr_w)
        curr_w = dp[i][curr_w][1]
    return "".join(reversed(res))