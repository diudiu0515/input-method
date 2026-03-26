import sys

def calculate_score(output_file, answer_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        output_lines = [line.strip() for line in f]
    with open(answer_file, 'r', encoding='utf-8') as f:
        answer_lines = [line.strip() for line in f]

    char_total = 0
    char_correct = 0
    sent_correct = 0
    
    for out, ans in zip(output_lines, answer_lines):
        if out == ans:
            sent_correct += 1
        
        char_total += len(ans)
        for i in range(min(len(out), len(ans))):
            if out[i] == ans[i]:
                char_correct += 1
                
    print(f"句子总数: {len(answer_lines)}")
    print(f"句准确率: {sent_correct / len(answer_lines) * 100:.2f}%")
    print(f"字准确率: {char_correct / char_total * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        calculate_score(sys.argv[1], "data/answer.txt")
    else:
        calculate_score("data/output.txt", "data/answer.txt")