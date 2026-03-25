import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    from src.score import calculate_score
    print("评测SINA模型:")
    calculate_score("output_sina.txt", "data/answer.txt")
    
    print("\n评测SMP模型:")
    calculate_score("output_smp.txt", "data/answer.txt")
    
    print("\n评测混合模型:")
    calculate_score("output_mixed.txt", "data/answer.txt")

if __name__ == "__main__":
    run_all_tests()
