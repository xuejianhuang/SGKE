import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# 读取CSV文件
def read_csv_file(filename):
    df = pd.read_csv(filename)
    return df

# 计算指标
def calculate_metrics(predictions, truths):
    precision, recall, f1_score, _ = precision_recall_fscore_support(truths, predictions, average=None)
    return precision, recall, f1_score

if __name__ == '__main__':

    # 示例用法
    filename = '../../LLM/Weibo/ChatGTP3.5_100.csv'  # 替换为你的CSV文件路径

    # 读取数据
    df = read_csv_file(filename)

    # 获取预测标签和真实标签
    predictions = df['predicted_label'].tolist()
    truths = df['true_label'].tolist()

    # 计算Precision、Recall、F1-score
    precision, recall, f1_score = calculate_metrics(predictions, truths)

    # 打印结果
    for i in range(len(precision)):
        print(f"Class {i}:")
        print(f"  Precision: {precision[i]:.3f}")
        print(f"  Recall   : {recall[i]:.3f}")
        print(f"  F1-score : {f1_score[i]:.3f}")
        print()
