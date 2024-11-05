import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体字体
if __name__ == '__main__':
    # 定义混淆矩阵 twitter
    # confusion_matrix = np.array([[0.982, 0.012, 0.006],
    #                              [0.030, 0.895, 0.075],
    #                              [0.007, 0.029, 0.964]])

    # # 定义混淆矩阵 weibo
    confusion_matrix = np.array([[0.967,0.015,0.019],
                                 [0.026,0.896,0.079 ],
                                 [0.026,0.061,0.914 ]])
    # 创建一个热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True,fmt=".3f", cmap="Reds", cbar=True,
                annot_kws={"size": 30},  # 设置热力图上数字的大小
                xticklabels=['非谣言', '谣言','未验证'],
                yticklabels=['非谣言', '谣言','未验证'])

    # 添加标题和标签
    #plt.title('Confusion Matrix')
    plt.xlabel('预测类别',fontsize=18)
    plt.ylabel('真实类别',fontsize=18)

    # 设置xticklabels和yticklabels的大小
    plt.xticks(fontsize=16)  # 设置x轴标签大小
    plt.yticks(fontsize=16)  # 设置y轴标签大小
    plt.savefig('weibo_confusion.png', dpi=600)
    #plt.savefig('twitter_confusion.png', dpi=600)


    # 显示图像
    plt.show()
