import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier   #导入KNN分类器
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. 读取数据集
# 请把 Social_Network_Ads.csv 放在这个 .py 文件同目录下，
# 或者把下面路径改成你的实际路径。
dataset = pd.read_csv('Social_Network_Ads.csv')

# 2. 选择特征和标签
# X 只取 Age 和 EstimatedSalary 两列
# y 取 Purchased 列（是否购买）
X = dataset.iloc[:, [2, 3]].values  #选第3、4列作为特征矩阵x
y = dataset.iloc[:, 4].values  #选第5列作为标签

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 4. 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. 训练 KNN 模型
# n_neighbors=5 表示看最近的 5 个邻居
# metric='minkowski', p=2 表示使用欧氏距离
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)  #训练KNN模型，也叫懒惰学习，只是把样本记住

# 6. 在测试集上做预测
y_pred = classifier.predict(X_test)  #预测过程就是把每个样本计算其和所有训练集样本的距离，找最近的5个距离，看这5个距离中哪一类最多，就把该样本判为这一类

# 7. 输出评估结果
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print('混淆矩阵 confusion_matrix =')
print(cm)
print('\n准确率 accuracy =', acc)
print('\n分类报告 classification_report =')
print(classification_report(y_test, y_pred))

# 8. 单个新样本预测示例
# 例子：年龄 30，预估薪资 87000
sample_pred = classifier.predict(scaler.transform([[30, 87000]]))
print('\n样本 [30, 87000] 的预测结果 =', sample_pred[0])


def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
    )

    grid_pred = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
    grid_pred = grid_pred.reshape(X1.shape)

    plt.contourf(X1, X2, grid_pred, alpha=0.3, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for j in np.unique(y_set):
        plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            label=f'Class {j}'
        )

    plt.title(title)
    plt.xlabel('Age (standardized)')
    plt.ylabel('Estimated Salary (standardized)')
    plt.legend()
    plt.show()


# 9. 可视化训练集结果
plot_decision_boundary(X_train, y_train, 'K-NN (Training set)')

# 10. 可视化测试集结果
plot_decision_boundary(X_test, y_test, 'K-NN (Test set)')