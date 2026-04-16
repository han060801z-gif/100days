import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap  #导入颜色工具来着色
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  #导入标准化工具
from sklearn.linear_model import LogisticRegression  #逻辑回归模型
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. 读取数据集
dataset = pd.read_csv('Social_Network_Ads.csv')

# 2. 选择特征和标签
# X 只取 Age 和 EstimatedSalary 两列
# y 取 Purchased 列（是否购买）
X = dataset.iloc[:, [2, 3]].values  #取第3列和第4列作为特征X
y = dataset.iloc[:, 4].values  #取第5列作为标签y

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 4. 特征缩放(标准化）
scaler = StandardScaler()  #创建标准化对象
X_train = scaler.fit_transform(X_train)  #在训练集上学习均值和标准差，把训练集标准化
X_test = scaler.transform(X_test)  #用训练集学到的规则变换测试集（不能直接处理测试集）

# 5. 训练逻辑回归模型
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 6. 在测试集上做预测
y_pred = classifier.predict(X_test)

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
sample_pred = classifier.predict(scaler.transform([[30, 87000]]))   #输入一个新用户，数据为30，87000，先标准化，然后再让模型预测
print('\n样本 [30, 87000] 的预测结果 =', sample_pred[0])

#到此，模型其实已经训练完了，但是我们还没办法看见模型分类的边界，再加上样本数量没办法完全覆盖平面上的所有边界
#因此，我们在网格上铺满点，每个点都问模型这里分为哪一类，再根据分类结果把该点的区域着色，这样就能画出边界
def plot_decision_boundary(X_set, y_set, title):  #定义一个函数来专门画分类边界图 set只是表示“这一组数据”的含义而已，实际上就是指的样本
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
    )  #在平面上铺很多小点，形成一个细密网格，来问模型每个位置的判断结果

    grid_pred = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)  #把网格里的点丢给模型，让模型判断类别。ravel是把二维网格拉平成一维，T是转置成每行一个点
    grid_pred = grid_pred.reshape(X1.shape)  #把预测结果重新变回和网格一样的二维形状

    plt.contourf(X1, X2, grid_pred, alpha=0.3, cmap=ListedColormap(('red', 'green')))  #画出分类区域，红色不买，绿色买
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())  #设定横纵坐标显示范围

#下面的for循环是把样本里的点取出来并画在图上
    for i, j in enumerate(np.unique(y_set)):  #找出y_set里所有不重复的类别值，并进行遍历（如0和1）
        plt.scatter(   #把下面取出的点都画出来
            X_set[y_set == j, 0],   #选出所有满足y_set == j的行，并取这些行的第0列。也就是取出属于第j类的所有样本的横坐标
            X_set[y_set == j, 1],   #同理，取出所有属于第j类的样本的纵坐标
            label=f'Class {j}'  #f-string,格式化字符串
        )

    plt.title(title)
    plt.xlabel('Age (standardized)')
    plt.ylabel('Estimated Salary (standardized)')
    plt.legend()
    plt.show()


# 9. 可视化训练集结果
plot_decision_boundary(X_train, y_train, 'Logistic Regression (Training set)')

# 10. 可视化测试集结果
plot_decision_boundary(X_test, y_test, 'Logistic Regression (Test set)')