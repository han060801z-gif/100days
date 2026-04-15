import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  #导入线性回归模型

# 1. 读取数据集
dataset = pd.read_csv('studentscores.csv')

# 2. 划分特征和标签
# X: 学习时间（Hours）
# y: 分数（Scores）
X = dataset.iloc[:, :1].values  #读取第1列作为X,不写0的原因是模型要求x是一个二维数组
y = dataset.iloc[:, 1].values   #读取第2列作为标签y

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 4. 训练简单线性回归模型
regressor = LinearRegression()  #创建一个线性回归对象
regressor.fit(X_train, y_train)  #拿regressor来训练训练集数据，学出一条最合适的直线

# 5. 用测试集做预测
y_pred = regressor.predict(X_test)

# 6. 可视化训练集结果
plt.scatter(X_train, y_train)  #画训练集散点图
plt.plot(X_train, regressor.predict(X_train)) #在散点图上画出拟合出来的直线
plt.title('Hours vs Scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

# 7. 可视化测试集结果
plt.scatter(X_test, y_test)
plt.plot(X_train, regressor.predict(X_train))  #画的还是训练集的拟合直线，不是重新对测试集画一条
plt.title('Hours vs Scores (Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

# 8. 打印一些结果，方便检查
print('截距 intercept =', regressor.intercept_)
print('斜率 coefficient =', regressor.coef_[0])

print('\n测试集真实值 y_test =')
print(y_test)

print('\n测试集预测值 y_pred =')
print(y_pred)