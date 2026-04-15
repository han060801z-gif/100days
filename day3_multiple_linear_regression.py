import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer  #用于对指定列做转换
from sklearn.preprocessing import OneHotEncoder  #把类别变量转化为数字
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. 读取数据集
dataset = pd.read_csv('50_Startups.csv')

# 2. 划分特征和标签
# X: 前 4 列（R&D Spend, Administration, Marketing Spend, State）
# y: 最后 1 列（Profit）
X = dataset.iloc[:, :-1].values  #取出前4列作为输入特征x，：-1表示取所有列但不要最后一列
y = dataset.iloc[:, 4].values   #取第5列作为y，即预测目标

# 3. 对类别变量 State 做独热编码
# drop='first' 表示自动删去一个虚拟变量，避免虚拟变量陷阱
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), [3])],   #使用独热编码，按“是1吗，是2吗，是3吗”的方式来拆成数字。drop=first是避免虚拟变量陷阱。因为在编码时，前两个问题知道了，第三个问题自然就知道了，如果保留的话就会使变量之间完全共线，所以需要删掉一个
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X), dtype=float)  #编码后转成float类型，方便回归计算

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 5. 训练多元线性回归模型
regressor = LinearRegression()
regressor.fit(X_train, y_train)  #根据训练集数据学出截距和每个变量前的系数

# 6. 用测试集做预测
y_pred = regressor.predict(X_test)

# 7. 打印结果，方便检查
print('截距 intercept =', regressor.intercept_)
print('回归系数 coefficients =', regressor.coef_)

print('\n测试集真实值 y_test =')
print(y_test)

print('\n测试集预测值 y_pred =')
print(y_pred)

#8.判断模型拟合效果
#8.1 R方 衡量模型能解释y的波动的比例，数值越接近1，说明拟合效果越好
print('训练集 R^2 =', regressor.score(X_train, y_train))
print('测试集 R^2 =', regressor.score(X_test, y_test))
#训练集高，测试集也高，比较好
#缺点：普通R方存在一个问题，自变量越多，往往容易变大，所以可以用调整后的R方

#8.2 调整后的R方
r2 = regressor.score(X_test, y_test)
n = X_test.shape[0]  #样本数
p = X_test.shape[1]  #自变量个数
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print('测试集 Adjusted R^2 =', adj_r2)

#9.图像
#9.1 横轴真实值，纵轴预测值，如果拟合的好，应该靠近45°的对角线
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('Actual vs Predicted')
plt.show()

#9.2残差图
#残差指真实值-预测值。如果在0附近随即散开，说明模型拟合效果不错，不要出现明显规律
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel('Predicted Profit')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()