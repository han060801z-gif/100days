import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer  #用来处理缺失值
from sklearn.compose import ColumnTransformer  #用来对指定列做处理
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler  #分别的用处为：把类别变量处理成数字列、把标签文字变成数字、把数据标准化
from sklearn.model_selection import train_test_split  #测试集、训练集划分工具

# 1. 读取数据集
# 请把 Data.csv 放在这个 .py 文件同目录下，或者把下面路径改成你的实际路径
dataset = pd.read_csv('Data.csv')

# 2. 划分特征和标签
# X 取前 3 列（Country, Age, Salary）
# y 取第 4 列（Purchased）
X = dataset.iloc[:, :-1].values  #第一个：表示所有行，第二个：-1表示从第一列取到倒数第2列，不取最后一列。取出这些数组并赋值给X
y = dataset.iloc[:, 3].values  #指把第四列取出来赋值给y，y是要预测的结果，也就是标签

# 3. 处理缺失值
# 对第 2、3 列（Age, Salary）中的空值用均值填充
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  #把缺失值看作np.nan，用mean来补空处，创建了imputer这一缺失值填补器
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
#X[:, 1:3] 表示所有行和第二列到第三列
#fit表示先看该列的已有数据，算出均值
#transform表示用这个均值去填空处

# 4. 编码类别特征
# 对第 1 列 Country （即把文字转为数字）做独热编码
# 例如 France/Germany/Spain -> [1,0,0] / [0,1,0] / [0,0,1]
ct = ColumnTransformer(   #引入工具，创建一个按列处理的工具
    transformers=[('encoder', OneHotEncoder(), [0])],   #OneHotEncoder()指定编码方式是独热编码，【0】表示第1列country
    remainder='passthrough'  #其他没指定的列原样保留，不动
)
X = np.array(ct.fit_transform(X))  #执行编码操作。fit先识别有哪几类，transform再把类别转为数字。最后用np.array转为numpy数组

# 对标签 y 做编码
# No/Yes -> 0/1
label_encoder_y = LabelEncoder()  #创建标签编码器变量
y = label_encoder_y.fit_transform(y)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 6. 特征缩放
# 这里只对特征做标准化，不对标签 y 做标准化
scaler = StandardScaler()  #创建一个标准化工具
X_train = scaler.fit_transform(X_train)  #用训练集算出每一列的均值和标准差，再把训练集标准化
X_test = scaler.transform(X_test)  #在训练集上学到标准化规则，再把这个规则应用到测试集上

# 打印结果，方便你检查是否运行成功
print('X_train =')
print(X_train)
print('\nX_test =')
print(X_test)
print('\ny_train =')
print(y_train)
print('\ny_test =')
print(y_test)