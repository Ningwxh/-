#coding=GBK
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)#异或
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='o', label='-1')
plt.legend()
plt.show()

'''
gamma ： float，optional（默认='auto'）
当前默认值为'auto'，它使用1 / n_features，如果gamma='scale'传递，则使用1 /（n_features * X.std（））作为gamma的值。
当前默认的gamma''auto'将在版本0.22中更改为'scale'。'auto_deprecated'，'auto'的弃用版本用作默认值，表示没有传递明确的gamma值。
coef0 ： float，optional（默认值= 0.0）
核函数中的独立项。它只在'poly'和'sigmoid'中很重要。
degree ：多项式poly核函数的维度
'''
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)#使用rbf核函数，映射到高维度寻找划分超平面，gamma过大会决策边界紧凑，可能会产生过拟合。
svm.fit(X_xor, y_xor)
#画图
h = 0.01
x_min, x_max = X_xor[:, 0].min() - 1, X_xor[:, 0].max() + 1
y_min, y_max = X_xor[:, 1].min() - 1, X_xor[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))#一维数组变为矩阵将第一个变为行向量，第二个为列向量
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='o', label='-1')
plt.show()
