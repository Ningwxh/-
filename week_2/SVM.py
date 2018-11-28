#coding=GBK
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
iris_X = iris.data[:100,[2, 3]]
iris_y = iris.target[:100]
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y,test_size=0.3, random_state=0)
std = StandardScaler()
std.fit(X_train)
X_train_std = std.transform(X_train)
X_test_std = std.transform(X_test)
'''
指定要在算法中使用的内核类型。它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'或者callable之一。如果没有给出，将使用'rbf'。
'''
svm = SVC(kernel='linear', C=1, random_state=0)#线性核，C的值较小时可以允许一些错误
svm.fit(X_train_std, y_train)
print(svm.score(X_test_std, y_test))

h = 0.01
x_min, x_max = iris_X[:, 0].min() - 1, iris_X[:, 0].max() + 1
y_min, y_max = iris_X[:, 1].min() - 1, iris_X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X_train_std[:, [0]], X_train_std[:, [1]], edgecolors='k')
plt.show()
