#感知器
import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
class Perceptron(object):
    '''
    eta: 学习速率
    n_iter: 迭代次数
    w_: fit后的每个特征的权重
    error_: 每次迭代预测错误数

    '''
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])#生成X列+1大的数组
        self.error_ = []
        for _ in range(self.n_iter):
            error = 0
            for xi ,target in zip(X, y):
                updata = self.eta * (target - self.predict(xi))
                self.w_[1:] += updata * xi#权重更新,w + △w
                self.w_[0] += updata
                error += int(updata!=0.0)#不为0的更新为预测错误
            self.error_.append(error)
        return self

    #权重与特征相乘，矩阵相乘，返回计算的值，一维数组
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_X = pd.DataFrame(iris.data).iloc[0:100,[0,2]].values
    iris_y = pd.DataFrame(iris.target).iloc[0:100].values
    iris_y = np.where(iris_y == 0,-1,1)
    plt.scatter(iris_X[:,0], iris_X[:,1],edgecolors='red', marker='o')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()
    ppn = Perceptron(eta=0.01, n_iter=5)
    ppn.fit(iris_X,iris_y)
    print(ppn.error_)
    l = list(range(1,len(ppn.error_)+1))
    plt.plot(l, ppn.error_, marker='o')
    plt.xlabel('n_iter')
    plt.ylabel('error Number')
    plt.show()

