#批处理梯度下降
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
class AdalineGB(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta #学习速率
        self.n_iter = n_iter #迭代次数
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1]) #初始化权重
        self.cost_ = []#每次迭代的误差平方和SSE
        for i in range(self.n_iter):
            output = self.nei_output(X)
            error = (y - output)[0]
            self.w_[1:] += self.eta * X.T.dot(error)
            self.w_[0] += self.eta * error.sum()
            cost = (error ** 2).sum()/2.0
            self.cost_.append(cost)
    def nei_output(self, X):
        return np.dot(X, self.w_[1:])+ self.w_[0]
    def activation(self, X):
        return self.nei_output(X)
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_X = pd.DataFrame(iris.data).iloc[0:100, [0, 2]].values
    iris_y = pd.DataFrame(iris.target).iloc[0:100].values
    iris_y = np.where(iris_y == 0, -1, 1)
    ada = AdalineGB(eta=0.0001, n_iter=50)
    ada.fit(iris_X, iris_y)
    l = list(range(1, len(ada.cost_) + 1))
    plt.plot(l, np.log10(ada.cost_), marker='o')
    plt.xlabel('n_iter')
    plt.ylabel('cost Number')
    plt.show()