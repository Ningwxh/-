
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier, test_id=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    X_test, y_test = X[test_id, :], y[test_id]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    if test_id:
        X_test, y_test = X[test_id, :], y[test_id]
        plt.scatter(X_test[:,0],X_test[:, 1], c='', s=55, alpha=0.1, linewidths=1, marker='o', label='test set')

if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_X = iris.data[:, [2, 3]]#取数据的2，3列特征
    iris_y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3, random_state=0)#划分数据集，分为训练集和测试集
    sc = StandardScaler()#特征缩放
    sc.fit(X_train)#训练得每个特征得样本均值和标准差
    X_train_sc = sc.transform(X_train)#对训练数据做标准化处理，即特征缩放
    X_test_sc = sc.transform(X_test)#测试集与训练集同步
    ppn = Perceptron(n_iter=50, eta0=0.1, random_state=0)#感知器模型，迭代次数50，学习速率为0.1每次迭代初始化重排训练集
    ppn.fit(X_train_sc, y_train)#训练
    y_yred = ppn.predict(X_test_sc)#用测试集进行预测
    X_combined_std = np.vstack((X_train_sc, X_test_sc))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_id=range(105, 150))
    plt.show()
