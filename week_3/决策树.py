#coding=GBK
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn import preprocessing

def draw(model,X,y):
    h = 0.01
    x_min, x_max = iris_X[:, 0].min() - 1, iris_X[:, 0].max() + 1
    y_min, y_max = iris_X[:, 1].min() - 1, iris_X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # 一维数组变为矩阵将第一个变为行向量，第二个为列向量
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[y == 2, 0], X[y == 2, 1], c='y', marker='*', label='2')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', marker='x', label='1')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', marker='o', label='0')
    plt.show()

iris = load_iris()
iris_X = iris.data[:, [2, 3]]
iris_y = iris.target
iris_X = preprocessing.scale(iris_X)#标准化
'''
criterion ： string，optional（default =“gini”）
衡量划节点分的功能。支持的标准是基尼杂质的“gini”和信息增益的“entropy”。
splitter ： string，optional（default =“best”）
用于在每个节点处选择拆分的策略。支持的策略是“best”选择最佳分割和“random”选择最佳随机分割
max_depth ： int或None，可选（默认=无）
树的最大深度。如果为None，则扩展节点直到所有叶子都是纯的或直到所有叶子包含少于min_samples_split样本。
min_samples_split ： int，float，optional（default = 2）
拆分内部节点所需的最小样本数：
'''
# tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
# tree.fit(iris_X, iris_y)
# draw(tree,iris_X,iris_y)
# export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])#安装graphviz做可视化树图 dot -Tpng tree.dot -o tree.png

'''
随机森林
    使用bootstrap抽样方法随机选择n个样本用于训练（随机可重复）
    （2）不重复随机选取d个特征，根据目标函数要求，如最大化信息增益，使用所选取的特征进行节点划分
    重复（2）多次
    汇总每课决策树的类标进行多数投票。
'''
'''
n_estimators ： 整数，可选（默认= 10）
森林里的树木数量
criterion :(同上)
max_depth :（同上）
n_jobs: int或None，可选（默认=无）
处理器内核数量
'''
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini', n_estimators=5, random_state=1,n_jobs=2)
forest.fit(iris_X,iris_y)
draw(forest, iris_X, iris_y)
