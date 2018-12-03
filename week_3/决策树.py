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
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # һά�����Ϊ���󽫵�һ����Ϊ���������ڶ���Ϊ������
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
iris_X = preprocessing.scale(iris_X)#��׼��
'''
criterion �� string��optional��default =��gini����
�������ڵ�ֵĹ��ܡ�֧�ֵı�׼�ǻ������ʵġ�gini������Ϣ����ġ�entropy����
splitter �� string��optional��default =��best����
������ÿ���ڵ㴦ѡ���ֵĲ��ԡ�֧�ֵĲ����ǡ�best��ѡ����ѷָ�͡�random��ѡ���������ָ�
max_depth �� int��None����ѡ��Ĭ��=�ޣ�
���������ȡ����ΪNone������չ�ڵ�ֱ������Ҷ�Ӷ��Ǵ��Ļ�ֱ������Ҷ�Ӱ�������min_samples_split������
min_samples_split �� int��float��optional��default = 2��
����ڲ��ڵ��������С��������
'''
# tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
# tree.fit(iris_X, iris_y)
# draw(tree,iris_X,iris_y)
# export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])#��װgraphviz�����ӻ���ͼ dot -Tpng tree.dot -o tree.png

'''
���ɭ��
    ʹ��bootstrap�����������ѡ��n����������ѵ����������ظ���
    ��2�����ظ����ѡȡd������������Ŀ�꺯��Ҫ���������Ϣ���棬ʹ����ѡȡ���������нڵ㻮��
    �ظ���2�����
    ����ÿ�ξ������������ж���ͶƱ��
'''
'''
n_estimators �� ��������ѡ��Ĭ��= 10��
ɭ�������ľ����
criterion :(ͬ��)
max_depth :��ͬ�ϣ�
n_jobs: int��None����ѡ��Ĭ��=�ޣ�
�������ں�����
'''
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini', n_estimators=5, random_state=1,n_jobs=2)
forest.fit(iris_X,iris_y)
draw(forest, iris_X, iris_y)
