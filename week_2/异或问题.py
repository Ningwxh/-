#coding=GBK
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)#���
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='o', label='-1')
plt.legend()
plt.show()

'''
gamma �� float��optional��Ĭ��='auto'��
��ǰĬ��ֵΪ'auto'����ʹ��1 / n_features�����gamma='scale'���ݣ���ʹ��1 /��n_features * X.std��������Ϊgamma��ֵ��
��ǰĬ�ϵ�gamma''auto'���ڰ汾0.22�и���Ϊ'scale'��'auto_deprecated'��'auto'�����ð汾����Ĭ��ֵ����ʾû�д�����ȷ��gammaֵ��
coef0 �� float��optional��Ĭ��ֵ= 0.0��
�˺����еĶ������ֻ��'poly'��'sigmoid'�к���Ҫ��
degree ������ʽpoly�˺�����ά��
'''
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)#ʹ��rbf�˺�����ӳ�䵽��ά��Ѱ�һ��ֳ�ƽ�棬gamma�������߽߱���գ����ܻ��������ϡ�
svm.fit(X_xor, y_xor)
#��ͼ
h = 0.01
x_min, x_max = X_xor[:, 0].min() - 1, X_xor[:, 0].max() + 1
y_min, y_max = X_xor[:, 1].min() - 1, X_xor[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))#һά�����Ϊ���󽫵�һ����Ϊ���������ڶ���Ϊ������
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='o', label='-1')
plt.show()
