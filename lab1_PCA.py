import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn import datasets
from sklearn.datasets import load_iris
import numpy as np


# 1.对原始数据零均值化（中心化），
# 2.求协方差矩阵，
# 3.对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间。


# 我们对于一组数据，如果它在某一坐标轴上的方差越大，说明坐标点越分散，该属性能够比较好的反映源数据。
# 所以在进行降维的时候，主要目的是找到一个超平面，它能使得数据点的分布方差呈最大
# 这样数据表现在新的坐标轴上时候已经足够分散了。


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        # 求协方差矩阵
        X = X - X.mean(axis=0)  # 0均值化
        # print("标准化后的数据为")
        # print(X)
        self.covariance = X.T.dot(X) / X.shape[0]  # 求协方差
        # print("协方差为")
        # print(self.covariance)
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # print("特征值和特征向量为")
        # print(eig_vals)
        # print("-------------------")
        # print(eig_vectors)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        # 对X进行降维
        return np.dot(X, self.components_)


# 调用
pca = PCA(n_components=2)  # 降维为2
x, y = load_iris(return_X_y=True)  # 导入四维数据 x:属性 y:类别
# print("-----------------------------------")
# print(x.shape)
# print("-----------------------------------")
# print(x)
# print("-----------------------------------")
# print(y)
# print("-----------------------------------")
reduced_x = pca.fit_transform(x)
print(reduced_x)  # 输出降维后的数据

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_x)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='pink', marker='x')
plt.scatter(blue_x, blue_y, c='green', marker='D')
plt.scatter(green_x, green_y, c='blue', marker='.')
plt.show()
