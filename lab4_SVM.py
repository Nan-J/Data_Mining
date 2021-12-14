from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 一、iris数据获取
iris = datasets.load_iris()  # 加载数据集
X = iris.data[:, :2]  # 取所有行的前两位元素
y = iris.target

# iris数据可视化
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='pink', marker='*')  # X中y==0的,索引为0的元素(共50个) ，即第一个元素为横坐标，索引为1的为
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='b', marker='o')
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='g', marker='+')
plt.title('the relationship between sepal and target classes')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()


# 三、模型的训练
X = iris.data[:, :2]
y = iris.target  # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lin_svc = svm.SVC(kernel='linear',C=0.1).fit(X_train,y_train)  # 核函数kernel为线性核函数
rbf_svc = svm.SVC(kernel='rbf', C=0.5,gamma=20,degree=3).fit(X_train, y_train)  #kernel为多项式核函数

# 四、模型的评估
print("Training_set_score in lin_svc：", format(lin_svc.score(X_train, y_train),'.3f'))
print("Testing_set_score in lin_svc：", format(lin_svc.score(X_test, y_test),'.3f'))
print("Training_set_score in rbf_svc：", format(rbf_svc.score(X_train, y_train), '.3f'))
print("Testing_set_score in rbf_svc：", format(rbf_svc.score(X_test, y_test), '.3f'))


# 分类结果可视化
# the step of the grid
h = .02
# to create the grid , so that we can plot the images on it
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# the title of the graph
titles = ['LinearSVC (linear kernel)',
          'SVC with polynomial (degree 3) kernel']

for i, clf in enumerate((lin_svc, rbf_svc)):
    # to plot the edge of different classes
    # to create a 1*2 grid , and set the i image as current image
    plt.subplot(1, 2, i + 1)
    # set the margin between different sub-plot
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # SVM input :xx and yy output: an array
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # to plot the result
    Z = Z.reshape(xx.shape)  # (220, 280)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
