import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DemoModel:

    def __init__(self, alpha, max_iters):
        self.alpha = alpha
        self.max_iters = max_iters

    def step(self, z):
        return np.where(z >= 0, 1, -1)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.w_ = np.zeros(1 + X.shape[1])
        self.loss_ = []
        for i in range(self.max_iters):
            loss = 0
            for x, label in zip(X, y):
                y_hat = self.step(np.dot(x, self.w_[1:]) + self.w_[0])
                loss += y_hat != label
                self.w_[0] += self.alpha * (label - y_hat) * 1
                self.w_[1:] += self.alpha * (label - y_hat) * x
            self.loss_.append(loss)

    def predict(self, X):
        return self.step(np.dot(X, self.w_[1:]) + self.w_[0])


X, y = datasets.load_iris(return_X_y=True)
X = X[0:100, :]
y = y[0:100]
y[y == 0] = -1
y[y == 1] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = DemoModel(alpha=0.1, max_iters=10)
model.fit(X_train, y_train)
weights = model.w_
all_loss = model.loss_
print('最终权重为: \n{}'.format(weights))
print('损失列表为: \n{}'.format(all_loss))

y_pred = model.predict(X_test)
print('预测值为: \n{}'.format(y_pred))
print('真实值为: \n{}'.format(y_test))
acc = accuracy_score(y_pred, y_test)
print('准确率为 :\n{}'.format(acc))

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = 'False'
plt.figure(figsize=(10, 7))
plt.scatter(np.arange(1, len(y_test) + 1), y_test, c='b', marker='o', linewidths=15, label='真实值')
plt.scatter(np.arange(1, len(y_pred) + 1), y_pred, c='r', marker='x', label='真实值')
plt.title('感知器二分类测试结果')
plt.xlabel('样品序号')
plt.ylabel('类别')
plt.legend()
plt.show()
