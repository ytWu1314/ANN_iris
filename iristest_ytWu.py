#导入第三方库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 读取鸢尾花数据集
iris = load_iris()
print("鸢尾花数据集的返回值：\n", iris)
# 返回值是一个继承自字典的Bench
print("鸢尾花的特征值:\n", iris["data"])
print("鸢尾花的目标值：\n", iris.target)
print("鸢尾花特征的名字：\n", iris.feature_names)
print("鸢尾花目标值的名字：\n", iris.target_names)
print("鸢尾花的描述：\n", iris.DESCR)

# 划分训练集和测试集
# x_train,x_test,y_train,y_test为训练集特征值、测试集特征值、训练集目标值、测试集目标值
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

transfer = StandardScaler()
X_train = transfer.fit_transform(X_train)
X_test = transfer.transform(X_test)
estimator = KNeighborsClassifier()
estimator.fit(X_train, y_train)
KNeighborsClassifier()

y_pre = estimator.predict(X_test)
print("预测值是:\n", y_pre)
print("预测值和真实值的对比是:\n", y_pre == y_test)

# 构建人工神经网络
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, input_shape=(4, ), activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# 编译模型
# 指定损失函数，优化器，评价指标
# 多分类问题loss使用交叉熵，评价指标为准确性
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics="accuracy")

history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test))

# 模型预测
y_pre_proba = model.predict(X_test)         # 返回每个类别的概率
y_pre = model.predict_classes(X_test)       # 返回最大概率的类别

from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,y_pre))

# 模型评分
scores = []
for i in range(len(y_test)):
    if y_pre[i] == y_test[i]:
        scores.append(1)
    else:
        scores.append(0)

accuracy = sum(scores) / len(scores)
print(accuracy)

# 绘制训练曲线
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss/acc')
plt.gca().set_ylim(0, 1)
plt.show()