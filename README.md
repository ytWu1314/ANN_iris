
iris(鸢尾花)特征为 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'，如下图所示：

![鸢尾花.png](https://github.com/ytWu1314/ANN_iris/blob/master/%E9%B8%A2%E5%B0%BE%E8%8A%B1.png)


使用以下代码可以读如三种不同种类的鸢尾花数据：
from sklearn.datasets import load_iris
iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

读入的 iris 为字典，X 每行有4列，分别为上述特征值，y 为 0、1、2，分别代表三种鸢尾花：'setosa' 'versicolor' 'virginica'。
构建并训练一个ANN 网络模型，作出模型训练时的训练集、测试集准确率、损失对比图，并使用训练好的模型预测新的鸢尾花数据种类，以此对模型进行评价。
