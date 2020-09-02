# example of binary classification task
from numpy import where
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot


# define dataset
def classification(centers):
    x, y = make_blobs(n_samples=1000, centers=centers,
                      random_state=1)  # использоются для генерации капель точек с гауссовым распределением.

    # summarize dataset shape
    print("summarize dataset shape")
    print(x.shape, y.shape)
    # summarize observations by class label
    counter = Counter(y)  # словарь подсчитывающий количество неизменяемых объектов
    print("counter")
    print(counter)
    # summarize first few examples
    for i in range(10):
        print(x[i], y[i])
    # plot the dataset and color the by class label
    for label, _ in counter.items():
        row_ix = where(y == label)[0]  # возвращает элементы, выбранные из x или y в зависимости от условия .
        pyplot.scatter(x[row_ix, 0], x[row_ix, 1],
                       label=str(label))  # диаграмма разброса y от x с разным размером или цветом маркера.

    pyplot.legend()
    pyplot.show()


classification(3)
