import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Функция для генерации двухклассовой выборки
def generate_data(n):
    # Класс 1: Центрирован вокруг (-5, -5)
    x1 = np.random.normal(-5, 3, n)
    y1 = np.random.normal(-5, 3, n)

    # Класс -1: Центрирован вокруг (5, 5)
    x2 = np.random.normal(5, 3, n)
    y2 = np.random.normal(5, 3, n)

    X = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))
    y = np.hstack((np.ones(n), -np.ones(n)))
    return X, y


# Функция для обучения SVM и рисования разделяющей прямой
def train_and_plot(X, y, new_point=None):
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    colors = np.where(y == 1, 'red', 'blue')
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Сетка для рисования границы решения
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Граница решения
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')

    # Добавление и классификация новой точки
    if new_point is not None:
        prediction = clf.predict([new_point])
        color = 'red' if prediction == 1 else 'blue'
        plt.scatter(new_point[0], new_point[1], c=color, s=100, edgecolor='k', marker='s')

    plt.show()


# Генерация выборки
X, y = generate_data(50)

# Обучение и визуализация
train_and_plot(X, y)

# Определение новой точки
new_point = np.random.uniform(-10, 10, 2)

# Обучение, визуализация с новой точкой
train_and_plot(X, y, new_point)
