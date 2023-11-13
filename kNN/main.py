import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


iris_data_path = 'Iris.csv'
iris_data = pd.read_csv(iris_data_path)
iris_data.drop('Id', axis=1, inplace=True)

# Отделение признаков от целевой переменной
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']

# Нормализация данных признаков
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Подбор оптимального k
cv_scores = []
k_range = range(1, 51)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
optimal_k = k_range[np.argmax(cv_scores)]


# Создание и отображение графиков
def create_and_show_plots(X, y, features, normalized=False):
    plt.figure(figsize=(18, 12))
    plot_index = 1  # Индекс для подграфиков
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            plt.subplot(2, 3, plot_index)
            sns.scatterplot(x=X.iloc[:, i], y=X.iloc[:, j], hue=y)
            plt.title(f"{'Normalized ' if normalized else ''}{features[i]} vs {features[j]}")
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plot_index += 1
    plt.tight_layout()
    plt.show()


# Визуализация для ненормализованных данных
features = X.columns.tolist()
create_and_show_plots(X, y, features)

# Визуализация для нормализованных данных
X_train_df = pd.DataFrame(X_train, columns=X.columns)
create_and_show_plots(X_train_df, y_train, features, normalized=True)

# Модель k-NN
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)


# Определение класса объекта
def predict_species(new_data):
    new_data_df = pd.DataFrame([new_data], columns=X.columns)
    new_data_normalized = scaler.transform(new_data_df)
    species = knn.predict(new_data_normalized)
    return species[0]


new_sample = [5.1, 3.5, 1.4, 0.2]
predicted_species = predict_species(new_sample)
print(f"Класс нового объекта: {predicted_species}")
