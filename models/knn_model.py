# models/knn_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def train_knn_model(csv_path, n_neighbors=5):
    # Загрузка данных
    df = pd.read_csv(csv_path)

    # Выделение признаков и целевой переменной
    X = df.drop("Купил", axis=1)
    y = df["Купил"]

    # Нормализация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Обучение модели KNN
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # Возвращаем модель и тестовые данные
    return model, X_test, y_test
