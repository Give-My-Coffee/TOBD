import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка набора данных
data = load_breast_cancer()
X, y = data.data, data.target

# Вывод информации о данных
print("=" * 50)
print("Задаие 1")
print("=" * 50)

print(f"Форма набора данных (X): {X.shape}")
print(f"Форма целевых значений (y): {y.shape}")
print(f"Целевое распределение: {np.bincount(y)}")
print(f"Метки классов: {data.target_names}")

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,       # 30% тестовых данных
    train_size=0.7,      # 70% тренировочных данных
    random_state=42,     # для воспроизводимости результатов
    stratify=y           # сохраняем пропорции классов в выборках
)

# Вывод информации о разделенных данных
print("\n" + "=" * 50)
print("РАЗДЕЛЕНИЕ ДАННЫХ")
print("=" * 50)

print(f"Обучающая выборка (X_train): {X_train.shape}")
print(f"Тестовая выборка (X_test): {X_test.shape}")
print(f"Обучающие целевые значения (y_train): {y_train.shape}")
print(f"Тестовые целевые значения (y_test): {y_test.shape}")

print(f"\nРаспределение в обучающей выборке: {np.bincount(y_train)}")
print(f"Распределение в тестовой выборке: {np.bincount(y_test)}")

# Проверка пропорций
print(f"\nПропорции классов:")
print(f"Обучающая: {np.bincount(y_train)[0]/len(y_train):.2%} / {np.bincount(y_train)[1]/len(y_train):.2%}")
print(f"Тестовая: {np.bincount(y_test)[0]/len(y_test):.2%} / {np.bincount(y_test)[1]/len(y_test):.2%}")
print(f"Исходная: {np.bincount(y)[0]/len(y):.2%} / {np.bincount(y)[1]/len(y):.2%}")

# Дополнительная информация
print(f"\nРазмер тестовой выборки: {len(X_test)} samples ({len(X_test)/len(X):.1%})")
print(f"Размер обучающей выборки: {len(X_train)} samples ({len(X_train)/len(X):.1%})")