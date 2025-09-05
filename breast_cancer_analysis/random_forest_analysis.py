import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("ЗАДАНИЕ 5")
print("=" * 60)

# 1. Загрузка данных
print("1. Загрузка и подготовка данных...")
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   Признаков: {X.shape[1]}, Образцов: {X.shape[0]}")
print(f"   Обучающая выборка: {X_train.shape}")
print(f"   Тестовая выборка: {X_test.shape}")

# 2. Эксперименты с разным количеством деревьев
print("\n2. Эксперименты с RandomForestClassifier...")
n_estimators_list = [50, 100, 200]
results = {}

for n_estimators in n_estimators_list:
    print(f"   Обучение RandomForest с {n_estimators} деревьями...")
    
    # Инициализация и обучение модели
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1  # Использовать все ядра процессора
    )
    
    rf_model.fit(X_train, y_train)
    
    # Прогнозирование и оценка
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results[n_estimators] = {
        'model': rf_model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"   Accuracy с {n_estimators} деревьями: {accuracy:.4f} ({accuracy:.2%})")

# 3. Сравнение результатов
print("\n3. Сравнение результатов для разного количества деревьев:")
print("   Деревья | Accuracy  | Улучшение")
print("   " + "-" * 30)

best_accuracy = 0
best_n_estimators = 0

for i, n_estimators in enumerate(n_estimators_list):
    accuracy = results[n_estimators]['accuracy']
    improvement = accuracy - results[n_estimators_list[0]]['accuracy'] if i > 0 else 0
    
    print(f"   {n_estimators:>7} | {accuracy:.6f} | {improvement:+.4f}" if i > 0 else f"   {n_estimators:>7} | {accuracy:.6f} | -")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_estimators = n_estimators

print(f"\n   Лучший результат: {best_accuracy:.4f} с {best_n_estimators} деревьями")

# 4. Выбор лучшей модели
print(f"\n4. Выбираем лучшую модель с {best_n_estimators} деревьями...")
best_rf_model = results[best_n_estimators]['model']
y_test_pred = best_rf_model.predict(X_test)

# Детальная оценка лучшей модели
print("\n5. Детальная оценка лучшей модели Random Forest:")
accuracy = accuracy_score(y_test, y_test_pred)
print(f"   Accuracy: {accuracy:.4f} ({accuracy:.2%})")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\n   Confusion Matrix:")
print("   " + " " * 15 + "Предсказано")
print("   " + " " * 10 + "Malignant  Benign")
print("   " + "Фактически" + " " * 5 + "-----------")
print(f"   Malignant    {cm[0, 0]:>8}    {cm[0, 1]:>6}")
print(f"   Benign       {cm[1, 0]:>8}    {cm[1, 1]:>6}")

# Classification Report
print("\n   Classification Report:")
print("   " + "=" * 50)
cr = classification_report(y_test, y_test_pred, 
                          target_names=data.target_names,
                          digits=4)
print(cr)
print("   " + "=" * 50)

# 6. Важность признаков (Feature Importances)
print("\n6. Анализ важности признаков...")

# Извлечение важности признаков
feature_importances = best_rf_model.feature_importances_
print(f"   Размерность важности признаков: {feature_importances.shape}")

# Создание DataFrame для удобства
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
})

# Сортировка по важности
importance_df = importance_df.sort_values('importance', ascending=False)

print("\n   Топ-10 самых важных признаков:")
for i, (feature, importance) in enumerate(zip(importance_df['feature'][:10], importance_df['importance'][:10])):
    print(f"   {i+1:2d}. {feature:25s}: {importance:.4f} ({importance:.2%})")

# 7. Визуализация важности признаков
print("\n7. Визуализация важности признаков...")

plt.figure(figsize=(12, 8))

# Барплот для топ-15 признаков
top_n = 15
top_features = importance_df.head(top_n)

plt.subplot(2, 1, 1)
bars = plt.barh(range(top_n), top_features['importance'][::-1], color='skyblue')
plt.yticks(range(top_n), top_features['feature'][::-1], fontsize=10)
plt.xlabel('Важность признака', fontsize=12)
plt.title(f'Топ-{top_n} самых важных признаков\nRandom Forest Classifier', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Добавление значений на барплот
for i, (bar, importance) in enumerate(zip(bars, top_features['importance'][::-1])):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{importance:.3f}', ha='left', va='center', fontsize=9)

# Круговой график для топ-10 признаков
plt.subplot(2, 1, 2)
top_10 = importance_df.head(10)
other_importance = importance_df['importance'][10:].sum()


explode = [0.1] + [0] * 9 + [0]  # 10 для топ-признаков + 1 для "Другие"

# Добавляем "Другие" для круговой диаграммы
pie_data = list(top_10['importance']) + [other_importance]
pie_labels = list(top_10['feature']) + ['Другие признаки']

plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90, explode=explode)
plt.axis('equal')
plt.title('Распределение важности признаков\n(Топ-10 + остальные)', fontsize=12)

plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
plt.savefig('feature_importances.pdf', bbox_inches='tight')
plt.show()
print("   ✅ Графики важности признаков сохранены")

# 8. Дополнительный анализ
print("\n8. Дополнительный анализ:")

# Статистика важности признаков
print(f"   Средняя важность: {feature_importances.mean():.4f}")
print(f"   Максимальная важность: {feature_importances.max():.4f}")
print(f"   Минимальная важность: {feature_importances.min():.4f}")
print(f"   Стандартное отклонение: {feature_importances.std():.4f}")

# Количество признаков с высокой важностью
high_importance = sum(feature_importances > 0.05)
medium_importance = sum((feature_importances > 0.01) & (feature_importances <= 0.05))
low_importance = sum(feature_importances <= 0.01)

print(f"\n   Признаков с высокой важностью (>0.05): {high_importance}")
print(f"   Признаков со средней важностью (0.01-0.05): {medium_importance}")
print(f"   Признаков с низкой важностью (≤0.01): {low_importance}")

# 9. Сохранение результатов
print("\n9. Сохранение результатов...")

# Сохранение важности признаков
importance_df.to_csv('feature_importances.csv', index=False)
print("   ✅ Важность признаков сохранена в 'feature_importances.csv'")

# Сохранение прогнозов
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'actual_label': [data.target_names[x] for x in y_test],
    'predicted_label': [data.target_names[x] for x in y_test_pred]
})
predictions_df.to_csv('random_forest_predictions.csv', index=False)
print("   ✅ Прогнозы сохранены в 'random_forest_predictions.csv'")

# Сохранение метрик
metrics_df = pd.DataFrame({
    'n_estimators': list(results.keys()),
    'accuracy': [results[n]['accuracy'] for n in results.keys()]
})
metrics_df.to_csv('random_forest_metrics.csv', index=False)
print("   ✅ Метрики сохранены в 'random_forest_metrics.csv'")

# 10. Итоговый вывод
print("\n10. Итоговые результаты Random Forest:")
print(f"   Лучшее количество деревьев: {best_n_estimators}")
print(f"   Лучшая точность: {best_accuracy:.4f} ({best_accuracy:.2%})")
print(f"   Самый важный признак: {importance_df['feature'].iloc[0]} ({importance_df['importance'].iloc[0]:.2%})")
print(f"   Количество информативных признаков: {high_importance + medium_importance}")
