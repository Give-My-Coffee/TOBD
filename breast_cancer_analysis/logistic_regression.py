import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("ЗАДАНИЕ 2")
print("=" * 60)

# Загрузка и разделение данных
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 1. Инициализация модели логистической регрессии
print("1. Инициализация модели LogisticRegression...")
model = LogisticRegression(
    random_state=42,    # для воспроизводимости результатов
    max_iter=10000,     # увеличиваем количество итераций для сходимости
    solver='liblinear'  # хороший solver для небольших datasets
)

print(f"   Параметры модели: {model}")

# 2. Обучение модели на обучающих данных
print("2. Обучение модели на тренировочных данных...")
model.fit(X_train, y_train)

print("   ✅ Модель успешно обучена!")

# 3. Получение информации о обученной модели
print("3. Информация о обученной модели:")

# Коэффициенты модели
print(f"   Количество коэффициентов: {len(model.coef_[0])}")
print(f"   Intercept (свободный член): {model.intercept_[0]:.4f}")

# Первые 5 коэффициентов для примера
print(f"   Первые 5 коэффициентов: {model.coef_[0][:5]}")

# 4. Предсказания на тренировочных данных
print("4. Делаем предсказания на тренировочных данных...")
y_train_pred = model.predict(X_train)

# 5. Оценка качества на тренировочных данных
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"   Accuracy на тренировочных данных: {train_accuracy:.4f} ({train_accuracy:.2%})")

# 6. Предсказания на тестовых данных
print("5. Делаем предсказания на тестовых данных...")
y_test_pred = model.predict(X_test)

# 7. Оценка качества на тестовых данных
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"   Accuracy на тестовых данных: {test_accuracy:.4f} ({test_accuracy:.2%})")

# 8. Детальная статистика
print("\n6. Детальная статистика модели:")
print(classification_report(y_test, y_test_pred, target_names=data.target_names))

# 9. Матрица ошибок
print("7. Матрица ошибок (Confusion Matrix):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data.target_names, 
            yticklabels=data.target_names)
plt.title('Матрица ошибок логистической регрессии')
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.savefig('confusion_matrix.png')
plt.show()

# 8. Прогнозирование на тестовом наборе данных
print("8. Прогнозирование результатов на тестовых данных...")
y_test_pred = model.predict(X_test)

print("   Первые 10 предсказаний:")
for i in range(10):
    actual = y_test[i]
    predicted = y_test_pred[i]
    actual_label = data.target_names[actual]
    predicted_label = data.target_names[predicted]
    status = "✓" if actual == predicted else "✗"
    print(f"   Образец {i}: {actual_label} -> {predicted_label} {status}")

# 9. Прогнозирование вероятностей
print("\n9. Прогнозирование вероятностей положительного исхода...")
y_test_proba = model.predict_proba(X_test)

print("   Форма массива вероятностей:", y_test_proba.shape)
print("   Первые 5 прогнозов вероятностей:")

for i in range(5):
    proba_0 = y_test_proba[i][0]  # Вероятность класса 0 (malignant)
    proba_1 = y_test_proba[i][1]  # Вероятность класса 1 (benign)
    actual_class = y_test[i]
    predicted_class = y_test_pred[i]
    
    print(f"   Образец {i}:")
    print(f"      P(malignant) = {proba_0:.4f} ({proba_0:.2%})")
    print(f"      P(benign)    = {proba_1:.4f} ({proba_1:.2%})")
    print(f"      Фактический класс: {data.target_names[actual_class]}")
    print(f"      Предсказанный класс: {data.target_names[predicted_class]}")
    print()

# 10. Анализ порога классификации
print("10. Анализ различных порогов классификации...")
print("   По умолчанию используется порог 0.5:")

# Считаем количество образцов с разной уверенностью модели
high_confidence = sum(1 for proba in y_test_proba if max(proba) > 0.8)
medium_confidence = sum(1 for proba in y_test_proba if 0.6 < max(proba) <= 0.8)
low_confidence = sum(1 for proba in y_test_proba if max(proba) <= 0.6)

print(f"   Высокая уверенность (>80%): {high_confidence} образцов")
print(f"   Средняя уверенность (60-80%): {medium_confidence} образцов")
print(f"   Низкая уверенность (≤60%): {low_confidence} образцов")

# 11. Пример изменения порога
print("\n11. Пример с измененным порогом классификации (0.7 для benign):")
custom_threshold = 0.7
y_test_custom = (y_test_proba[:, 1] >= custom_threshold).astype(int)

print("   Первые 5 образцов с порогом 0.7:")
for i in range(5):
    proba_benign = y_test_proba[i][1]
    original_pred = y_test_pred[i]
    custom_pred = y_test_custom[i]
    
    print(f"   Образец {i}: P(benign)={proba_benign:.3f}")
    print(f"      Стандартное предсказание: {data.target_names[original_pred]}")
    print(f"      С порогом 0.7: {data.target_names[custom_pred]}")
    if original_pred != custom_pred:
        print(f"      ⚠️  Изменилось предсказание!")
    print()

# 12. Сохранение результатов
print("12. Сохранение прогнозов и вероятностей...")
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'prob_malignant': y_test_proba[:, 0],
    'prob_benign': y_test_proba[:, 1],
    'actual_label': [data.target_names[x] for x in y_test],
    'predicted_label': [data.target_names[x] for x in y_test_pred]
})

# Добавляем информацию об уверенности
results_df['confidence'] = results_df[['prob_malignant', 'prob_benign']].max(axis=1)
results_df['is_correct'] = results_df['actual'] == results_df['predicted']

print("   Первые 5 строк результатов:")
print(results_df.head().to_string())

# Сохраняем в CSV
results_df.to_csv('logistic_regression_predictions.csv', index=False)
print("   ✅ Результаты сохранены в 'logistic_regression_predictions.csv'")

# 13. Статистика по прогнозам
print("\n13. Статистика прогнозов:")
correct_predictions = sum(results_df['is_correct'])
total_predictions = len(results_df)
accuracy = correct_predictions / total_predictions

print(f"   Правильных прогнозов: {correct_predictions}/{total_predictions}")
print(f"   Точность: {accuracy:.4f} ({accuracy:.2%})")
print(f"   Средняя уверенность модели: {results_df['confidence'].mean():.4f}")
print(f"   Уверенность правильных прогнозов: {results_df[results_df['is_correct']]['confidence'].mean():.4f}")
print(f"   Уверенность ошибок: {results_df[~results_df['is_correct']]['confidence'].mean():.4f}")
# 14. Оценка модели с помощью метрик
print("14. Оценка модели с помощью метрик...")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Вычисление accuracy_score
accuracy = accuracy_score(y_test, y_test_pred)
print(f"   Accuracy Score: {accuracy:.4f} ({accuracy:.2%})")

# 15. Confusion Matrix
print("\n15. Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print("   Матрица ошибок:")
print("   " + " " * 15 + "Предсказано")
print("   " + " " * 10 + "Malignant  Benign")
print("   " + "Фактически" + " " * 5 + "-----------")
print(f"   Malignant    {cm[0, 0]:>8}    {cm[0, 1]:>6}")
print(f"   Benign       {cm[1, 0]:>8}    {cm[1, 1]:>6}")

# Визуализация Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=data.target_names, 
            yticklabels=data.target_names,
            cbar_kws={'label': 'Количество образцов'})
plt.title('Confusion Matrix - Логистическая регрессия\n(Rak Grudi)', fontsize=14, fontweight='bold')
plt.ylabel('Истинные значения', fontsize=12)
plt.xlabel('Предсказанные значения', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Матрица ошибок сохранена в 'confusion_matrix_detailed.png'")

# 16. Детальный анализ Confusion Matrix
print("\n16. Анализ Confusion Matrix:")
tn, fp, fn, tp = cm.ravel()
print(f"   True Negative (TN):  {tn} - правильно предсказаны malignant")
print(f"   False Positive (FP): {fp} - benign ошибочно предсказаны как malignant")
print(f"   False Negative (FN): {fn} - malignant ошибочно предсказаны как benign")
print(f"   True Positive (TP):  {tp} - правильно предсказаны benign")

# Вычисление дополнительных метрик
precision_malignant = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_malignant = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_malignant = 2 * (precision_malignant * recall_malignant) / (precision_malignant + recall_malignant) if (precision_malignant + recall_malignant) > 0 else 0

precision_benign = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_benign = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_benign = 2 * (precision_benign * recall_benign) / (precision_benign + recall_benign) if (precision_benign + recall_benign) > 0 else 0

print(f"\n   Метрики для malignant:")
print(f"   - Precision: {precision_malignant:.4f} ({precision_malignant:.2%})")
print(f"   - Recall:    {recall_malignant:.4f} ({recall_malignant:.2%})")
print(f"   - F1-score:  {f1_malignant:.4f} ({f1_malignant:.2%})")

print(f"\n   Метрики для benign:")
print(f"   - Precision: {precision_benign:.4f} ({precision_benign:.2%})")
print(f"   - Recall:    {recall_benign:.4f} ({recall_benign:.2%})")
print(f"   - F1-score:  {f1_benign:.4f} ({f1_benign:.2%})")

# 17. Classification Report
print("\n17. Classification Report:")
print("   " + "=" * 50)
cr = classification_report(y_test, y_test_pred, 
                          target_names=data.target_names,
                          digits=4)
print(cr)
print("   " + "=" * 50)

# 18. Интерпретация результатов
print("\n18. Интерпретация результатов:")
print("   Precision (Точность):")
print("   - Доля правильно предсказанных positive среди всех предсказанных positive")
print("   Recall (Полнота):")
print("   - Доля правильно предсказанных positive среди всех фактических positive")
print("   F1-score:")
print("   - Гармоническое среднее между Precision и Recall")

# 19. Сохранение всех метрик
print("\n19. Сохранение метрик в файл...")
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision_malignant', 'Recall_malignant', 'F1_malignant', 
               'Precision_benign', 'Recall_benign', 'F1_benign'],
    'Value': [accuracy, precision_malignant, recall_malignant, f1_malignant,
              precision_benign, recall_benign, f1_benign],
    'Percentage': [f"{accuracy:.2%}", f"{precision_malignant:.2%}", f"{recall_malignant:.2%}", f"{f1_malignant:.2%}",
                   f"{precision_benign:.2%}", f"{recall_benign:.2%}", f"{f1_benign:.2%}"]
})

metrics_df.to_csv('model_metrics.csv', index=False)
print("   ✅ Метрики сохранены в 'model_metrics.csv'")

# 20. Итоговая оценка модели
print("\n20. Итоговая оценка модели:")
if accuracy > 0.9:
    print("   🎉 Отличная точность модели! (>90%)")
elif accuracy > 0.8:
    print("   👍 Хорошая точность модели! (>80%)")
else:
    print("   ⚠️  Точность модели требует улучшения")

print(f"   Общая точность: {accuracy:.2%}")
print(f"   Ошибки классификации: {fp + fn} из {len(y_test)} образцов")
print(f"   Правильные предсказания: {tp + tn} из {len(y_test)} образцов")