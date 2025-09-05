import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("ЗАДАНИЕ 4")
print("=" * 60)

# 1. Загрузка данных
print("1. Загрузка и подготовка данных...")
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   Обучающая выборка: {X_train.shape}")
print(f"   Тестовая выборка: {X_test.shape}")

# 3. Масштабирование данных (важно для SVM)
print("\n2. Масштабирование данных...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("   ✅ Данные масштабированы")

# 4. Инициализация и обучение линейного SVM
print("\n3. Инициализация и обучение линейного SVM...")
svm_model = SVC(
    kernel='linear',          # Линейное ядро
    random_state=42,          # Для воспроизводимости
    probability=True          # Для возможности получения вероятностей
)

print(f"   Параметры модели: {svm_model}")

# Обучение модели
svm_model.fit(X_train_scaled, y_train)
print("   ✅ SVM модель обучена!")

# 5. Прогнозирование
print("\n4. Прогнозирование на тестовых данных...")
y_test_pred = svm_model.predict(X_test_scaled)
y_test_proba = svm_model.predict_proba(X_test_scaled)

print("   Первые 10 предсказаний:")
for i in range(10):
    actual = y_test[i]
    predicted = y_test_pred[i]
    actual_label = data.target_names[actual]
    predicted_label = data.target_names[predicted]
    status = "✓" if actual == predicted else "✗"
    print(f"   Образец {i}: {actual_label} -> {predicted_label} {status}")

# 6. Оценка эффективности с помощью accuracy_score
print("\n5. Оценка эффективности модели...")
accuracy = accuracy_score(y_test, y_test_pred)
print(f"   Accuracy Score: {accuracy:.4f} ({accuracy:.2%})")

# 7. Confusion Matrix
print("\n6. Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print("   Матрица ошибок:")
print("   " + " " * 15 + "Предсказано")
print("   " + " " * 10 + "Malignant  Benign")
print("   " + "Фактически" + " " * 5 + "-----------")
print(f"   Malignant    {cm[0, 0]:>8}    {cm[0, 1]:>6}")
print(f"   Benign       {cm[1, 0]:>8}    {cm[1, 1]:>6}")

# Визуализация Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data.target_names, 
            yticklabels=data.target_names,
            cbar_kws={'label': 'Количество образцов'})
plt.title('Confusion Matrix - Линейный SVM\n(Рак груди)', fontsize=14, fontweight='bold')
plt.ylabel('Истинные значения', fontsize=12)
plt.xlabel('Предсказанные значения', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Матрица ошибок сохранена в 'svm_confusion_matrix.png'")

# 8. Classification Report
print("\n7. Classification Report:")
print("   " + "=" * 60)
cr = classification_report(y_test, y_test_pred, 
                          target_names=data.target_names,
                          digits=4)
print(cr)
print("   " + "=" * 60)

# 9. Детальный анализ матрицы ошибок
print("\n8. Детальный анализ:")
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
print(f"   - Precision: {precision_malignant:.4f}")
print(f"   - Recall:    {recall_malignant:.4f}")
print(f"   - F1-score:  {f1_malignant:.4f}")

print(f"\n   Метрики для benign:")
print(f"   - Precision: {precision_benign:.4f}")
print(f"   - Recall:    {recall_benign:.4f}")
print(f"   - F1-score:  {f1_benign:.4f}")

# 10. Информация о опорных векторах
print("\n9. Информация о опорных векторах:")
print(f"   Количество опорных векторов: {len(svm_model.support_vectors_)}")
print(f"   Индексы опорных векторов: {svm_model.support_[:10]}...")  # Первые 10
print(f"   Коэффициенты: {svm_model.coef_[0][:5]}...")  # Первые 5 коэффициентов

# 11. Сравнение с логистической регрессией
print("\n10. Сравнение с логистической регрессией:")
# Для сравнения нужно иметь результаты LR, можно добавить если есть

# 12. Сохранение результатов
print("\n11. Сохранение результатов...")
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'prob_malignant': y_test_proba[:, 0],
    'prob_benign': y_test_proba[:, 1],
    'actual_label': [data.target_names[x] for x in y_test],
    'predicted_label': [data.target_names[x] for x in y_test_pred]
})

results_df['confidence'] = results_df[['prob_malignant', 'prob_benign']].max(axis=1)
results_df['is_correct'] = results_df['actual'] == results_df['predicted']

results_df.to_csv('svm_predictions.csv', index=False)
print("   ✅ Результаты сохранены в 'svm_predictions.csv'")

# 13. Итоговая оценка
print("\n12. Итоговая оценка модели SVM:")
if accuracy > 0.95:
    print("   🎉 Отличная точность! (>95%)")
elif accuracy > 0.9:
    print("   👍 Очень хорошая точность! (>90%)")
elif accuracy > 0.85:
    print("   ✅ Хорошая точность! (>85%)")
else:
    print("   ⚠️  Точность требует улучшения")

print(f"   Общая точность: {accuracy:.2%}")
print(f"   Количество опорных векторов: {len(svm_model.support_vectors_)}")
print(f"   Размер обучающей выборки: {len(X_train)}")

from sklearn.model_selection import GridSearchCV

print("=" * 60)
print("4: НЕЛИНЕЙНЫЙ SVM И НАСТРОЙКА")
print("=" * 60)

# 13. Нелинейный SVM с радиально-базисным ядром (RBF)
print("\n13. Нелинейный SVM с RBF ядром...")

# Инициализация RBF SVM
rbf_svm = SVC(
    kernel='rbf',           # Радиально-базисное ядро
    random_state=42,
    probability=True
)

print(f"   Параметры RBF SVM: {rbf_svm}")

# Обучение RBF SVM
rbf_svm.fit(X_train_scaled, y_train)
print("   ✅ RBF SVM модель обучена!")

# Прогнозирование
y_test_pred_rbf = rbf_svm.predict(X_test_scaled)

# Оценка эффективности
accuracy_rbf = accuracy_score(y_test, y_test_pred_rbf)
print(f"   Accuracy RBF SVM: {accuracy_rbf:.4f} ({accuracy_rbf:.2%})")

# Сравнение с линейным SVM
print(f"   Accuracy Linear SVM: {accuracy:.4f} ({accuracy:.2%})")

if accuracy_rbf > accuracy:
    print("   🔥 RBF SVM показал лучшую точность!")
elif accuracy_rbf < accuracy:
    print("   ℹ️  Linear SVM показал лучшую точность")
else:
    print("   ⚖️  Модели показали одинаковую точность")

# 14. Настройка гиперпараметров с GridSearchCV
print("\n14. Настройка гиперпараметров с GridSearchCV...")

# Определение пространства поиска параметров
param_grid = {
    'C': [0.1, 1, 10, 100],           # Параметр регуляризации
    'gamma': [0.001, 0.01, 0.1, 1],   # Параметр ядра RBF
    'kernel': ['rbf']                 # Используем только RBF ядро
}

print("   Пространство поиска параметров:")
print(f"   C: {param_grid['C']}")
print(f"   gamma: {param_grid['gamma']}")

# Инициализация GridSearchCV
grid_search = GridSearchCV(
    estimator=SVC(random_state=42, probability=True),
    param_grid=param_grid,
    cv=5,                    # 5-кратная кросс-валидация
    scoring='accuracy',      # Метрика для оптимизации
    n_jobs=-1,               # Использовать все ядра процессора
    verbose=1                # Вывод процесса поиска
)

print("   Запуск поиска по сетке...")
# Обучение GridSearchCV
grid_search.fit(X_train_scaled, y_train)

print("   ✅ Поиск по сетке завершен!")

# 15. Результаты GridSearchCV
print("\n15. Результаты настройки гиперпараметров:")

# Лучшие параметры
print(f"   Лучшие параметры: {grid_search.best_params_}")
print(f"   Лучшая точность: {grid_search.best_score_:.4f} ({grid_search.best_score_:.2%})")

# Лучшая модель
best_svm = grid_search.best_estimator_
print(f"   Лучшая модель: {best_svm}")

# 16. Оценка лучшей модели на тестовых данных
print("\n16. Оценка лучшей модели на тестовых данных...")

# Прогнозирование
y_test_pred_best = best_svm.predict(X_test_scaled)

# Метрики
accuracy_best = accuracy_score(y_test, y_test_pred_best)
print(f"   Accuracy лучшей модели: {accuracy_best:.4f} ({accuracy_best:.2%})")

# Confusion Matrix
cm_best = confusion_matrix(y_test, y_test_pred_best)
print("\n   Confusion Matrix лучшей модели:")
print("   " + " " * 15 + "Предсказано")
print("   " + " " * 10 + "Malignant  Benign")
print("   " + "Фактически" + " " * 5 + "-----------")
print(f"   Malignant    {cm_best[0, 0]:>8}    {cm_best[0, 1]:>6}")
print(f"   Benign       {cm_best[1, 0]:>8}    {cm_best[1, 1]:>6}")

# Classification Report
print("\n   Classification Report лучшей модели:")
print("   " + "=" * 50)
cr_best = classification_report(y_test, y_test_pred_best, 
                               target_names=data.target_names,
                               digits=4)
print(cr_best)
print("   " + "=" * 50)

# 17. Сравнение всех моделей
print("\n17. Сравнение всех SVM моделей:")
print("   Модель           | Accuracy | Параметры")
print("   " + "-" * 45)
print(f"   Linear SVM      | {accuracy:.4f}  | kernel='linear'")
print(f"   RBF SVM         | {accuracy_rbf:.4f}  | kernel='rbf' (по умолчанию)")
print(f"   Optimized RBF   | {accuracy_best:.4f}  | {grid_search.best_params_}")

# 18. Анализ результатов GridSearch
print("\n18. Анализ результатов GridSearchCV:")

# Результаты всех комбинаций
results_df = pd.DataFrame(grid_search.cv_results_)
print("   Топ-5 комбинаций параметров:")
top_results = results_df[['params', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score').head()
print(top_results.to_string(index=False))

# Визуализация результатов GridSearch
plt.figure(figsize=(12, 8))
scores = results_df['mean_test_score'].values.reshape(len(param_grid['C']), len(param_grid['gamma']))

plt.imshow(scores, cmap='viridis', aspect='auto')
plt.colorbar(label='Accuracy')
plt.xticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'])
plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
plt.xlabel('Gamma')
plt.ylabel('C')
plt.title('Результаты GridSearchCV\nТочность для разных комбинаций C и gamma', fontsize=14)

# Добавление значений в ячейки
for i in range(len(param_grid['C'])):
    for j in range(len(param_grid['gamma'])):
        plt.text(j, i, f'{scores[i, j]:.3f}', 
                ha='center', va='center', 
                color='white' if scores[i, j] > 0.5 else 'black')

plt.tight_layout()
plt.savefig('grid_search_results.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Визуализация GridSearch сохранена в 'grid_search_results.png'")

# 19. Сохранение лучшей модели
print("\n19. Сохранение результатов...")

# Сохранение прогнозов лучшей модели
best_results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred_best,
    'actual_label': [data.target_names[x] for x in y_test],
    'predicted_label': [data.target_names[x] for x in y_test_pred_best]
})

best_results_df.to_csv('best_svm_predictions.csv', index=False)
print("   ✅ Прогнозы лучшей модели сохранены в 'best_svm_predictions.csv'")

# Сохранение параметров
params_df = pd.DataFrame([grid_search.best_params_])
params_df['best_score'] = grid_search.best_score_
params_df['test_accuracy'] = accuracy_best
params_df.to_csv('best_svm_parameters.csv', index=False)
print("   ✅ Параметры лучшей модели сохранены в 'best_svm_parameters.csv'")

# 20. Итоговый вывод
print("\n20. Итоговые результаты:")
print(f"   Лучшая модель: SVM с параметрами {grid_search.best_params_}")
print(f"   Точность на кросс-валидации: {grid_search.best_score_:.2%}")
print(f"   Точность на тестовых данных: {accuracy_best:.2%}")
print(f"   Улучшение по сравнению с Linear SVM: {((accuracy_best - accuracy) / accuracy * 100):.1f}%")
