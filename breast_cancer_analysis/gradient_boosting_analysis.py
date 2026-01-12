# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("=" * 60)
print("ЗАДАНИЕ 5")
print("=" * 60)

# 1. Загрузка данных
print("1. Загрузка и подготовка данных...")
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   Обучающая выборка: {X_train.shape}")
print(f"   Тестовая выборка: {X_test.shape}")

# 2. Эксперименты с разными параметрами Gradient Boosting
print("\n2. Эксперименты с GradientBoostingClassifier...")

# Параметры для экспериментов
n_estimators_list = [50, 100, 200]
learning_rates = [0.01, 0.1, 0.2]
results = []

start_time = time.time()

for n_estimators in n_estimators_list:
    for learning_rate in learning_rates:
        print(f"   Обучение GBM: {n_estimators} trees, LR={learning_rate}...")
        
        # Инициализация и обучение модели
        gbm_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
            subsample=0.8,  # Stochastic Gradient Boosting
            max_depth=3     # Ограничение глубины деревьев
        )
        
        gbm_model.fit(X_train, y_train)
        
        # Прогнозирование и оценка
        y_pred = gbm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Сохраняем результаты
        result = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'model': gbm_model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        results.append(result)
        
        print(f"   Accuracy: {accuracy:.4f} ({accuracy:.2%})")

end_time = time.time()
print(f"   Время обучения: {end_time - start_time:.2f} секунд")

# 3. Сравнение результатов
print("\n3. Сравнение результатов Gradient Boosting:")
print("   Trees | Learning Rate | Accuracy  ")
print("   " + "-" * 35)

best_accuracy = 0
best_params = {}

for result in results:
    n_est = result['n_estimators']
    lr = result['learning_rate']
    acc = result['accuracy']
    
    print(f"   {n_est:>5} | {lr:>13} | {acc:.6f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_params = {'n_estimators': n_est, 'learning_rate': lr}
        best_model = result['model']

print(f"\n   Лучший результат: {best_accuracy:.4f} с параметрами: {best_params}")

# 4. Детальная оценка лучшей модели
print(f"\n4. Детальная оценка лучшей модели GBM ({best_params})...")
y_test_pred = best_model.predict(X_test)

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

# 5. Анализ кривой обучения
print("\n5. Анализ кривой обучения...")

# Получаем staged predictions для анализа развития модели
train_scores = []
test_scores = []

for i, y_pred in enumerate(best_model.staged_predict(X_test)):
    acc = accuracy_score(y_test, y_pred)
    test_scores.append(acc)

for i, y_pred in enumerate(best_model.staged_predict(X_train)):
    acc = accuracy_score(y_train, y_pred)
    train_scores.append(acc)

# Визуализация кривой обучения
plt.figure(figsize=(12, 6))

plt.plot(train_scores, label='Обучающая выборка', linewidth=2)
plt.plot(test_scores, label='Тестовая выборка', linewidth=2)
plt.xlabel('Количество деревьев', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Кривая обучения Gradient Boosting\n', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=best_accuracy, color='r', linestyle='--', alpha=0.7, label=f'Лучшая точность: {best_accuracy:.3f}')

plt.tight_layout()
plt.savefig('gbm_learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Кривая обучения сохранена в 'gbm_learning_curve.png'")

# 6. Важность признаков для GBM
print("\n6. Анализ важности признаков для Gradient Boosting...")

feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': data.feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("\n   Топ-10 самых важных признаков (GBM):")
for i, (feature, importance) in enumerate(zip(importance_df['feature'][:10], importance_df['importance'][:10])):
    print(f"   {i+1:2d}. {feature:25s}: {importance:.4f} ({importance:.2%})")

# Сравнение с Random Forest
print("\n7. Сравнение с Random Forest:")
# (Предполагая, что у тебя есть результаты RF)
rf_accuracy = 0.9415  # Из предыдущих результатов
improvement = ((accuracy - rf_accuracy) / rf_accuracy) * 100

print(f"   Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy:.2%})")
print(f"   Gradient Boosting Accuracy: {accuracy:.4f} ({accuracy:.2%})")
print(f"   Разница: {accuracy - rf_accuracy:+.4f} ({improvement:+.1f}%)")

if accuracy > rf_accuracy:
    print("   🎉 Gradient Boosting показал лучшую точность!")
elif accuracy < rf_accuracy:
    print("   ℹ️  Random Forest показал лучшую точность")
else:
    print("   ⚖️  Модели показали одинаковую точность")

# 8. Дополнительный анализ
print("\n8. Дополнительный анализ GBM:")

# Информация о модели
print(f"   Количество деревьев: {best_params['n_estimators']}")
print(f"   Learning rate: {best_params['learning_rate']}")
print(f"   Глубина деревьев: {best_model.max_depth}")
print(f"   Subsample: {best_model.subsample}")

# Анализ переобучения
train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
print(f"   Accuracy на обучающих данных: {train_accuracy:.4f} ({train_accuracy:.2%})")
print(f"   Разница train-test: {train_accuracy - accuracy:.4f}")

if train_accuracy - accuracy > 0.05:
    print("   ⚠️  Возможное переобучение")
else:
    print("   ✅ Хорошее обобщение")

# 9. Сохранение результатов
print("\n9. Сохранение результатов...")

# Сохранение всех результатов
all_results_df = pd.DataFrame([{
    'n_estimators': r['n_estimators'],
    'learning_rate': r['learning_rate'],
    'accuracy': r['accuracy']
} for r in results])

all_results_df.to_csv('gbm_all_results.csv', index=False)
print("   ✅ Все результаты экспериментов сохранены в 'gbm_all_results.csv'")

# Сохранение прогнозов лучшей модели
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'actual_label': [data.target_names[x] for x in y_test],
    'predicted_label': [data.target_names[x] for x in y_test_pred]
})
predictions_df.to_csv('gbm_predictions.csv', index=False)
print("   ✅ Прогнозы сохранены в 'gbm_predictions.csv'")

# Сохранение важности признаков
importance_df.to_csv('gbm_feature_importances.csv', index=False)
print("   ✅ Важность признаков сохранена в 'gbm_feature_importances.csv'")

# 10. Итоговый вывод
print("\n10. Итоговые результаты Gradient Boosting:")
print(f"   Лучшие параметры: {best_params}")
print(f"   Лучшая точность: {best_accuracy:.4f} ({best_accuracy:.2%})")
print(f"   Самый важный признак: {importance_df['feature'].iloc[0]} ({importance_df['importance'].iloc[0]:.2%})")
print(f"   Время обучения: {end_time - start_time:.2f} секунд")
