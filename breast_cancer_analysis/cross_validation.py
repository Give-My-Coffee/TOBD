import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("=" * 70)
print("ПРОВЕРКА ЛУЧШИХ МОДЕЛЕЙ")
print("=" * 70)

# 1. Загрузка и подготовка данных
print("1. Загрузка и подготовка данных...")
data = load_breast_cancer()
X, y = data.data, data.target

# Масштабирование данных (важно для SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Размер данных: {X.shape}")
print(f"   Классы: {np.bincount(y)}")
print(f"   Масштабирование применено")

# 2. Выбор двух лучших моделей из предыдущего анализа
print("\n2. Выбор моделей для перекрестной проверки...")
print("   🥇 Linear SVM - лучшая точность (98.25%)")
print("   🥈 Gradient Boosting - вторая по точности (97.66%)")

# Инициализация моделей с лучшими параметрами
models = {
    'Linear SVM': SVC(kernel='linear', random_state=42, probability=True),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.2,
        random_state=42,
        subsample=0.8,
        max_depth=3
    )
}

# 3. Настройка перекрестной проверки
print("\n3. Настройка перекрестной проверки...")
k = 5  # 5-кратная перекрестная проверка
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

print(f"   Метод: {k}-кратная стратифицированная перекрестная проверка")
print(f"   Количество фолдов: {k}")
print(f"   Стратификация: Да (сохранение пропорций классов)")

# 4. Выполнение перекрестной проверки
print("\n4. Выполнение перекрестной проверки...")
results = {}

for model_name, model in models.items():
    print(f"   Проверка модели: {model_name}")
    
    start_time = time.time()
    
    # Выполнение cross-validation
    cv_scores = cross_val_score(
        estimator=model,
        X=X_scaled if 'SVM' in model_name else X,  # Для SVM используем масштабированные данные
        y=y,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1  # Использовать все ядра процессора
    )
    
    end_time = time.time()
    
    results[model_name] = {
        'scores': cv_scores,
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'time': end_time - start_time
    }
    
    print(f"     Время выполнения: {results[model_name]['time']:.2f} сек")
    print(f"     Scores: {cv_scores}")

# 5. Анализ результатов
print("\n5. Анализ результатов перекрестной проверки:")
print("   Модель           | Средняя точность | Стандартное отклонение | Время (сек)")
print("   " + "-" * 80)

for model_name, result in results.items():
    print(f"   {model_name:16s} | {result['mean_accuracy']:15.4f} | {result['std_accuracy']:19.4f} | {result['time']:10.2f}")

# 6. Детальная статистика
print("\n6. Детальная статистика по фолдам:")
for model_name, result in results.items():
    print(f"\n   {model_name}:")
    for i, score in enumerate(result['scores']):
        print(f"     Фолд {i+1}: {score:.4f} ({score:.2%})")

# 7. Визуализация результатов
print("\n7. Визуализация результатов перекрестной проверки...")

plt.figure(figsize=(14, 10))

# График 1: Сравнение точности по фолдам
plt.subplot(2, 2, 1)
x_pos = np.arange(k)
width = 0.35

for i, (model_name, result) in enumerate(results.items()):
    plt.bar(x_pos + i*width, result['scores'], width, label=model_name, alpha=0.8)

plt.xlabel('Номер фолда', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Точность по фолдам перекрестной проверки', fontsize=14, fontweight='bold')
plt.xticks(x_pos + width/2, [f'Фолд {i+1}' for i in range(k)])
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: Сравнение средних значений
plt.subplot(2, 2, 2)
model_names = list(results.keys())
means = [results[name]['mean_accuracy'] for name in model_names]
stds = [results[name]['std_accuracy'] for name in model_names]

bars = plt.bar(model_names, means, yerr=stds, capsize=10, alpha=0.8, color=['skyblue', 'lightgreen'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Средняя точность ± стандартное отклонение', fontsize=14, fontweight='bold')
plt.ylim(0.9, 1.0)

# Добавление значений на bars
for bar, mean, std in zip(bars, means, stds):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{mean:.4f} (±{std:.4f})', ha='center', va='bottom', fontsize=10)

# График 3: Boxplot распределения точности
plt.subplot(2, 2, 3)
all_scores = [results[name]['scores'] for name in model_names]
plt.boxplot(all_scores, labels=model_names)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Распределение точности по фолдам', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# График 4: Сравнение со оригинальными результатами
plt.subplot(2, 2, 4)
# Оригинальные результаты из предыдущего анализа
original_accuracies = {
    'Linear SVM': 0.9825,
    'Gradient Boosting': 0.9766
}

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, [original_accuracies[name] for name in model_names], 
        width, label='Оригинальная оценка', alpha=0.8)
plt.bar(x + width/2, means, width, label='CV оценка', alpha=0.8)

plt.xlabel('Модели', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Сравнение с оригинальной оценкой', fontsize=14, fontweight='bold')
plt.xticks(x, model_names)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Визуализации сохранены в 'cross_validation_results.png'")

# 8. Статистический анализ
print("\n8. Статистический анализ:")

# Проверка значимости различий
from scipy import stats

linear_svm_scores = results['Linear SVM']['scores']
gb_scores = results['Gradient Boosting']['scores']

# t-test для парных выборок
t_stat, p_value = stats.ttest_rel(linear_svm_scores, gb_scores)

print(f"   t-статистика: {t_stat:.4f}")
print(f"   p-value: {p_value:.4f}")

if p_value < 0.05:
    print("   📊 Различия статистически значимы (p < 0.05)")
    if t_stat > 0:
        print("   ✅ Linear SVM значимо лучше Gradient Boosting")
    else:
        print("   ✅ Gradient Boosting значимо лучше Linear SVM")
else:
    print("   📊 Различия не статистически значимы (p ≥ 0.05)")

# 9. Анализ стабильности моделей
print("\n9. Анализ стабильности моделей:")
print("   Модель           | Min Accuracy | Max Accuracy | Range    | CV Стабильность")
print("   " + "-" * 75)

for model_name, result in results.items():
    min_acc = np.min(result['scores'])
    max_acc = np.max(result['scores'])
    range_acc = max_acc - min_acc
    
    stability = "Высокая" if range_acc < 0.02 else "Средняя" if range_acc < 0.05 else "Низкая"
    
    print(f"   {model_name:16s} | {min_acc:11.4f} | {max_acc:11.4f} | {range_acc:8.4f} | {stability}")

# 10. Рекомендации на основе CV
print("\n10. Рекомендации на основе перекрестной проверки:")

best_cv_model = max(results.items(), key=lambda x: x[1]['mean_accuracy'])[0]
best_cv_accuracy = results[best_cv_model]['mean_accuracy']
best_cv_std = results[best_cv_model]['std_accuracy']

print(f"   🏆 Лучшая модель по CV: {best_cv_model}")
print(f"   Средняя точность: {best_cv_accuracy:.4f} ({best_cv_accuracy:.2%})")
print(f"   Стандартное отклонение: {best_cv_std:.4f}")
print(f"   Доверительный интервал (95%): {best_cv_accuracy:.4f} ± {1.96*best_cv_std:.4f}")

print(f"\n   💡 Интерпретация:")
print(f"   Модель показывает стабильную точность в диапазоне")
print(f"   от {best_cv_accuracy - 1.96*best_cv_std:.4f} до {best_cv_accuracy + 1.96*best_cv_std:.4f}")

# 11. Сохранение результатов
print("\n11. Сохранение результатов...")

cv_results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Mean_Accuracy': [results[name]['mean_accuracy'] for name in results],
    'Std_Accuracy': [results[name]['std_accuracy'] for name in results],
    'Min_Accuracy': [np.min(results[name]['scores']) for name in results],
    'Max_Accuracy': [np.max(results[name]['scores']) for name in results],
    'Time_Seconds': [results[name]['time'] for name in results]
})

cv_results_df.to_csv('cross_validation_results.csv', index=False)
print("   ✅ Результаты CV сохранены в 'cross_validation_results.csv'")

# Сохранение детальных scores
detailed_scores = {}
for model_name, result in results.items():
    for i, score in enumerate(result['scores']):
        detailed_scores[f'{model_name}_Fold_{i+1}'] = score

pd.DataFrame.from_dict(detailed_scores, orient='index', columns=['Accuracy']).to_csv('detailed_cv_scores.csv')
print("   ✅ Детальные scores сохранены в 'detailed_cv_scores.csv'")

print("\n" + "=" * 70)
print("ПРОВЕРКА ЗАВЕРШЕНА! 🎯")
print("=" * 70)