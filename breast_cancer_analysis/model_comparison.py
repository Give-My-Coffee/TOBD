import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from openpyxl.styles import PatternFill
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer

print("=" * 60)
print("ЗАДАНИЕ 6")
print("=" * 60)

# 1. Загрузка данных (если нужно перезагрузить)
print("1. Загрузка данных...")
data = load_breast_cancer()
X, y = data.data, data.target

# Разделение данных 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Загрузка результатов предыдущих моделей
print("2. Загрузка результатов моделей...")

# Предположим, что у нас есть следующие результаты (замени на реальные значения)


models_results = {
    'Logistic Regression': {
        'accuracy': 0.9474,
        'precision': 0.9474,  # Среднее значение
        'recall': 0.9474,     # Среднее значение  
        'f1': 0.9474,         # Среднее значение
        'predictions': None    # Можно загрузить из файла
    },
    'Linear SVM': {
        'accuracy': 0.9825,
        'precision': 0.9825,
        'recall': 0.9825,
        'f1': 0.9825,
        'predictions': None
    },
    'RBF SVM (Default)': {
        'accuracy': 0.9766,
        'precision': 0.9766,
        'recall': 0.9766,
        'f1': 0.9766,
        'predictions': None
    },
    'RBF SVM (Optimized)': {
        'accuracy': 0.9708,
        'precision': 0.9736,  # macro avg из вывода
        'recall': 0.9641,
        'f1': 0.9685,
        'predictions': None
    },
    'Random Forest (200 trees)': {
        'accuracy': 0.9415,
        'precision': 0.9414,  # weighted avg
        'recall': 0.9415,
        'f1': 0.9413,
        'predictions': None
    },
    'Gradient Boosting': {
        'accuracy': 0.9766,   # Предположим лучший результат
        'precision': 0.9766,
        'recall': 0.9766,
        'f1': 0.9766,
        'predictions': None
    }
}

# 3. Создание сводной таблицы
print("3. Создание сводной таблицы...")

comparison_df = pd.DataFrame({
    'Model': list(models_results.keys()),
    'Accuracy': [models_results[model]['accuracy'] for model in models_results],
    'Precision': [models_results[model]['precision'] for model in models_results],
    'Recall': [models_results[model]['recall'] for model in models_results],
    'F1-Score': [models_results[model]['f1'] for model in models_results]
})

# Сортировка по Accuracy
comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

print("\nСводная таблица моделей (отсортировано по Accuracy):")
print("=" * 85)
print(comparison_df.to_string(index=False))
print("=" * 85)

# 4. Визуализация сравнения моделей
print("4. Визуализация сравнения моделей...")

plt.figure(figsize=(15, 12))

# График 1: Accuracy всех моделей
plt.subplot(2, 2, 1)
bars = plt.barh(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
plt.xlabel('Accuracy', fontsize=12)
plt.title('Сравнение Accuracy моделей', fontsize=14, fontweight='bold')
plt.xlim(0.9, 1.0)
plt.gca().invert_yaxis()  # Лучшая модель была наверху

# Добавление значений на bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=10)

# График 2: Все метрики для топ-моделей
plt.subplot(2, 2, 2)
top_models = comparison_df.head(4)  # Топ-4 модели
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

x = np.arange(len(top_models))
width = 0.2

for i, metric in enumerate(metrics):
    offset = width * i - width * (len(metrics) - 1) / 2
    plt.bar(x + offset, top_models[metric], width, label=metric)

plt.xlabel('Модели', fontsize=12)
plt.ylabel('Значение метрики', fontsize=12)
plt.title('Детальное сравнение топ-моделей', fontsize=14, fontweight='bold')
plt.xticks(x, top_models['Model'], rotation=45, ha='right')
plt.legend()
plt.ylim(0.9, 1.0)

# График 3: Radar chart для сравнения моделей
plt.subplot(2, 2, 3, polar=True)  

categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Замыкаем круг

for i, model in enumerate(comparison_df['Model']):
    values = comparison_df[comparison_df['Model'] == model][categories].values[0].tolist()
    values += values[:1]  # Замыкаем круг
    plt.polar(angles, values, linewidth=2, label=model, marker='o')

plt.xticks(angles[:-1], categories)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
plt.ylim(0, 1)
plt.title('Radar Chart: Сравнение моделей', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.3, 1.1))

# График 4: Heatmap корреляции метрик
plt.subplot(2, 2, 4)
correlation_matrix = comparison_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'label': 'Корреляция'})
plt.title('Корреляция между метриками', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('model_comparison.pdf', bbox_inches='tight')
plt.show()
print("   ✅ Визуализации сохранены в 'model_comparison.png'")

# 5. Детальный анализ
print("\n5. Детальный анализ результатов:")

# Лучшая модель
best_model = comparison_df.iloc[0]
print(f"   🏆 Лучшая модель: {best_model['Model']}")
print(f"   Accuracy: {best_model['Accuracy']:.4f} ({best_model['Accuracy']:.2%})")

# Разница между лучшей и второй моделью
if len(comparison_df) > 1:
    second_best = comparison_df.iloc[1]
    difference = best_model['Accuracy'] - second_best['Accuracy']
    print(f"   Преимущество над второй моделью: {difference:.4f} ({difference/.01:.1f}%)")

# Анализ стабильности моделей
print(f"\n   Стабильность моделей (разброс метрик):")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    std_dev = comparison_df[metric].std()
    print(f"   {metric}: σ = {std_dev:.4f}")

# 6. Рекомендации по выбору модели
print("\n6. Рекомендации по выбору модели:")

print("   📊 На основе Accuracy:")
for i, (model, acc) in enumerate(zip(comparison_df['Model'], comparison_df['Accuracy'])):
    print(f"   {i+1}. {model}: {acc:.4f}")

print("\n   ⚖️  Выбор зависит от задачи:")
print("   - Для максимальной точности: Linear SVM")
print("   - Для интерпретируемости: Logistic Regression или Random Forest")
print("   - Для важности признаков: Random Forest")
print("   - Для производительности: нужно тестировать на новых данных")

# 7. Сохранение результатов сравнения
print("\n7. Сохранение результатов сравнения...")

# Сохранение сводной таблицы
comparison_df.to_csv('model_comparison.csv', index=False)
print("   ✅ Сводная таблица сохранена в 'model_comparison.csv'")

# Сохранение в Excel с форматированием
with pd.ExcelWriter('model_comparison.xlsx', engine='openpyxl') as writer:
    comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)
    
    # Добавляем условное форматирование
    workbook = writer.book
    worksheet = writer.sheets['Model Comparison']
    
    # Форматирование для лучших значений
    for col in range(2, 6):  # Колонки с метриками
        for row in range(2, len(comparison_df) + 2):
            cell = worksheet.cell(row=row, column=col)
            if cell.value == comparison_df.iloc[:, col-1].max():
                cell.fill = openpyxl.styles.PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")

print("   ✅ Excel файл с форматированием сохранен")

# 8. Итоговый вывод
print("\n8. Итоговый вывод сравнения:")
print(f"   Всего протестировано моделей: {len(comparison_df)}")
print(f"   Диапазон Accuracy: {comparison_df['Accuracy'].min():.4f} - {comparison_df['Accuracy'].max():.4f}")
print(f"   Средняя Accuracy: {comparison_df['Accuracy'].mean():.4f}")

best_model_name = comparison_df.iloc[0]['Model']
best_accuracy = comparison_df.iloc[0]['Accuracy']
print(f"   🎯 Рекомендуемая модель: {best_model_name} с точностью {best_accuracy:.2%}")
