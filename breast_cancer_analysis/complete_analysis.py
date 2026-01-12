import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("ПОЛНЫЙ АНАЛИЗ РАКА ГРУДИ С LOGISTIC REGRESSION")
print("=" * 60)

# 1. Загрузка и подготовка данных
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Данные загружены: {X.shape}")
print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# 2. Обучение модели
print("\n2. Обучение логистической регрессии...")
model = LogisticRegression(random_state=42, max_iter=10000)
model.fit(X_train, y_train)
print("   ✅ Модель обучена!")

# 3. Прогнозирование
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)

# 4. Оценка модели
accuracy = accuracy_score(y_test, y_test_pred)
print(f"\n3. Accuracy: {accuracy:.4f} ({accuracy:.2%})")

# 5. ROC кривая и AUC
print("\n4. Вычисление ROC кривой и AUC...")
y_test_proba_benign = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba_benign, pos_label=1)
roc_auc = roc_auc_score(y_test, y_test_proba_benign)

print(f"   AUC: {roc_auc:.4f}")

# 6. Построение ROC кривой
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC кривая (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Случайный классификатор (AUC = 0.5)')

# Настройка графика
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC кривая - Логистическая регрессия\n(Рак груди)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)

# Добавляем оптимальную точку
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_fpr = fpr[optimal_idx]
optimal_tpr = tpr[optimal_idx]

plt.scatter(optimal_fpr, optimal_tpr, color='red', s=100, 
           label=f'Оптимальная точка\n(Порог = {optimal_threshold:.3f})')

plt.tight_layout()
plt.savefig('roc_curve_complete.png', dpi=300, bbox_inches='tight')
plt.show()

print("   ✅ ROC кривая построена и сохранена!")
print(f"   Оптимальный порог: {optimal_threshold:.3f}")

# 7. Вывод всех результатов
print("\n5. Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(f"   Malignant правильно: {cm[0, 0]}")
print(f"   Malignant ошибки: {cm[0, 1]}")
print(f"   Benign ошибки: {cm[1, 0]}")
print(f"   Benign правильно: {cm[1, 1]}")

print("\n6. Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=data.target_names))

print("✅ Все задания завершены!")