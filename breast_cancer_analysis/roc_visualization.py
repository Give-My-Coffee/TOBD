import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
import numpy as np

print("=" * 60)
print("ЗАДАНИЕ 3")
print("=" * 60)

# 1. Вычисление ROC кривой и AUC
print("1. Вычисление ROC кривой и AUC...")

# Получаем вероятности для положительного класса (benign)
y_test_proba_benign = model.predict_proba(X_test)[:, 1]

# Вычисляем ROC кривую
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba_benign, pos_label=1)

# Вычисляем AUC (Area Under Curve)
roc_auc = roc_auc_score(y_test, y_test_proba_benign)

print(f"   Количество точек на ROC кривой: {len(fpr)}")
print(f"   AUC (Area Under Curve): {roc_auc:.4f}")
print(f"   Первые 5 пороговых значений: {thresholds[:5]}")
print(f"   Первые 5 FPR: {fpr[:5]}")
print(f"   Первые 5 TPR: {tpr[:5]}")

# 2. Построение ROC кривой
print("\n2. Построение ROC кривой...")

plt.figure(figsize=(10, 8))

# ROC кривая
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC кривая (AUC = {roc_auc:.3f})')

# Диагональная линия (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Случайный классификатор (AUC = 0.5)')

# Настройка графика
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)\n(1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)\n(Sensitivity/Recall)', fontsize=12)
plt.title('ROC кривая - Логистическая регрессия\n(Рак груди)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)

# Добавляем координатную сетку
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')

# 3. Добавление дополнительной информации
print("3. Добавление дополнительной информации на график...")

# Находим оптимальный порог (ближайший к左上 углу)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_fpr = fpr[optimal_idx]
optimal_tpr = tpr[optimal_idx]

# Отмечаем оптимальную точку на графике
plt.scatter(optimal_fpr, optimal_tpr, color='red', s=100, 
           label=f'Оптимальная точка\n(Порог = {optimal_threshold:.3f})',
           zorder=5)

# Добавляем аннотацию
plt.annotate(f'Порог: {optimal_threshold:.3f}\nFPR: {optimal_fpr:.3f}\nTPR: {optimal_tpr:.3f}',
             xy=(optimal_fpr, optimal_tpr), xytext=(optimal_fpr + 0.2, optimal_tpr - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# 4. Сохранение и отображение графика
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.savefig('roc_curve.pdf', bbox_inches='tight')
print("   ✅ ROC кривая сохранена в 'roc_curve.png' и 'roc_curve.pdf'")
plt.show()

# 5. Детальный анализ ROC кривой
print("\n4. Детальный анализ ROC кривой:")
print(f"   Площадь под кривой (AUC): {roc_auc:.4f}")

# Интерпретация AUC
if roc_auc >= 0.9:
    interpretation = "Отличная разделяющая способность"
elif roc_auc >= 0.8:
    interpretation = "Хорошая разделяющая способность"
elif roc_auc >= 0.7:
    interpretation = "Приемлемая разделяющая способность"
else:
    interpretation = "Плохая разделяющая способность"

print(f"   Интерпретация: {interpretation}")

# 6. Анализ различных порогов
print("\n5. Анализ различных пороговых значений:")
print("   Порог | TPR (Sensitivity) | FPR (1-Specificity) | Youden's J")
print("   " + "-" * 65)

youden_j = tpr - fpr
for i in range(0, len(thresholds), len(thresholds)//10):  # Каждый 10-й порог
    if i < len(thresholds):
        print(f"   {thresholds[i]:.3f} | {tpr[i]:.3f}               | {fpr[i]:.3f}                | {youden_j[i]:.3f}")

# 7. Сохранение данных ROC кривой
print("\n6. Сохранение данных ROC кривой...")
roc_data = pd.DataFrame({
    'threshold': thresholds,
    'fpr': fpr,
    'tpr': tpr,
    'youden_j': youden_j
})
roc_data.to_csv('roc_curve_data.csv', index=False)
print("   ✅ Данные ROC кривой сохранены в 'roc_curve_data.csv'")

# 8. Сравнение с идеальным классификатором
print("\n7. Сравнение с идеальным классификатором:")
print(f"   Наш классификатор: AUC = {roc_auc:.4f}")
print(f"   Идеальный классификатор: AUC = 1.0000")
print(f"   Случайный классификатор: AUC = 0.5000")

# 9. Практическое применение
print("\n8. Практическое применение:")
print(f"   Оптимальный порог: {optimal_threshold:.3f}")
print(f"   При этом пороге:")
print(f"   - True Positive Rate (Sensitivity): {optimal_tpr:.3f} ({optimal_tpr:.1%})")
print(f"   - False Positive Rate: {optimal_fpr:.3f} ({optimal_fpr:.1%})")
print(f"   - Youden's J statistic: {youden_j[optimal_idx]:.3f}")

print("\n✅ Задание 3 завершено! ROC кривая построена и проанализирована.")