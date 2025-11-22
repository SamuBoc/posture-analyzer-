import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el modelo entrenado
with open('../entrega2/models/model_random_forest.pkl', 'rb') as f:
    model_data = pickle.load(f)

rf_model = model_data['model']
feature_names = model_data['feature_names']

# Obtener importancias
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Crear dataframe con resultados
importance_df = pd.DataFrame({
    'feature': [feature_names[i] for i in indices],
    'importance': importances[indices]
})

# Guardar resultados completos
importance_df.to_csv('feature_importances.csv', index=False)
print(f"Guardado: feature_importances.csv")

# Top 20 features
top20 = importance_df.head(20)
print("\nTop 20 features más importantes:")
print(top20)

# Gráfico de top 20
plt.figure(figsize=(10, 8))
plt.barh(range(20), top20['importance'].values)
plt.yticks(range(20), top20['feature'].values)
plt.xlabel('Importancia')
plt.title('Top 20 Features más Importantes')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('top20_features.png', dpi=300, bbox_inches='tight')
print("Guardado: top20_features.png")

# Análisis de features acumuladas
cumsum = np.cumsum(importances[indices])
n_features_90 = np.argmax(cumsum >= 0.90) + 1
n_features_95 = np.argmax(cumsum >= 0.95) + 1

print(f"\nAnálisis:")
print(f"- Features necesarias para 90% de importancia: {n_features_90}")
print(f"- Features necesarias para 95% de importancia: {n_features_95}")
print(f"- Total de features actuales: {len(feature_names)}")

# Gráfico de importancia acumulada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumsum)+1), cumsum*100, 'b-')
plt.axhline(y=90, color='r', linestyle='--', label='90%')
plt.axhline(y=95, color='g', linestyle='--', label='95%')
plt.axvline(x=n_features_90, color='r', linestyle=':', alpha=0.5)
plt.axvline(x=n_features_95, color='g', linestyle=':', alpha=0.5)
plt.xlabel('Número de Features')
plt.ylabel('Importancia Acumulada (%)')
plt.title('Importancia Acumulada de Features')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('cumulative_importance.png', dpi=300, bbox_inches='tight')
print("Guardado: cumulative_importance.png")

plt.show()
