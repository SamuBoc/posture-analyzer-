import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Cargar datos preprocesados
df = pd.read_csv('../entrega2/dataset_preprocessed.csv')

# Separar features y labels
features_to_drop = ['frame', 'timestamp', 'label', 'video_id']
X = df.drop(columns=features_to_drop)
y = df['label']
video_ids = df['video_id']

# Split por video (mismo que en entrega 2)
unique_videos = df[['video_id', 'label']].drop_duplicates()
test_videos = []
for label in unique_videos['label'].unique():
    videos_of_class = unique_videos[unique_videos['label'] == label]['video_id'].tolist()
    test_videos.append(videos_of_class[-1])

train_mask = ~video_ids.isin(test_videos)
test_mask = video_ids.isin(test_videos)

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"Train: {len(X_train)} frames, Test: {len(X_test)} frames")

# Cargar scaler y label encoder de entrega 2
with open('../entrega2/models/model_random_forest.pkl', 'rb') as f:
    model_data = pickle.load(f)
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']

# Convertir labels a números
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline: modelo actual (172 features)
print("\n=== BASELINE (172 features) ===")
rf_baseline = model_data['model']
y_pred_baseline = rf_baseline.predict(X_test_scaled)
acc_baseline = accuracy_score(y_test, y_pred_baseline)
f1_baseline = f1_score(y_test, y_pred_baseline, average='weighted')
print(f"Accuracy: {acc_baseline:.4f}")
print(f"F1-Score: {f1_baseline:.4f}")

# Reducción con PCA a diferentes dimensiones
n_components_list = [50, 75, 100]
results = {
    'n_features': [172],
    'accuracy': [acc_baseline],
    'f1_score': [f1_baseline],
    'method': ['Original']
}

for n_comp in n_components_list:
    print(f"\n=== PCA con {n_comp} componentes ===")

    # Aplicar PCA
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    var_explained = pca.explained_variance_ratio_.sum()
    print(f"Varianza explicada: {var_explained:.4f}")

    # Entrenar modelo con features reducidas
    rf_pca = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_pca.fit(X_train_pca, y_train)

    y_pred_pca = rf_pca.predict(X_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    f1_pca = f1_score(y_test, y_pred_pca, average='weighted')

    print(f"Accuracy: {acc_pca:.4f}")
    print(f"F1-Score: {f1_pca:.4f}")

    results['n_features'].append(n_comp)
    results['accuracy'].append(acc_pca)
    results['f1_score'].append(f1_pca)
    results['method'].append(f'PCA-{n_comp}')

# Guardar resultados
results_df = pd.DataFrame(results)
results_df.to_csv('comparison_results.csv', index=False)
print("\nGuardado: comparison_results.csv")
print(results_df)

# Gráfico comparativo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy vs Features
ax1.plot(results['n_features'], [a*100 for a in results['accuracy']],
         'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Número de Features')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy vs Número de Features')
ax1.grid(alpha=0.3)
for i, txt in enumerate(results['method']):
    ax1.annotate(f"{results['accuracy'][i]*100:.1f}%",
                (results['n_features'][i], results['accuracy'][i]*100),
                textcoords="offset points", xytext=(0,10), ha='center')

# F1-Score vs Features
ax2.plot(results['n_features'], [f*100 for f in results['f1_score']],
         'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Número de Features')
ax2.set_ylabel('F1-Score (%)')
ax2.set_title('F1-Score vs Número de Features')
ax2.grid(alpha=0.3)
for i, txt in enumerate(results['method']):
    ax2.annotate(f"{results['f1_score'][i]*100:.1f}%",
                (results['n_features'][i], results['f1_score'][i]*100),
                textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('feature_reduction_comparison.png', dpi=300, bbox_inches='tight')
print("Guardado: feature_reduction_comparison.png")

plt.show()
