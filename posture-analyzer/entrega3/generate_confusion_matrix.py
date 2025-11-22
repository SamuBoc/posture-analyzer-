import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos y modelo
df = pd.read_csv('../entrega2/dataset_preprocessed.csv')

features_to_drop = ['frame', 'timestamp', 'label', 'video_id']
X = df.drop(columns=features_to_drop)
y = df['label']
video_ids = df['video_id']

# Split por video
unique_videos = df[['video_id', 'label']].drop_duplicates()
test_videos = []
for label in unique_videos['label'].unique():
    videos_of_class = unique_videos[unique_videos['label'] == label]['video_id'].tolist()
    test_videos.append(videos_of_class[-1])

test_mask = video_ids.isin(test_videos)
X_test = X[test_mask]
y_test = y[test_mask]

# Cargar modelo
with open('../entrega2/models/model_random_forest.pkl', 'rb') as f:
    model_data = pickle.load(f)

scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
rf_model = model_data['model']

# Predecir
X_test_scaled = scaler.transform(X_test)
y_test_encoded = label_encoder.transform(y_test)
y_pred = rf_model.predict(X_test_scaled)

# Crear confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred)
class_names = label_encoder.classes_

# Visualizar
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Número de Predicciones'})
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.title('Matriz de Confusión - Random Forest (172 features)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('IMAGEN_NECESARIA_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Guardado: IMAGEN_NECESARIA_confusion_matrix.png")
plt.show()
