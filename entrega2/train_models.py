"""
Script para entrenar y evaluar múltiples modelos de clasificación.

Incluye:
- Random Forest
- SVM (Support Vector Machine)
- XGBoost

Con ajuste de hiperparámetros usando GridSearchCV.

Uso:
    python train_models.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare_data(data_path):
    """
    Carga y prepara el dataset para entrenamiento.

    IMPORTANTE: Hace split por VIDEO, no por frame, para evitar data leakage.

    Args:
        data_path: Path al CSV preprocesado

    Returns:
        X_train, X_test, y_train, y_test, label_encoder, scaler
    """
    print("\n" + "="*70)
    print("CARGA Y PREPARACIÓN DE DATOS")
    print("="*70)

    # Cargar datos
    df = pd.read_csv(data_path)
    print(f"\nDataset cargado: {len(df):,} frames")
    print(f"Videos únicos: {df['video_id'].nunique()}")

    # Separar features y labels
    features_to_drop = ['frame', 'timestamp', 'label', 'video_id']
    X = df.drop(columns=features_to_drop)
    y = df['label']
    video_ids = df['video_id']

    print(f"Features: {X.shape[1]}")
    print(f"Clases: {y.unique().tolist()}")

    # Distribución de clases
    print(f"\nDistribución de clases:")
    class_counts = y.value_counts()
    for label, count in class_counts.items():
        print(f"  {label:20s}: {count:5d} ({count/len(y)*100:5.1f}%)")

    # Distribución de videos por clase
    print(f"\nDistribución de videos por clase:")
    video_class_counts = df.groupby('label')['video_id'].nunique()
    for label, count in video_class_counts.items():
        print(f"  {label:20s}: {count} videos")

    # Codificar labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # SPLIT POR VIDEO (no por frame!)
    print(f"\n{'='*70}")
    print("SPLIT POR VIDEO (evitar data leakage)")
    print(f"{'='*70}")

    # Obtener videos únicos por clase
    unique_videos = df[['video_id', 'label']].drop_duplicates()

    # Seleccionar 1 video de cada clase para test (total: 5 videos)
    # Esto es más apropiado para datasets pequeños
    test_videos = []
    train_videos = []

    for label in unique_videos['label'].unique():
        videos_of_class = unique_videos[unique_videos['label'] == label]['video_id'].tolist()
        # Seleccionar el último video de cada clase para test
        test_videos.append(videos_of_class[-1])
        train_videos.extend(videos_of_class[:-1])

    print(f"\nEstrategia: 1 video por clase en TEST, resto en TRAIN")

    print(f"\nVideos en TRAIN: {len(train_videos)}")
    for video in sorted(train_videos):
        label = df[df['video_id'] == video]['label'].iloc[0]
        frames = len(df[df['video_id'] == video])
        print(f"  {video:30s} ({label:15s}): {frames:4d} frames")

    print(f"\nVideos en TEST: {len(test_videos)}")
    for video in sorted(test_videos):
        label = df[df['video_id'] == video]['label'].iloc[0]
        frames = len(df[df['video_id'] == video])
        print(f"  {video:30s} ({label:15s}): {frames:4d} frames")

    # Separar datos por video
    train_mask = video_ids.isin(train_videos)
    test_mask = video_ids.isin(test_videos)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y_encoded[train_mask]
    y_test = y_encoded[test_mask]

    print(f"\nSplit realizado:")
    print(f"  Train: {len(X_train):,} frames ({len(X_train)/len(X)*100:.1f}%) de {len(train_videos)} videos")
    print(f"  Test:  {len(X_test):,} frames ({len(X_test)/len(X)*100:.1f}%) de {len(test_videos)} videos")

    # Verificar que no hay videos compartidos
    assert set(train_videos).isdisjoint(set(test_videos)), "¡ERROR! Hay videos en train y test"
    print(f"\n✓ Verificado: No hay videos compartidos entre train y test")

    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n✓ Features normalizadas con StandardScaler")

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler, X.columns.tolist()


def train_random_forest(X_train, y_train):
    """
    Entrena Random Forest con ajuste de hiperparámetros.

    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento

    Returns:
        Mejor modelo entrenado
    """
    print("\n" + "="*70)
    print("ENTRENAMIENTO: RANDOM FOREST")
    print("="*70)

    # Grid de hiperparámetros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    print(f"\nBúsqueda de hiperparámetros:")
    print(f"  n_estimators: {param_grid['n_estimators']}")
    print(f"  max_depth: {param_grid['max_depth']}")
    print(f"  min_samples_split: {param_grid['min_samples_split']}")
    print(f"  min_samples_leaf: {param_grid['min_samples_leaf']}")

    # Modelo base
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Grid search con validación cruzada
    print(f"\nIniciando GridSearchCV (5-fold CV)...")
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✓ Entrenamiento completado")
    print(f"\nMejores hiperparámetros:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nMejor score (CV): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_svm(X_train, y_train):
    """
    Entrena SVM con ajuste de hiperparámetros.

    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento

    Returns:
        Mejor modelo entrenado
    """
    print("\n" + "="*70)
    print("ENTRENAMIENTO: SVM")
    print("="*70)

    # Grid de hiperparámetros
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'poly']
    }

    print(f"\nBúsqueda de hiperparámetros:")
    print(f"  C: {param_grid['C']}")
    print(f"  gamma: {param_grid['gamma']}")
    print(f"  kernel: {param_grid['kernel']}")

    # Modelo base
    svm = SVC(random_state=42)

    # Grid search con validación cruzada
    print(f"\nIniciando GridSearchCV (3-fold CV)...")
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=3,  # 3-fold para SVM (más lento)
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✓ Entrenamiento completado")
    print(f"\nMejores hiperparámetros:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nMejor score (CV): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_xgboost(X_train, y_train):
    """
    Entrena XGBoost con ajuste de hiperparámetros.

    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento

    Returns:
        Mejor modelo entrenado
    """
    print("\n" + "="*70)
    print("ENTRENAMIENTO: XGBOOST")
    print("="*70)

    # Grid de hiperparámetros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    print(f"\nBúsqueda de hiperparámetros:")
    print(f"  n_estimators: {param_grid['n_estimators']}")
    print(f"  max_depth: {param_grid['max_depth']}")
    print(f"  learning_rate: {param_grid['learning_rate']}")
    print(f"  subsample: {param_grid['subsample']}")
    print(f"  colsample_bytree: {param_grid['colsample_bytree']}")

    # Modelo base
    xgb = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    # Grid search con validación cruzada
    print(f"\nIniciando GridSearchCV (5-fold CV)...")
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✓ Entrenamiento completado")
    print(f"\nMejores hiperparámetros:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nMejor score (CV): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    """
    Evalúa un modelo y muestra métricas.

    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Labels de test
        label_encoder: LabelEncoder usado
        model_name: Nombre del modelo

    Returns:
        Dict con métricas
    """
    print("\n" + "="*70)
    print(f"EVALUACIÓN: {model_name.upper()}")
    print("="*70)

    # Predicciones
    y_pred = model.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    print(f"\nMétricas globales:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Classification report
    print(f"\nReporte por clase:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        digits=3
    ))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Guardar métricas
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classes': label_encoder.classes_.tolist()
    }

    return metrics, cm


def plot_confusion_matrix(cm, classes, model_name, output_dir):
    """
    Genera y guarda matriz de confusión.

    Args:
        cm: Confusion matrix
        classes: Nombres de clases
        model_name: Nombre del modelo
        output_dir: Directorio de salida
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Cantidad'}
    )
    plt.title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Verdadero', fontsize=12)
    plt.xlabel('Predicho', fontsize=12)
    plt.tight_layout()

    # Guardar
    output_path = output_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Matriz de confusión guardada: {output_path.name}")


def save_model(model, scaler, label_encoder, feature_names, model_name, output_dir):
    """
    Guarda modelo entrenado y componentes.

    Args:
        model: Modelo entrenado
        scaler: StandardScaler usado
        label_encoder: LabelEncoder usado
        feature_names: Nombres de features
        model_name: Nombre del modelo
        output_dir: Directorio de salida
    """
    model_filename = f'model_{model_name.lower().replace(" ", "_")}.pkl'
    model_path = output_dir / model_filename

    # Guardar todo en un dict
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat()
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Modelo guardado: {model_filename} ({size_mb:.2f} MB)")


def compare_models(all_metrics, output_dir):
    """
    Compara todos los modelos y genera gráficos.

    Args:
        all_metrics: Lista de dicts con métricas
        output_dir: Directorio de salida
    """
    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS")
    print("="*70)

    # Crear DataFrame comparativo
    df_metrics = pd.DataFrame(all_metrics)

    print(f"\nComparación:")
    print(df_metrics[['model_name', 'accuracy', 'precision', 'recall', 'f1_score']].to_string(index=False))

    # Gráfico de barras comparativo
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(df_metrics['model_name'], df_metrics[metric])

        # Colorear barras
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_ylabel(label, fontsize=12)
        ax.set_ylim([0, 1.0])
        ax.set_title(f'{label} por Modelo', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    comparison_path = output_dir / 'model_comparison.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Gráfico comparativo guardado: {comparison_path.name}")

    # Guardar métricas en JSON
    metrics_path = output_dir / 'all_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"✓ Métricas guardadas: {metrics_path.name}")

    # Determinar mejor modelo
    best_model_idx = df_metrics['accuracy'].idxmax()
    best_model = df_metrics.iloc[best_model_idx]

    print(f"\n{'='*70}")
    print(f"MEJOR MODELO: {best_model['model_name'].upper()}")
    print(f"{'='*70}")
    print(f"  Accuracy:  {best_model['accuracy']:.4f}")
    print(f"  Precision: {best_model['precision']:.4f}")
    print(f"  Recall:    {best_model['recall']:.4f}")
    print(f"  F1-Score:  {best_model['f1_score']:.4f}")


def main():
    """Función principal."""
    script_dir = Path(__file__).parent
    data_path = script_dir / 'dataset_preprocessed.csv'
    output_dir = script_dir / 'models'

    # Crear directorio de salida
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN")
    print("="*70)

    # 1. Cargar y preparar datos
    X_train, X_test, y_train, y_test, label_encoder, scaler, feature_names = load_and_prepare_data(data_path)

    # 2. Entrenar modelos
    models = {}

    # Random Forest
    models['Random Forest'] = train_random_forest(X_train, y_train)

    # SVM
    models['SVM'] = train_svm(X_train, y_train)

    # XGBoost
    models['XGBoost'] = train_xgboost(X_train, y_train)

    # 3. Evaluar y guardar modelos
    all_metrics = []

    for model_name, model in models.items():
        # Evaluar
        metrics, cm = evaluate_model(model, X_test, y_test, label_encoder, model_name)
        all_metrics.append(metrics)

        # Graficar matriz de confusión
        plot_confusion_matrix(cm, label_encoder.classes_, model_name, output_dir)

        # Guardar modelo
        save_model(model, scaler, label_encoder, feature_names, model_name, output_dir)

    # 4. Comparar modelos
    compare_models(all_metrics, output_dir)

    print("\n" + "="*70)
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("="*70)
    print(f"\nModelos guardados en: {output_dir}")
    print(f"Total de modelos: {len(models)}")


if __name__ == "__main__":
    main()
