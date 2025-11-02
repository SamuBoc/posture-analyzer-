"""
Script para preprocesar el dataset unificado.

Incluye:
- Filtrado de frames sin detección (NaN)
- Normalización de landmarks por tamaño del cuerpo
- Imputación de valores faltantes
- Feature engineering (velocidades, ángulos, inclinación)

Uso:
    python preprocess_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d


def filter_missing_landmarks(df, threshold=0.9):
    """
    Filtra frames con demasiados landmarks faltantes.

    Args:
        df: DataFrame con landmarks
        threshold: Proporción mínima de landmarks válidos requeridos

    Returns:
        DataFrame filtrado
    """
    print("\n" + "="*70)
    print("FILTRADO DE FRAMES CON LANDMARKS FALTANTES")
    print("="*70)

    # Contar landmarks válidos por frame
    visibility_cols = [col for col in df.columns if col.endswith('_visibility')]

    # Calcular proporción de landmarks válidos
    valid_proportions = df[visibility_cols].notna().mean(axis=1)

    # Filtrar frames
    mask = valid_proportions >= threshold
    df_filtered = df[mask].copy()

    removed = len(df) - len(df_filtered)
    print(f"\nFrames originales: {len(df)}")
    print(f"Frames removidos: {removed} ({removed/len(df)*100:.1f}%)")
    print(f"Frames conservados: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}%)")

    return df_filtered


def impute_missing_values(df):
    """
    Imputa valores faltantes usando interpolación lineal.

    Args:
        df: DataFrame con landmarks

    Returns:
        DataFrame con valores imputados
    """
    print("\n" + "="*70)
    print("IMPUTACIÓN DE VALORES FALTANTES")
    print("="*70)

    df_imputed = df.copy()

    # Obtener columnas de landmarks (x, y, z, visibility)
    landmark_cols = [col for col in df.columns if not col in ['frame', 'timestamp', 'label']]

    # Contar valores faltantes antes
    missing_before = df_imputed[landmark_cols].isna().sum().sum()

    # Interpolar por grupo (por video/secuencia)
    # Asumir que frames consecutivos del mismo label pertenecen a la misma secuencia
    for col in landmark_cols:
        df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')

    # Si aún quedan NaN, rellenar con la mediana
    for col in landmark_cols:
        if df_imputed[col].isna().any():
            median_val = df_imputed[col].median()
            df_imputed[col].fillna(median_val, inplace=True)

    missing_after = df_imputed[landmark_cols].isna().sum().sum()

    print(f"\nValores faltantes antes: {missing_before:,}")
    print(f"Valores faltantes después: {missing_after:,}")
    print(f"Valores imputados: {missing_before - missing_after:,}")

    return df_imputed


def normalize_landmarks(df):
    """
    Normaliza landmarks por el tamaño del cuerpo.

    Usa la distancia entre hombros como referencia para normalizar
    todas las coordenadas, haciendo el modelo invariante al tamaño.

    Args:
        df: DataFrame con landmarks

    Returns:
        DataFrame con landmarks normalizados
    """
    print("\n" + "="*70)
    print("NORMALIZACIÓN DE LANDMARKS")
    print("="*70)

    df_normalized = df.copy()

    # Calcular distancia entre hombros como referencia
    shoulder_dist = np.sqrt(
        (df_normalized['left_shoulder_x'] - df_normalized['right_shoulder_x'])**2 +
        (df_normalized['left_shoulder_y'] - df_normalized['right_shoulder_y'])**2 +
        (df_normalized['left_shoulder_z'] - df_normalized['right_shoulder_z'])**2
    )

    # Evitar división por cero
    shoulder_dist = shoulder_dist.replace(0, 1)

    # Normalizar todas las coordenadas x, y, z
    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

    for landmark in landmark_names:
        for coord in ['x', 'y', 'z']:
            col = f'{landmark}_{coord}'
            if col in df_normalized.columns:
                df_normalized[col] = df_normalized[col] / shoulder_dist

    print(f"\n✓ Landmarks normalizados por distancia entre hombros")
    print(f"  Distancia promedio hombros: {shoulder_dist.mean():.3f}")
    print(f"  Landmarks normalizados: {len(landmark_names)}")

    return df_normalized


def calculate_velocities(df):
    """
    Calcula velocidades de landmarks clave.

    Args:
        df: DataFrame con landmarks normalizados

    Returns:
        DataFrame con features de velocidad agregadas
    """
    print("\n" + "="*70)
    print("CÁLCULO DE VELOCIDADES")
    print("="*70)

    df_velocities = df.copy()

    # Landmarks clave para velocidad
    key_landmarks = [
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]

    velocity_features = []

    for landmark in key_landmarks:
        # Velocidad en x, y, z
        for coord in ['x', 'y', 'z']:
            col = f'{landmark}_{coord}'
            vel_col = f'{landmark}_{coord}_vel'

            # Diferencia con frame anterior
            df_velocities[vel_col] = df_velocities[col].diff()

            # Primer frame tiene velocidad 0
            df_velocities[vel_col].fillna(0, inplace=True)

            velocity_features.append(vel_col)

        # Velocidad total (magnitud)
        vel_total = f'{landmark}_vel_total'
        df_velocities[vel_total] = np.sqrt(
            df_velocities[f'{landmark}_x_vel']**2 +
            df_velocities[f'{landmark}_y_vel']**2 +
            df_velocities[f'{landmark}_z_vel']**2
        )
        velocity_features.append(vel_total)

    print(f"\n✓ Features de velocidad calculadas: {len(velocity_features)}")
    print(f"  Landmarks con velocidad: {len(key_landmarks)}")

    return df_velocities


def calculate_angles(df):
    """
    Calcula ángulos de articulaciones clave.

    Args:
        df: DataFrame con landmarks

    Returns:
        DataFrame con features de ángulos agregadas
    """
    print("\n" + "="*70)
    print("CÁLCULO DE ÁNGULOS")
    print("="*70)

    df_angles = df.copy()

    def angle_between_points(p1, p2, p3):
        """Calcula ángulo formado por 3 puntos (p1-p2-p3)."""
        # Vectores
        v1 = p1 - p2
        v2 = p3 - p2

        # Producto punto y magnitudes
        dot_product = (v1 * v2).sum(axis=1)
        magnitude1 = np.sqrt((v1**2).sum(axis=1))
        magnitude2 = np.sqrt((v2**2).sum(axis=1))

        # Ángulo en radianes
        cos_angle = dot_product / (magnitude1 * magnitude2 + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.arccos(cos_angle)

        return angles

    # Definir ángulos de articulaciones
    joint_angles = {
        'left_elbow_angle': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow_angle': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_knee_angle': ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee_angle': ('right_hip', 'right_knee', 'right_ankle'),
        'left_hip_angle': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip_angle': ('right_shoulder', 'right_hip', 'right_knee'),
    }

    angle_features = []

    for angle_name, (p1_name, p2_name, p3_name) in joint_angles.items():
        # Extraer coordenadas
        p1 = df_angles[[f'{p1_name}_x', f'{p1_name}_y', f'{p1_name}_z']].values
        p2 = df_angles[[f'{p2_name}_x', f'{p2_name}_y', f'{p2_name}_z']].values
        p3 = df_angles[[f'{p3_name}_x', f'{p3_name}_y', f'{p3_name}_z']].values

        # Calcular ángulo
        angles = angle_between_points(p1, p2, p3)
        df_angles[angle_name] = angles
        angle_features.append(angle_name)

    print(f"\n✓ Features de ángulos calculadas: {len(angle_features)}")
    for angle in angle_features:
        print(f"  {angle:25s}: {df_angles[angle].mean():.2f} rad promedio")

    return df_angles


def calculate_trunk_inclination(df):
    """
    Calcula inclinación del tronco.

    Args:
        df: DataFrame con landmarks

    Returns:
        DataFrame con features de inclinación agregadas
    """
    print("\n" + "="*70)
    print("CÁLCULO DE INCLINACIÓN DEL TRONCO")
    print("="*70)

    df_trunk = df.copy()

    # Punto medio de hombros
    shoulder_mid_x = (df_trunk['left_shoulder_x'] + df_trunk['right_shoulder_x']) / 2
    shoulder_mid_y = (df_trunk['left_shoulder_y'] + df_trunk['right_shoulder_y']) / 2
    shoulder_mid_z = (df_trunk['left_shoulder_z'] + df_trunk['right_shoulder_z']) / 2

    # Punto medio de caderas
    hip_mid_x = (df_trunk['left_hip_x'] + df_trunk['right_hip_x']) / 2
    hip_mid_y = (df_trunk['left_hip_y'] + df_trunk['right_hip_y']) / 2
    hip_mid_z = (df_trunk['left_hip_z'] + df_trunk['right_hip_z']) / 2

    # Vector del tronco (de cadera a hombro)
    trunk_x = shoulder_mid_x - hip_mid_x
    trunk_y = shoulder_mid_y - hip_mid_y
    trunk_z = shoulder_mid_z - hip_mid_z

    # Inclinación respecto al eje Y (vertical)
    trunk_inclination = np.arctan2(
        np.sqrt(trunk_x**2 + trunk_z**2),
        trunk_y + 1e-8
    )

    df_trunk['trunk_inclination'] = trunk_inclination
    df_trunk['trunk_inclination_degrees'] = np.degrees(trunk_inclination)

    print(f"\n✓ Inclinación del tronco calculada")
    print(f"  Inclinación promedio: {df_trunk['trunk_inclination_degrees'].mean():.1f}°")
    print(f"  Inclinación mínima: {df_trunk['trunk_inclination_degrees'].min():.1f}°")
    print(f"  Inclinación máxima: {df_trunk['trunk_inclination_degrees'].max():.1f}°")

    return df_trunk


def main():
    """Función principal."""
    script_dir = Path(__file__).parent
    input_path = script_dir / 'dataset_unified.csv'
    output_path = script_dir / 'dataset_preprocessed.csv'

    print("="*70)
    print("PREPROCESAMIENTO Y FEATURE ENGINEERING")
    print("="*70)
    print(f"\nDataset de entrada: {input_path.name}")

    # Cargar dataset
    df = pd.read_csv(input_path)
    print(f"\nDataset cargado:")
    print(f"  Frames: {len(df):,}")
    print(f"  Columnas: {len(df.columns)}")
    print(f"  Clases: {df['label'].unique().tolist()}")

    # 1. Filtrar frames con muchos landmarks faltantes
    df = filter_missing_landmarks(df, threshold=0.5)

    # 2. Imputar valores faltantes
    df = impute_missing_values(df)

    # 3. Normalizar landmarks
    df = normalize_landmarks(df)

    # 4. Feature engineering
    df = calculate_velocities(df)
    df = calculate_angles(df)
    df = calculate_trunk_inclination(df)

    # 5. Guardar dataset preprocesado
    print("\n" + "="*70)
    print("GUARDANDO DATASET PREPROCESADO")
    print("="*70)

    df.to_csv(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n✓ Dataset preprocesado guardado: {output_path.name}")
    print(f"  Frames finales: {len(df):,}")
    print(f"  Features totales: {len(df.columns)}")
    print(f"  Tamaño: {size_mb:.1f} MB")

    # Resumen de features
    print(f"\nResumen de features:")
    landmark_features = len([c for c in df.columns if any(c.endswith(s) for s in ['_x', '_y', '_z', '_visibility'])])
    velocity_features = len([c for c in df.columns if '_vel' in c])
    angle_features = len([c for c in df.columns if 'angle' in c])
    trunk_features = len([c for c in df.columns if 'trunk' in c])

    print(f"  Landmarks originales: {landmark_features}")
    print(f"  Velocidades: {velocity_features}")
    print(f"  Ángulos: {angle_features}")
    print(f"  Inclinación tronco: {trunk_features}")
    print(f"  Metadata: 2 (frame, timestamp)")
    print(f"  Label: 1")

    print("\n" + "="*70)
    print("¡Preprocesamiento completado!")
    print("="*70)


if __name__ == "__main__":
    main()
