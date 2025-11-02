"""
Script para unificar todos los CSVs de landmarks en un único dataset.

Uso:
    python unify_dataset.py
"""

import pandas as pd
from pathlib import Path
import numpy as np

def unify_csvs(csv_dir, output_path):
    """
    Unifica todos los CSVs en un único dataset.

    Args:
        csv_dir: Directorio con los CSVs
        output_path: Path al archivo de salida
    """
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob('*.csv'))

    if not csv_files:
        print(f"No se encontraron archivos CSV en {csv_dir}")
        return

    print("="*70)
    print("UNIFICACIÓN DE DATASETS")
    print("="*70)
    print(f"\nArchivos CSV encontrados: {len(csv_files)}")

    all_data = []

    for csv_file in csv_files:
        print(f"  Leyendo: {csv_file.name}")
        df = pd.read_csv(csv_file)

        # Agregar columna video_id (nombre del archivo sin extensión)
        df['video_id'] = csv_file.stem

        all_data.append(df)

    # Concatenar todos los dataframes
    unified_df = pd.concat(all_data, ignore_index=True)

    print(f"\n✓ Dataset unificado creado")
    print(f"  Total de frames: {len(unified_df):,}")
    print(f"  Total de columnas: {len(unified_df.columns)}")
    print(f"  Clases: {unified_df['label'].unique().tolist()}")

    # Distribución por clase
    print(f"\nDistribución por clase:")
    class_dist = unified_df['label'].value_counts().sort_index()
    for label, count in class_dist.items():
        percentage = (count / len(unified_df)) * 100
        print(f"  {label:20s}: {count:5d} frames ({percentage:5.1f}%)")

    # Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unified_df.to_csv(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Dataset guardado: {output_path.name} ({size_mb:.1f} MB)")

    return unified_df


def main():
    script_dir = Path(__file__).parent
    csv_dir = script_dir / 'data_set_csv'
    output_path = script_dir / 'dataset_unified.csv'

    unify_csvs(csv_dir, output_path)

    print("\n" + "="*70)
    print("¡Unificación completada!")
    print("="*70)


if __name__ == "__main__":
    main()
