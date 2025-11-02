"""
Script para procesar todos los videos y extraer landmarks a CSV.

Procesa automáticamente todos los videos en videos_data_set/ y guarda
los landmarks en data_set_csv/ con la columna 'label' agregada.

Uso:
    python process_all_videos.py
"""

import sys
from pathlib import Path
import pandas as pd
import cv2
import mediapipe as mp
from tqdm import tqdm

# Agregar el path de entrega1 para importar el extractor
sys.path.insert(0, str(Path(__file__).parent.parent / 'entrega1' / 'src' / 'data'))
from extract_landmarks import PoseLandmarkExtractor


def extract_activity_from_filename(video_path):
    """
    Extrae la actividad del nombre del archivo.

    Ejemplo: 'caminarFrente_01.mp4' -> 'caminarFrente'
    """
    filename = video_path.stem  # Nombre sin extensión
    activity = filename.rsplit('_', 1)[0]  # Split por último '_'
    return activity


def process_video_with_label(video_path, output_path, extractor):
    """
    Procesa un video y agrega la columna 'label' al CSV.

    Args:
        video_path: Path al video de entrada
        output_path: Path al CSV de salida
        extractor: Instancia de PoseLandmarkExtractor
    """
    # Extraer actividad del nombre
    activity = extract_activity_from_filename(video_path)

    print(f"\n{'='*70}")
    print(f"Procesando: {video_path.name}")
    print(f"Actividad: {activity}")
    print(f"{'='*70}")

    # Extraer landmarks
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: No se pudo abrir {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")

    all_landmarks = []
    frame_idx = 0
    frames_detectados = 0

    with tqdm(total=total_frames, desc="Extrayendo landmarks") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar con MediaPipe
            results = extractor.pose.process(frame_rgb)

            # Extraer landmarks
            if results.pose_landmarks:
                landmarks_data = extractor._extract_landmarks_data(
                    results.pose_landmarks,
                    frame_idx,
                    fps
                )
                all_landmarks.append(landmarks_data)
                frames_detectados += 1
            else:
                # Frame sin detección
                landmarks_data = extractor._create_empty_row(frame_idx, fps)
                all_landmarks.append(landmarks_data)

            frame_idx += 1
            pbar.update(1)

    cap.release()

    # Convertir a DataFrame
    df = pd.DataFrame(all_landmarks)

    # Agregar columna 'label'
    df['label'] = activity

    # Guardar CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Guardado: {output_path.name}")
    print(f"  Frames totales: {len(df)}")
    print(f"  Frames con detección: {frames_detectados} ({frames_detectados/len(df)*100:.1f}%)")
    print(f"  Frames sin detección: {len(df) - frames_detectados}")
    print(f"  Columnas: {len(df.columns)}")

    return True


def main():
    """Función principal."""
    # Directorios
    script_dir = Path(__file__).parent
    videos_dir = script_dir / 'videos_data_set'
    output_dir = script_dir / 'data_set_csv'

    # Verificar que existe el directorio de videos
    if not videos_dir.exists():
        print(f"Error: No se encuentra el directorio {videos_dir}")
        return

    # Obtener lista de videos
    videos = sorted(videos_dir.glob('*.mp4'))

    if not videos:
        print(f"No se encontraron videos en {videos_dir}")
        return

    print("="*70)
    print("PROCESAMIENTO BATCH DE VIDEOS")
    print("="*70)
    print(f"\nDirectorio de entrada: {videos_dir}")
    print(f"Directorio de salida: {output_dir}")
    print(f"Total de videos: {len(videos)}")

    # Contar videos por actividad
    actividades = {}
    for video in videos:
        activity = extract_activity_from_filename(video)
        actividades[activity] = actividades.get(activity, 0) + 1

    print("\nDistribución de actividades:")
    for activity, count in sorted(actividades.items()):
        print(f"  {activity:20s}: {count} videos")

    print("\n" + "="*70)
    print("\nIniciando procesamiento automático...")

    # Crear extractor
    print("\nInicializando MediaPipe...")
    extractor = PoseLandmarkExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Procesar cada video
    resultados = {
        'exitosos': 0,
        'fallidos': 0,
        'total_frames': 0
    }

    for idx, video_path in enumerate(videos, 1):
        print(f"\n{'#'*70}")
        print(f"VIDEO {idx}/{len(videos)}")
        print(f"{'#'*70}")

        # Nombre del CSV de salida
        output_name = video_path.stem + '.csv'
        output_path = output_dir / output_name

        # Procesar
        try:
            success = process_video_with_label(video_path, output_path, extractor)
            if success:
                resultados['exitosos'] += 1
                # Contar frames
                df = pd.read_csv(output_path)
                resultados['total_frames'] += len(df)
            else:
                resultados['fallidos'] += 1
        except Exception as e:
            print(f"\n❌ Error procesando {video_path.name}: {e}")
            resultados['fallidos'] += 1

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"Videos procesados exitosamente: {resultados['exitosos']}/{len(videos)}")
    print(f"Videos con errores: {resultados['fallidos']}")
    print(f"Total de frames extraídos: {resultados['total_frames']}")
    print(f"CSVs guardados en: {output_dir}")
    print("="*70)

    # Listar archivos generados
    csv_files = sorted(output_dir.glob('*.csv'))
    if csv_files:
        print(f"\nArchivos CSV generados ({len(csv_files)}):")
        for csv_file in csv_files:
            size_kb = csv_file.stat().st_size / 1024
            print(f"  {csv_file.name:30s} ({size_kb:,.1f} KB)")

    print("\n¡Procesamiento completado!")


if __name__ == "__main__":
    main()
