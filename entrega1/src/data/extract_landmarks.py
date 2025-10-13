"""
Script para extraer landmarks corporales de videos usando MediaPipe Pose.

Este script procesa videos y extrae las coordenadas de 33 puntos del cuerpo
en cada frame. Los datos se guardan en formato CSV para análisis posterior.

Uso:
    python extract_landmarks.py --input video.mp4 --output landmarks.csv
    python extract_landmarks.py --webcam  # Para usar cámara en tiempo real
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm


class PoseLandmarkExtractor:
    """Clase para extraer landmarks de poses usando MediaPipe."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Inicializa el extractor de landmarks.

        Args:
            min_detection_confidence: Confianza mínima para detectar persona (0-1)
            min_tracking_confidence: Confianza mínima para tracking (0-1)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # 0=lite, 1=full, 2=heavy
        )

        # Nombres de los 33 landmarks
        self.landmark_names = [
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

    def extract_from_video(self, video_path, visualize=False):
        """
        Extrae landmarks de un video.

        Args:
            video_path: Ruta al archivo de video
            visualize: Si True, muestra el video con landmarks dibujados

        Returns:
            DataFrame con landmarks de cada frame
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Procesando: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")

        all_landmarks = []
        frame_idx = 0

        with tqdm(total=total_frames, desc="Extrayendo landmarks") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # MediaPipe requiere RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Procesar frame
                results = self.pose.process(frame_rgb)

                # Extraer landmarks
                if results.pose_landmarks:
                    landmarks_data = self._extract_landmarks_data(
                        results.pose_landmarks,
                        frame_idx,
                        fps
                    )
                    all_landmarks.append(landmarks_data)

                    # Visualizar si se solicita
                    if visualize:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS
                        )
                        cv2.imshow('MediaPipe Pose', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    # Frame sin detección, guardar NaN
                    landmarks_data = self._create_empty_row(frame_idx, fps)
                    all_landmarks.append(landmarks_data)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        if visualize:
            cv2.destroyAllWindows()

        # Convertir a DataFrame
        df = pd.DataFrame(all_landmarks)
        return df

    def extract_from_webcam(self):
        """
        Extrae landmarks de la webcam en tiempo real.
        Presiona 'q' para salir.
        """
        cap = cv2.VideoCapture(0)

        print("Iniciando cámara... Presiona 'q' para salir")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            # Dibujar landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                # Mostrar algunas coordenadas clave
                landmarks = results.pose_landmarks.landmark
                nose = landmarks[0]
                left_wrist = landmarks[15]
                right_wrist = landmarks[16]

                # Texto con info
                cv2.putText(frame, f"Nose: ({nose.x:.2f}, {nose.y:.2f})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"L Wrist: ({left_wrist.x:.2f}, {left_wrist.y:.2f})",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"R Wrist: ({right_wrist.x:.2f}, {right_wrist.y:.2f})",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Pose - Webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _extract_landmarks_data(self, pose_landmarks, frame_idx, fps):
        """Extrae datos de landmarks a un diccionario."""
        data = {
            'frame': frame_idx,
            'timestamp': frame_idx / fps if fps > 0 else 0
        }

        # Extraer x, y, z, visibility de cada landmark
        for idx, landmark_name in enumerate(self.landmark_names):
            landmark = pose_landmarks.landmark[idx]
            data[f'{landmark_name}_x'] = landmark.x
            data[f'{landmark_name}_y'] = landmark.y
            data[f'{landmark_name}_z'] = landmark.z
            data[f'{landmark_name}_visibility'] = landmark.visibility

        return data

    def _create_empty_row(self, frame_idx, fps):
        """Crea una fila con NaN para frames sin detección."""
        data = {
            'frame': frame_idx,
            'timestamp': frame_idx / fps if fps > 0 else 0
        }

        for landmark_name in self.landmark_names:
            data[f'{landmark_name}_x'] = np.nan
            data[f'{landmark_name}_y'] = np.nan
            data[f'{landmark_name}_z'] = np.nan
            data[f'{landmark_name}_visibility'] = 0.0

        return data

    def __del__(self):
        """Limpia recursos."""
        if hasattr(self, 'pose'):
            self.pose.close()


def process_video_file(input_path, output_path, visualize=False):
    """
    Procesa un video y guarda los landmarks en CSV.

    Args:
        input_path: Ruta al video de entrada
        output_path: Ruta al CSV de salida
        visualize: Si True, muestra visualización
    """
    extractor = PoseLandmarkExtractor()

    # Extraer landmarks
    df = extractor.extract_from_video(input_path, visualize=visualize)

    # Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nLandmarks guardados en: {output_path}")
    print(f"  Total frames procesados: {len(df)}")
    print(f"  Frames con detección: {df['nose_visibility'].notna().sum()}")
    print(f"  Frames sin detección: {df['nose_visibility'].isna().sum()}")


def main():
    parser = argparse.ArgumentParser(
        description='Extrae landmarks corporales de videos usando MediaPipe Pose'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Ruta al video de entrada'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Ruta al archivo CSV de salida'
    )

    parser.add_argument(
        '--webcam', '-w',
        action='store_true',
        help='Usar webcam en tiempo real'
    )

    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Mostrar visualización durante el procesamiento'
    )

    args = parser.parse_args()

    if args.webcam:
        # Modo webcam
        extractor = PoseLandmarkExtractor()
        extractor.extract_from_webcam()

    elif args.input and args.output:
        # Modo archivo
        process_video_file(args.input, args.output, args.visualize)

    else:
        parser.print_help()
        print("\nEjemplos de uso:")
        print("  python extract_landmarks.py --webcam")
        print("  python extract_landmarks.py --input video.mp4 --output landmarks.csv")
        print("  python extract_landmarks.py -i video.mp4 -o landmarks.csv --visualize")


if __name__ == "__main__":
    main()
