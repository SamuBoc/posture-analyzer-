"""
Sistema de inferencia en tiempo real con webcam.

Usa MediaPipe para detectar poses en tiempo real y clasifica
la actividad usando el mejor modelo entrenado.

Uso:
    python realtime_inference.py --model models/model_random_forest.pkl
    python realtime_inference.py --model models/model_xgboost.pkl
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import argparse
from pathlib import Path
from collections import deque


class RealtimeActivityClassifier:
    """Clasificador de actividades en tiempo real."""

    def __init__(self, model_path):
        """
        Inicializa el clasificador.

        Args:
            model_path: Path al modelo entrenado (.pkl)
        """
        # Cargar modelo
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_name = model_data['model_name']

        print(f"✓ Modelo cargado: {self.model_name}")
        print(f"  Clases: {self.label_encoder.classes_.tolist()}")

        # Inicializar MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        # Nombres de landmarks
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

        # Buffer para landmarks previos (para calcular velocidades)
        self.prev_landmarks = None

        # Buffer para suavizar predicciones
        self.predictions_buffer = deque(maxlen=10)

        # Colores para cada clase
        self.class_colors = {
            'caminarEspalda': (255, 0, 0),      # Azul
            'caminarFrente': (0, 255, 0),       # Verde
            'girar': (0, 255, 255),             # Amarillo
            'levantarse': (255, 0, 255),        # Magenta
            'sentarse': (255, 128, 0)           # Naranja
        }

    def extract_landmarks(self, pose_landmarks):
        """
        Extrae landmarks de la pose detectada.

        Args:
            pose_landmarks: Landmarks detectados por MediaPipe

        Returns:
            Dict con valores de landmarks
        """
        landmarks_data = {}

        # Extraer x, y, z, visibility
        for idx, landmark_name in enumerate(self.landmark_names):
            landmark = pose_landmarks.landmark[idx]
            landmarks_data[f'{landmark_name}_x'] = landmark.x
            landmarks_data[f'{landmark_name}_y'] = landmark.y
            landmarks_data[f'{landmark_name}_z'] = landmark.z
            landmarks_data[f'{landmark_name}_visibility'] = landmark.visibility

        return landmarks_data

    def normalize_landmarks(self, landmarks_data):
        """
        Normaliza landmarks por distancia entre hombros.

        Args:
            landmarks_data: Dict con landmarks

        Returns:
            Dict con landmarks normalizados
        """
        # Calcular distancia entre hombros
        shoulder_dist = np.sqrt(
            (landmarks_data['left_shoulder_x'] - landmarks_data['right_shoulder_x'])**2 +
            (landmarks_data['left_shoulder_y'] - landmarks_data['right_shoulder_y'])**2 +
            (landmarks_data['left_shoulder_z'] - landmarks_data['right_shoulder_z'])**2
        )

        if shoulder_dist == 0:
            shoulder_dist = 1

        # Normalizar coordenadas
        normalized = landmarks_data.copy()
        for landmark_name in self.landmark_names:
            for coord in ['x', 'y', 'z']:
                key = f'{landmark_name}_{coord}'
                normalized[key] = normalized[key] / shoulder_dist

        return normalized

    def calculate_features(self, landmarks_data):
        """
        Calcula features adicionales (velocidades, ángulos, etc.).

        Args:
            landmarks_data: Dict con landmarks normalizados

        Returns:
            Dict con todas las features
        """
        features = landmarks_data.copy()

        # 1. Velocidades (si hay landmarks previos)
        key_landmarks = [
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]

        if self.prev_landmarks is not None:
            for landmark in key_landmarks:
                for coord in ['x', 'y', 'z']:
                    curr_val = landmarks_data[f'{landmark}_{coord}']
                    prev_val = self.prev_landmarks[f'{landmark}_{coord}']
                    features[f'{landmark}_{coord}_vel'] = curr_val - prev_val

                # Velocidad total
                vel_x = features[f'{landmark}_x_vel']
                vel_y = features[f'{landmark}_y_vel']
                vel_z = features[f'{landmark}_z_vel']
                features[f'{landmark}_vel_total'] = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        else:
            # Primer frame: velocidad = 0
            for landmark in key_landmarks:
                for coord in ['x', 'y', 'z']:
                    features[f'{landmark}_{coord}_vel'] = 0.0
                features[f'{landmark}_vel_total'] = 0.0

        # 2. Ángulos de articulaciones
        def angle_between_points(p1, p2, p3):
            """Calcula ángulo formado por 3 puntos."""
            v1 = np.array([p1[f'{k}'] for k in ['x', 'y', 'z']])
            v2 = np.array([p2[f'{k}'] for k in ['x', 'y', 'z']])
            v3 = np.array([p3[f'{k}'] for k in ['x', 'y', 'z']])

            vec1 = v1 - v2
            vec2 = v3 - v2

            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            return angle

        # Definir puntos para cada ángulo
        def get_point(landmark_name):
            return {
                'x': landmarks_data[f'{landmark_name}_x'],
                'y': landmarks_data[f'{landmark_name}_y'],
                'z': landmarks_data[f'{landmark_name}_z']
            }

        joint_angles = {
            'left_elbow_angle': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'right_elbow_angle': ('right_shoulder', 'right_elbow', 'right_wrist'),
            'left_knee_angle': ('left_hip', 'left_knee', 'left_ankle'),
            'right_knee_angle': ('right_hip', 'right_knee', 'right_ankle'),
            'left_hip_angle': ('left_shoulder', 'left_hip', 'left_knee'),
            'right_hip_angle': ('right_shoulder', 'right_hip', 'right_knee'),
        }

        for angle_name, (p1_name, p2_name, p3_name) in joint_angles.items():
            p1 = get_point(p1_name)
            p2 = get_point(p2_name)
            p3 = get_point(p3_name)
            features[angle_name] = angle_between_points(p1, p2, p3)

        # 3. Inclinación del tronco
        shoulder_mid = {
            'x': (landmarks_data['left_shoulder_x'] + landmarks_data['right_shoulder_x']) / 2,
            'y': (landmarks_data['left_shoulder_y'] + landmarks_data['right_shoulder_y']) / 2,
            'z': (landmarks_data['left_shoulder_z'] + landmarks_data['right_shoulder_z']) / 2
        }

        hip_mid = {
            'x': (landmarks_data['left_hip_x'] + landmarks_data['right_hip_x']) / 2,
            'y': (landmarks_data['left_hip_y'] + landmarks_data['right_hip_y']) / 2,
            'z': (landmarks_data['left_hip_z'] + landmarks_data['right_hip_z']) / 2
        }

        trunk_x = shoulder_mid['x'] - hip_mid['x']
        trunk_y = shoulder_mid['y'] - hip_mid['y']
        trunk_z = shoulder_mid['z'] - hip_mid['z']

        trunk_inclination = np.arctan2(
            np.sqrt(trunk_x**2 + trunk_z**2),
            trunk_y + 1e-8
        )

        features['trunk_inclination'] = trunk_inclination
        features['trunk_inclination_degrees'] = np.degrees(trunk_inclination)

        # Guardar landmarks actuales para próximo frame
        self.prev_landmarks = landmarks_data.copy()

        return features

    def predict(self, features):
        """
        Predice la actividad a partir de features.

        Args:
            features: Dict con todas las features

        Returns:
            Clase predicha y probabilidad
        """
        # Ordenar features según el orden del modelo
        feature_values = [features.get(name, 0.0) for name in self.feature_names]

        # Convertir a array y escalar
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predecir
        y_pred = self.model.predict(X_scaled)[0]
        predicted_class = self.label_encoder.inverse_transform([y_pred])[0]

        # Obtener probabilidad (si el modelo lo soporta)
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_scaled)[0]
            confidence = probas[y_pred]
        else:
            confidence = 1.0

        # Añadir a buffer para suavizar
        self.predictions_buffer.append(predicted_class)

        # Predicción suavizada (moda del buffer)
        from collections import Counter
        smoothed_pred = Counter(self.predictions_buffer).most_common(1)[0][0]

        return smoothed_pred, confidence

    def draw_info(self, frame, predicted_class, confidence):
        """
        Dibuja información en el frame.

        Args:
            frame: Frame de video
            predicted_class: Clase predicha
            confidence: Confianza de la predicción

        Returns:
            Frame con información dibujada
        """
        # Fondo semi-transparente para el texto
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Rectángulo superior
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Título
        cv2.putText(
            frame,
            "Clasificador de Actividades en Tiempo Real",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # Actividad predicha
        color = self.class_colors.get(predicted_class, (255, 255, 255))
        cv2.putText(
            frame,
            f"Actividad: {predicted_class}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

        # Confianza
        cv2.putText(
            frame,
            f"Confianza: {confidence:.1%}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Instrucciones
        cv2.putText(
            frame,
            "Presiona 'q' para salir",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        return frame

    def run(self):
        """Ejecuta el clasificador en tiempo real con webcam."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: No se pudo abrir la webcam")
            return

        print("\n" + "="*70)
        print("CLASIFICACIÓN EN TIEMPO REAL")
        print("="*70)
        print(f"\nModelo: {self.model_name}")
        print("Presiona 'q' para salir")
        print("\nActividades reconocibles:")
        for idx, activity in enumerate(self.label_encoder.classes_, 1):
            print(f"  {idx}. {activity}")
        print("\n" + "="*70)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)

            # Convertir a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar con MediaPipe
            results = self.pose.process(frame_rgb)

            # Clasificar si hay detección
            if results.pose_landmarks:
                # Dibujar landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                # Extraer y procesar landmarks
                landmarks_data = self.extract_landmarks(results.pose_landmarks)
                landmarks_norm = self.normalize_landmarks(landmarks_data)
                features = self.calculate_features(landmarks_norm)

                # Predecir
                predicted_class, confidence = self.predict(features)

                # Dibujar información
                frame = self.draw_info(frame, predicted_class, confidence)
            else:
                # Sin detección
                cv2.putText(
                    frame,
                    "No se detecta persona",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )

            # Mostrar frame
            cv2.imshow('Clasificador de Actividades', frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Clasificación terminada")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Clasificador de actividades en tiempo real con webcam'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/model_random_forest.pkl',
        help='Path al modelo entrenado (.pkl)'
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Error: No se encuentra el modelo en {model_path}")
        print("\nModelos disponibles:")
        models_dir = Path(__file__).parent / 'models'
        if models_dir.exists():
            for model_file in models_dir.glob('*.pkl'):
                print(f"  - {model_file}")
        return

    # Crear clasificador y ejecutar
    classifier = RealtimeActivityClassifier(model_path)
    classifier.run()


if __name__ == "__main__":
    main()
