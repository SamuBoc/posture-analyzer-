import cv2
import mediapipe as mp
import os

# Configurar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Videos a procesar (uno de cada actividad)
videos = {
    'caminarFrente': '../entrega2/videos_data_set/caminarFrente_02.mp4',
    'sentarse': '../entrega2/videos_data_set/sentarse_02.mp4',
    'levantarse': '../entrega2/videos_data_set/levantarse_02.mp4',
    'landmarks': '../entrega2/videos_data_set/girar_01.mp4'
}

def extract_frame_with_landmarks(video_path, activity_name, output_name):
    """Extrae un frame del medio del video con landmarks dibujados"""
    cap = cv2.VideoCapture(video_path)

    # Obtener el total de frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    # Ir al frame del medio
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    success, frame = cap.read()
    if not success:
        print(f"❌ Error leyendo {video_path}")
        cap.release()
        return

    # Convertir BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar con MediaPipe
    results = pose.process(frame_rgb)

    # Dibujar landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Agregar etiqueta de actividad (excepto para landmarks general)
        if activity_name != 'landmarks':
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Actividad: {activity_name}"
            # Fondo del texto
            (text_width, text_height), _ = cv2.getTextSize(text, font, 1.0, 2)
            cv2.rectangle(frame, (10, 10), (text_width + 20, text_height + 30), (0, 0, 0), -1)
            # Texto
            cv2.putText(frame, text, (15, 35), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # Guardar
    cv2.imwrite(output_name, frame)
    print(f"✅ Guardado: {output_name}")

    cap.release()

# Procesar cada video
print("Extrayendo frames con landmarks...\n")

extract_frame_with_landmarks(
    videos['caminarFrente'],
    'caminarFrente',
    'IMAGEN_NECESARIA_realtime_caminarFrente.png'
)

extract_frame_with_landmarks(
    videos['sentarse'],
    'sentarse',
    'IMAGEN_NECESARIA_realtime_sentarse.png'
)

extract_frame_with_landmarks(
    videos['levantarse'],
    'levantarse',
    'IMAGEN_NECESARIA_realtime_levantarse.png'
)

extract_frame_with_landmarks(
    videos['landmarks'],
    'landmarks',
    'IMAGEN_NECESARIA_landmarks_extraction.png'
)

pose.close()
print("\n✅ Todas las imágenes generadas exitosamente!")
