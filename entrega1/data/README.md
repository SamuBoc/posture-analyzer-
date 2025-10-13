# Directorio de Datos

Este directorio contiene los datos del proyecto.

## Estructura

### `raw/`
Videos originales grabados.

**Formato esperado:**
- Resolución: 1280x720 o 1920x1080
- FPS: 30
- Formato: MP4
- Nomenclatura: `{actividad}_{persona}_{velocidad}_{angulo}_{repeticion}.mp4`

Ejemplo:
```
caminar_hacia_p1_normal_frontal_01.mp4
sentarse_p2_lenta_diagonal_02.mp4
girar_p3_rapida_perfil_01.mp4
```

### `processed/`
Landmarks extraídos de los videos usando MediaPipe.

**Formato:**
- CSV con columnas: frame, timestamp, y para cada landmark: x, y, z, visibility
- Total de columnas: 2 + (33 landmarks × 4 atributos) = 134 columnas

## Cómo usar

1. Coloca tus videos en `raw/`
2. Ejecuta el script de extracción:
   ```bash
   python ../src/data/extract_landmarks.py --input raw/tu_video.mp4 --output processed/landmarks.csv
   ```
3. Los landmarks se guardarán en `processed/`

## Nota

Los archivos de video (.mp4) y datos procesados (.csv) están en `.gitignore` por su tamaño.
Solo subir archivos pequeños o ejemplos si es necesario.
