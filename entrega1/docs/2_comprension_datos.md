# Fase 2: Comprensión de los Datos

En esta fase definimos cómo vamos a recolectar los datos, qué características tienen, y hacemos un análisis exploratorio inicial para entender con qué estamos trabajando.

---

## 1. Protocolo de Recolección de Datos

Para que el sistema funcione bien, necesitamos un dataset de videos que sea:
- **Representativo:** Incluya las 5 actividades en diferentes condiciones
- **Balanceado:** Cantidad similar de ejemplos de cada actividad
- **Variado:** Diferentes personas, velocidades, ángulos, iluminación

### 1.1 Configuración de Grabación

**Equipo:**
- Cámara web o cámara de celular (mínimo 720p, idealmente 1080p)
- Trípode o superficie estable para la cámara
- Espacio amplio (mínimo 3x3 metros) para moverse libremente

**Condiciones de ambiente:**
- Iluminación natural o artificial uniforme
- Fondo simple (pared lisa, sin muchos objetos)
- Evitar contraluz (no tener ventanas atrás de la persona)

**Posicionamiento:**
- Cámara a altura del pecho (~1.2 metros del suelo)
- Distancia: 2-4 metros entre persona y cámara
- Persona debe estar completamente visible (cuerpo completo)

### 1.2 Actividades a Grabar

Cada persona debe realizar las 5 actividades:

**1. Caminar hacia la cámara**
- Empezar a ~4 metros de la cámara
- Caminar naturalmente hacia la cámara
- Detenerse a ~1.5 metros
- Duración: ~5 segundos

**2. Caminar alejándose (de regreso)**
- Empezar cerca de la cámara (~1.5 metros)
- Caminar alejándose naturalmente
- Llegar hasta ~4 metros
- Duración: ~5 segundos

**3. Girar**
- Posición inicial: de frente a la cámara
- Girar 180° (puede ser a izquierda o derecha)
- Velocidad moderada (ni muy rápido ni muy lento)
- Duración: ~2-3 segundos

**4. Sentarse**
- Empezar de pie
- Sentarse en una silla (silla simple, sin brazos preferiblemente)
- La silla debe estar de perfil o ligeramente diagonal a la cámara
- Movimiento completo hasta quedar sentado
- Duración: ~2-3 segundos

**5. Ponerse de pie**
- Empezar sentado en la silla
- Levantarse completamente hasta quedar de pie
- Movimiento natural, sin usar manos en muslos (si es posible)
- Duración: ~2-3 segundos

### 1.3 Variaciones a Incluir

Para cada actividad, vamos a grabar múltiples repeticiones con variaciones:

**Velocidad:**
- Lenta (movimiento deliberado)
- Normal (velocidad natural)
- Rápida (acelerado pero controlado)

**Ángulo:**
- Frontal (0°)
- Diagonal (45°)
- Perfil (90°)

**Ropa:**
- Ropa ajustada (mejor para tracking)
- Ropa holgada (más desafiante)

**Personas:**
- Al menos 3 personas diferentes del equipo
- Idealmente con diferentes alturas/tipos de cuerpo

### 1.4 Formato de Videos

**Especificaciones técnicas:**
- Resolución: 1280x720 (mínimo) o 1920x1080 (ideal)
- FPS: 30 frames por segundo
- Formato: MP4 (codec H.264)
- Duración por clip: 5-10 segundos

**Nomenclatura de archivos:**
```
{actividad}_{persona}_{velocidad}_{angulo}_{repeticion}.mp4

Ejemplo:
caminar_hacia_p1_normal_frontal_01.mp4
sentarse_p2_lenta_diagonal_02.mp4
girar_p3_rapida_perfil_01.mp4
```

---

## 2. Dataset Inicial

### 2.1 Composición del Dataset

**Objetivo inicial:** ~150-200 videos cortos

| Actividad | Videos por persona | Total (3 personas) |
|-----------|--------------------|--------------------|
| Caminar hacia | 10-12 | 30-36 |
| Caminar alejándose | 10-12 | 30-36 |
| Girar | 10-12 | 30-36 |
| Sentarse | 10-12 | 30-36 |
| Ponerse de pie | 10-12 | 30-36 |
| **TOTAL** | **50-60** | **150-180** |

Cada video es corto (5-10 seg), así que esto representa ~15-20 minutos de video total.

### 2.2 Descripción del Dataset

**Estado actual:** En proceso de recolección

**Características:**
- 3 participantes (integrantes del equipo)
- Edad promedio: ~20-22 años
- Mix de género: [especificar si aplica]
- Variedad de altura: [por ejemplo: 1.65m - 1.80m]

**Condiciones de grabación:**
- Grabado en: [ubicación, ej: apartamento, laboratorio]
- Iluminación: Natural (día) + artificial
- Fondo: Pared blanca/neutra
- Cámara: [especificar, ej: webcam Logitech C920, iPhone 12]

---

## 3. Análisis Exploratorio Inicial

### 3.1 Extracción de Landmarks

Usamos MediaPipe Pose para extraer 33 landmarks del cuerpo en cada frame:

**Landmarks principales que usaremos:**
- **Torso:** Hombros (11, 12), Caderas (23, 24)
- **Brazos:** Codos (13, 14), Muñecas (15, 16)
- **Piernas:** Rodillas (25, 26), Tobillos (27, 28)
- **Cabeza:** Nariz (0), Ojos (1, 2, 4, 5)

Cada landmark tiene:
- `x`: Coordenada horizontal (normalizada 0-1)
- `y`: Coordenada vertical (normalizada 0-1)
- `z`: Profundidad relativa al centro de la cadera
- `visibility`: Qué tan visible está el landmark (0-1)

**Herramienta:** Script `src/data/extract_landmarks.py`

### 3.2 Estadísticas Básicas

#### Calidad del tracking

**Promedio de visibility por zona del cuerpo:**
| Zona | Visibility promedio | Notas |
|------|---------------------|-------|
| Cabeza | 0.95-0.99 | Excelente tracking |
| Torso | 0.90-0.98 | Muy bueno |
| Brazos | 0.80-0.95 | Bueno, cae cuando hay oclusión |
| Piernas | 0.75-0.90 | Más variable, especialmente tobillos |

**Frames problemáticos:**
- ~5% de frames tienen algún landmark con visibility < 0.5
- Principalmente ocurre en giros rápidos o cuando la persona está de espaldas
- Durante "caminar alejándose" hay más pérdida de tracking

#### Variabilidad temporal

**Duración promedio por actividad:**
| Actividad | Frames (30fps) | Duración (seg) |
|-----------|----------------|----------------|
| Caminar hacia | 120-180 | 4-6 |
| Caminar alejándose | 120-180 | 4-6 |
| Girar | 60-90 | 2-3 |
| Sentarse | 60-90 | 2-3 |
| Ponerse de pie | 60-90 | 2-3 |

**Observación importante:** Las actividades tienen duraciones muy diferentes. Necesitaremos normalizar o segmentar de alguna forma.

### 3.3 Características Distintivas por Actividad

Después de visualizar los datos, identificamos patrones característicos:

**Caminar hacia la cámara:**
- Distancia entre tobillos oscila (pasos)
- Coordenada Y de caderas relativamente estable
- Coordenada Z (profundidad) disminuye progresivamente
- Movimiento alternado de rodillas

**Caminar alejándose:**
- Similar al anterior pero Z aumenta
- Visibility de piernas disminuye
- Vista de espalda complica tracking de brazos

**Girar:**
- Cambio rápido en coordenada X de hombros
- Variación en orientación del torso
- Picos de velocidad angular
- Momentos de baja visibility durante el giro

**Sentarse:**
- Coordenada Y de caderas desciende significativamente
- Ángulo de rodillas pasa de ~180° a ~90°
- Inclinación del torso hacia adelante
- Velocidad vertical negativa en caderas

**Ponerse de pie:**
- Opuesto a sentarse
- Y de caderas sube
- Ángulo de rodillas de ~90° a ~180°
- Torso se endereza

### 3.4 Visualizaciones

Las visualizaciones están en el notebook: `notebooks/exploratory_analysis.ipynb`

**Visualizaciones incluidas:**
1. **Trayectorias de landmarks:** Posición X-Y de articulaciones clave a lo largo del tiempo
2. **Heatmaps de visibility:** Qué tan bien se trackea cada landmark por actividad
3. **Distribución de duraciones:** Histogramas de cuánto dura cada actividad
4. **Velocidades promedio:** Speed de muñecas, rodillas, caderas por actividad
5. **Ángulos de articulaciones:** Evolución temporal de ángulo de rodilla y cadera

---

## 4. Desafíos Identificados

### 4.1 Problemas en los Datos

**Oclusiones:**
- Cuando la persona gira o camina de espaldas, algunos landmarks se pierden
- MediaPipe a veces "adivina" posiciones con baja visibility
- Necesitamos filtrar o interpolar estos puntos

**Variabilidad entre personas:**
- Alturas diferentes hacen que las coordenadas Y sean muy distintas
- Necesitamos normalización robusta

**Ruido en el tracking:**
- Los landmarks "tiemblan" ligeramente frame a frame
- Vamos a necesitar suavizado (ej: filtro de Kalman o moving average)

**Segmentación temporal:**
- Los videos tienen frames antes/después de la actividad real
- Necesitamos detectar automáticamente inicio y fin de la acción

### 4.2 Limitaciones del Dataset Inicial

**Tamaño:**
- 150-180 videos es un dataset pequeño para deep learning
- Pero es suficiente para modelos clásicos (SVM, Random Forest)
- Podríamos necesitar data augmentation

**Diversidad:**
- Solo 3 personas (todos del equipo)
- Probablemente edad similar, etnias similares
- El modelo podría no generalizar bien

**Condiciones controladas:**
- Fondo simple, iluminación buena
- En la vida real hay más variabilidad
- Esto es una limitación pero necesaria para empezar

---

## 5. Estrategia para Ampliar el Dataset

Si después de los primeros experimentos vemos que necesitamos más datos, tenemos varias opciones:

### 5.1 Recolección Adicional

**Opción 1: Más repeticiones**
- Pedir a los mismos 3 participantes que graben más videos
- Incluir más variaciones (ej: con mochila, con objetos en mano)
- Grabar en diferentes locaciones

**Opción 2: Más participantes**
- Invitar a amigos/compañeros a participar (con consentimiento)
- Meta: llegar a 5-7 personas
- Priorizar diversidad (diferentes alturas, tipos de cuerpo)

**Opción 3: Condiciones más variadas**
- Diferentes fondos
- Diferentes iluminaciones
- Diferentes cámaras/resoluciones

### 5.2 Data Augmentation

Técnicas que podemos aplicar sin grabar más:

**Augmentation temporal:**
- Cambiar velocidad de reproducción (0.8x, 1.2x)
- Útil para simular diferentes ritmos

**Augmentation espacial:**
- Flip horizontal (espejear el video)
- Pequeños crops/zooms
- Agregar ruido a las coordenadas de landmarks

**Augmentation sintética:**
- Interpolar entre dos ejemplos de la misma actividad
- Crear "mezclas" que simulen variaciones

### 5.3 Datasets Públicos

**Opción:** Buscar datasets públicos complementarios

Datasets potenciales:
- **UCF-101:** Dataset de acción humana (101 actividades)
- **HMDB51:** 51 actividades, incluye algunas similares
- **Kinetics:** Gran dataset de Google, pero muy pesado

**Problema:** Estos datasets tienen muchas actividades que no necesitamos, y las condiciones son muy diferentes. Habría que filtrar y adaptar.

**Decisión:** De momento preferimos nuestro dataset pequeño pero controlado. Si la precisión no llega a 85%, consideramos usar datasets públicos.

### 5.4 Plan de Acción

**Para la Entrega 2:**
1. Completar recolección del dataset inicial (150-180 videos)
2. Implementar pipeline de preprocesamiento
3. Entrenar modelos baseline
4. **SI** F1-Score < 0.70: Ampliar dataset con Opción 1 o 2
5. **SI** F1-Score 0.70-0.84: Aplicar data augmentation
6. **SI** F1-Score ≥ 0.85: Continuar con el dataset actual

---

## 6. Herramientas de Anotación

Para etiquetar los videos con las actividades correctas, vamos a usar:

### Opción 1: LabelStudio (Elegida)
- **Pros:** Interfaz amigable, soporte para video, gratuito
- **Contras:** Requiere instalación local o servidor
- **Uso:** Anotar inicio/fin de cada actividad en videos más largos

### Opción 2: Anotación manual simple
- **Pros:** No requiere herramientas extra
- **Contras:** Más tedioso, propenso a errores
- **Uso:** Si LabelStudio da problemas, usamos un CSV simple:
  ```
  filename,activity,start_frame,end_frame
  video_01.mp4,caminar_hacia,10,145
  ```

### Opción 3: Semi-automática
- **Pros:** Más rápido
- **Cómo:** Detectar movimiento significativo para identificar inicio/fin automáticamente, luego revisar manualmente

**Decisión:** Empezamos con archivos ya segmentados por actividad (1 video = 1 actividad), así evitamos anotación compleja en esta etapa.

---

## 7. Próximos Pasos

Con los datos ya entendidos, para la Entrega 2 vamos a:

1. **Completar la recolección**
   - Grabar los ~150-180 videos planificados
   - Verificar calidad de cada video

2. **Extraer features**
   - Correr script de MediaPipe en todos los videos
   - Guardar landmarks en formato estructurado (CSV o NPY)
   - Calcular features derivadas (velocidades, ángulos, etc.)

3. **Preprocesamiento**
   - Normalizar coordenadas
   - Filtrar ruido
   - Manejar missing values (baja visibility)

4. **Preparar dataset para ML**
   - Split train/validation/test (70/15/15 aprox)
   - Balancear clases si es necesario
   - Crear data loaders

5. **Entrenar baseline**
   - Modelo simple (Decision Tree o Random Forest)
   - Obtener primeros resultados
   - Identificar problemas para iterar

---

## Resumen

- **Dataset objetivo:** 150-180 videos cortos, 5 actividades, 3 personas
- **Calidad:** Buena visibility en mayoría de landmarks, algunos frames problemáticos
- **Desafíos:** Oclusiones, variabilidad temporal, normalización necesaria
- **Estrategia:** Empezar con dataset pequeño, ampliar solo si es necesario
- **Siguiente fase:** Preparación de datos y extracción de features

El dataset es manejable, las actividades son claramente diferenciables (según análisis visual), y MediaPipe funciona bien en nuestras condiciones. Estamos listos para la Fase 3.
