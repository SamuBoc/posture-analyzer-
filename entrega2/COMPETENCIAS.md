# Evidencias de Competencias - Entrega 2
## Proyecto: Sistema de Clasificación de Actividades Humanas

---

## Resumen

Este documento compila las evidencias de las competencias evaluadas en la Entrega 2 del curso de Inteligencia Artificial I, según los indicadores de desempeño PI1, PI2 y PI3.

**Proyecto:** Sistema de clasificación de actividades humanas usando MediaPipe Pose y Machine Learning.

**Fecha:** Noviembre 2025

---

## Competencia 1: Ética

### Indicador PI1: Reconocimiento de implicaciones éticas

**Definición:** Reconocer, con respecto al código de ética profesional, las implicaciones de las actividades o decisiones desarrolladas.

### Evidencia en el Proyecto

 **Ubicación en el informe:** Sección 3 - Consideraciones Éticas (páginas 3-7 del informe)

### Situaciones Éticas Identificadas

1. **Privacidad de datos de video**
   - Dilema: Procesamiento de video captura datos biométricos sensibles
   - Principio ético: Respeto a la privacidad y confidencialidad
   - Medida adoptada: Solo usar videos propios, procesamiento local sin cloud

2. **Sesgos en modelos de clasificación**
   - Dilema: Modelo entrenado con 1 persona → discriminación no intencionada
   - Principio ético: Responsabilidad social y profesional
   - Medida adoptada: Documentar explícitamente las limitaciones y sesgos

3. **Consentimiento informado**
   - Dilema: Uso de videos sin autorización viola autonomía
   - Principio ético: Respeto a la dignidad y autonomía humana
   - Medida adoptada: Consentimiento explícito de todos los participantes

4. **Transparencia de limitaciones**
   - Dilema: 76.6% accuracy significa 23.4% de error
   - Principio ético: Honestidad e integridad profesional
   - Medida adoptada: Documentar limitaciones claramente en README e informe

5. **Uso responsable del sistema**
   - Dilema: Sistema podría usarse para vigilancia no autorizada
   - Principio ético: Bienestar y seguridad pública
   - Medida adoptada: Advertencias de uso solo educativo, no médico ni vigilancia

### Relación con Código de Ética Profesional

| Situación | Principio del Código | Justificación |
|-----------|---------------------|---------------|
| Privacidad de videos | Respeto a la privacidad | Videos contienen datos biométricos personales |
| Sesgos (1 persona) | Responsabilidad social | Documentar limitaciones es responsabilidad profesional |
| Consentimiento | Respeto a la dignidad humana | Derecho a decidir sobre uso de imagen y datos |
| Transparencia | Honestidad e integridad | Comunicar capacidades y limitaciones claramente |
| Uso responsable | Bienestar público | Considerar mal uso y prevenir daño |

### Medidas Concretas Implementadas

1. **Protección de privacidad:** Solo videos del equipo, procesamiento local
2. **Documentación de sesgos:** Limitaciones explícitas en README e informe
3. **Consentimiento:** Autorización de todos los miembros del equipo
4. **Advertencias:** Sistema solo con fines educativos, no médico
5. **Código abierto:** Transparencia total del código y metodología

### Reflexión

> "Aunque este proyecto es pequeño y académico, aprendimos que **incluso sistemas simples tienen implicaciones éticas**. La transparencia es más importante que la perfección, y como desarrolladores somos responsables de documentar riesgos y limitaciones."

---

## Competencia 1: Impactos

### Indicador PI2: Explicación de impactos globales, sociales, ambientales y económicos

**Definición:** Explicar adecuadamente los impactos globales, ambientales, sociales y económicos de la solución de ingeniería.

### Evidencia en el Proyecto

**Ubicación en el informe:** Sección 4 - Análisis de Impactos (páginas 8-12 del informe)

### Matriz Completa de Impactos

#### 1. Impactos Sociales

**Positivos:**
- Rehabilitación física accesible remotamente
- Monitoreo deportivo sin sensores costosos
- Educación física mejorada
- Accesibilidad (solo requiere webcam)

**Negativos:**
- Vigilancia no autorizada
- Discriminación por físico (modelo entrenado con 1 persona)
- Exclusión digital (requiere computadora)
- Percepción de vigilancia constante

**Mitigación:**
- Consentimiento obligatorio antes de grabación
- Expandir dataset con múltiples personas diversas
- Documentar sesgos explícitamente
- Indicador visual cuando cámara está activa

#### 2. Impactos Económicos

**Positivos:**
- Bajo costo (~$20-50 USD webcam)
- Software open source (gratis)
- Accesible para fisioterapeutas independientes
- Reducción costos en salud (seguimiento en casa)

**Negativos:**
- Posible reemplazo de trabajo humano
- Brecha digital (no todos tienen computadora)
- Requiere conocimientos técnicos para instalación

**Mitigación:**
- Posicionar como herramienta complementaria, no sustituto
- Capacitación gratuita y documentación accesible
- Interfaz simplificada para no-técnicos
- Colaboración con profesionales de salud

#### 3. Impactos Ambientales

**Positivos:**
- No requiere hardware especial (reduce e-waste)
- Procesamiento local (menos consumo data centers)
- Reutiliza cámaras existentes

**Negativos:**
- Consumo energético CPU/GPU (30-50W continuo)
- Entrenamiento costoso (0.5-0.8 kWh para GridSearchCV)
- Requiere hardware relativamente moderno

**Datos cuantitativos:**
- Entrenamiento completo: ~500-800 Wh
- Inferencia tiempo real: ~30-50W
- Comparación: Data center consumiría ~100-150W + overhead red

**Mitigación:**
- Optimización de modelos (quantización, pruning)
- Modo de bajo consumo (reducir FPS sin movimiento)
- Compartir modelos pre-entrenados (evitar re-entrenamiento)
- Documentar consumo energético para usuarios

#### 4. Impactos Globales

**Positivos:**
- Aplicable en cualquier país (offline, no cloud)
- Código abierto (accesible mundialmente)
- Independiente de idioma (trabaja con movimientos)
- Útil en países en desarrollo (acceso limitado a fisioterapia)

**Negativos:**
- Sesgo cultural (actividades reflejan contexto occidental)
- Regulaciones de privacidad (GDPR, LGPD, CCPA)
- Brecha digital global (37% población sin internet)
- Expectativas culturales diferentes (no todos aceptan ser grabados)

**Mitigación:**
- Cumplimiento GDPR (procesamiento local, no almacenar)
- Expandir actividades culturalmente diversas
- Documentación multiidioma
- Respetar diferencias culturales en uso

### Evaluación de Riesgos

| Riesgo | Probabilidad | Severidad | Prioridad |
|--------|-------------|-----------|-----------|
| Vigilancia sin consentimiento | Alta | Alta | CRÍTICO |
| Discriminación por físico | Alta | Media | ALTO |
| Incumplimiento GDPR | Media | Alta | ALTO |
| Reemplazo trabajo humano | Baja | Media | MEDIO |
| Consumo energético excesivo | Media | Baja | BAJO |

### Conclusiones sobre Impactos

> "Creemos que un sistema como este tiene **potencial neto positivo** si se usa responsablemente, pero requiere controles de acceso, expansión del dataset, cumplimiento regulatorio, y optimización continua."

---

## Competencia 2: Solución de Problemas (Matemática)

### Indicador PI3: Solución de problemas complejos aplicando matemáticas

**Definición:** Resolver problemas complejos proponiendo estrategias compatibles con su formulación y aplicando matemáticas.

### Evidencia en el Proyecto

- **Ubicación en el informe:** Sección 5 - Fundamentos Matemáticos y Computacionales (páginas 12-35 del informe)

### 1. Formulación Matemática del Problema

**Problema:** Clasificación multiclase supervisada sobre series temporales

**Entrada:**
- Secuencia de landmarks: `L = {l₁, l₂, ..., lₙ}` donde `lᵢ ∈ ℝ^(33×4)`

**Salida:**
- Clase: `y ∈ {C₁, C₂, C₃, C₄, C₅}`

**Objetivo:**
```
f*(x) = argmax P(y = cₖ | x)
         k∈{1,2,3,4,5}
```

**Complejidad:** Problema no trivial por:
- Alta dimensionalidad (132 features base)
- Variabilidad temporal (misma actividad a diferentes velocidades)
- Similitud entre clases (levantarse ≈ sentarse inverso)
- Ruido en datos (visibilidad de MediaPipe variable)
- Necesidad de generalización

### 2. Aplicación de Fundamentos Matemáticos

#### A. Normalización por Distancia de Hombros

**Fundamento teórico:** Invarianza a la escala

**Fórmula:**
```
d_hombros = ||P_izq - P_der||₂ = √((x_izq - x_der)² + (y_izq - y_der)² + (z_izq - z_der)²)
x_norm = x / d_hombros
```

**Justificación:** División por factor de escala constante hace al modelo invariante a distancia de cámara y tamaño de persona.

#### B. Cálculo de Ángulos (Álgebra Lineal)

**Fundamento teórico:** Producto punto de vectores

**Fórmula:**
```
v₁ = P₁ - P₂,  v₂ = P₃ - P₂
θ = arccos((v₁ · v₂) / (||v₁|| × ||v₂||))
```

**Justificación:** Producto punto mide similitud direccional. Ángulo es independiente de posición absoluta.

**Aplicación:** 6 ángulos calculados (codos, rodillas, caderas)

#### C. Features de Velocidad (Cálculo)

**Fundamento teórico:** Derivada discreta (aproximación primer orden)

**Fórmula:**
```
v_x(t) = x(t) - x(t-1)  ≈ dx/dt
||v(t)|| = √(v_x² + v_y² + v_z²)
```

**Justificación:** Con frame rate constante, `v ∝ ΔP`. Captura dinamismo del movimiento.

**Aplicación:** 32 features de velocidad (8 landmarks clave × 4 valores)

#### D. Inclinación del Tronco (Trigonometría)

**Fundamento teórico:** Proyección y ángulos

**Fórmula:**
```
trunk_vector = shoulder_mid - hip_mid
θ = arctan2(√(trunk_x² + trunk_z²), trunk_y)
```

**Justificación:** `arctan2` maneja cuadrantes correctamente. Numerador = proyección horizontal, denominador = componente vertical.

### 3. Justificación de Modelos (Teoría de ML)

#### A. Random Forest (Ensemble Learning)

**Fundamento matemático:**
```
f_RF(x) = mode{T₁(x), T₂(x), ..., T_n(x)}
```

**Principio:** Bagging reduce varianza. Cada árbol entrenado con muestra bootstrap.

**Impureza Gini:**
```
G = 1 - Σ p_k²
```

**Complejidad:**
- Entrenamiento: `O(n_trees × m × n × log(n))`
- Predicción: `O(n_trees × depth)`

**Justificación para nuestro caso:** Con dataset pequeño (2,625 samples), RF es robusto porque promedia múltiples árboles independientes, reduciendo varianza sin overfitting.

#### B. SVM (Teoría de Optimización)

**Fundamento matemático:** Maximización de margen

**Kernel RBF:**
```
K(x, x') = exp(-γ ||x - x'||²)
```

**Principio:** Kernel trick proyecta datos a espacio de alta dimensión donde son linealmente separables.

**Multiclase:** Estrategia one-vs-one (10 clasificadores binarios para 5 clases)

**Justificación:** Efectivo cuando `#features (175) >> #samples (2,625)`.

#### C. XGBoost (Gradient Boosting)

**Fundamento matemático:** Optimización iterativa

**Fórmula:**
```
F(x) = Σ f_t(x)
f_t = argmin Σ L(yᵢ, F_{t-1}(xᵢ) + f_t(xᵢ)) + Ω(f_t)
```

**Principio:** Boosting reduce sesgo. Cada árbol corrige errores del anterior.

**Gradiente:**
```
g_i = ∂L/∂F_{t-1}(x_i)
h_i = ∂²L/∂F_{t-1}(x_i)²
```

**Justificación:** Regularización `Ω(f_t)` previene overfitting mejor que Gradient Boosting clásico.

### 4. Métricas de Evaluación (Estadística)

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN) = 76.6%
```

#### F1-Score (Media armónica)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall) = 74.5%
```

#### Intervalo de Confianza (95%)
```
σ = √(p(1-p)/n) = √(0.766 × 0.234 / 1241) = 0.012
IC 95% = 76.6% ± 1.96 × 0.012 = [74.2%, 79.0%]
```

**Interpretación:** Con 95% de confianza, el accuracy real del modelo está entre 74% y 79%.

### 5. Estrategia de Validación (Teoría de Conjuntos)

**Problema identificado:** Data leakage

**Solución matemática:** Split por video garantiza conjuntos disjuntos:
```
V_train ∩ V_test = ∅
```

**Justificación:** Frames consecutivos son casi idénticos (diferencia < 33ms). Split por frame violaría la independencia del test set. Split por video garantiza validación honesta con videos completamente nuevos.

**Resultado:** Test accuracy (76.6%) es realista y reproducible, no inflado por data leakage.

---

## Resumen de Evidencias por Competencia

| Competencia | Indicador | Evidencia Principal | Ubicación en Informe |
|-------------|-----------|---------------------|---------------------|
| **Ética** | PI1 | Tabla de decisiones éticas, medidas adoptadas, reflexión | Sección 3 (págs. 3-7) |
| **Impactos** | PI2 | Matriz 4D de impactos con mitigaciones, evaluación riesgos | Sección 4 (págs. 8-12) |
| **Matemática** | PI3 | Formulación formal, justificación features, teoría modelos | Sección 5 (págs. 12-35) |
