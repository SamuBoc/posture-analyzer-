# Fase 1: Comprensión del Problema

Este documento cubre la primera fase de CRISP-DM donde definimos qué problema vamos a resolver, por qué es importante, y cómo vamos a medir si lo logramos.

---

## 1. Definición del Problema

Queremos desarrollar un sistema que pueda detectar y clasificar automáticamente cinco actividades humanas básicas a partir de video en tiempo real:

1. **Caminar hacia la cámara**
2. **Caminar alejándose (de regreso)**
3. **Girar**
4. **Sentarse**
5. **Ponerse de pie**

El sistema debe:
- Funcionar en tiempo real (procesar video mientras se captura)
- Analizar la postura de la persona (inclinaciones, ángulos de articulaciones)
- Alcanzar una precisión mínima del 85% (medida con F1-Score)

### ¿Qué tipo de problema es esto en ML?

Desde la perspectiva de machine learning, estamos ante un problema de **clasificación multiclase supervisada** sobre **series de tiempo multivariadas**.

- **Clasificación multiclase supervisada:** Vamos a entrenar un modelo con ejemplos etiquetados (videos donde ya sabemos qué actividad se está haciendo) para que aprenda a predecir la categoría correcta en datos nuevos.

- **Series de tiempo multivariadas:** No es un problema de clasificar una imagen estática. El input es una secuencia de frames a lo largo del tiempo, y en cada frame tenemos múltiples valores (coordenadas x, y, z de todas las articulaciones del cuerpo).

### ¿Por qué es interesante?

Este tipo de sistema tiene aplicaciones reales en:
- **Fisioterapia:** Monitorear ejercicios de rehabilitación
- **Deporte:** Analizar técnica y rendimiento
- **Ergonomía:** Evaluar posturas en ambientes laborales
- **Asistencia médica:** Detectar caídas o movimientos anormales en adultos mayores

---

## 2. Preguntas de Investigación

Estas son las preguntas clave que guiarán todo el proyecto:

### Pregunta Principal
**¿Es posible clasificar con alta precisión (>85% F1-Score) un conjunto de actividades humanas usando únicamente los datos de landmarks corporales extraídos de video?**

Esta pregunta es fundamental porque define si el enfoque es viable. Si MediaPipe puede darnos suficiente información de las articulaciones para distinguir entre actividades, entonces el proyecto tiene sentido.

### Pregunta de Modelado
**¿Cuál de los modelos de clasificación supervisada (SVM, Random Forest, XGBoost) funciona mejor para detectar actividades humanas en términos de precisión, recall y F1-Score?**

Queremos saber no solo cuál modelo da mejores resultados, sino también entender por qué. ¿Es mejor uno más simple como Random Forest o necesitamos algo más complejo como XGBoost?

### Pregunta de Features
**¿Cuáles son las características más importantes para distinguir entre las actividades? ¿Son más útiles los ángulos de articulaciones, las velocidades, las posiciones relativas, o la inclinación del tronco?**

Esto nos ayudará a entender qué aspectos del movimiento humano son más informativos. Además, nos permitirá reducir dimensionalidad si hay features que no aportan mucho.

---

## 3. Objetivos del Proyecto

### Objetivo General
Desarrollar e implementar un sistema de clasificación de actividades humanas que funcione en tiempo real, con una precisión mínima del 85% (F1-Score), y que además proporcione métricas posturales útiles.

### Objetivos Específicos

1. **Recolección de Datos**
   - Capturar un dataset de videos con al menos 3-5 personas diferentes
   - Incluir variaciones en velocidad, ángulos de cámara y condiciones de iluminación
   - Asegurar balance entre las 5 clases de actividades

2. **Preprocesamiento**
   - Implementar pipeline de extracción de landmarks con MediaPipe
   - Normalizar coordenadas para que sean independientes de la distancia a la cámara
   - Aplicar filtros para reducir ruido en el tracking
   - Generar features derivadas (velocidades, ángulos, inclinaciones)

3. **Modelado**
   - Entrenar y comparar al menos 3 modelos diferentes (SVM, Random Forest, XGBoost)
   - Optimizar hiperparámetros de cada modelo
   - Implementar validación cruzada

4. **Evaluación**
   - Alcanzar F1-Score ≥ 0.85 en el conjunto de test
   - Identificar en qué actividades el modelo tiene más dificultades
   - Medir latencia de inferencia (debe ser capaz de procesar en tiempo real)

5. **Despliegue**
   - Crear interfaz gráfica simple para visualización
   - Mostrar actividad detectada + métricas posturales en tiempo real
   - Documentar limitaciones y casos de uso recomendados

---

## 4. Métricas de Éxito

Para saber si el proyecto fue exitoso, vamos a usar estas métricas:

### Métricas Principales (Clasificación)

**F1-Score** (Métrica principal)
- Es el promedio armónico entre precisión y recall
- Útil cuando hay desbalance de clases
- **Criterio de éxito:** F1-Score ≥ 0.85

**Precisión (Precision)**
- De todas las veces que el modelo predijo una actividad, ¿cuántas acertó?
- Importante para evitar falsos positivos

**Exhaustividad (Recall)**
- De todas las veces que ocurrió una actividad, ¿cuántas veces la detectó?
- Importante para no perder eventos

**Matriz de Confusión**
- Nos muestra en qué pares de actividades el modelo se confunde más
- Ej: ¿confunde "sentarse" con "ponerse de pie"?

### Métricas Técnicas

**Latencia de Inferencia**
- Tiempo que tarda en procesar un frame y dar resultado
- Medida en FPS (frames por segundo)
- **Objetivo:** Mantener al menos 15-20 FPS para que se sienta fluido

**Tiempo de Entrenamiento**
- Importante para iterar rápido durante desarrollo
- No es crítico, pero preferimos modelos que no tarden horas

---

## 5. Análisis de Aspectos Éticos

Aunque es un proyecto académico, es importante considerar las implicaciones éticas de crear sistemas que analizan personas.

### 5.1 Privacidad y Consentimiento

**El problema:**
- Estamos procesando imágenes/videos de personas
- Hay información sensible sobre cómo se mueve alguien, su cuerpo, etc.
- Podría usarse para identificar o monitorear personas sin su conocimiento

**Qué vamos a hacer:**
- **Solo usamos videos de nosotros mismos** (los integrantes del equipo)
- Todos los que aparecen en los videos dieron consentimiento explícito
- No compartimos videos públicamente
- Si eventualmente usamos videos de terceros, pediremos permiso por escrito
- Los datos se mantienen en local, no se suben a ningún servicio en la nube

**Referencia:** El trabajo de Lee et al. (2023) sobre marcos éticos en investigación con participantes humanos enfatiza la importancia del consentimiento informado y la transparencia [1].

### 5.2 Sesgos y Equidad

**El problema:**
- Si solo entrenamos con un tipo de persona (ej: solo hombres jóvenes), el modelo podría no funcionar bien para otros grupos
- Hay sesgos históricos en datasets de visión por computadora
- Diferentes tipos de cuerpo, edades, géneros, etnias se mueven diferente

**Qué vamos a hacer:**
- Aunque nuestro dataset es pequeño y del equipo, intentamos incluir variedad en:
  - Ropa (no siempre la misma)
  - Velocidad de movimientos
  - Formas de realizar cada actividad
- **Documentamos claramente las limitaciones:** en el reporte final vamos a decir explícitamente que el modelo se entrenó con un grupo pequeño y homogéneo, y probablemente no generaliza bien
- Si el modelo falla con ciertos grupos, lo reportamos honestamente

**Referencia:** Sharma y Singh (2023) discuten consideraciones éticas en visión por computadora, incluyendo la importancia de datasets diversos y la documentación de limitaciones [2].

### 5.3 Uso Indebido

**El problema:**
- Esta tecnología podría usarse para vigilancia masiva
- Podría usarse para monitorear empleados sin su consentimiento
- Podría usarse para discriminar (ej: en procesos de contratación basados en cómo camina alguien)

**Qué vamos a hacer:**
- Dejamos claro que esto es un proyecto educativo
- En la documentación incluiremos una sección de "Usos NO recomendados"
- Discutiremos abiertamente las limitaciones y riesgos
- No promocionamos esto como una herramienta de vigilancia

**Nuestra postura:**
Creemos que la tecnología puede usarse para bien (fisioterapia, deporte) pero reconocemos que también puede usarse mal. La responsabilidad es documentar ambos lados y fomentar uso ético.

### 5.4 Transparencia

**Qué vamos a hacer:**
- Código abierto (en GitHub)
- Documentación completa de cómo funciona
- Explicación clara de qué puede y qué NO puede hacer el sistema
- Reportar métricas de error, no solo de éxito

---

## 6. Restricciones y Limitaciones

### Restricciones Técnicas
- **Tiempo:** 5 semanas de desarrollo (3 entregas)
- **Hardware:** Solo laptops (sin GPUs potentes)
- **Datos:** Dataset pequeño (solo el equipo aparece en videos)

### Limitaciones Esperadas
- El modelo probablemente no generaliza bien a personas muy diferentes del equipo
- Funciona mejor en condiciones de iluminación similares a las de entrenamiento
- Necesita que la persona esté de frente o de perfil a la cámara
- No funciona bien con ropa muy holgada que oculte el cuerpo
- No detecta actividades parciales (ej: solo medio sentarse)

### Simplificaciones
- Solo 5 actividades (la realidad tiene muchas más)
- Asumimos una persona a la vez (no múltiples personas)
- No consideramos contexto (ej: si hay una silla cerca para sentarse)

---

## 7. Siguientes Pasos

Con el problema bien definido, ahora vamos a:

1. **Definir protocolo de recolección de datos** (Fase 2)
   - ¿Cuántos videos por actividad?
   - ¿Qué variaciones incluir?
   - ¿Desde qué ángulos grabar?

2. **Recolectar dataset inicial** (Fase 2)
   - Grabar primeros videos
   - Probar extracción de landmarks
   - Identificar problemas

3. **Análisis exploratorio** (Fase 2)
   - Visualizar landmarks
   - Calcular estadísticas básicas
   - Entender variabilidad

4. **Estrategia para más datos** (Fase 2)
   - ¿Necesitamos más videos?
   - ¿Podemos hacer augmentation?

---

## Referencias

[1] M. K. Lee, J. T. Biega, A. L. Cunliffe, D. Williams, D. Schmit, y T. K. Lee, "A Contextual Ethics Framework for Human Participant AI Research," *arXiv preprint arXiv:2311.01254*, 2023.

[2] S. Sharma y S. Singh, "Ethical Considerations in Artificial Intelligence: A Comprehensive Discussion from the Perspective of Computer Vision," en *2023 3rd International Conference on Advance Computing and Innovative Technologies in Engineering (ICACITE)*, 2023, pp. 1812–1817.
