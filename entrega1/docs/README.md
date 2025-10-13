# Entrega 1 - Comprensión del Problema y los Datos

## Metodología: CRISP-DM

Para este proyecto decidimos usar CRISP-DM (Cross-Industry Standard Process for Data Mining) porque es la metodología más utilizada en proyectos de analítica de datos y nos permite trabajar de forma estructurada sin perder flexibilidad.

La metodología tiene 6 fases, pero para esta primera entrega nos enfocamos en las dos primeras.

---

## Fases del Proyecto

### Fase 1: Comprensión del Problema ✓
**¿Qué hacemos aquí?**
- Definir claramente qué problema vamos a resolver
- Traducir el objetivo general a un problema técnico de ML
- Establecer preguntas de investigación
- Definir métricas para medir el éxito
- Analizar aspectos éticos

**Estado:** Completado
**Documento:** [1_comprension_problema.md](./1_comprension_problema.md)

---

### Fase 2: Comprensión de los Datos ✓
**¿Qué hacemos aquí?**
- Recolectar un dataset inicial de videos
- Explorar los datos para entender su estructura
- Identificar problemas de calidad
- Extraer primeros insights
- Definir estrategia para conseguir más datos

**Estado:** Completado
**Documento:** [2_comprension_datos.md](./2_comprension_datos.md)

---

### Fase 3: Preparación de Datos (Entrega 2)
**¿Qué haremos?**
- Etiquetar/anotar los videos con las actividades
- Extraer landmarks con MediaPipe
- Limpiar datos (filtrar ruido)
- Normalizar coordenadas
- Feature engineering (calcular velocidades, ángulos, etc.)

**Estado:** Pendiente

---

### Fase 4: Modelado (Entrega 2)
**¿Qué haremos?**
- Seleccionar modelos (SVM, Random Forest, XGBoost)
- Entrenar los modelos
- Ajustar hiperparámetros
- Comparar resultados
- Iterar si es necesario

**Estado:** Pendiente

---

### Fase 5: Evaluación (Entrega 3)
**¿Qué haremos?**
- Evaluar modelos con métricas definidas
- Comparar con criterios de éxito
- Analizar errores
- Validar que responde a las preguntas de interés

**Estado:** Pendiente

---

### Fase 6: Despliegue (Entrega 3)
**¿Qué haremos?**
- Crear interfaz gráfica para visualización
- Implementar sistema en tiempo real
- Documentar todo
- Hacer video demo
- Entregar reporte final

**Estado:** Pendiente

---

## Documentos de esta Entrega

1. **[Comprensión del Problema](./1_comprension_problema.md)**
   - Definición del problema
   - Preguntas de investigación
   - Métricas de éxito
   - Análisis ético

2. **[Comprensión de los Datos](./2_comprension_datos.md)**
   - Dataset inicial
   - Análisis exploratorio
   - Protocolo de recolección
   - Estrategia para ampliar datos

---

## Estructura de Código

```
entrega1/
├── src/
│   └── data/
│       └── extract_landmarks.py    # Script para extraer landmarks con MediaPipe
├── notebooks/
│   └── exploratory_analysis.ipynb  # Análisis exploratorio de datos
└── data/
    ├── raw/                         # Videos originales
    └── processed/                   # Landmarks extraídos
```

---

## Siguiente Pasos

Para la Entrega 2 necesitamos:
1. Ampliar el dataset según la estrategia definida
2. Anotar todos los videos con las actividades
3. Implementar pipeline completo de preprocesamiento
4. Entrenar los primeros modelos
5. Obtener métricas iniciales
