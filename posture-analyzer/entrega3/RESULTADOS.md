# Resultados de Feature Reduction - Entrega 3

## Resumen Ejecutivo

En esta entrega realizamos análisis de reducción de características para mejorar el modelo de clasificación de actividades humanas. Los resultados muestran que **reducir de 172 a 75 features con PCA mejora significativamente el performance**.

---

## 1. Análisis de Feature Importance

### Top 20 Features Más Importantes

Las features más relevantes según Random Forest son:

1. **right_foot_index_x** (2.88%)
2. **left_knee_x** (2.68%)
3. **right_elbow_visibility** (2.59%)
4. **left_heel_y** (2.28%)
5. **left_foot_index_x** (2.21%)

**Hallazgos:**
- Las posiciones de **pies, rodillas y tobillos** son las más importantes
- Las coordenadas **X (horizontales)** e **Y (verticales)** son más relevantes que Z
- La **visibility** de articulaciones también es informativa

### Análisis de Importancia Acumulada

- **102 features** capturan el **90%** de la importancia total
- **121 features** capturan el **95%** de la importancia total
- Total actual: **172 features**

**Conclusión:** Se puede reducir ~40% de features (de 172 a 102) manteniendo el 90% de la información.

---

## 2. Reducción con PCA

### Resultados Comparativos

| Método | # Features | Accuracy | F1-Score | Varianza Explicada |
|--------|-----------|----------|----------|-------------------|
| **Original** | 172 | 76.6% | 74.5% | - |
| PCA-50 | 50 | 69.1% | 68.4% | 99.9% |
| **PCA-75**  | **75** | **81.3%** | **81.2%** | **99.99%** |
| PCA-100 | 100 | 72.9% | 72.1% | 100.0% |

### Hallazgo Principal

**Reducir de 172 a 75 features con PCA mejora el modelo:**

- **Accuracy:** 76.6% → 81.3% **(+4.7%)**
- **F1-Score:** 74.5% → 81.2% **(+6.7%)**
- **Reducción de dimensionalidad:** 56% menos features

### ¿Por Qué Mejoró?

1. **Eliminación de ruido:** PCA elimina variaciones irrelevantes en los datos
2. **Decorrelación:** PCA crea componentes independientes, reduciendo multicolinealidad
3. **Regularización implícita:** Menos features = menor riesgo de overfitting
4. **Mejor generalización:** El modelo se enfoca en patrones principales

---

## 3. Comparación con Objetivo

### Meta del Proyecto

- **Objetivo inicial:** F1-Score ≥ 85%
- **Resultado Entrega 2:** 74.5% F1-Score
- **Resultado Entrega 3:** **81.2% F1-Score** (PCA-75)

### Progreso

- **Brecha inicial:** 10.5% (de 74.5% a 85%)
- **Brecha actual:** 3.8% (de 81.2% a 85%)
- **Mejora lograda:** 6.7 puntos porcentuales

**Estamos mucho más cerca del objetivo.**

---

## 4. Rendimiento por Clase (PCA-75)

Comparación con modelo original:

| Clase | F1 Original | F1 PCA-75 | Cambio |
|-------|-------------|-----------|--------|
| caminarEspalda | 93.2% | - | - |
| caminarFrente | 87.2% | - | - |
| girar | 63.1% | - | - |
| levantarse | 44.0% | - | - |
| sentarse | 76.4% | - | - |

*(Nota: Análisis detallado por clase pendiente de generar con script adicional)*

---

## 5. Conclusiones

### Lecciones Aprendidas

1. **Más features ≠ mejor modelo:** 172 features tenían redundancia y ruido
2. **PCA es efectivo:** Reducción de dimensionalidad mejora generalización
3. **El sweet spot es 75 componentes:** Ni muy pocos (50) ni muchos (100)

### Implicaciones Prácticas

**Ventajas de usar PCA-75:**
-  Mejor accuracy (+4.7%)
-  Mejor F1-Score (+6.7%)
-  Inferencia más rápida (56% menos features)
-  Menor riesgo de overfitting
-  Menor uso de memoria

**Desventaja:**
-  Menor interpretabilidad (componentes PCA vs features originales)

---

## 6. Recomendaciones Finales

### Para Mejorar Aún Más (llegar a 85%)

1. **Aumentar el dataset:**
   - Grabar más videos de "levantarse" (clase problemática)
   - Incluir más personas (actualmente solo 1)

2. **Data augmentation:**
   - Flip horizontal
   - Variaciones de velocidad

3. **Ensemble methods:**
   - Combinar Random Forest + XGBoost
   - Voting classifier

### Modelo Recomendado para Despliegue

**Random Forest con PCA-75 componentes:**
- Accuracy: 81.3%
- F1-Score: 81.2%
- Features: 75 (reducción de 56%)
- Balance óptimo entre performance y eficiencia

---

## Archivos Generados

1. `feature_importances.csv` - Importancia de todas las features
2. `top20_features.png` - Gráfico de top 20 features
3. `cumulative_importance.png` - Importancia acumulada
4. `comparison_results.csv` - Comparación de métodos
5. `feature_reduction_comparison.png` - Gráficos comparativos

---

**Fecha:** Noviembre 2024
**Proyecto:** Clasificación de Actividades Humanas con MediaPipe
**Equipo:** [Tu equipo]
