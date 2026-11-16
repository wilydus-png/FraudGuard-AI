
# 💳 FraudGuard AI

## Sistema inteligente para la detección y análisis de fraude en transacciones con tarjetas de débito mediante Machine Learning e inteligencia artificial generativa

**Universidad Yachay Tech**  
Maestría en Ciencia de Datos  
Asignatura: Fundamentos de Inteligencia Artificial  
Docente: Manuel Eugenio Morocho Cayamcela, Ph.D.

---

## 🚀 Aplicación en línea

Accede a FraudGuard AI:

https://fraudguard-ai-yachay.streamlit.app

La aplicación está desplegada en Streamlit Community Cloud y permite explorar indicadores de fraude, realizar consultas mediante inteligencia artificial generativa y ejecutar predicciones utilizando un modelo de Machine Learning entrenado.

---

## 1. Descripción del proyecto

FraudGuard AI es un prototipo académico orientado al análisis y detección de operaciones potencialmente fraudulentas realizadas con tarjetas de débito.

El sistema integra análisis de datos, aprendizaje automático e inteligencia artificial generativa para identificar patrones transaccionales, generar puntuaciones de riesgo y apoyar la priorización de alertas de fraude.

El proyecto busca demostrar cómo la incorporación de características históricas y de comportamiento puede mejorar la capacidad de detección frente a modelos predictivos tradicionales.

---

## 2. Objetivo general

Desarrollar un sistema inteligente para el análisis y la detección de fraude en transacciones con tarjetas de débito, mediante técnicas de Machine Learning, ingeniería de características e inteligencia artificial generativa, que permita identificar patrones de riesgo y priorizar operaciones para su revisión.

### Objetivos específicos

- Analizar el comportamiento de las transacciones fraudulentas y legítimas.
- Construir características predictoras relacionadas con el historial transaccional de las tarjetas y los clientes.
- Entrenar y comparar modelos de clasificación supervisada.
- Evaluar el desempeño predictivo y la eficiencia operativa de los modelos.
- Integrar el modelo seleccionado en una aplicación web interactiva.
- Implementar un asistente inteligente para consultar indicadores de fraude mediante lenguaje natural.

---

## 3. Fuente de datos

Se utiliza el dataset público:

**Financial Transactions Dataset: Analytics**

Disponible en Kaggle:

https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets

El conjunto de datos contiene información histórica sobre transacciones financieras, tarjetas, clientes, establecimientos comerciales y etiquetas de fraude.

Para el desarrollo del proyecto se seleccionaron exclusivamente transacciones con tarjetas de débito, excluyendo tarjetas de crédito y débito prepagadas.

### Resumen del conjunto de datos

| Indicador | Resultado |
|---|---:|
| Historial completo de débito | 8.280.996 |
| Transacciones etiquetadas | 5.548.196 |
| Transacciones sin etiqueta | 2.732.800 |
| Transacciones fraudulentas etiquetadas | 7.464 |
| Transacciones legítimas etiquetadas | 5.540.732 |
| Tasa general de fraude | 0,1345 % |

Las transacciones sin etiqueta se utilizaron para ampliar el historial de comportamiento, sin considerarlas automáticamente como legítimas.

Los datos son de naturaleza sintética y no representan estadísticas oficiales de fraude de Ecuador.

---

## 4. Metodología

El proyecto se desarrolló mediante las siguientes etapas:

1. Carga, integración y limpieza de datos.
2. Análisis exploratorio de transacciones.
3. Identificación de patrones de fraude.
4. División cronológica de los datos.
5. Ingeniería de características históricas.
6. Entrenamiento de modelos de clasificación.
7. Optimización de hiperparámetros y ponderaciones.
8. Evaluación de métricas predictivas y operativas.
9. Selección del modelo y del umbral operativo.
10. Integración y despliegue en Streamlit.

### División cronológica

| Conjunto | Transacciones | Fraudes |
|---|---:|---:|
| Entrenamiento | 3.884.576 | 5.803 |
| Validación | 832.889 | 464 |
| Prueba | 830.731 | 1.197 |
| Total | 5.548.196 | 7.464 |

La separación cronológica permite estudiar la capacidad del modelo para identificar fraude en períodos posteriores a los utilizados durante el entrenamiento.

El preprocesamiento se ajusta sobre los datos de entrenamiento y se aplica posteriormente a validación y prueba.

---

## 5. Análisis exploratorio de datos

El análisis descriptivo permite identificar patrones asociados con:

- Modalidades transaccionales.
- Categorías comerciales MCC.
- Horarios de mayor incidencia de fraude.
- Evolución mensual del comportamiento fraudulento.
- Montos transaccionales.
- Frecuencia de operaciones.
- Uso habitual de establecimientos comerciales.
- Cambios en el comportamiento de tarjetas y clientes.

Uno de los principales hallazgos fue la existencia de cambios temporales en las modalidades de fraude, lo que evidencia la importancia de utilizar características de comportamiento y evaluaciones cronológicas.

### Tasa de fraude por modalidad

| Modalidad | Tasa de fraude |
|---|---:|
| Online | 0,7666 % |
| Chip | 0,0829 % |
| Swipe | 0,0282 % |

Estos resultados corresponden exclusivamente al dataset analizado.

---

## 6. Ingeniería de características

Se desarrolló una segunda versión del conjunto predictivo con 84 características agrupadas en:

| Grupo | Características |
|---|---:|
| Variables originales | 13 |
| Perfil histórico de tarjeta | 30 |
| Perfil comercial | 18 |
| Perfil del cliente | 10 |
| Aceleración transaccional | 5 |
| Perfil geográfico | 8 |
| Total | 84 |

Las características históricas buscan identificar desviaciones respecto al comportamiento habitual de las tarjetas y clientes.

Entre las características más relevantes se encuentran:

- Número de transacciones previas en un comercio.
- Frecuencia de operaciones en un comercio durante 90 días.
- Aceleración del monto transaccional en 24 horas.
- Frecuencia histórica de categorías MCC.
- Tiempo desde la transacción anterior.
- Relación entre el monto actual y los montos históricos.

El modelo final utiliza una selección de 69 características predictoras.

---

## 7. Modelos de Machine Learning

Se evaluaron diferentes técnicas de clasificación supervisada:

- Regresión logística base.
- Regresión logística con historial.
- Red neuronal Perceptrón Multicapa (MLP).
- XGBoost con características históricas.
- XGBoost con perfil completo.
- XGBoost con características de comportamiento.
- XGBoost con diferentes tamaños de muestra y ponderaciones.

El proceso experimental permitió identificar que la ingeniería de características históricas y comerciales produjo una mejora sustancial en la capacidad predictiva.

### Modelo seleccionado

**XGB-5 — Ponderación moderada**

Características del entrenamiento:

| Parámetro | Valor |
|---|---:|
| Algoritmo | XGBoost |
| Variables predictoras | 69 |
| Transacciones legítimas de entrenamiento | 1.000.000 |
| Transacciones fraudulentas de entrenamiento | 5.803 |
| Total de registros de entrenamiento | 1.005.803 |
| Peso de clase positiva | 13,1272 |
| Umbral operativo equilibrado | 0,413096 |

El modelo genera una puntuación relativa de riesgo para cada transacción.

Las puntuaciones no representan probabilidades calibradas de fraude.

---

## 8. Resultados del modelo predictivo

El modelo seleccionado fue evaluado sobre 830.731 transacciones correspondientes al período comprendido entre junio de 2018 y octubre de 2019.

### Métricas finales

| Indicador | Resultado |
|---|---:|
| ROC-AUC | 0,931422 |
| Average Precision | 0,096328 |
| Precision | 6,9266 % |
| Recall | 34,5865 % |
| F1-score | 0,115417 |
| Fraudes existentes | 1.197 |
| Fraudes detectados | 414 |
| Fraudes no detectados | 783 |
| Falsos positivos | 5.563 |
| Total alertas | 5.977 |
| Tasa de alertas | 0,7195 % |

### Matriz de confusión

| Clase real | Sin alerta | Revisión requerida |
|---|---:|---:|
| Legítima | 823.971 | 5.563 |
| Fraudulenta | 783 | 414 |

El modelo permite identificar más de un tercio de los fraudes revisando menos del 1 % de las operaciones, utilizando el umbral equilibrado seleccionado en validación.

---

## 9. Comparación con el modelo inicial

| Indicador | MLP inicial | XGB-5 |
|---|---:|---:|
| ROC-AUC | 0,7680 | 0,9314 |
| Precision | 0,81 % | 6,93 % |
| Recall | 14,70 % | 34,59 % |
| Fraudes detectados | 176 | 414 |
| Falsos positivos | 21.521 | 5.563 |
| Total alertas | 21.697 | 5.977 |

Principales mejoras:

- Incremento del 135,23 % en los fraudes detectados.
- Reducción del 74,15 % en los falsos positivos.
- Reducción del 72,45 % en el volumen de alertas.
- Mejora considerable en la capacidad de discriminación.

Los resultados respaldan la importancia de incorporar perfiles históricos y comerciales al desarrollo de modelos de fraude.

---

## 10. Evaluación según capacidad operativa

Se evaluó la capacidad de priorización del modelo mediante diferentes porcentajes de transacciones revisadas.

| Operaciones revisadas | Alertas | Fraudes detectados | Precision | Recall |
|---|---:|---:|---:|---:|
| 0,1 % | 831 | 166 | 19,98 % | 13,87 % |
| 0,5 % | 4.154 | 359 | 8,64 % | 29,99 % |
| 1 % | 8.307 | 481 | 5,79 % | 40,18 % |
| 2 % | 16.615 | 649 | 3,91 % | 54,22 % |
| 5 % | 41.537 | 879 | 2,12 % | 73,43 % |

Al revisar únicamente el 1 % de las transacciones con mayor puntuación de riesgo, el modelo identifica el 40,18 % de los fraudes del conjunto de prueba.

Esta evaluación demuestra la utilidad del modelo para priorizar operaciones bajo capacidades limitadas de revisión.

Los escenarios Top-K corresponden a evaluaciones retrospectivas de priorización y no a garantías de desempeño futuro.

---

## 11. Inteligencia artificial generativa

FraudGuard AI incorpora un asistente inteligente que permite realizar consultas en lenguaje natural sobre los indicadores del dataset.

Se utilizan dos modelos:

- openai/gpt-oss-20b
- openai/gpt-oss-120b

Ambos modelos son ejecutados mediante la API de Groq.

El asistente permite:

- Consultar indicadores generales de fraude.
- Comparar modalidades transaccionales.
- Identificar categorías comerciales de mayor riesgo.
- Analizar patrones horarios y temporales.
- Generar recomendaciones de monitoreo basadas en los datos disponibles.

Durante el desarrollo se realizaron 16 consultas experimentales, correspondientes a ocho preguntas evaluadas con cada modelo.

La evaluación considera precisión factual, pertinencia, coherencia y capacidad para reconocer información no disponible.

El asistente generativo no sustituye al modelo predictivo. Las puntuaciones de riesgo son calculadas directamente mediante XGBoost.

---

## 12. Funcionalidades de la aplicación

La aplicación FraudGuard AI está organizada en cuatro módulos principales.

### Dashboard

Visualización interactiva de los principales indicadores de fraude.

### Datos

Exploración, filtrado y descarga de indicadores transaccionales.

### Asistente inteligente

Consultas en lenguaje natural mediante modelos GPT-OSS ejecutados a través de Groq.

### Modelo predictivo

Evaluación de transacciones mediante el modelo XGB-5.

Permite:

- Analizar transacciones de demostración.
- Cargar archivos CSV con las características requeridas.
- Seleccionar diferentes escenarios operativos.
- Generar puntuaciones de riesgo.
- Clasificar operaciones que requieren revisión.
- Visualizar el ranking de transacciones.
- Consultar la matriz de confusión cuando existen etiquetas reales.
- Descargar los resultados predictivos.

---

## 13. Validación funcional en Streamlit Cloud

Se verificó el funcionamiento del modelo exportado mediante una muestra reproducible de 5.000 transacciones del conjunto de prueba.

| Indicador | Resultado |
|---|---:|
| Transacciones evaluadas | 5.000 |
| Fraudes existentes | 8 |
| Alertas generadas | 38 |
| Fraudes detectados | 4 |
| Falsos positivos | 34 |
| Fraudes no detectados | 4 |
| Precision | 10,53 % |
| Recall | 50,00 % |
| F1-score | 0,1739 |

Los resultados obtenidos en Streamlit Community Cloud coinciden con los resultados de la aplicación ejecutada localmente.

Esta prueba verifica la reproducibilidad funcional del modelo desplegado.

Las métricas de la muestra de demostración no sustituyen los resultados de la evaluación retrospectiva completa.

---

## 14. Tecnologías utilizadas

- Python
- Pandas
- NumPy
- DuckDB
- Scikit-learn
- XGBoost
- Joblib
- Plotly
- Streamlit
- Groq API
- GPT-OSS
- GitHub
- Streamlit Community Cloud

---

## 15. Estructura del repositorio

```text
FraudGuard-AI/
│
├── app.py
├── modulo_predictivo.py
├── requirements.txt
├── README.md
│
├── procesados/
│   └── indicadores_fraudguard.csv
│
└── modelo_predictivo/
    ├── fraudguard_xgb5.json
    ├── preprocesador_xgb5.joblib
    ├── configuracion_xgb5.json
    └── transacciones_demo.csv
```

---

## 16. Instalación y ejecución local

Clonar el repositorio:

```bash
git clone https://github.com/wilydus-png/FraudGuard-AI.git
```

Ingresar a la carpeta:

```bash
cd FraudGuard-AI
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Configurar la clave privada de Groq mediante el archivo:

```text
.streamlit/secrets.toml
```

Contenido:

```toml
GROQ_API_KEY = "TU_CLAVE_PRIVADA"
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

El módulo predictivo no requiere una clave de Groq para ejecutar el modelo XGBoost.

La clave es necesaria únicamente para utilizar el asistente generativo.

Las credenciales privadas no deben publicarse en GitHub.

---

## 17. Limitaciones

- El dataset utilizado es sintético y no representa directamente el comportamiento de fraude de una institución financiera ecuatoriana.
- El conjunto de datos presenta un elevado desbalance entre transacciones legítimas y fraudulentas.
- Existen cambios temporales en los patrones de fraude.
- Las puntuaciones de riesgo no están calibradas como probabilidades reales de fraude.
- El modelo requiere características históricas previamente calculadas.
- La evaluación final es retrospectiva y el conjunto de prueba fue examinado durante etapas anteriores del proyecto.
- No se realizó una validación externa sobre datos reales de otra institución financiera.
- El asistente generativo puede producir respuestas incorrectas y requiere verificación de sus resultados.

FraudGuard AI constituye un prototipo académico de apoyo al monitoreo transaccional.

No debe utilizarse como un sistema autónomo de autorización, rechazo o bloqueo de operaciones financieras.

---

## 18. Conclusiones

El desarrollo de FraudGuard AI permitió demostrar la utilidad de combinar análisis exploratorio, ingeniería de características, aprendizaje automático e inteligencia artificial generativa para el estudio del fraude financiero.

La incorporación de características relacionadas con el comportamiento histórico de las tarjetas, clientes y comercios permitió mejorar sustancialmente el desempeño de los modelos iniciales.

El modelo XGB-5 alcanzó un ROC-AUC de 0,9314 y permitió detectar 414 fraudes generando 5.977 alertas sobre el conjunto de prueba.

Asimismo, la evaluación por capacidad operativa mostró que es posible detectar el 40,18 % de los fraudes revisando únicamente el 1 % de las operaciones con mayor puntuación de riesgo.

La implementación en Streamlit Community Cloud demuestra la viabilidad técnica de integrar los componentes descriptivos, predictivos y generativos en una aplicación web funcional.

Como trabajos futuros se propone realizar validaciones con datos reales, mejorar la calibración de puntuaciones, incorporar mecanismos de monitoreo temporal y desarrollar estrategias de optimización basadas en los costos financieros del fraude.

---

## Proyecto académico

Universidad Yachay Tech  
Maestría en Ciencia de Datos  
Fundamentos de Inteligencia Artificial  

FraudGuard AI — 2026
