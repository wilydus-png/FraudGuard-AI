
# 💳 FraudGuard AI

## Sistema inteligente para la detección y análisis de fraude en transacciones con tarjetas de débito

---

## 🎓 Información académica

- **Universidad:** Universidad Yachay Tech
- **Programa:** Maestría en Ciencia de Datos
- **Asignatura:** Fundamentos de Inteligencia Artificial
- **Docente:** Manuel Eugenio Morocho Cayamcela, Ph.D.
- **Actividad:** Actividad 12 – Implementación y evaluación de un asistente inteligente
- **Proyecto:** FraudGuard AI
- **Año:** 2026

---

## 🌐 Acceso a la aplicación

La aplicación se encuentra publicada en Streamlit Community Cloud y puede ser utilizada directamente desde un navegador web.

### [🚀 Abrir FraudGuard AI](https://fraudguard-ai-yachay.streamlit.app)

### [📂 Repositorio del proyecto](https://github.com/wilydus-png/FraudGuard-AI)

---

## 📌 Descripción del proyecto

FraudGuard AI es una aplicación interactiva desarrollada para analizar transacciones con tarjetas de débito e identificar patrones asociados al fraude financiero mediante herramientas de ciencia de datos e inteligencia artificial.

El sistema integra un dashboard interactivo, indicadores estadísticos y un asistente inteligente basado en modelos de lenguaje, permitiendo explorar los resultados del análisis mediante consultas en lenguaje natural.

La aplicación constituye una primera etapa de un proyecto más amplio orientado a la incorporación de modelos de Machine Learning para la detección de transacciones potencialmente fraudulentas.

---

## 🎯 Objetivo general

Desarrollar un sistema inteligente que permita analizar patrones de fraude en transacciones con tarjetas de débito mediante técnicas de ciencia de datos, visualización interactiva e inteligencia artificial generativa.

### Objetivos específicos

- Procesar e integrar información transaccional de un conjunto de datos público.
- Identificar modalidades, categorías comerciales y períodos con mayores tasas de fraude.
- Construir un dashboard interactivo para visualizar indicadores financieros.
- Implementar un asistente inteligente capaz de responder preguntas sobre los indicadores analizados.
- Evaluar comparativamente dos modelos de lenguaje considerando precisión, relevancia, coherencia y eficiencia computacional.

---

## 📊 Conjunto de datos

El proyecto utiliza el conjunto de datos público **Financial Transactions Dataset: Analytics**, disponible en Kaggle.

**Fuente:**

https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets

Para el análisis se seleccionaron exclusivamente transacciones correspondientes a tarjetas de débito, excluyendo tarjetas de crédito y tarjetas prepagadas.

### Indicadores generales

| Indicador | Resultado |
|-----------|----------:|
| Transacciones analizadas | 5.548.196 |
| Transacciones fraudulentas | 7.464 |
| Transacciones legítimas | 5.540.732 |
| Tasa general de fraude | 0,13453 % |
| Monto transaccional neto | $218.308.270,15 |
| Monto fraudulento neto | $839.885,36 |
| Indicadores agregados | 255 |

Los indicadores se encuentran organizados en cinco dimensiones:

- General.
- Modalidad transaccional.
- Categoría comercial MCC.
- Hora del día.
- Mes.

---

## 📈 Funcionalidades de la aplicación

### 1. Dashboard interactivo

Permite visualizar:

- Total de transacciones analizadas.
- Número de transacciones fraudulentas y legítimas.
- Tasa general de fraude.
- Monto transaccional neto.
- Monto fraudulento neto.
- Tasa de fraude por modalidad transaccional.
- Top 10 de categorías comerciales MCC con mayor tasa de fraude.
- Comportamiento del fraude por hora.
- Evolución mensual del fraude.

Las visualizaciones se desarrollaron mediante Plotly.

### 2. Exploración de datos

La aplicación permite:

- Consultar los indicadores agregados.
- Filtrar los datos por dimensión.
- Descargar los indicadores en formato CSV.
- Cargar archivos CSV adicionales que cumplan con la estructura requerida.

### 3. Asistente inteligente

El asistente permite realizar consultas en lenguaje natural sobre los indicadores disponibles.

Entre sus funcionalidades se incluyen:

- Interpretación de indicadores financieros.
- Comparación de modalidades transaccionales.
- Análisis de categorías comerciales MCC.
- Identificación de patrones temporales.
- Generación de recomendaciones de monitoreo.
- Reconocimiento de preguntas fuera del contexto del dataset.

---

## 🤖 Modelos de inteligencia artificial

Para la implementación del asistente inteligente se utilizaron dos modelos de lenguaje de pesos abiertos desarrollados por OpenAI y ejecutados mediante la infraestructura de Groq.

| Modelo | Identificador |
|--------|---------------|
| GPT-OSS-20B | `openai/gpt-oss-20b` |
| GPT-OSS-120B | `openai/gpt-oss-120b` |

### Proveedor de inferencia

**Groq Cloud**

https://console.groq.com/

### Modelos disponibles en Hugging Face

- [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)
- [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)

Inicialmente se configuró la API oficial de OpenAI. Sin embargo, debido a las restricciones de cuota de la cuenta utilizada, se implementó una alternativa mediante Groq.

La solución utiliza la biblioteca OpenAI de Python con una interfaz compatible con Groq, permitiendo ejecutar consultas dentro de los límites del servicio gratuito.

---

## 🧪 Actividad 12 – Evaluación comparativa de modelos

Para evaluar el desempeño del asistente inteligente se diseñó un experimento compuesto por ocho preguntas, ejecutadas con cada uno de los dos modelos.

Se realizaron un total de **16 evaluaciones experimentales**.

### Categorías evaluadas

1. Indicadores generales.
2. Modalidad transaccional.
3. Categorías comerciales MCC.
4. Análisis temporal.
5. Recomendaciones de monitoreo.
6. Preguntas fuera del contexto.
7. Información sobre fraude en Ecuador no disponible en los indicadores.
8. Información sobre Visa Risk Manager no disponible en los indicadores.

### Metodología

Ambos modelos recibieron las mismas preguntas, instrucciones y dimensiones de información.

Se mantuvieron constantes los siguientes parámetros:

- Temperatura: 0,2.
- Esfuerzo de razonamiento: bajo.
- Límite máximo de respuesta: 2.048 tokens.

La evaluación consideró:

- Precisión.
- Relevancia.
- Coherencia.
- Tiempo de respuesta.
- Consumo de tokens.
- Capacidad para reconocer información no disponible.

---

## 📋 Resultados de la evaluación

### Evaluación cuantitativa

| Indicador | GPT-OSS-20B | GPT-OSS-120B |
|-----------|------------:|-------------:|
| Consultas completadas | 8 | 8 |
| Errores de ejecución | 0 | 0 |
| Ejecución exitosa | 100 % | 100 % |
| Tiempo promedio de respuesta | 0,7525 s | 1,1842 s |
| Tokens promedio | 1.559,25 | 1.647,75 |
| Tokens totales | 12.474 | 13.182 |

### Evaluación cualitativa

Se aplicó una rúbrica de evaluación de 10 puntos por respuesta.

| Criterio | Puntaje máximo |
|----------|---------------:|
| Precisión | 4 |
| Relevancia | 3 |
| Coherencia | 3 |
| Total | 10 |

### Resultados consolidados

| Criterio | GPT-OSS-20B | GPT-OSS-120B |
|----------|------------:|-------------:|
| Precisión | 26/32 | 29/32 |
| Relevancia | 24/24 | 24/24 |
| Coherencia | 23/24 | 24/24 |
| Puntaje total | 73/80 | 77/80 |
| Promedio | 9,125/10 | 9,625/10 |
| Porcentaje del puntaje máximo | 91,25 % | 96,25 % |

### Principales hallazgos

- Ambos modelos completaron exitosamente todas las consultas.
- GPT-OSS-20B presentó menores tiempos de respuesta y consumo de tokens.
- GPT-OSS-120B obtuvo una mayor puntuación cualitativa.
- Ambos modelos interpretaron correctamente los indicadores generales y temporales.
- Ambos reconocieron las limitaciones del dataset en las preguntas de control.
- Se identificaron errores de filtrado, ordenamiento y cálculo de porcentajes, especialmente en consultas relacionadas con categorías comerciales MCC.

Los resultados corresponden exclusivamente a las preguntas evaluadas y no representan una medición general del rendimiento de los modelos.

---

## 🔍 Principales hallazgos del análisis de fraude

### Fraude por modalidad

Las transacciones Online presentan la mayor tasa de fraude del conjunto de datos analizado.

Este resultado evidencia la importancia de fortalecer el análisis y monitoreo de las transacciones no presenciales.

### Categorías comerciales con mayor tasa de fraude

Para categorías con al menos 1.000 transacciones:

| MCC | Tasa de fraude |
|-----|---------------:|
| 5045 | 11,2512 % |
| 5732 | 8,8544 % |
| 5712 | 7,4186 % |
| 5816 | 6,8657 % |
| 5094 | 5,8882 % |

### Comportamiento temporal

Las dos horas con mayor tasa de fraude fueron:

| Hora | Tasa de fraude |
|------|---------------:|
| 11:00 | 0,234318 % |
| 10:00 | 0,233861 % |

Estos horarios corresponden a los registros del dataset y no deben interpretarse directamente como horarios de mayor riesgo en Ecuador.

---

## 🛠️ Tecnologías utilizadas

| Tecnología | Aplicación |
|------------|------------|
| Python | Desarrollo del proyecto |
| Pandas | Procesamiento y análisis de datos |
| NumPy | Operaciones numéricas |
| DuckDB | Procesamiento de grandes volúmenes de datos |
| Plotly | Visualizaciones interactivas |
| Streamlit | Desarrollo de la aplicación web |
| Groq API | Inferencia de modelos de lenguaje |
| GPT-OSS-20B | Asistente inteligente |
| GPT-OSS-120B | Modelo comparativo |
| GitHub | Control de versiones |
| Streamlit Community Cloud | Despliegue de la aplicación |

---

## 📂 Estructura del repositorio

```text
FraudGuard-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── procesados/
    └── indicadores_fraudguard.csv
```

### Descripción de los archivos

- **app.py:** código principal del dashboard y asistente inteligente.
- **requirements.txt:** dependencias necesarias para ejecutar la aplicación.
- **README.md:** documentación general del proyecto.
- **indicadores_fraudguard.csv:** indicadores agregados utilizados por la aplicación.

Los archivos originales de transacciones y las bases de procesamiento no se incluyen en este repositorio debido a su tamaño.

---

## ⚙️ Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/wilydus-png/FraudGuard-AI.git
```

### 2. Ingresar a la carpeta del proyecto

```bash
cd FraudGuard-AI
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar la API Key de Groq

Crear la carpeta:

```text
.streamlit
```

Dentro de esta carpeta, crear el archivo:

```text
secrets.toml
```

Agregar el siguiente contenido:

```toml
GROQ_API_KEY = "TU_CLAVE_PRIVADA_DE_GROQ"
```

La clave debe obtenerse desde:

https://console.groq.com/keys

**Importante:** no publicar el archivo `secrets.toml` ni compartir la clave privada.

### 5. Ejecutar la aplicación

```bash
python -m streamlit run app.py
```

La aplicación estará disponible normalmente en:

```text
http://localhost:8501
```

---

## ☁️ Despliegue en la nube

La aplicación se encuentra desplegada en Streamlit Community Cloud.

El proceso de publicación se realizó mediante:

1. Creación del repositorio en GitHub.
2. Publicación del código fuente y los indicadores agregados.
3. Configuración de las dependencias.
4. Configuración segura de la API Key de Groq.
5. Despliegue de la aplicación.

**Aplicación publicada:**

https://fraudguard-ai-yachay.streamlit.app

---

## 📝 Conclusiones

FraudGuard AI demuestra la integración de técnicas de ciencia de datos e inteligencia artificial generativa para facilitar el análisis de fraude en transacciones con tarjetas de débito.

La comparación experimental permitió identificar que GPT-OSS-20B presenta una mayor eficiencia en tiempo de respuesta y consumo de tokens, mientras que GPT-OSS-120B obtuvo una mejor puntuación cualitativa en las consultas evaluadas.

Sin embargo, ambos modelos presentaron limitaciones relacionadas con operaciones de filtrado, ordenamiento y cálculos porcentuales.

Por esta razón, se propone complementar los modelos de lenguaje con operaciones estadísticas determinísticas realizadas mediante Python, Pandas o DuckDB, permitiendo que la inteligencia artificial generativa se concentre principalmente en interpretar y explicar resultados previamente calculados.

---

## 🚀 Trabajos futuros

Como parte de las siguientes etapas del proyecto se propone:

- Implementar modelos de Machine Learning para la detección de fraude.
- Evaluar modelos de regresión logística y redes neuronales.
- Incorporar métricas adecuadas para conjuntos de datos desbalanceados.
- Integrar predicciones de riesgo transaccional en la aplicación.
- Mejorar la precisión del chatbot mediante cálculos determinísticos.
- Implementar mecanismos de validación automática de respuestas.
- Desplegar y evaluar la aplicación en AWS.
- Incorporar controles de acceso y límites de consultas para proteger los recursos de inferencia.

---

## ⚠️ Consideraciones y limitaciones

Este proyecto tiene fines exclusivamente académicos.

Los resultados corresponden a un conjunto de datos público y no representan estadísticas oficiales del sistema financiero ecuatoriano ni pérdidas financieras confirmadas.

Los modelos de lenguaje pueden generar respuestas incorrectas, por lo que sus resultados deben contrastarse con los indicadores originales.

La evaluación experimental se realizó con ocho preguntas por modelo y una única ejecución de cada pregunta.

Actualmente, la aplicación permite analizar e interpretar indicadores históricos. El componente de predicción de fraude mediante Machine Learning se encuentra contemplado como una etapa posterior del proyecto.

---

## 🎓 Créditos académicos

Proyecto desarrollado en el marco de la asignatura Fundamentos de Inteligencia Artificial de la Maestría en Ciencia de Datos de la Universidad Yachay Tech.

**FraudGuard AI — Ciencia de datos e inteligencia artificial aplicadas al análisis de fraude financiero.**

