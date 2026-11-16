
# ============================================================
# FRAUDGUARD AI
# MÓDULO PREDICTIVO DE DETECCIÓN DE FRAUDE
# Universidad Yachay Tech
# Fundamentos de Inteligencia Artificial
# ============================================================


# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from pathlib import Path
from time import perf_counter
from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 2. CONFIGURACIÓN GENERAL
# ============================================================

RUTA_BASE = Path(__file__).resolve().parent

RUTA_MODELO = RUTA_BASE / "modelo_predictivo"

ARCHIVO_MODELO = (
    RUTA_MODELO / "fraudguard_xgb5.json"
)

ARCHIVO_PREPROCESADOR = (
    RUTA_MODELO / "preprocesador_xgb5.joblib"
)

ARCHIVO_CONFIGURACION = (
    RUTA_MODELO / "configuracion_xgb5.json"
)

ARCHIVO_DEMO = (
    RUTA_MODELO / "transacciones_demo.csv"
)

MAX_FILAS = 20000

MAX_ARCHIVO_MB = 10


# ============================================================
# 3. CARGA DEL MODELO PREDICTIVO
# ============================================================

@st.cache_resource
def cargar_modelo_predictivo():

    archivos_requeridos = [

        ARCHIVO_MODELO,

        ARCHIVO_PREPROCESADOR,

        ARCHIVO_CONFIGURACION
    ]

    for archivo in archivos_requeridos:

        if not archivo.exists():

            raise FileNotFoundError(
                f"No se encontró: {archivo.name}"
            )

    # --------------------------------------------------------
    # Cargar configuración.
    # --------------------------------------------------------

    with open(

        ARCHIVO_CONFIGURACION,

        "r",

        encoding="utf-8"

    ) as archivo:

        configuracion = json.load(archivo)

    # --------------------------------------------------------
    # Cargar modelo XGBoost.
    # --------------------------------------------------------

    modelo = XGBClassifier()

    modelo.load_model(
        str(ARCHIVO_MODELO)
    )

    # Limitar recursos en la nube.

    modelo.set_params(n_jobs=2)

    # --------------------------------------------------------
    # Cargar preprocesador.
    # --------------------------------------------------------

    preprocesador = joblib.load(
        ARCHIVO_PREPROCESADOR
    )

    return (
        modelo,
        preprocesador,
        configuracion
    )


# ============================================================
# 4. CARGA DE TRANSACCIONES
# ============================================================

@st.cache_data
def cargar_transacciones_demo(categoricas):

    if not ARCHIVO_DEMO.exists():

        raise FileNotFoundError(
            "No se encontró el archivo de demostración."
        )

    tipos = {

        columna: "string"

        for columna in categoricas
    }

    return pd.read_csv(

        ARCHIVO_DEMO,

        dtype=tipos
    )


# ============================================================
# 5. VALIDACIÓN Y PREPROCESAMIENTO
# ============================================================

def preparar_transacciones(df, configuracion):

    variables = configuracion[
        "variables_predictoras"
    ]

    numericas = configuracion[
        "variables_numericas"
    ]

    categoricas = configuracion[
        "variables_categoricas"
    ]

    # --------------------------------------------------------
    # 5.1. Validar cantidad de registros.
    # --------------------------------------------------------

    if df.empty:

        raise ValueError(
            "El archivo no contiene transacciones."
        )

    if len(df) > MAX_FILAS:

        raise ValueError(

            f"El máximo permitido es "
            f"{MAX_FILAS:,} transacciones."
        )

    # --------------------------------------------------------
    # 5.2. Verificar columnas requeridas.
    # --------------------------------------------------------

    faltantes = [

        columna

        for columna in variables

        if columna not in df.columns
    ]

    if faltantes:

        raise ValueError(

            "Faltan características requeridas: "

            + ", ".join(faltantes)
        )

    # --------------------------------------------------------
    # 5.3. Seleccionar únicamente predictores.
    # --------------------------------------------------------

    X = df[variables].copy()

    # --------------------------------------------------------
    # 5.4. Preparar variables numéricas.
    # --------------------------------------------------------

    for columna in numericas:

        original = X[columna]

        convertido = pd.to_numeric(

            original,

            errors="coerce"
        )

        invalidos = (

            original.notna()

            & convertido.isna()
        )

        if invalidos.any():

            raise ValueError(

                f"La variable {columna} contiene "

                "valores no numéricos."
            )

        X[columna] = (

            convertido

            .replace(
                [np.inf, -np.inf],
                np.nan
            )

            .astype("float32")
        )

    # --------------------------------------------------------
    # 5.5. Preparar variables categóricas.
    # --------------------------------------------------------

    for columna in categoricas:

        X[columna] = (

            X[columna]

            .astype("string")

            .fillna("DESCONOCIDO")

            .astype(str)
        )

    return X


# ============================================================
# 6. GENERACIÓN DE PREDICCIONES
# ============================================================

def generar_predicciones(

    df,

    modelo,

    preprocesador,

    configuracion,

    umbral
):

    inicio = perf_counter()

    # --------------------------------------------------------
    # Preparar características.
    # --------------------------------------------------------

    X = preparar_transacciones(

        df,

        configuracion
    )

    # --------------------------------------------------------
    # Aplicar el preprocesador entrenado.
    # --------------------------------------------------------

    X_transformado = (

        preprocesador.transform(X)

        .astype(
            np.float32,
            copy=False
        )
    )

    # --------------------------------------------------------
    # Ejecutar XGBoost.
    # --------------------------------------------------------

    scores = modelo.predict_proba(

        X_transformado

    )[:, 1]

    if not np.isfinite(scores).all():

        raise ValueError(
            "Se obtuvieron puntuaciones no válidas."
        )

    # --------------------------------------------------------
    # Aplicar umbral operativo.
    # --------------------------------------------------------

    alertas = (

        scores >= umbral

    ).astype(np.int8)

    # --------------------------------------------------------
    # Construir resultados.
    # --------------------------------------------------------

    resultados = pd.DataFrame({

        "ID transacción":

            df["transaction_id"].to_numpy()

            if "transaction_id" in df.columns

            else np.arange(1, len(df) + 1),

        "Monto":

            pd.to_numeric(
                df["amount"],
                errors="coerce"
            ).to_numpy(),

        "MCC":

            df["mcc"].to_numpy(),

        "Score de riesgo":

            scores,

        "Alerta":

            alertas,

        "Clasificación":

            np.where(

                alertas == 1,

                "Revisión requerida",

                "Sin alerta"
            )
    })

    if "transaction_date" in df.columns:

        resultados.insert(

            1,

            "Fecha",

            df["transaction_date"].to_numpy()
        )

    # Conservar etiqueta únicamente para evaluación.

    if "is_fraud" in df.columns:

        etiquetas = pd.to_numeric(

            df["is_fraud"],

            errors="coerce"
        )

        if (

            etiquetas.notna().all()

            and etiquetas.isin([0, 1]).all()
        ):

            resultados["Fraude real"] = (

                etiquetas.to_numpy(dtype=np.int8)
            )

    # Ordenar desde mayor riesgo.

    resultados = (

        resultados.sort_values(

            "Score de riesgo",

            ascending=False,

            kind="stable"
        )

        .reset_index(drop=True)
    )

    tiempo = perf_counter() - inicio

    return resultados, tiempo


# ============================================================
# 7. RESULTADOS HISTÓRICOS DEL MODELO
# ============================================================

def mostrar_desempeno_historico(configuracion):

    metricas = configuracion[
        "resultados_prueba"
    ]

    st.subheader(
        "Desempeño histórico del modelo"
    )

    st.caption(

        "Evaluación retrospectiva sobre "
        "830.731 transacciones del período "
        "junio de 2018 a octubre de 2019. "
        "Escenario equilibrado."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(

        "ROC-AUC",

        f"{metricas['roc_auc']:.4f}"
    )

    col2.metric(

        "Average Precision",

        f"{metricas['average_precision']:.4f}"
    )

    col3.metric(

        "Precision",

        f"{metricas['precision'] * 100:.2f}%"
    )

    col4.metric(

        "Recall",

        f"{metricas['recall'] * 100:.2f}%"
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(

        "Fraudes detectados",

        f"{metricas['fraudes_detectados']:,}"
    )

    col6.metric(

        "Total alertas",

        f"{metricas['total_alertas']:,}"
    )

    col7.metric(

        "Recall@Top 1%",

        f"{metricas['recall_top_1'] * 100:.2f}%"
    )

    col8.metric(

        "Variables predictoras",

        configuracion["numero_predictores"]
    )

    with st.expander(
        "Ver detalles de la evaluación histórica"
    ):

        tabla = pd.DataFrame({

            "Indicador": [

                "Transacciones evaluadas",

                "Fraudes existentes",

                "Fraudes detectados",

                "Fraudes no detectados",

                "Falsos positivos",

                "Total alertas",

                "Fraudes detectados en Top 1%",

                "F1-score"
            ],

            "Resultado": [

                f"{metricas['transacciones']:,}",

                f"{metricas['fraudes_totales']:,}",

                f"{metricas['fraudes_detectados']:,}",

                f"{metricas['fraudes_no_detectados']:,}",

                f"{metricas['falsos_positivos']:,}",

                f"{metricas['total_alertas']:,}",

                f"{metricas['fraudes_top_1']:,}",

                f"{metricas['f1_score']:.6f}"
            ]
        })

        st.dataframe(

            tabla,

            hide_index=True,

            use_container_width=True
        )

        st.caption(

            "Estos indicadores corresponden al umbral "
            "equilibrado fijado en validación. "
            "Recall@Top 1% corresponde a una evaluación "
            "retrospectiva de priorización."
        )


# ============================================================
# 8. VISUALIZACIÓN DE NUEVAS PREDICCIONES
# ============================================================

def mostrar_resultados_predictivos(

    resultados,

    tiempo,

    umbral
):

    st.subheader(
        "Resultados del análisis predictivo"
    )

    total = len(resultados)

    alertas = int(
        resultados["Alerta"].sum()
    )

    sin_alerta = total - alertas

    tasa_alertas = (
        alertas / total * 100
    )

    # --------------------------------------------------------
    # 8.1. Indicadores de las nuevas predicciones.
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(

        "Transacciones evaluadas",

        f"{total:,}"
    )

    col2.metric(

        "Alertas generadas",

        f"{alertas:,}"
    )

    col3.metric(

        "Sin alerta",

        f"{sin_alerta:,}"
    )

    col4.metric(

        "Tasa de alertas",

        f"{tasa_alertas:.2f}%"
    )

    st.caption(

        f"Predicción ejecutada en {tiempo:.3f} segundos "
        f"con un umbral de {umbral:.6f}."
    )

    # --------------------------------------------------------
    # 8.2. Distribución de puntuaciones.
    # --------------------------------------------------------

    st.markdown(
        "#### Distribución de puntuaciones de riesgo"
    )

    figura = px.histogram(

        resultados,

        x="Score de riesgo",

        nbins=40,

        color_discrete_sequence=[
            "#2563EB"
        ],

        labels={

            "Score de riesgo":
                "Puntuación de riesgo"
        }
    )

    figura.add_vline(

        x=umbral,

        line_dash="dash",

        line_color="#DC2626",

        annotation_text="Umbral operativo"
    )

    figura.update_layout(

        template="plotly_white",

        height=380,

        yaxis_title="Transacciones",

        xaxis_title="Puntuación de riesgo",

        bargap=0.05
    )

    st.plotly_chart(

        figura,

        use_container_width=True
    )

    # --------------------------------------------------------
    # 8.3. Ranking de transacciones.
    # --------------------------------------------------------

    st.markdown(
        "#### Ranking de transacciones por riesgo"
    )

    st.caption(

        "Las transacciones están ordenadas "
        "desde la mayor hasta la menor puntuación."
    )

    filtro = st.radio(

        "Mostrar",

        [

            "Todas las transacciones",

            "Solo alertas",

            "Solo transacciones sin alerta"
        ],

        horizontal=True,

        key="fraudguard_filtro_predicciones"
    )

    if filtro == "Solo alertas":

        datos_mostrar = resultados[
            resultados["Alerta"] == 1
        ]

    elif filtro == "Solo transacciones sin alerta":

        datos_mostrar = resultados[
            resultados["Alerta"] == 0
        ]

    else:

        datos_mostrar = resultados

    st.dataframe(

        datos_mostrar,

        hide_index=True,

        use_container_width=True,

        column_config={

            "Score de riesgo":

                st.column_config.ProgressColumn(

                    "Score de riesgo",

                    min_value=0.0,

                    max_value=1.0,

                    format="%.4f"
                ),

            "Monto":

                st.column_config.NumberColumn(

                    "Monto ($)",

                    format="$%.2f"
                )
        }
    )

    # --------------------------------------------------------
    # 8.4. Descargar predicciones.
    # --------------------------------------------------------

    st.download_button(

        label="Descargar resultados predictivos",

        data=resultados.to_csv(

            index=False

        ).encode("utf-8-sig"),

        file_name="predicciones_fraudguard.csv",

        mime="text/csv",

        key="fraudguard_descarga_predicciones"
    )


# ============================================================
# 9. EVALUACIÓN SI EXISTEN ETIQUETAS REALES
# ============================================================

def mostrar_evaluacion_muestra(resultados):

    if "Fraude real" not in resultados.columns:

        st.info(

            "El archivo no contiene etiquetas reales "
            "válidas de fraude. Se muestran puntuaciones "
            "y alertas, pero no métricas de clasificación."
        )

        return

    y_real = resultados[
        "Fraude real"
    ].to_numpy()

    y_pred = resultados[
        "Alerta"
    ].to_numpy()

    # --------------------------------------------------------
    # 9.1. Calcular matriz de confusión.
    # --------------------------------------------------------

    matriz = confusion_matrix(

        y_real,

        y_pred,

        labels=[0, 1]
    )

    tn, fp, fn, tp = matriz.ravel()

    st.subheader(

        "Evaluación de las transacciones cargadas"
    )

    st.caption(

        "Estas métricas corresponden exclusivamente "
        "a las transacciones analizadas en esta sesión, "
        "no al conjunto histórico completo de prueba."
    )

    total_fraudes = int(y_real.sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(

        "Fraudes existentes",

        total_fraudes
    )

    col2.metric(

        "Fraudes detectados",

        int(tp)
    )

    col3.metric(

        "Fraudes no detectados",

        int(fn)
    )

    col4.metric(

        "Falsos positivos",

        int(fp)
    )

    # --------------------------------------------------------
    # 9.2. Métricas de clasificación.
    # --------------------------------------------------------

    precision = precision_score(

        y_real,

        y_pred,

        zero_division=0
    )

    recall = recall_score(

        y_real,

        y_pred,

        zero_division=0
    )

    f1 = f1_score(

        y_real,

        y_pred,

        zero_division=0
    )

    col5, col6, col7 = st.columns(3)

    col5.metric(

        "Precision",

        f"{precision * 100:.2f}%"

        if y_pred.sum() > 0

        else "No definida"
    )

    col6.metric(

        "Recall",

        f"{recall * 100:.2f}%"

        if total_fraudes > 0

        else "No definido"
    )

    col7.metric(

        "F1-score",

        f"{f1:.4f}"
    )

    # --------------------------------------------------------
    # 9.3. Matriz de confusión.
    # --------------------------------------------------------

    figura_matriz = px.imshow(

        matriz,

        x=[

            "Sin alerta",

            "Revisión requerida"
        ],

        y=[

            "Legítima",

            "Fraudulenta"
        ],

        text_auto=True,

        color_continuous_scale="Blues",

        labels={

            "x": "Predicción",

            "y": "Clase real",

            "color": "Transacciones"
        }
    )

    figura_matriz.update_layout(

        template="plotly_white",

        height=400,

        title="Matriz de confusión de la muestra"
    )

    st.plotly_chart(

        figura_matriz,

        use_container_width=True
    )

    if total_fraudes < 30:

        st.warning(

            "La muestra contiene pocos fraudes. "
            "Sus métricas pueden variar considerablemente "
            "y no representan una nueva evaluación "
            "independiente del modelo."
        )


# ============================================================
# 10. INTERFAZ PRINCIPAL DEL MÓDULO PREDICTIVO
# ============================================================

def mostrar_modulo_predictivo():

    st.header(

        "🛡️ Modelo predictivo de fraude"
    )

    st.markdown(

        """
        Evaluación inteligente de transacciones con tarjetas
        de débito mediante **Machine Learning**.

        El modelo XGB-5 analiza características transaccionales
        e históricas para identificar operaciones que requieren
        revisión prioritaria.
        """
    )

    # --------------------------------------------------------
    # 10.1. Cargar modelo y configuración.
    # --------------------------------------------------------

    try:

        (

            modelo,

            preprocesador,

            configuracion

        ) = cargar_modelo_predictivo()

    except Exception as error:

        st.error(

            f"No fue posible cargar el modelo predictivo: "
            f"{error}"
        )

        return

    st.success(

        "Modelo XGB-5 cargado correctamente | "
        f"{configuracion['numero_predictores']} "
        "características predictoras."
    )

    # --------------------------------------------------------
    # 10.2. Mostrar resultados históricos.
    # --------------------------------------------------------

    mostrar_desempeno_historico(
        configuracion
    )

    st.divider()

    # --------------------------------------------------------
    # 10.3. Configuración del análisis.
    # --------------------------------------------------------

    st.subheader(

        "Configuración del análisis predictivo"
    )

    origen = st.radio(

        "Origen de las transacciones",

        [

            "Transacciones de demostración",

            "Cargar archivo CSV"
        ],

        key="fraudguard_origen_predictivo"
    )

    umbrales = configuracion[
        "umbrales_operativos"
    ]

    escenarios = list(
        umbrales.keys()
    )

    escenario_default = configuracion[
        "escenario_operativo"
    ]

    indice_default = (

        escenarios.index(escenario_default)

        if escenario_default in escenarios

        else 0
    )

    escenario = st.selectbox(

        "Escenario operativo",

        escenarios,

        index=indice_default,

        key="fraudguard_escenario_predictivo"
    )

    umbral = float(
        umbrales[escenario]
    )

    descripcion_escenarios = {

        "Cobertura alta":

            "Prioriza la detección, "
            "aceptando una mayor cantidad de alertas.",

        "Equilibrado":

            "Busca un equilibrio entre detección "
            "de fraude y volumen de alertas.",

        "Selectivo":

            "Prioriza alertas con puntuaciones "
            "de riesgo más elevadas."
    }

    st.info(

        f"{descripcion_escenarios.get(escenario, '')} "
        f"Umbral: {umbral:.6f}."
    )

    # --------------------------------------------------------
    # 10.4. Cargar transacciones.
    # --------------------------------------------------------

    df = None

    if origen == "Transacciones de demostración":

        try:

            df = cargar_transacciones_demo(

                tuple(

                    configuracion[
                        "variables_categoricas"
                    ]
                )
            )

            st.write(

                f"Dataset disponible: "
                f"**{len(df):,} transacciones**."
            )

            st.caption(

                "Muestra reproducible del conjunto "
                "histórico de prueba."
            )

            st.download_button(

                "Descargar CSV de ejemplo",

                data=df.to_csv(

                    index=False

                ).encode("utf-8-sig"),

                file_name="transacciones_demo.csv",

                mime="text/csv",

                key="fraudguard_descarga_demo"
            )

        except Exception as error:

            st.error(

                f"No se pudieron cargar los ejemplos: "
                f"{error}"
            )

            return

    else:

        st.warning(

            "Utiliza únicamente datos públicos, sintéticos "
            "o debidamente anonimizados."
        )

        archivo = st.file_uploader(

            "Cargar transacciones con características calculadas",

            type=["csv"],

            key="fraudguard_csv_predictivo"
        )

        st.caption(

            f"Máximo: {MAX_ARCHIVO_MB} MB y "
            f"{MAX_FILAS:,} transacciones."
        )

        if archivo is None:

            st.info(

                "Carga un CSV con las 69 características "
                "predictoras requeridas por el modelo."
            )

            with st.expander(
                "Ver características requeridas"
            ):

                st.dataframe(

                    pd.DataFrame({

                        "Característica":

                            configuracion[
                                "variables_predictoras"
                            ]
                    }),

                    hide_index=True,

                    use_container_width=True
                )

            return

        if archivo.size > MAX_ARCHIVO_MB * 1024**2:

            st.error(
                "El archivo supera el tamaño permitido."
            )

            return

        try:

            tipos = {

                columna: "string"

                for columna in configuracion[
                    "variables_categoricas"
                ]
            }

            df = pd.read_csv(

                archivo,

                dtype=tipos,

                nrows=MAX_FILAS + 1
            )

            if len(df) > MAX_FILAS:

                st.error(

                    f"El archivo supera las "
                    f"{MAX_FILAS:,} transacciones."
                )

                return

        except Exception as error:

            st.error(

                f"No fue posible leer el CSV: {error}"
            )

            return

    # --------------------------------------------------------
    # 10.5. Vista previa de transacciones.
    # --------------------------------------------------------

    if df is not None:

        with st.expander(
            "Vista previa de los datos de entrada"
        ):

            st.dataframe(

                df.head(20),

                use_container_width=True,

                hide_index=True
            )

    # --------------------------------------------------------
    # 10.6. Identificar la configuración actual.
    # --------------------------------------------------------

    firma_datos = (

        origen,

        escenario,

        len(df),

        int(

            pd.util.hash_pandas_object(

                df,

                index=True

            ).sum()
        )
    )

    # Evitar mostrar resultados de una carga anterior.

    if st.session_state.get(

        "fraudguard_firma_predictiva"

    ) != firma_datos:

        st.session_state.pop(

            "fraudguard_resultados_predictivos",

            None
        )

        st.session_state[
            "fraudguard_firma_predictiva"
        ] = firma_datos

    # --------------------------------------------------------
    # 10.7. Ejecutar predicción.
    # --------------------------------------------------------

    if st.button(

        "Ejecutar análisis predictivo",

        type="primary",

        key="fraudguard_ejecutar_prediccion"
    ):

        try:

            with st.spinner(

                "Evaluando transacciones con XGBoost..."
            ):

                resultados, tiempo = (

                    generar_predicciones(

                        df,

                        modelo,

                        preprocesador,

                        configuracion,

                        umbral
                    )
                )

            st.session_state[

                "fraudguard_resultados_predictivos"

            ] = {

                "resultados": resultados,

                "tiempo": tiempo,

                "umbral": umbral
            }

        except Exception as error:

            st.error(

                f"No fue posible realizar la predicción: "
                f"{error}"
            )

            return

    # --------------------------------------------------------
    # 10.8. Mostrar resultados persistentes.
    # --------------------------------------------------------

    salida = st.session_state.get(

        "fraudguard_resultados_predictivos"
    )

    if salida is not None:

        st.divider()

        mostrar_resultados_predictivos(

            salida["resultados"],

            salida["tiempo"],

            salida["umbral"]
        )

        st.divider()

        mostrar_evaluacion_muestra(

            salida["resultados"]
        )

    # --------------------------------------------------------
    # 10.9. Consideraciones metodológicas.
    # --------------------------------------------------------

    st.divider()

    st.caption(

        "FraudGuard AI es un prototipo académico. "
        "Las puntuaciones de riesgo no son probabilidades "
        "calibradas de fraude. Las predicciones se utilizan "
        "como apoyo para priorizar revisiones y no implican "
        "bloqueos automáticos de operaciones. "
        "El desempeño futuro en instituciones financieras "
        "requiere validación adicional."
    )


# ============================================================
# FIN DEL MÓDULO PREDICTIVO
# ============================================================
