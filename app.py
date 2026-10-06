from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_seleccionado.joblib"
METADATA_PATH = BASE_DIR / "metadata_modelo.json"

st.set_page_config(
    page_title="Evaluación de riesgo de DM2",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# Estilos
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 42%, #f8fafc 100%);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 1.7rem 1.9rem;
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            background: rgba(255,255,255,0.92);
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.3rem;
        }
        .hero-kicker {
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            color: #2563eb;
            margin-bottom: .45rem;
        }
        .hero h1 {
            margin: 0;
            color: #0f172a;
            font-size: 2.15rem;
            line-height: 1.12;
        }
        .hero p {
            color: #475569;
            margin: .7rem 0 0 0;
            font-size: 1rem;
            line-height: 1.6;
        }
        .section-title {
            color: #0f172a;
            font-size: 1.18rem;
            font-weight: 750;
            margin: .15rem 0 .2rem 0;
        }
        .section-copy {
            color: #64748b;
            font-size: .92rem;
            margin-bottom: 1rem;
        }
        .result-card {
            border-radius: 22px;
            padding: 1.6rem 1.7rem;
            margin-top: .4rem;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.07);
        }
        .result-low {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
        }
        .result-high {
            background: #fff1f2;
            border: 1px solid #fecdd3;
        }
        .result-eyebrow {
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            color: #64748b;
        }
        .result-title {
            margin: .35rem 0 .5rem 0;
            font-size: 1.65rem;
            font-weight: 800;
            color: #0f172a;
        }
        .result-text {
            color: #334155;
            line-height: 1.6;
            margin: 0;
        }
        .score-box {
            margin-top: 1rem;
            padding: .9rem 1rem;
            border-radius: 14px;
            background: rgba(255,255,255,.75);
            border: 1px solid rgba(148,163,184,.28);
        }
        .score-label {
            color: #64748b;
            font-size: .8rem;
            font-weight: 650;
        }
        .score-value {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: .15rem;
        }
        .clinical-note {
            margin-top: 1rem;
            color: #475569;
            font-size: .86rem;
            line-height: 1.55;
        }
        .footer-note {
            color: #64748b;
            font-size: .82rem;
            line-height: 1.55;
        }
        .author-note {
            color: #94a3b8;
            font-size: .76rem;
            margin-top: .45rem;
        }
        /* Mantener legibles las etiquetas y desplegables también con tema oscuro del navegador. */
        div[data-testid="stNumberInput"] label,
        div[data-testid="stNumberInput"] label p,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary p {
            color: #334155 !important;
            opacity: 1 !important;
        }
        div[data-testid="stNumberInput"] label {
            font-weight: 650 !important;
        }
        div[data-testid="stForm"] {
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 1.35rem 1.4rem 1.1rem 1.4rem;
            background: white;
            box-shadow: 0 8px 26px rgba(15, 23, 42, 0.045);
        }
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Carga de artefactos
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


try:
    model, metadata = load_artifacts()
except Exception as exc:
    st.error("No se han podido cargar los artefactos del modelo.")
    st.code(str(exc))
    st.stop()

FEATURES = metadata["features"]
THRESHOLD = float(metadata.get("decision_function_threshold", 0.0))


# -----------------------------------------------------------------------------
# Cabecera
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">TFM · Inteligencia Artificial aplicada a la Sanidad</div>
        <h1>Evaluación de riesgo de diabetes mellitus tipo 2</h1>
        <p>
            Herramienta académica de apoyo basada en un modelo SVM con kernel RBF.
            Introduzca las variables del paciente para obtener la clasificación generada
            por el modelo previamente entrenado y validado.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.55, 1], gap="large")


# -----------------------------------------------------------------------------
# Formulario
# -----------------------------------------------------------------------------
with left:
    st.markdown('<div class="section-title">Datos del paciente</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Complete las ocho variables empleadas por el modelo. Los valores deben corresponder a la información clínica disponible.</div>',
        unsafe_allow_html=True,
    )

    with st.form("patient_form"):
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            pregnancies = st.number_input(
                "Embarazos",
                min_value=0,
                step=1,
                value=0,
                help="Número de embarazos registrados.",
            )
            glucose = st.number_input(
                "Glucosa (mg/dL)",
                min_value=0.0,
                step=1.0,
                value=100.0,
                help="Concentración de glucosa plasmática.",
            )
            blood_pressure = st.number_input(
                "Presión arterial diastólica (mmHg)",
                min_value=0.0,
                step=1.0,
                value=70.0,
                help="Presión arterial diastólica.",
            )
            skin_thickness = st.number_input(
                "Grosor del pliegue cutáneo (mm)",
                min_value=0.0,
                step=1.0,
                value=20.0,
                help="Grosor del pliegue cutáneo del tríceps.",
            )

        with c2:
            insulin = st.number_input(
                "Insulina (µU/mL)",
                min_value=0.0,
                step=1.0,
                value=80.0,
                help="Insulina sérica a las 2 horas.",
            )
            bmi = st.number_input(
                "Índice de masa corporal (kg/m²)",
                min_value=0.0,
                step=0.1,
                value=25.0,
                format="%.1f",
                help="Índice de masa corporal (IMC).",
            )
            pedigree = st.number_input(
                "Función de antecedentes familiares",
                min_value=0.0,
                step=0.01,
                value=0.50,
                format="%.3f",
                help="Diabetes Pedigree Function (DPF): índice del conjunto de datos que resume la carga de antecedentes familiares de diabetes. Introduzca el valor disponible para esta variable; no es un número de familiares.",
            )
            age = st.number_input(
                "Edad (años)",
                min_value=1,
                step=1,
                value=35,
                help="Edad del paciente en años.",
            )

        submitted = st.form_submit_button("Analizar paciente", type="primary")


# -----------------------------------------------------------------------------
# Panel de resultado
# -----------------------------------------------------------------------------
with right:
    st.markdown('<div class="section-title">Resultado del modelo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">La salida se interpreta según la regla de decisión original del modelo.</div>',
        unsafe_allow_html=True,
    )

    if not submitted:
        st.info("Introduzca los datos y pulse **Analizar paciente** para generar una evaluación.")

        with st.expander("Información del modelo"):
            st.write(f"**Modelo:** {metadata.get('model_name', 'SVM RBF')}")
            st.write("**Pipeline:** imputación → escalado → SVM-RBF")
            st.write(f"**Umbral de decisión:** {THRESHOLD:.1f}")
            st.write("**Variables de entrada:** 8")

    else:
        values = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
        }

        # Se fuerza exactamente el orden de variables registrado en los metadatos.
        patient_df = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)

        prediction = int(model.predict(patient_df)[0])
        decision_score = float(model.decision_function(patient_df)[0])

        # La clasificación desplegada conserva la regla científica congelada.
        positive = decision_score > THRESHOLD

        # Comprobación defensiva: predict() y la regla del score deben ser coherentes.
        if prediction != int(positive):
            st.warning(
                "Se ha detectado una discrepancia interna entre la predicción del modelo y su función de decisión. "
                "El resultado no debe interpretarse hasta revisar el entorno de ejecución."
            )
        elif positive:
            st.markdown(
                f"""
                <div class="result-card result-high">
                    <div class="result-eyebrow">Clasificación del modelo · Resultado positivo</div>
                    <div class="result-title">Perfil compatible con mayor riesgo de DM2</div>
                    <p class="result-text">
                        Para los datos introducidos, el modelo clasifica el caso en la clase positiva.
                        Este resultado indica una señal de riesgo según el patrón aprendido por el modelo y
                        requiere interpretación clínica; no constituye por sí mismo un diagnóstico de diabetes.
                    </p>
                    <div class="score-box">
                        <div class="score-label">Puntuación de decisión SVM</div>
                        <div class="score-value">{decision_score:.4f}</div>
                    </div>
                    <div class="clinical-note">
                        Regla aplicada: puntuación de decisión &gt; {THRESHOLD:.1f} → clase positiva.
                        La puntuación mostrada es un <strong>score del SVM</strong>, no una probabilidad calibrada.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-low">
                    <div class="result-eyebrow">Clasificación del modelo · Resultado negativo</div>
                    <div class="result-title">Perfil no clasificado como positivo para DM2</div>
                    <p class="result-text">
                        Para los datos introducidos, el modelo clasifica el caso en la clase negativa.
                        El resultado no permite descartar diabetes ni sustituye una valoración clínica,
                        pruebas diagnósticas o seguimiento profesional cuando estén indicados.
                    </p>
                    <div class="score-box">
                        <div class="score-label">Puntuación de decisión SVM</div>
                        <div class="score-value">{decision_score:.4f}</div>
                    </div>
                    <div class="clinical-note">
                        Regla aplicada: puntuación de decisión ≤ {THRESHOLD:.1f} → clase negativa.
                        La puntuación mostrada es un <strong>score del SVM</strong>, no una probabilidad calibrada.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Ver datos analizados"):
            display_df = patient_df.rename(
                columns={
                    "Pregnancies": "Embarazos",
                    "Glucose": "Glucosa",
                    "BloodPressure": "Presión arterial",
                    "SkinThickness": "Pliegue cutáneo",
                    "Insulin": "Insulina",
                    "BMI": "IMC",
                    "DiabetesPedigreeFunction": "Antecedentes familiares",
                    "Age": "Edad",
                }
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Nota de uso
# -----------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div class="footer-note">
        <strong>Uso académico.</strong> Esta aplicación forma parte de un Trabajo Fin de Máster y presenta
        la salida de un modelo de aprendizaje automático previamente desarrollado. No es un producto sanitario,
        no establece diagnósticos y no debe utilizarse como sustituto del criterio de un profesional sanitario.
        <div class="author-note">Desarrollado por Jorge Sanjuan Canto · Trabajo Fin de Máster</div>
    </div>
    """,
    unsafe_allow_html=True,
)
