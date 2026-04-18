import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import google.generativeai as genai
# --- CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(page_title="Asistente Estadístico Pro", layout="wide")

# Inicializar estados de sesión para evitar "pantallas negras" y pérdida de datos
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ia_response' not in st.session_state:
    st.session_state.ia_response = ""

# Paleta de colores centralizada
COLOR_HIST      = "#4C9BE8"
COLOR_BOX       = "#56C596"
COLOR_PROB      = "#F5A623"
COLOR_CURVA     = "#1A1A2E"
COLOR_RECHAZO   = "#E84C4C"
COLOR_Z_OK      = "#27AE60"
COLOR_CRITICO   = "#F39C12"

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

st.title("📊 Aplicación Interactiva de Estadística y Probabilidad")

# --- SIDEBAR: GESTIÓN DE DATOS ---
st.sidebar.header("1. Configuración de Datos")
opcion_datos = st.sidebar.radio("Origen de datos:", ("Subir CSV", "Generar Datos Sintéticos"))

if opcion_datos == "Subir CSV":
    archivo = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
    if archivo is not None:
        st.session_state.df = pd.read_csv(archivo)
else:
    with st.sidebar.expander("Parámetros Sintéticos"):
        n_s = st.slider("n", 30, 1000, 100)
        mu_s = st.number_input("Media", value=50.0)
        sigma_s = st.number_input("Desv. Estándar", value=10.0)
    if st.sidebar.button("Generar nuevos datos"):
        data = np.random.normal(mu_s, sigma_s, n_s)
        st.session_state.df = pd.DataFrame(data, columns=["Variable_Generada"])

# --- CUERPO PRINCIPAL ---
if st.session_state.df is not None:
    df = st.session_state.df
    variable = st.selectbox("Selecciona la variable:", df.columns)
    datos = df[variable].dropna()

    # --- MÓDULO: ANÁLISIS VISUAL ---
    st.header("📈 Distribución de los Datos")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.histplot(datos, kde=True, ax=ax, color=COLOR_HIST)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.boxplot(x=datos, ax=ax, color=COLOR_BOX)
        st.pyplot(fig)

    # --- MÓDULO: PRUEBA DE HIPÓTESIS Z ---
    st.write("---")
    st.header("🔬 Prueba de Hipótesis (Z-Test)")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        mu_0 = st.number_input("Hipótesis Nula (μ₀):", value=float(datos.mean()))
        alpha = st.select_slider("Significancia (α):", options=[0.01, 0.05, 0.10], value=0.05)
    with col_p2:
        tipo_cola = st.radio("Cola:", ["Bilateral", "Superior", "Inferior"])

    # Cálculos Estadísticos
    x_barra = datos.mean()
    sigma = datos.std()
    n = len(datos)
    z_calc = (x_barra - mu_0) / (sigma / np.sqrt(n))

    if tipo_cola == "Bilateral":
        p_v = 2 * (1 - stats.norm.cdf(abs(z_calc)))
        z_crit = stats.norm.ppf(1 - alpha/2)
        rechazo = abs(z_calc) > z_crit
    elif tipo_cola == "Superior":
        p_v = 1 - stats.norm.cdf(z_calc)
        z_crit = stats.norm.ppf(1 - alpha)
        rechazo = z_calc > z_crit
    else:
        p_v = stats.norm.cdf(z_calc)
        z_crit = stats.norm.ppf(alpha)
        rechazo = z_calc < z_crit

    # Mostrar Resultados
    res1, res2, res3 = st.columns(3)
    res1.metric("Z Calculado", f"{z_calc:.4f}")
    res2.metric("p-valor", f"{p_v:.4f}")
    res3.metric("¿Rechaza H₀?", "SÍ" if rechazo else "NO")

    if rechazo:
        st.error(f"Se rechaza la hipótesis nula con α={alpha}")
    else:
        st.success(f"No hay evidencia suficiente para rechazar H₀")


    # --- MÓDULO 6: INTERPRETACIÓN CON IA (VERSIÓN 2026) ---
    st.write("---")
    st.header("🤖 Interpretación con Inteligencia Artificial")
    
    api_key_input = st.text_input("Introduce tu Gemini API Key:", type="password")

    if st.button("✨ Interpretar resultados con IA", type="primary"):
        import requests
        import json

        clave_limpia = api_key_input.strip()

        if not clave_limpia:
            st.error("❌ Por favor, introduce una API Key.")
        else:
            # USAMOS EL NOMBRE EXACTO DE TU DIAGNÓSTICO: gemini-2.5-flash
            # Y la versión estable v1
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={clave_limpia}"
            
            headers = {'Content-Type': 'application/json'}
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Analiza como experto en estadística: Variable {variable}, Media {x_barra:.2f}, p-valor {p_v:.4f}. Decisión: {'Rechazar H0' if rechazo else 'No rechazar H0'}. Explica qué significa esto de forma sencilla."
                    }]
                }]
            }

            try:
                with st.spinner("🧠 Consultando a Gemini 2.5 Flash..."):
                    response = requests.post(url, headers=headers, data=json.dumps(payload))
                    res_json = response.json()

                if response.status_code == 200:
                    texto_ia = res_json['candidates'][0]['content']['parts'][0]['text']
                    st.session_state.ia_response = texto_ia
                    st.success("✅ ¡Análisis completado con Gemini 2.5!")
                else:
                    # Si falla el 2.5, intentamos con el 2.0 que también vimos en tu lista
                    st.warning("Intentando con Gemini 2.0...")
                    url_2 = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={clave_limpia}"
                    response = requests.post(url_2, headers=headers, data=json.dumps(payload))
                    res_json = response.json()
                    texto_ia = res_json['candidates'][0]['content']['parts'][0]['text']
                    st.session_state.ia_response = texto_ia
            
            except Exception as e:
                st.error(f"❌ Error crítico: {res_json.get('error', {}).get('message', 'Error de conexión')}")

    if st.session_state.ia_response:
        st.info(st.session_state.ia_response)