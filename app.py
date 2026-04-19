import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import json

# --- CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(page_title="Prueba de hipotesis ", layout="wide")

# Inicializar estados de sesión
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ia_response' not in st.session_state:
    st.session_state.ia_response = ""

# Paleta de colores (mantengo la tuya)
COLOR_HIST      = "#4C9BE8"
COLOR_BOX       = "#56C596"
COLOR_Z_OK      = "#27AE60"
COLOR_CRITICO   = "#F39C12"
COLOR_RECHAZO   = "#E84C4C"

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
})

st.title("📊 Aplicación Prueba De Hipótesis")

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

    # Tabla de datos
    st.subheader("📋 Datos Cargados")
    st.dataframe(df.style.format({variable: "{:.2f}"}), use_container_width=True, height=320)

    # Análisis visual
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

    # ====================== PRUEBA Z ======================
    st.write("---")
    st.header("🔬 Prueba de Hipótesis (Z-Test)")

    # Estadísticos clave visibles
    st.subheader("📊 Estadísticos de la Muestra")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tamaño de muestra (n)", len(datos))
    with col2:
        x_barra = datos.mean()
        st.metric("Media muestral (x̄)", f"{x_barra:.4f}")
    with col3:
        s_muestral = datos.std(ddof=1)
        st.metric("Desv. Estándar Muestral (s)", f"{s_muestral:.4f}")

    # ==================== SIGMA CONFIGURABLE ====================
    with col4:
        usar_sigma_conocida = st.checkbox("Usar σ poblacional conocida", value=True)
        if usar_sigma_conocida:
            sigma = st.number_input("Valor de σ (desviación estándar poblacional)", 
                                  value=8.0, step=0.1, min_value=0.1)
        else:
            sigma = s_muestral
        st.metric("Valor usado (σ)", f"{sigma:.4f}")

    # Configuración de la prueba
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        mu_0 = st.number_input("Hipótesis Nula (μ₀):", value=48.0, step=0.1)
        alpha = st.select_slider("Significancia (α):", options=[0.01, 0.05, 0.10], value=0.05)
    with col_p2:
        tipo_cola = st.radio("Tipo de prueba:", ["Bilateral", "Superior", "Inferior"], horizontal=True)

    # Cálculo Z
    n = len(datos)
    z_calc = (x_barra - mu_0) / (sigma / np.sqrt(n))

    # Cálculo de p-valor y z crítico
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
    st.subheader("📈 Resultados del Z-Test")
    res1, res2, res3 = st.columns(3)
    res1.metric("Estadístico Z calculado", f"{z_calc:.4f}")
    res2.metric("p-valor", f"{p_v:.6f}")
    res3.metric("¿Rechaza H₀?", "✅ SÍ" if rechazo else "❌ NO")

    if rechazo:
        st.error(f"🔴 Se rechaza la hipótesis nula con α = {alpha}")
    else:
        st.success(f"🟢 No hay evidencia suficiente para rechazar H₀")

    # ====================== GRÁFICA (MANTENIDA COMO LA TENÍAS) ======================
    st.subheader("📉 Distribución Normal Estándar - Regiones de Rechazo")

    fig, ax = plt.subplots(figsize=(10, 4))   # ← Tamaño y estilo como lo tenías antes

    # Curva normal
    x = np.linspace(-4.5, 4.5, 500)
    y = stats.norm.pdf(x)
    ax.plot(x, y, color='#000000', linewidth=2.8, label='Distribución Normal Estándar (Z)')

    # Sombrear región de rechazo
    if tipo_cola == "Bilateral":
        ax.fill_between(x, y, where=(x <= -z_crit) | (x >= z_crit), 
                       color='red', alpha=0.35, label='Región de rechazo (α)')

        # Líneas críticas en NEGRO
        ax.axvline(-z_crit, color='black', linestyle='--', linewidth=2.0)
        ax.axvline(z_crit, color='black', linestyle='--', linewidth=2.0)

    elif tipo_cola == "Superior":
        ax.fill_between(x, y, where=(x >= z_crit), color=COLOR_RECHAZO, alpha=0.35)
        ax.axvline(z_crit, color='black', linestyle='--', linewidth=2.0)
        ax.text(z_crit - 0.15, 0.08, f'{z_crit:.2f}', color='red', fontsize=11, fontweight='bold')
    else:  # Inferior
        ax.fill_between(x, y, where=(x <= z_crit), color=COLOR_RECHAZO, alpha=0.35)
        ax.axvline(z_crit, color='black', linestyle='--', linewidth=2.0)
        ax.text(z_crit - 0.15, 0.08, f'{z_crit:.2f}', color='red', fontsize=11, fontweight='bold')

    # Línea de Z calculado (verde)
    ax.axvline(z_calc, color=COLOR_Z_OK, linestyle='-', linewidth=3.2, 
               label=f'Z calculado = {z_calc:.3f}')

    # Línea de la media bajo H₀ 
    ax.axvline(0, color='#FF0000', linestyle='-', linewidth=2.5, alpha=0.85, 
               label=f'Media bajo H₀ (μ₀ = {mu_0})')

    ax.set_title('Prueba Z Bilateral - Distribución Normal con Regiones Críticas', 
                 fontsize=15, pad=15)
    ax.set_xlabel('Valores Z', fontsize=12)
    ax.set_ylabel('Densidad de Probabilidad', fontsize=12)
    
    ax.legend(fontsize=10.5, loc='upper right')
    ax.grid(True, alpha=0.25)

    st.pyplot(fig)
    st.caption("• Zona roja = Región de rechazo | • Línea verde = Z calculado | • Línea negra = Media según H₀")

    # --- INTERPRETACIÓN CON IA ---
    st.write("---")
    st.header("🤖 Interpretación con Inteligencia Artificial")
    
    api_key_input = st.text_input("Introduce tu Gemini API Key:", type="password")

    if st.button("✨ Interpretar resultados con IA", type="primary"):
        clave_limpia = api_key_input.strip()
        if not clave_limpia:
            st.error("❌ Por favor, introduce una API Key.")
        else:
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={clave_limpia}"
            
            headers = {'Content-Type': 'application/json'}
            
            prompt_ia = f"""
            Analiza como experto en estadística la siguiente prueba Z:
            - Variable: {variable}
            - Media muestral: {x_barra:.4f}
            - Hipótesis nula: μ = {mu_0}
            - σ usada: {sigma:.4f}
            - Z calculado: {z_calc:.4f}
            - p-valor: {p_v:.6f}
            - Decisión: {'Rechazar H0' if rechazo else 'No rechazar H0'}
            - Nivel de significancia: {alpha}

            Explica el resultado de forma profesional y luego una explicación sencilla para público general.
            """

            payload = {"contents": [{"parts": [{"text": prompt_ia}]}]}

            try:
                with st.spinner("🧠 Consultando a Gemini 2.5 Flash..."):
                    response = requests.post(url, headers=headers, data=json.dumps(payload))
                    res_json = response.json()

                if response.status_code == 200:
                    texto_ia = res_json['candidates'][0]['content']['parts'][0]['text']
                    st.session_state.ia_response = texto_ia
                    st.success("✅ ¡Análisis completado!")
            except Exception as e:
                st.error(f"❌ Error de conexión: {str(e)}")

    if st.session_state.ia_response:
        st.info(st.session_state.ia_response)