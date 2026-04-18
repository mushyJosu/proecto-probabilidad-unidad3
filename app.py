import streamlit as st        # Crea la interfaz web (botones, sliders, títulos).
import pandas as pd           # Manipula tablas de datos (como Excel en código).
import numpy as np            # Realiza cálculos matemáticos y genera datos aleatorios.
import matplotlib.pyplot as plt # El "lienzo" base para dibujar gráficas.
import seaborn as sns         # El "pincel" que hace que las gráficas se vean bonitas.
from scipy import stats       # La calculadora estadística (calcula Z, p-value, etc.).
from google import genai      #IA

# AÑADIDO (Commit 7): Paleta de colores centralizada para todas las gráficas.
COLOR_HIST      = "#4C9BE8"
COLOR_BOX       = "#56C596"
COLOR_PROB      = "#F5A623"
COLOR_CURVA     = "#1A1A2E"
COLOR_RECHAZO   = "#E84C4C"
COLOR_Z_RECHAZO = "#E84C4C"
COLOR_Z_OK      = "#27AE60"
COLOR_CRITICO   = "#F39C12"

# AÑADIDO (Commit 7): Estilo global de Matplotlib más limpio y profesional.
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.facecolor":   "white",
    "axes.facecolor":     "#F9F9F9",
})

st.set_page_config(page_title="Asistente Estadístico", layout="wide")
st.title("📊 Aplicación Interactiva de Estadística y Probabilidad")

st.sidebar.header("Configuración de Datos")
opcion_datos = st.sidebar.radio("Selecciona origen de datos:", ("Subir CSV", "Generar Datos Sintéticos"))

df = None

if opcion_datos == "Subir CSV":
    archivo = st.file_uploader("Cargar archivo CSV", type=["csv"])
    if archivo is not None:
        df = pd.read_csv(archivo)

else:
    n_sintetico      = st.sidebar.slider("Tamaño de muestra (n)", 30, 1000, 100)
    mu_sintetico     = st.sidebar.number_input("Media real", value=50.0)
    sigma_sintetico  = st.sidebar.number_input("Desviación estándar real", value=10.0)

    if st.sidebar.button("Generar datos"):
        data = np.random.normal(mu_sintetico, sigma_sintetico, n_sintetico)
        df   = pd.DataFrame(data, columns=["Variable_Generada"])

# Sidebar de configuración de prueba (arriba para que alpha y tipo_cola
# estén disponibles cuando el Módulo 5 los necesite).
st.sidebar.write("---")
st.sidebar.header("⚙️ Configuración de Prueba")

alpha = st.sidebar.select_slider(
    "Nivel de significancia (α):",
    options=[0.01, 0.05, 0.10],
    value=0.05,
    help="Probabilidad de rechazar la hipótesis nula siendo esta verdadera."
)

tipo_cola = st.sidebar.radio(
    "Tipo de cola:",
    options=["Bilateral (Dos colas)", "Unilateral Superior", "Unilateral Inferior"],
    index=0
)

st.sidebar.info(f"Configuración actual: α={alpha} | {tipo_cola}")

if df is not None:
    st.write("### Vista previa de los datos")
    st.dataframe(df.head())

    variable      = st.selectbox("Selecciona la variable para analizar:", df.columns)
    datos_analizar = df[variable].dropna()

    # --- MÓDULO 2: VISUALIZACIÓN DE DISTRIBUCIONES ---
    st.write("---")
    st.header("📈 Análisis Visual de la Distribución")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Histograma y KDE")
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(datos_analizar, kde=True, ax=ax_hist, color=COLOR_HIST)
        ax_hist.set_title(f"Distribución de {variable}", fontsize=13, fontweight="bold")
        ax_hist.set_xlabel(variable, fontsize=10)
        ax_hist.set_ylabel("Frecuencia", fontsize=10)
        st.pyplot(fig_hist)

    with col2:
        st.subheader("Boxplot (Diagrama de Caja)")
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x=datos_analizar, ax=ax_box, color=COLOR_BOX)
        ax_box.set_title(f"Boxplot de {variable}", fontsize=13, fontweight="bold")
        ax_box.set_xlabel(variable, fontsize=10)
        st.pyplot(fig_box)

    # --- MÓDULO 3: ESTADÍSTICAS DESCRIPTIVAS Y PRUEBAS ---
    st.write("---")
    st.header("📊 Análisis Estadístico Detallado")

    sesgo    = datos_analizar.skew()
    curtosis = datos_analizar.kurtosis()

    estadistico, p_valor = stats.shapiro(datos_analizar)

    col_met1, col_met2, col_met3 = st.columns(3)

    with col_met1:
        st.metric("Sesgo (Skewness)", f"{sesgo:.2f}")
        st.caption("0 = Simétrico. (+) = Derecha, (-) = Izquierda")

    with col_met2:
        st.metric("Curtosis", f"{curtosis:.2f}")
        st.caption(">0 = Puntiaguda, <0 = Plana")

    with col_met3:
        es_normal = "Sí ✅" if p_valor > 0.05 else "No ❌"
        st.metric("¿Es Normal?", es_normal)
        st.caption(f"P-Valor Shapiro-Wilk: {p_valor:.4f}")

    Q1 = datos_analizar.quantile(0.25)
    Q3 = datos_analizar.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = datos_analizar[(datos_analizar < limite_inferior) | (datos_analizar > limite_superior)]

    st.write(f"**Detección de Outliers:** Se han encontrado **{len(outliers)}** valores atípicos.")
    if len(outliers) > 0:
        st.write(outliers)

    # --- MÓDULO 4: CÁLCULO DE PROBABILIDADES ---
    st.write("---")
    st.header("🎲 Calculadora de Probabilidades")
    st.write("Calcula la probabilidad acumulada $P(X / leq x)$ basándose en la media y desviación de tus datos.")

    mu_calc    = datos_analizar.mean()
    sigma_calc = datos_analizar.std()

    x_value = st.number_input(
        "Ingresa un valor para calcular P(X ≤ x):",
        value=float(mu_calc),
        step=0.1
    )

    probabilidad = stats.norm.cdf(x_value, mu_calc, sigma_calc)
    st.success(f"La probabilidad de que un valor sea menor o igual a **{x_value}** es del **{probabilidad*100:.2f}%**")

    fig_prob, ax_prob = plt.subplots(figsize=(7, 4))
    x_axis = np.linspace(mu_calc - 4*sigma_calc, mu_calc + 4*sigma_calc, 200)
    ax_prob.plot(x_axis, stats.norm.pdf(x_axis, mu_calc, sigma_calc),
                 color=COLOR_CURVA, lw=2, label="Curva Normal")

    x_fill = np.linspace(mu_calc - 4*sigma_calc, x_value, 200)
    ax_prob.fill_between(
        x_fill, stats.norm.pdf(x_fill, mu_calc, sigma_calc),
        color=COLOR_PROB, alpha=0.5,
        label=f"P(X ≤ {x_value:.2f}) = {probabilidad*100:.2f}%"
    )

    ax_prob.set_title(f"Área bajo la curva para X ≤ {x_value}", fontsize=11, fontweight="bold")
    ax_prob.set_xlabel("Valor de X", fontsize=10)
    ax_prob.set_ylabel("Densidad de probabilidad", fontsize=10)
    ax_prob.legend(fontsize=8)
    ax_prob.tick_params(labelsize=8)

    col_graf1, col_graf2, col_graf3 = st.columns([1, 2, 1])
    with col_graf2:
        st.pyplot(fig_prob)

    # --- MÓDULO 5: PRUEBA DE HIPÓTESIS Z ---
    st.write("---")
    st.header("🔬 Prueba de Hipótesis (Distribución Z)")
    st.write("""
    La prueba Z compara la media de tu muestra contra un valor hipotético (μ₀)
    para decidir si hay evidencia suficiente para rechazar la hipótesis nula.
    """)

    if tipo_cola == "Bilateral (Dos colas)":
        st.latex(r"H_0: \mu = \mu_0 \qquad H_a: \mu \neq \mu_0")
    elif tipo_cola == "Unilateral Superior":
        st.latex(r"H_0: \mu \leq \mu_0 \qquad H_a: \mu > \mu_0")
    else:
        st.latex(r"H_0: \mu \geq \mu_0 \qquad H_a: \mu < \mu_0")

    st.write("---")

    col_z1, col_z2 = st.columns(2)

    with col_z1:
        mu_0 = st.number_input(
            "Valor hipotético de la media (μ₀):",
            value=float(datos_analizar.mean()),
            step=0.1,
            help="El valor que quieres poner a prueba (tu hipótesis nula)."
        )

    with col_z2:
        usar_sigma_conocida = st.checkbox(
            "¿Conoces la desviación estándar poblacional (σ)?",
            value=False,
            help="Si no la conoces, se usará la desviación de tu muestra (s)."
        )
        if usar_sigma_conocida:
            sigma_conocida = st.number_input("Ingresa σ (poblacional):", value=float(datos_analizar.std()), step=0.1)
        else:
            sigma_conocida = datos_analizar.std()

    # Cálculo Z → (x̄ - μ₀) / (σ / √n)
    n              = len(datos_analizar)
    x_barra        = datos_analizar.mean()
    sigma          = sigma_conocida
    error_estandar = sigma / np.sqrt(n)
    z_calculado    = (x_barra - mu_0) / error_estandar

    # p-valor según tipo de cola
    if tipo_cola == "Bilateral (Dos colas)":
        p_valor_z = 2 * (1 - stats.norm.cdf(abs(z_calculado)))
    elif tipo_cola == "Unilateral Superior":
        p_valor_z = 1 - stats.norm.cdf(z_calculado)
    else:
        p_valor_z = stats.norm.cdf(z_calculado)

    # Z crítico según alpha y tipo de cola
    if tipo_cola == "Bilateral (Dos colas)":
        z_critico = stats.norm.ppf(1 - alpha / 2)
    elif tipo_cola == "Unilateral Superior":
        z_critico = stats.norm.ppf(1 - alpha)
    else:
        z_critico = stats.norm.ppf(alpha)

    # Decisión: ¿cae Z en la región de rechazo?
    if tipo_cola == "Bilateral (Dos colas)":
        rechazar = abs(z_calculado) > z_critico
    elif tipo_cola == "Unilateral Superior":
        rechazar = z_calculado > z_critico
    else:
        rechazar = z_calculado < z_critico

    st.write("### Resultados de la Prueba")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        st.metric("Media muestral (x̄)", f"{x_barra:.4f}")
    with col_r2:
        st.metric("Z calculado", f"{z_calculado:.4f}")
    with col_r3:
        st.metric("Z crítico", f"±{z_critico:.4f}" if tipo_cola == "Bilateral (Dos colas)" else f"{z_critico:.4f}")
    with col_r4:
        st.metric("p-valor", f"{p_valor_z:.4f}")

    if rechazar:
        st.error(f"🚫 **Se RECHAZA H₀** | Z={z_calculado:.4f} cae en la región de rechazo | p={p_valor_z:.4f} < α={alpha}")
    else:
        st.success(f"✅ **No se rechaza H₀** | Z={z_calculado:.4f} no cae en la región de rechazo | p={p_valor_z:.4f} ≥ α={alpha}")

    st.write("### Fórmula aplicada:")
    st.latex(
        rf"Z = \frac{{\bar{{x}} - \mu_0}}{{\sigma / \sqrt{{n}}}} = "
        rf"\frac{{{x_barra:.4f} - {mu_0:.4f}}}{{{sigma:.4f} / \sqrt{{{n}}}}} = {z_calculado:.4f}"
    )

    st.write("### Gráfica de la Prueba Z")
    fig_z, ax_z = plt.subplots(figsize=(9, 4))
    x_z = np.linspace(-4, 4, 300)
    ax_z.plot(x_z, stats.norm.pdf(x_z), color=COLOR_CURVA, lw=2)

    if tipo_cola == "Bilateral (Dos colas)":
        x_rej_left = x_z[x_z <= -z_critico]
        ax_z.fill_between(x_rej_left, stats.norm.pdf(x_rej_left),
                          color=COLOR_RECHAZO, alpha=0.4, label=f"Región de rechazo (α/2={alpha/2})")
        x_rej_right = x_z[x_z >= z_critico]
        ax_z.fill_between(x_rej_right, stats.norm.pdf(x_rej_right),
                          color=COLOR_RECHAZO, alpha=0.4)
    elif tipo_cola == "Unilateral Superior":
        x_rej = x_z[x_z >= z_critico]
        ax_z.fill_between(x_rej, stats.norm.pdf(x_rej),
                          color=COLOR_RECHAZO, alpha=0.4, label=f"Región de rechazo (α={alpha})")
    else:
        x_rej = x_z[x_z <= z_critico]
        ax_z.fill_between(x_rej, stats.norm.pdf(x_rej),
                          color=COLOR_RECHAZO, alpha=0.4, label=f"Región de rechazo (α={alpha})")

    color_z_linea = COLOR_Z_RECHAZO if rechazar else COLOR_Z_OK
    ax_z.axvline(x=z_calculado, color=color_z_linea, lw=2, linestyle="--",
                 label=f"Z calculado = {z_calculado:.4f}")

    if tipo_cola == "Bilateral (Dos colas)":
        ax_z.axvline(x= z_critico, color=COLOR_CRITICO, lw=1.5, linestyle=":", label=f"Z crítico = ±{z_critico:.4f}")
        ax_z.axvline(x=-z_critico, color=COLOR_CRITICO, lw=1.5, linestyle=":")
    else:
        ax_z.axvline(x=z_critico, color=COLOR_CRITICO, lw=1.5, linestyle=":", label=f"Z crítico = {z_critico:.4f}")

    ax_z.set_title("Distribución Z Estándar con Región de Rechazo", fontsize=12, fontweight="bold")
    ax_z.set_xlabel("Valor Z", fontsize=10)
    ax_z.set_ylabel("Densidad de probabilidad", fontsize=10)
    ax_z.legend(fontsize=8)

    col_gz1, col_gz2, col_gz3 = st.columns([1, 3, 1])
    with col_gz2:
        st.pyplot(fig_z)

              # --- MÓDULO 6: INTERPRETACIÓN CON IA (GEMINI) ---
    st.write("---")
    st.header("🤖 Interpretación con Inteligencia Artificial")

    api_key_input = st.text_input("Introduce tu Gemini API Key:", type="password")

    if st.button("✨ Interpretar resultados con IA", type="primary"):
        st.write("🚀 Botón presionado - Empezando...")   # Primer debug visible

        if not api_key_input or api_key_input.strip() == "":
            st.error("❌ Debes pegar tu API Key")
        else:
            try:
                st.write("🔑 API Key detectada")   # Debug 2

                client = genai.Client(api_key=api_key_input.strip())

                st.write("✅ Cliente Gemini creado")   # Debug 3

                # Prompt más corto y seguro
                prompt = f"""
Analiza de forma clara y en español estos resultados estadísticos:

Variable: {variable}
Sesgo: {sesgo:.2f}
Curtosis: {curtosis:.2f}
Normalidad: {es_normal} (p={p_valor:.4f})
Outliers: {len(outliers)}
Prueba Z: {tipo_cola}
Z calculado: {z_calculado:.4f}
p-valor: {p_valor_z:.4f}
Decisión: {"Se rechaza H0" if rechazar else "No se rechaza H0"}

Explica qué significan estos números de manera sencilla.
"""

                st.write("📤 Enviando prompt a Gemini...")   # Debug 4

                with st.spinner("Gemini pensando (espera 5-10 segundos)..."):
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-lite",   # Modelo más estable actualmente
                        contents=prompt
                    )

                st.success("✅ ¡Respuesta recibida!")
                st.markdown("### 📝 Interpretación de Gemini")
                st.markdown(response.text)

            except Exception as e:
                st.error("❌ Error detectado")
                st.error(f"Tipo: {type(e).__name__}")
                st.error(f"Detalle: {str(e)}")