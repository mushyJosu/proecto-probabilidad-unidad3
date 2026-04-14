import streamlit as st        # Crea la interfaz web (botones, sliders, títulos).
import pandas as pd           # Manipula tablas de datos (como Excel en código).
import numpy as np            # Realiza cálculos matemáticos y genera datos aleatorios.
import matplotlib.pyplot as plt # El "lienzo" base para dibujar gráficas.
import seaborn as sns         # El "pincel" que hace que las gráficas se vean bonitas.
from scipy import stats       # La calculadora estadística (calcula Z, p-value, etc.).

st.set_page_config(page_title="Asistente Estadístico", layout="wide") # Define el nombre en la pestaña del navegador y usa todo el ancho de la pantalla.
st.title("📊 Aplicación Interactiva de Estadística y Probabilidad") # El encabezado principal que verá el usuario.

st.sidebar.header("Configuración de Datos") # Crea una sección en la barra izquierda.
opcion_datos = st.sidebar.radio("Selecciona origen de datos:", ("Subir CSV", "Generar Datos Sintéticos")) # Un interruptor para decidir si el usuario usa su archivo o datos de prueba.

df = None # Creamos una variable vacía para guardar los datos después.

if opcion_datos == "Subir CSV":
    archivo = st.file_uploader("Cargar archivo CSV", type=["csv"]) # Crea el botón de "Arrastrar y soltar".
    if archivo is not None:
        df = pd.read_csv(archivo) # Si el usuario sube algo, Pandas lo lee y lo convierte en tabla.

else:
    n_sintetico = st.sidebar.slider("Tamaño de muestra (n)", 30, 1000, 100) # Selector para cuántos números crear.
    mu_sintetico = st.sidebar.number_input("Media real", value=50.0) # Define el "centro" de tus datos falsos.
    sigma_sintetico = st.sidebar.number_input("Desviación estándar real", value=10.0) # Define qué tan "dispersos" están.
    
    if st.sidebar.button("Generar datos"):
        # Crea una lista de números que siguen una distribución normal perfecta.
        data = np.random.normal(mu_sintetico, sigma_sintetico, n_sintetico) 
        df = pd.DataFrame(data, columns=["Variable_Generada"]) # Los guarda en una tabla para que la app pueda usarlos.

if df is not None: # Si ya hay datos (por CSV o por generación)...
    st.write("### Vista previa de los datos")
    st.dataframe(df.head()) # Muestra las primeras 5 filas para confirmar que todo está bien.
    
    # Crea un menú desplegable para que el usuario elija qué columna quiere analizar.
    variable = st.selectbox("Selecciona la variable para analizar:", df.columns)
    
    # Extrae la columna elegida y borra celdas vacías para no romper los cálculos.
    datos_analizar = df[variable].dropna()

    # --- MÓDULO 2: VISUALIZACIÓN DE DISTRIBUCIONES ---
st.write("---") # Crea una línea divisoria visual en la app.
st.header("📈 Análisis Visual de la Distribución")

# Creamos dos columnas para que los gráficos aparezcan uno al lado del otro
col1, col2 = st.columns(2)

with col1:
    st.subheader("Histograma y KDE")
    fig_hist, ax_hist = plt.subplots() # Crea la figura base de Matplotlib.
    
    # sns.histplot dibuja las barras y la curva de densidad (kde=True)
    sns.histplot(datos_analizar, kde=True, ax=ax_hist, color="skyblue")
    
    ax_hist.set_title(f"Distribución de {variable}")
    st.pyplot(fig_hist) # Renderiza la gráfica en la interfaz de Streamlit.

with col2:
    st.subheader("Boxplot (Diagrama de Caja)")
    fig_box, ax_box = plt.subplots()
    
    # El boxplot muestra la mediana, los cuartiles y puntos fuera de los bigotes (outliers)
    sns.boxplot(x=datos_analizar, ax=ax_box, color="lightgreen")
    
    ax_box.set_title(f"Boxplot de {variable}")
    st.pyplot(fig_box)

# --- MÓDULO 3: ESTADÍSTICAS DESCRIPTIVAS Y PRUEBAS ---
st.write("---")
st.header("📊 Análisis Estadístico Detallado")

# Cálculo de métricas clave usando SciPy y Pandas
sesgo = datos_analizar.skew()
curtosis = datos_analizar.kurtosis()

# Prueba de Normalidad de Shapiro-Wilk
# estadistico: qué tan cerca está de la normal / p_valor: probabilidad de error
estadistico, p_valor = stats.shapiro(datos_analizar)

col_met1, col_met2, col_met3 = st.columns(3)

with col_met1:
    st.metric("Sesgo (Skewness)", f"{sesgo:.2f}")
    st.caption("0 = Simétrico. (+) = Derecha, (-) = Izquierda")

with col_met2:
    st.metric("Curtosis", f"{curtosis:.2f}")
    st.caption(">0 = Puntiaguda, <0 = Plana")

with col_met3:
    # Interpretación automática de la normalidad (Alpha = 0.05)
    es_normal = "Sí ✅" if p_valor > 0.05 else "No ❌"
    st.metric("¿Es Normal?", es_normal)
    st.caption(f"P-Valor: {p_valor:.4f}")

# Detección de Outliers (Valores Atípicos) usando el Rango Intercuartílico (IQR)
Q1 = datos_analizar.quantile(0.25)
Q3 = datos_analizar.quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = datos_analizar[(datos_analizar < limite_inferior) | (datos_analizar > limite_superior)]

st.write(f"**Detección de Outliers:** Se han encontrado **{len(outliers)}** valores atípicos.")
if len(outliers) > 0:
    st.write(outliers) # Muestra la lista de los valores locos si existen

# --- MÓDULO 4: CÁLCULO DE PROBABILIDADES (DISTRIBUCIÓN NORMAL) ---
st.write("---")
st.header("🎲 Calculadora de Probabilidades")
st.write("Calcula la probabilidad acumulada $P(X \leq x)$ basándose en la media y desviación de tus datos.")

# Usamos la media y desviación real de tus datos actuales
mu_calc = datos_analizar.mean()
sigma_calc = datos_analizar.std()

# Entrada del usuario para el valor a evaluar
x_value = st.number_input(f"Ingresa un valor para calcular P(X ≤ x):", 
                          value=float(mu_calc), 
                          step=0.1)

# Cálculo de la Probabilidad usando la Función de Distribución Acumulada (CDF)
probabilidad = stats.norm.cdf(x_value, mu_calc, sigma_calc)

st.success(f"La probabilidad de que un valor sea menor o igual a **{x_value}** es del **{probabilidad*100:.2f}%**")

# --- Gráfico de Probabilidad Sombreada (Tamaño Mediano) ---
# Usamos figsize=(7, 4) para que sea más compacta y no tan alta
fig_prob, ax_prob = plt.subplots(figsize=(7, 4)) 

x_axis = np.linspace(mu_calc - 4*sigma_calc, mu_calc + 4*sigma_calc, 100)
y_axis = stats.norm.pdf(x_axis, mu_calc, sigma_calc)

ax_prob.plot(x_axis, y_axis, color='black', lw=2, label='Curva Normal')

# Sombreado del área de probabilidad
x_fill = np.linspace(mu_calc - 4*sigma_calc, x_value, 100)
y_fill = stats.norm.pdf(x_fill, mu_calc, sigma_calc)
ax_prob.fill_between(x_fill, y_fill, color='orange', alpha=0.5, label='Área de Probabilidad')

ax_prob.set_title(f"Área bajo la curva para X ≤ {x_value}", fontsize=10)
ax_prob.legend(fontsize=8)
ax_prob.tick_params(labelsize=8) # Hace los números de los ejes más pequeños

# Centrar la gráfica en Streamlit usando columnas
col_graf1, col_graf2, col_graf3 = st.columns([1, 2, 1])
with col_graf2:
    st.pyplot(fig_prob)