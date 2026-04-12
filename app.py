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