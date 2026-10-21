"""
Módulo de Estimación de Emisiones GEI - Sector 4.B: Tratamiento Biológico de Residuos Sólidos
=============================================================================================

Descripción Técnica:
    Herramienta interactiva para el cálculo, análisis y calibración de las emisiones
    de Gases de Efecto Invernadero (GEI) producidas por el tratamiento biológico
    de residuos sólidos urbanos (compostaje y digestión anaerobia), siguiendo las
    directrices del IPCC 2006 (Volumen 5: Residuos, Capítulo 4).

Marco del Proyecto:
    Trabajo de Fin de Grado (TFG) - Grado en Ingeniería Electrónica Industrial y Automática.
    Escuela de Ingenierías Industriales (EII), Universidad de Valladolid (UVa).
    Desarrollado para cubrir el balance del sector residuos en el modelo de evaluación
    integrada WILIAM (Horizon 2020 LOCOMOTION, Grant Agreement No. 821105).

Metodología Implementada:
    - Nivel Metodológico: IPCC 2006 Tier 1 estricto con factores de emisión por defecto.
    - Balance de Masa: Población x Generación municipal per cápita x Fracción desviada a tratamiento.
    - Gases Cuantificados: Metano (CH4) y Óxido Nitroso (N2O) en base húmeda.
    - Algoritmo de Calibración: Optimización numérica basada en el Método de Powell
      (scipy.optimize.minimize) sobre la suma de errores cuadráticos (SSE) frente a inventarios.

Fuentes de Datos Principales:
    - Emisiones históricas: Base de datos EDGAR v8.0 (1970-2023) / Inventarios Level 2 IPCC.
    - Demografía: Censos oficiales Eurostat y Banco Mundial (World Development Indicators).
    - Residuos per cápita: Estadísticas oficiales municipales Eurostat / OECD.

Autor:
    Javier Hernández Cepeda
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PARÁMETROS DE CONFIGURACIÓN Y CONSTANTES IPCC TIER 1 (SECTOR 4.B)
# =============================================================================
st.set_page_config(
    page_title="Simulador y Calibrador IPCC Tier 1 - Sector 4B",
    layout="wide",
    initial_sidebar_state="expanded"
)

AÑOS_ANALISIS = np.arange(2005, 2024)
LONGITUD_ANIOS = len(AÑOS_ANALISIS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_EMIS = os.path.join(BASE_DIR, 'data', 'HISTORICAL_GHG_EMISSIONS_REGIONS_IPCC_SECTORS_LEVEL_2.csv')
PATH_POP = os.path.join(BASE_DIR, 'data', 'Population_Regions.csv')
PATH_MSW = os.path.join(BASE_DIR, 'data', 'HISTORICAL_MUNICIPAL_WASTE_PER_CAPITA.csv')

LISTA_REGIONES = [
    'AUSTRIA', 'BELGIUM', 'BULGARIA', 'CROATIA', 'CYPRUS', 'CZECH_REPUBLIC', 'DENMARK',
    'ESTONIA', 'FINLAND', 'FRANCE', 'GERMANY', 'GREECE', 'HUNGARY', 'IRELAND', 'ITALY',
    'LATVIA', 'LITHUANIA', 'LUXEMBOURG', 'MALTA', 'NETHERLANDS', 'POLAND', 'PORTUGAL',
    'ROMANIA', 'SLOVAKIA', 'SLOVENIA', 'SPAIN', 'SWEDEN', 'EU27', 'CHINA', 'EASOC',
    'INDIA', 'LATAM', 'RUSSIA', 'USMCA', 'LROW', 'UK'
]

EU27_COUNTRIES = [
    'AUSTRIA', 'BELGIUM', 'BULGARIA', 'CROATIA', 'CYPRUS', 'CZECH_REPUBLIC', 'DENMARK',
    'ESTONIA', 'FINLAND', 'FRANCE', 'GERMANY', 'GREECE', 'HUNGARY', 'IRELAND', 'ITALY',
    'LATVIA', 'LITHUANIA', 'LUXEMBOURG', 'MALTA', 'NETHERLANDS', 'POLAND', 'PORTUGAL',
    'ROMANIA', 'SLOVAKIA', 'SLOVENIA', 'SPAIN', 'SWEDEN'
]

FRACCION_COMPOSTAJE_IPCC = {
    'AUSTRIA': 0.32, 'BELGIUM': 0.21, 'DENMARK': 0.18, 'ESTONIA': 0.08, 'FINLAND': 0.13,
    'FRANCE': 0.17, 'GERMANY': 0.17, 'IRELAND': 0.04, 'LATVIA': 0.01, 'LITHUANIA': 0.02,
    'LUXEMBOURG': 0.19, 'MALTA': 0.025, 'NETHERLANDS': 0.24, 'PORTUGAL': 0.07, 'SPAIN': 0.12,
    'BULGARIA': 0.04, 'CZECH_REPUBLIC': 0.04, 'HUNGARY': 0.04, 'POLAND': 0.04, 'ROMANIA': 0.04,
    'SLOVAKIA': 0.04, 'RUSSIA': 0.04, 'CROATIA': 0.03, 'CYPRUS': 0.03, 'GREECE': 0.03,
    'ITALY': 0.03, 'SLOVENIA': 0.03, 'SWEDEN': 0.09, 'UK': 0.09, 'EU27': 0.15, 'CHINA': 0.025,
    'EASOC': 0.025, 'INDIA': 0.01, 'LATAM': 0.025, 'USMCA': 0.13, 'LROW': 0.01
}

# Factores de emisión por defecto IPCC 2006 (Vol. 5, Cap. 4, Tabla 4.1 en g/kg residuo en base húmeda)
EF_COMP_CH4_DEF = 4.0
EF_COMP_N2O_DEF = 0.24
EF_ANA_CH4 = 0.8
EF_ANA_N2O = 0.0


# =============================================================================
# 2. CARGA DE BASES DE DATOS Y EXTRACCIÓN TEMPORAL (2005-2023)
# =============================================================================
@st.cache_data
def load_all_data():
    """
    Carga y almacena en caché los conjuntos de datos históricos estructurados del Sector 4.B.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tupla que contiene:
            - df_emis: Matriz de emisiones históricas EDGAR/IPCC Level 2.
            - df_pop: Series temporales de población regional (2005-2023).
            - df_msw: Series temporales de generación municipal de residuos per cápita (kg/hab/año).
    """
    df_emis = pd.read_csv(PATH_EMIS, header=None, skiprows=1)
    df_pop = pd.read_csv(PATH_POP)
    df_msw = pd.read_csv(PATH_MSW, header=None)
    return df_emis, df_pop, df_msw

def get_target_4B(df_emis, region, gas):
    """
    Extrae el vector de emisiones históricas declaradas para una región y gas específicos.

    Args:
        df_emis (pd.DataFrame): DataFrame con el inventario de emisiones históricas.
        region (str): Código o nombre identificador de la región analizada (e.g., 'SPAIN', 'EU27').
        gas (str): Denominación del gas trazador ('CH4' o 'N2O').

    Returns:
        np.ndarray: Vector unidimensional de emisiones observadas (Gg gas / año)
            durante el periodo de análisis 2005-2023. Devuelve ceros si no hay datos.
    """
    mask = (df_emis.iloc[:, 0].astype(str).str.strip() == region.strip()) & \
           (df_emis.iloc[:, 1].astype(str).str.strip() == '4_B__Biological_Treatment_of_Solid_Waste') & \
           (df_emis.iloc[:, 2].astype(str).str.strip() == gas.strip())
    fila = df_emis[mask]
    if fila.empty:
        return np.zeros(LONGITUD_ANIOS)
    target = fila.iloc[0, 3:3+LONGITUD_ANIOS].values.astype(float)
    return np.nan_to_num(target, nan=0.0)

def get_msw_mass_Gg(df_pop, df_msw, region):
    """
    Calcula la masa total anual de residuos sólidos urbanos generada en una región geográfica.

    Fundamento Matemático:
        M_total(t) = [Población(t) · Generación_pc(t)] · 10⁻⁶
        donde Población se expresa en habitantes, Generación_pc en kg/hab/año y M_total en Gg/año.
        Para la región agregada 'EU27', realiza la suma demográfica de los 27 Estados Miembros.

    Args:
        df_pop (pd.DataFrame): Matriz con las series demográficas por país/región.
        df_msw (pd.DataFrame): Matriz de generación per cápita municipal.
        region (str): Región o país de análisis.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tupla con:
            - total_Gg: Masa total generada anual (Gg residuo / año).
            - pop_array: Serie temporal de población (habitantes).
            - msw_array: Serie temporal de generación de residuos per cápita (kg/hab/año).
    """
    try:
        region_clean = region.strip()
        pop_array = np.zeros(LONGITUD_ANIOS)
        
        # 1. Extracción de población para el periodo 2005-2023
        if region_clean == 'EU27':
            mask_pop = df_pop.iloc[:, 0].astype(str).str.strip().isin(EU27_COUNTRIES)
            pop_df = df_pop[mask_pop]
            if pop_df.empty:
                return np.zeros(LONGITUD_ANIOS), np.zeros(LONGITUD_ANIOS), np.zeros(LONGITUD_ANIOS)
            for i, year in enumerate(AÑOS_ANALISIS):
                year_str, year_int = str(year), int(year)
                suma_poblacion = 0.0
                for _, row in pop_df.iterrows():
                    if year_str in row.index: suma_poblacion += float(row[year_str])
                    elif year_int in row.index: suma_poblacion += float(row[year_int])
                pop_array[i] = suma_poblacion
        else:
            pop_df = df_pop[df_pop.iloc[:, 0].astype(str).str.strip() == region_clean]
            if pop_df.empty:
                return np.zeros(LONGITUD_ANIOS), np.zeros(LONGITUD_ANIOS), np.zeros(LONGITUD_ANIOS)
            pop_series = pop_df.iloc[0]
            for i, year in enumerate(AÑOS_ANALISIS):
                year_str, year_int = str(year), int(year)
                pop_val = float(pop_series[year_str]) if year_str in pop_series.index else (float(pop_series[year_int]) if year_int in pop_series.index else 0.0)
                pop_array[i] = pop_val

        # 2. Extracción de residuos per cápita (2005-2023)
        msw_df = df_msw[df_msw.iloc[:, 0].astype(str).str.strip() == region_clean]
        if msw_df.empty:
            return np.zeros(LONGITUD_ANIOS), np.zeros(LONGITUD_ANIOS), np.zeros(LONGITUD_ANIOS)
        
        msw_series = msw_df.iloc[0]
        msw_valores = pd.to_numeric(msw_series.values[1:LONGITUD_ANIOS+1], errors='coerce')
        msw_array = np.nan_to_num(msw_valores, nan=0.0)
        if len(msw_array) < LONGITUD_ANIOS:
            msw_array = np.pad(msw_array, (0, LONGITUD_ANIOS - len(msw_array)), 'constant')
        
        # 3. Masa total generada (Gg) = [Población · (kg/hab/año)] / 10^6
        total_Gg = (pop_array * msw_array) / 1e6
        return np.nan_to_num(total_Gg, nan=0.0), pop_array, msw_array
        
    except Exception as e:
        st.error(f"Error procesando datos para {region}: {e}")
        return np.zeros(LONGITUD_ANIOS), np.zeros(LONGITUD_ANIOS), np.zeros(LONGITUD_ANIOS)


# =============================================================================
# 3. MOTOR MATEMÁTICO IPCC TIER 1 Y OPTIMIZACIÓN NUMÉRICA (POWELL)
# =============================================================================
def simular_4B(M_compost_Gg, M_anaerobic_Gg, ef_ch4, ef_n2o):
    """
    Ejecuta el balance de masa de emisiones según las Directrices IPCC 2006 (Tier 1).

    Fundamento Matemático (IPCC 2006 Vol. 5, Cap. 4, Ec. 4.1 y 4.2):
        E_CH4(t) = [M_comp(t) · EF_comp,CH4 + M_anaerob(t) · EF_anaerob,CH4] · 10⁻³
        E_N2O(t) = [M_comp(t) · EF_comp,N2O + M_anaerob(t) · EF_anaerob,N2O] · 10⁻³

    Args:
        M_compost_Gg (np.ndarray): Masa de residuo derivada a compostaje (Gg en base húmeda / año).
        M_anaerobic_Gg (np.ndarray): Masa derivada a digestión anaerobia (Gg en base húmeda / año).
        ef_ch4 (float): Factor de emisión de metano para compostaje (g CH4 / kg residuo húmedo).
        ef_n2o (float): Factor de emisión de óxido nitroso para compostaje (g N2O / kg residuo húmedo).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - ch4_total: Emisiones totales simuladas de metano (Gg CH4 / año).
            - n2o_total: Emisiones totales simuladas de óxido nitroso (Gg N2O / año).
    """
    ch4_total = (M_compost_Gg * ef_ch4 * 1e-3) + (M_anaerobic_Gg * EF_ANA_CH4 * 1e-3)
    n2o_total = (M_compost_Gg * ef_n2o * 1e-3) + (M_anaerobic_Gg * EF_ANA_N2O * 1e-3)
    return ch4_total, n2o_total

def funcion_objetivo(params, M_compost_Gg, M_anaerobic_Gg, target_ch4, target_n2o):
    """
    Función de pérdida escalar que evalúa la suma de errores cuadráticos (SSE) multigas.

    Fundamento Matemático:
        J(ef_CH4, ef_N2O) = ∑ (CH4_sim(t) - CH4_obs(t))² + ∑ (N2O_sim(t) - N2O_obs(t))²

    Args:
        params (list[float] | tuple[float, float]): Vector de parámetros [ef_ch4, ef_n2o].
        M_compost_Gg (np.ndarray): Masa enviada a compostaje (Gg / año).
        M_anaerobic_Gg (np.ndarray): Masa enviada a digestión anaerobia (Gg / año).
        target_ch4 (np.ndarray): Serie observada de emisiones de metano (Gg CH4 / año).
        target_n2o (np.ndarray): Serie observada de emisiones de óxido nitroso (Gg N2O / año).

    Returns:
        float: Valor escalar del error cuadrático combinado (Gg²).
    """
    ef_ch4, ef_n2o = params
    ch4_sim, n2o_sim = simular_4B(M_compost_Gg, M_anaerobic_Gg, ef_ch4, ef_n2o)
    error_ch4 = np.sum((ch4_sim - target_ch4) ** 2)
    error_n2o = np.sum((n2o_sim - target_n2o) ** 2)
    return error_ch4 + error_n2o


# =============================================================================
# 4. APLICACIÓN Y PANEL DE CONTROL INTERACTIVO (STREAMLIT)
# =============================================================================
def render_glosario_4b():
    """Renderiza el Glosario Técnico, Parámetros y Fundamento Normativo para el Sector 4.B (Tratamiento Biológico)."""
    with st.expander("📖 Glosario Técnico, Parámetros y Fundamento Normativo (IPCC 2006)", expanded=False):
        st.markdown(
            """
            ### Marco Metodológico y Normativo — Sector 4.B: Tratamiento Biológico de Residuos Sólidos (IPCC 2006 Vol. 5, Cap. 4)
            El cálculo de emisiones de metano ($\text{CH}_4$) y óxido nitroso ($\text{N}_2\text{O}$) en procesos biológicos
            se realiza conforme a la metodología **IPCC Tier 1**, discriminando entre compostaje aerobio (pilas y túneles) 
            y digestión anaerobia estanca. El módulo implementa la calibración numérica de los factores de emisión regionales mediante el **Método de Powell** frente a los inventarios de **EDGAR v8.0**.
            """
        )

        tab_act, tab_params, tab_metr = st.tabs([
            "📊 Variables de Entrada y Actividad",
            "⚙️ Parámetros del Modelo y Calibración (IPCC Vol. 5, Cap. 4)",
            "📈 Métricas de Evaluación y Salidas"
        ])

        with tab_act:
            st.markdown("#### 1. Variables de Actividad y Flujos de Masa")
            data_act = [
                {
                    "Variable": "Población regional",
                    "Símbolo": "P(t)",
                    "Unidades": "Habitantes",
                    "Fuente de Datos": "Eurostat / Banco Mundial (WDI)",
                    "Significado e Interpretación Técnica": "Censo demográfico anual de la región analizada bajo la partición territorial del modelo WILIAM."
                },
                {
                    "Variable": "Generación municipal per cápita",
                    "Símbolo": "MSW_pc(t)",
                    "Unidades": "kg / hab / año",
                    "Fuente de Datos": "Eurostat / OECD Environmental Stats",
                    "Significado e Interpretación Técnica": "Tasa anual per cápita de generación bruta de residuos sólidos urbanos municipales."
                },
                {
                    "Variable": "Masa total municipal generada",
                    "Símbolo": "M_total(t)",
                    "Unidades": "Gg / año",
                    "Fuente de Datos": "Calculado: P(t) · MSW_pc(t) · 10⁻⁶",
                    "Significado e Interpretación Técnica": "Masa total anual de residuo municipal generada en base húmeda en la región."
                },
                {
                    "Variable": "Fracción a compostaje aerobio",
                    "Símbolo": "ϕ_comp",
                    "Unidades": "Adimensional (0 - 1)",
                    "Fuente de Datos": "Eurostat / Directivas Marco de Residuos",
                    "Significado e Interpretación Técnica": "Porcentaje de residuo municipal derivado a plantas de compostaje en pilas o reactores aireados."
                },
                {
                    "Variable": "Fracción a digestión anaerobia",
                    "Símbolo": "ϕ_ana",
                    "Unidades": "Adimensional (0 - 1)",
                    "Fuente de Datos": "Estadísticas energéticas / Biogás",
                    "Significado e Interpretación Técnica": "Porcentaje de residuo orgánico derivado a biodigestores anaerobios herméticos para biometano."
                },
                {
                    "Variable": "Masa derivada a compostaje",
                    "Símbolo": "M_compost(t)",
                    "Unidades": "Gg / año",
                    "Fuente de Datos": "Calculado: M_total(t) · ϕ_comp",
                    "Significado e Interpretación Técnica": "Flujo másico anual en base húmeda sometido a tratamiento biológico aerobio."
                },
                {
                    "Variable": "Masa derivada a digestión anaerobia",
                    "Símbolo": "M_anaerobic(t)",
                    "Unidades": "Gg / año",
                    "Fuente de Datos": "Calculado: M_total(t) · ϕ_ana",
                    "Significado e Interpretación Técnica": "Flujo másico anual en base húmeda tratado en digestores anaerobios estancos."
                },
                {
                    "Variable": "Emisiones históricas observadas",
                    "Símbolo": "E_obs(t)",
                    "Unidades": "Gg CH4 / año y Gg N2O / año",
                    "Fuente de Datos": "EDGAR v8.0 / IPCC Sector 4_B",
                    "Significado e Interpretación Técnica": "Series de inventario de referencia para la calibración simultánea de los factores de emisión."
                }
            ]
            st.dataframe(pd.DataFrame(data_act), use_container_width=True, hide_index=True)
            st.caption("Nota metodológica: Las masas se expresan de forma estricta en Gigagramos (Gg = 1000 t) en base húmeda.")

        with tab_params:
            st.markdown("#### 2. Parámetros de Proceso y Factores de Emisión (IPCC Vol. 5, Cap. 4)")
            data_params = [
                {
                    "Parámetro": "Factor de emisión CH4 en compostaje",
                    "Símbolo": "EF_CH4,comp",
                    "Valor por Defecto IPCC": "4.0 g/kg",
                    "Rango de Calibración": "[0.5, 15.0] g/kg",
                    "Unidades": "g CH4 / kg residuo húmedo",
                    "Naturaleza": "Calibrado Powell 2D",
                    "Cita Normativa y Justificación Técnica": "Tabla 4.1 IPCC Vol. 5 Cap. 4. Cuantifica las emisiones fugitivas debidas a la formación de micro-bolsas anóxicas temporales y gradientes de difusión de oxígeno en pilas de compostaje aerobio."
                },
                {
                    "Parámetro": "Factor de emisión N2O en compostaje",
                    "Símbolo": "EF_N2O,comp",
                    "Valor por Defecto IPCC": "0.24 g/kg",
                    "Rango de Calibración": "[0.01, 1.50] g/kg",
                    "Unidades": "g N2O / kg residuo húmedo",
                    "Naturaleza": "Calibrado Powell 2D",
                    "Cita Normativa y Justificación Técnica": "Tabla 4.1 IPCC Vol. 5 Cap. 4. Emisiones intermedias procedentes de fases acopladas de nitrificación y desnitrificación durante la maduración y curado de la masa orgánica."
                },
                {
                    "Parámetro": "Factor de emisión CH4 en digestión anaerobia",
                    "Símbolo": "EF_CH4,ana",
                    "Valor por Defecto IPCC": "0.8 g/kg",
                    "Rango de Calibración": "Fijo (0.8 g/kg)",
                    "Unidades": "g CH4 / kg residuo húmedo",
                    "Naturaleza": "Estático IPCC Tier 1",
                    "Cita Normativa y Justificación Técnica": "Tabla 4.1 IPCC Vol. 5 Cap. 4. Digestores herméticos industriales; contempla únicamente fugas residuales menores en bridas, sellos mecánicos y almacenamiento abierto del digestato."
                },
                {
                    "Parámetro": "Factor de emisión N2O en digestión anaerobia",
                    "Símbolo": "EF_N2O,ana",
                    "Valor por Defecto IPCC": "0.0 g/kg",
                    "Rango de Calibración": "Fijo (0.0 g/kg)",
                    "Unidades": "g N2O / kg residuo húmedo",
                    "Naturaleza": "Estático IPCC Tier 1",
                    "Cita Normativa y Justificación Técnica": "Tabla 4.1 IPCC Vol. 5 Cap. 4. Condiciones de anoxia estricta en reactores estancos que impiden termodinámicamente la formación y desprendimiento de óxido nitroso."
                },
                {
                    "Parámetro": "Factor de conversión dimensional",
                    "Símbolo": "10⁻³",
                    "Valor por Defecto IPCC": "0.001",
                    "Rango de Calibración": "Constante",
                    "Unidades": "kg / g",
                    "Naturaleza": "Estático Dimensional",
                    "Cita Normativa y Justificación Técnica": "Conversión directa de gramos por kilogramo (g/kg) a Gigagramos totales (Gg)."
                }
            ]
            st.dataframe(pd.DataFrame(data_params), use_container_width=True, hide_index=True)

        with tab_metr:
            st.markdown("#### 3. Métricas de Evaluación y Salidas del Modelo")
            data_metr = [
                {
                    "Métrica / Salida": "Emisión anual de Metano",
                    "Símbolo": "E_CH4,4B(t)",
                    "Formulación Matemática": "[M_compost · EF_CH4,comp + M_anaerobic · EF_CH4,ana] · 10⁻³",
                    "Unidades": "Gg CH4 / año",
                    "Interpretación Ingenieril": "Emisiones directas de metano calculadas por balance de masa Tier 1 (Ecuación 4.1 del IPCC 2006)."
                },
                {
                    "Métrica / Salida": "Emisión anual de Óxido Nitroso",
                    "Símbolo": "E_N2O,4B(t)",
                    "Formulación Matemática": "[M_compost · EF_N2O,comp + M_anaerobic · EF_N2O,ana] · 10⁻³",
                    "Unidades": "Gg N2O / año",
                    "Interpretación Ingenieril": "Emisiones directas de óxido nitroso calculadas por balance de masa Tier 1 (Ecuación 4.2 del IPCC 2006)."
                },
                {
                    "Métrica / Salida": "Emisiones agregadas en CO2 equivalente",
                    "Símbolo": "E_Total,CO2-eq(t)",
                    "Formulación Matemática": "27.9 · E_CH4 + 273 · E_N2O",
                    "Unidades": "Gg CO2-eq / año",
                    "Interpretación Ingenieril": "Potencial de Calentamiento Global a 100 años bajo la métrica AR6 GWP100 estandarizada en WILIAM."
                },
                {
                    "Métrica / Salida": "Suma de Errores Cuadráticos",
                    "Símbolo": "SSE",
                    "Formulación Matemática": "∑ (CH4_sim - CH4_obs)² + ∑ (N2O_sim - N2O_obs)²",
                    "Unidades": "Gg²",
                    "Interpretación Ingenieril": "Función de coste combinada multigas minimizada por el algoritmo de Powell."
                },
                {
                    "Métrica / Salida": "Error Cuadrático Medio por Gas",
                    "Símbolo": "MSE",
                    "Formulación Matemática": "(1/N) ∑ (E_sim - E_obs)²",
                    "Unidades": "Gg²",
                    "Interpretación Ingenieril": "Dispersión media residual cuadrática reportada de forma individual para metano y óxido nitroso."
                }
            ]
            st.dataframe(pd.DataFrame(data_metr), use_container_width=True, hide_index=True)

def main():
    """
    Punto de entrada de la aplicación Streamlit para el Sector 4.B.
    Gestiona los elementos reactivos del panel lateral, la ejecución del optimizador
    de Powell, el renderizado de gráficos Plotly y las tablas de auditoría.
    """
    st.title("Simulador y Calibrador IPCC Tier 1 - Sector 4B (2005-2023)")
    st.markdown("Aplicación interactiva para el modelado y calibración mediante el **algoritmo de Powell** (mínimos cuadrados) de las emisiones del sector **4B** (Tratamiento Biológico de Residuos).")

    render_glosario_4b()

    df_emis, df_pop, df_msw = load_all_data()

    default_idx = LISTA_REGIONES.index('SPAIN') if 'SPAIN' in LISTA_REGIONES else 0
    region_seleccionada = st.sidebar.selectbox("Selecciona Región WILIAM:", LISTA_REGIONES, index=default_idx)
    
    fraccion_compost_def = FRACCION_COMPOSTAJE_IPCC.get(region_seleccionada, 0.05)
    fraccion_compost = st.sidebar.slider("Fracción a Compostaje", 0.0, 1.0, float(fraccion_compost_def), 0.01)
    fraccion_anaerobico = st.sidebar.slider("Fracción a Digestión Anaeróbica", 0.0, 1.0, 0.0, 0.01)

    total_msw_Gg, pop_array, msw_array = get_msw_mass_Gg(df_pop, df_msw, region_seleccionada)
    M_compost_Gg = total_msw_Gg * fraccion_compost
    M_anaerobic_Gg = total_msw_Gg * fraccion_anaerobico

    target_ch4 = get_target_4B(df_emis, region_seleccionada, 'CH4')
    target_n2o = get_target_4B(df_emis, region_seleccionada, 'N2O')

    if 'ef_ch4_calib' not in st.session_state:
        st.session_state['ef_ch4_calib'] = EF_COMP_CH4_DEF
    if 'ef_n2o_calib' not in st.session_state:
        st.session_state['ef_n2o_calib'] = EF_COMP_N2O_DEF
    if 'optimizado' not in st.session_state:
        st.session_state['optimizado'] = False

    st.sidebar.markdown("---")
    st.sidebar.header("Calibración del Modelo")
    if st.sidebar.button("Ejecutar Optimización (Powell)", type="primary"):
        params_iniciales = [EF_COMP_CH4_DEF, EF_COMP_N2O_DEF]
        bounds = [(0.5, 15.0), (0.01, 1.5)]
        
        resultado = minimize(
            funcion_objetivo,
            params_iniciales,
            args=(M_compost_Gg, M_anaerobic_Gg, target_ch4, target_n2o),
            method='Powell',
            bounds=bounds,
            options={'xtol': 1e-6, 'ftol': 1e-6}
        )
        
        if resultado.success:
            st.session_state['ef_ch4_calib'] = float(resultado.x[0])
            st.session_state['ef_n2o_calib'] = float(resultado.x[1])
            st.session_state['optimizado'] = True
            st.sidebar.success("¡Optimización completada con éxito!")
        else:
            st.sidebar.warning("La optimización no convergió completamente, pero se aplicaron los mejores resultados.")
            st.session_state['ef_ch4_calib'] = float(resultado.x[0])
            st.session_state['ef_n2o_calib'] = float(resultado.x[1])
            st.session_state['optimizado'] = True

    if st.sidebar.button("Restaurar Valores por Defecto (IPCC)"):
        st.session_state['ef_ch4_calib'] = EF_COMP_CH4_DEF
        st.session_state['ef_n2o_calib'] = EF_COMP_N2O_DEF
        st.session_state['optimizado'] = False
        st.sidebar.info("Valores restablecidos a los por defecto de IPCC Tier 1.")

    ch4_simulado, n2o_simulado = simular_4B(
        M_compost_Gg, M_anaerobic_Gg, 
        st.session_state['ef_ch4_calib'], 
        st.session_state['ef_n2o_calib']
    )

    mse_ch4 = np.mean((ch4_simulado - target_ch4) ** 2)
    mse_n2o = np.mean((n2o_simulado - target_n2o) ** 2)

    st.markdown(f"### Resultados del Análisis para: **{region_seleccionada}** (2005-2023)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("EF CH4 Compostaje", f"{st.session_state['ef_ch4_calib']:.3f} g/kg", delta="Calibrado" if st.session_state['optimizado'] else "IPCC Default")
    kpi2.metric("EF N2O Compostaje", f"{st.session_state['ef_n2o_calib']:.3f} g/kg", delta="Calibrado" if st.session_state['optimizado'] else "IPCC Default")
    kpi3.metric("MSE CH4", f"{mse_ch4:.4f}")
    kpi4.metric("MSE N2O", f"{mse_n2o:.4f}")

    col1, col2 = st.columns(2)
    
    with col1:
        fig_ch4 = go.Figure()
        fig_ch4.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=ch4_simulado, mode='lines+markers', name='CH4 Simulado', line=dict(color='#1f77b4', width=2)))
        fig_ch4.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=target_ch4, mode='lines+markers', name='CH4 Inventario Real', line=dict(color='#d62728', width=2, dash='dash')))
        fig_ch4.update_layout(
            title=f"Emisiones de Metano (CH4) - {region_seleccionada}",
            xaxis_title="Año",
            yaxis_title="Emisiones (Gg CH4)",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_ch4, use_container_width=True)

    with col2:
        fig_n2o = go.Figure()
        fig_n2o.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=n2o_simulado, mode='lines+markers', name='N2O Simulado', line=dict(color='#2ca02c', width=2)))
        fig_n2o.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=target_n2o, mode='lines+markers', name='N2O Inventario Real', line=dict(color='#ff7f0e', width=2, dash='dash')))
        fig_n2o.update_layout(
            title=f"Emisiones de Óxido Nitroso (N2O) - {region_seleccionada}",
            xaxis_title="Año",
            yaxis_title="Emisiones (Gg N2O)",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_n2o, use_container_width=True)

    st.markdown("### Tabla de Datos y Auditoría (2005-2023)")
    df_debug = pd.DataFrame({
        'Año': AÑOS_ANALISIS,
        'Población (Hab)': pop_array,
        'Residuos per Cápita (kg)': msw_array,
        'Masa Total Generada (Gg)': total_msw_Gg,
        'Masa Compostaje (Gg)': M_compost_Gg,
        'CH4 Simulado (Gg)': ch4_simulado,
        'CH4 Real (Gg)': target_ch4,
        'N2O Simulado (Gg)': n2o_simulado,
        'N2O Real (Gg)': target_n2o
    })
    st.dataframe(df_debug.style.format(precision=3), use_container_width=True)

if __name__ == "__main__":
    main()
