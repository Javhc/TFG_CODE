"""
Módulo de Estimación de Emisiones GEI - Sector 4.C: Incineración y Quema Abierta de Residuos
=============================================================================================

Descripción Técnica:
    Herramienta interactiva para la cuantificación, caracterización estequiométrica
    y calibración de las emisiones de Gases de Efecto Invernadero (CO2 fósil, CH4 y N2O)
    procedentes de la combustión térmica y quema incontrolada de residuos sólidos urbanos,
    conforme a las directrices del IPCC 2006 (Volumen 5: Residuos, Capítulo 5).

Marco del Proyecto:
    Trabajo de Fin de Grado (TFG) - Grado en Ingeniería Electrónica Industrial y Automática.
    Escuela de Ingenierías Industriales (EII), Universidad de Valladolid (UVa).
    Desarrollado para cerrar el balance de emisiones térmicas del sector residuos en el modelo
    de evaluación integrada WILIAM (Horizon 2020 LOCOMOTION, Grant Agreement No. 821105).

Metodología Implementada:
    - Nivel Metodológico: IPCC 2006 Tier 1 estricto con balance de carbono fósil.
    - Discriminación de Carbono: Separación estricta de emisiones neutras biogénicas vs fósiles.
    - Balance Estequiométrico: Conversión C -> CO2 mediante ratio molecular (44/12).
    - Factores de Oxidación (OF): 100% (OF = 1.0) para incineración y 71% (OF = 0.71) para quema abierta.
    - Algoritmo de Calibración: Método de Powell (scipy.optimize.minimize) sobre la suma de
      errores cuadráticos (SSE) frente a inventarios históricos internacionales.

Fuentes de Datos Principales:
    - Emisiones históricas: Base de datos EDGAR v8.0 (1970-2023) / Inventarios Level 2 IPCC.
    - Demografía: Series temporales Eurostat y Banco Mundial (World Development Indicators).
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
# 1. PARÁMETROS DE CONFIGURACIÓN Y CONSTANTES IPCC TIER 1 (SECTOR 4.C)
# =============================================================================
st.set_page_config(
    page_title="Simulador y Calibrador IPCC Tier 1 - Sector 4C",
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

# Parámetros y factores por defecto IPCC 2006 (Vol. 5, Cap. 5, Tablas 5.2 a 5.6)
DM_DEF = 0.80       # Fracción de materia seca en residuos municipales húmedos (80%)
CF_DEF = 0.50       # Fracción de carbono total en materia seca (50%)
FCF_DEF = 0.90      # Fracción de carbono fósil respecto al carbono total (90%)
OF_INC_DEF = 1.0    # Factor de oxidación completo en plantas de incineración controladas (100%)
OF_BURN_DEF = 0.71  # Factor de oxidación incompleta para quema incontrolada a cielo abierto (71%)

EF_CH4_INC_DEF = 5.0   # Factor de emisión de metano: 5.0 kg CH4 / Gg residuo incinerado
EF_N2O_INC_DEF = 0.1   # Factor de emisión de óxido nitroso: 0.1 kg N2O / Gg residuo incinerado


# =============================================================================
# 2. CARGA DE BASES DE DATOS Y EXTRACCIÓN TEMPORAL (2005-2023)
# =============================================================================
@st.cache_data
def load_all_data():
    """
    Carga y almacena en caché los conjuntos de datos históricos estructurados del Sector 4.C.

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

def get_target_4C(df_emis, region, gas):
    """
    Extrae el vector de emisiones históricas declaradas del inventario para una región y gas.

    Args:
        df_emis (pd.DataFrame): DataFrame con el inventario de emisiones históricas.
        region (str): Código o nombre identificador de la región analizada (e.g., 'SPAIN', 'EU27').
        gas (str): Denominación del gas trazador ('CO2', 'CH4' o 'N2O').

    Returns:
        np.ndarray: Vector de emisiones observadas (Gg gas / año) durante el periodo 2005-2023.
    """
    mask = (df_emis.iloc[:, 0].astype(str).str.strip() == region.strip()) & \
           (df_emis.iloc[:, 1].astype(str).str.strip() == '4_C__Incineration_and_Open_Burning_of_Waste') & \
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
        
        # 1. Extracción y agregación demográfica para el periodo 2005-2023
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
def simular_4C(M_inc_Gg, M_burn_Gg, fcf, ef_ch4, ef_n2o):
    """
    Ejecuta el balance de masa estequiométrico de combustión térmica (IPCC 2006 Tier 1).

    Fundamento Matemático (IPCC 2006 Vol. 5, Cap. 5, Ec. 5.1 y 5.2):
        CO2_inc(t)  = M_inc(t) · DM · CF · FCF · OF_inc · (44 / 12)
        CO2_burn(t) = M_burn(t) · DM · CF · FCF · OF_burn · (44 / 12)
        CO2_total(t) = CO2_inc(t) + CO2_burn(t)
        CH4_total(t) = [M_inc(t) · EF_ch4 + M_burn(t) · (EF_ch4 · 1.3)] · 10⁻³
        N2O_total(t) = [M_inc(t) · EF_n2o + M_burn(t) · (EF_n2o · 1.5)] · 10⁻³

    Args:
        M_inc_Gg (np.ndarray): Masa enviada a plantas de incineración (Gg / año).
        M_burn_Gg (np.ndarray): Masa sometida a quema a cielo abierto (Gg / año).
        fcf (float): Fracción de carbono fósil en la materia seca (adimensional).
        ef_ch4 (float): Factor de emisión de metano por incineración (kg CH4 / Gg residuo).
        ef_n2o (float): Factor de emisión de óxido nitroso por incineración (kg N2O / Gg residuo).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - co2_total: Emisiones de dióxido de carbono fósil (Gg CO2 / año).
            - ch4_total: Emisiones de metano (Gg CH4 / año).
            - n2o_total: Emisiones de óxido nitroso (Gg N2O / año).
    """
    co2_inc = M_inc_Gg * DM_DEF * CF_DEF * fcf * OF_INC_DEF * (44.0 / 12.0)
    co2_burn = M_burn_Gg * DM_DEF * CF_DEF * fcf * OF_BURN_DEF * (44.0 / 12.0)
    co2_total = co2_inc + co2_burn

    ch4_total = (M_inc_Gg * ef_ch4 * 1e-3) + (M_burn_Gg * (ef_ch4 * 1.3) * 1e-3)
    n2o_total = (M_inc_Gg * ef_n2o * 1e-3) + (M_burn_Gg * (ef_n2o * 1.5) * 1e-3)

    return co2_total, ch4_total, n2o_total

def funcion_objetivo(params, M_inc_Gg, M_burn_Gg, target_co2, target_ch4, target_n2o):
    """
    Función de pérdida escalar que evalúa la suma de errores cuadráticos (SSE) multigás.

    Fundamento Matemático:
        J(FCF, EF_CH4, EF_N2O) = ∑ (CO2_sim - CO2_obs)² + ∑ (CH4_sim - CH4_obs)² + ∑ (N2O_sim - N2O_obs)²

    Args:
        params (list[float] | tuple[float, float, float]): Vector de parámetros [fcf, ef_ch4, ef_n2o].
        M_inc_Gg (np.ndarray): Masa de residuo incinerado (Gg / año).
        M_burn_Gg (np.ndarray): Masa de residuo sometido a quema abierta (Gg / año).
        target_co2 (np.ndarray): Serie observada de CO2 fósil (Gg CO2 / año).
        target_ch4 (np.ndarray): Serie observada de CH4 (Gg CH4 / año).
        target_n2o (np.ndarray): Serie observada de N2O (Gg N2O / año).

    Returns:
        float: Error cuadrático escalar multigas ponderado (Gg²).
    """
    fcf, ef_ch4, ef_n2o = params
    co2_sim, ch4_sim, n2o_sim = simular_4C(M_inc_Gg, M_burn_Gg, fcf, ef_ch4, ef_n2o)
    
    error_co2 = np.sum((co2_sim - target_co2) ** 2)
    error_ch4 = np.sum((ch4_sim - target_ch4) ** 2)
    error_n2o = np.sum((n2o_sim - target_n2o) ** 2)
    return error_co2 + error_ch4 + error_n2o


# =============================================================================
# 4. APLICACIÓN Y PANEL DE CONTROL INTERACTIVO (STREAMLIT)
# =============================================================================
def render_glosario_4c():
    """Renderiza el Glosario Técnico, Parámetros y Fundamento Normativo para el Sector 4.C (Incineración y Quema Abierta)."""
    with st.expander("📖 Glosario Técnico, Parámetros y Fundamento Normativo (IPCC 2006)", expanded=False):
        st.markdown(
            """
            ### Marco Metodológico y Normativo — Sector 4.C: Incineración y Quema Abierta de Residuos (IPCC 2006 Vol. 5, Cap. 5)
            El balance de emisiones térmicas se rige por la estequiometría de combustión y la discriminación de carbono fósil según el **IPCC Tier 1**.
            En inventarios de GEI, el $\text{CO}_2$ de origen biogénico (madera, papel, materia orgánica) se considera de ciclo cerrado neutro;
            **únicamente el carbono de origen fósil (plásticos, fibras sintéticas, caucho) computa como emisión neta de GEI**.
            El módulo incluye la optimización tridimensional simultánea ($\text{FCF}$, $\text{EF}_{\text{CH}_4}$ y $\text{EF}_{\text{N}_2\text{O}}$) 
            mediante el **Método de Powell** frente a los inventarios de **EDGAR v8.0**.
            """
        )

        tab_act, tab_params, tab_metr = st.tabs([
            "📊 Variables de Entrada y Actividad",
            "⚙️ Parámetros Estequiométricos y Calibración (IPCC Vol. 5, Cap. 5)",
            "📈 Métricas de Evaluación y Salidas"
        ])

        with tab_act:
            st.markdown("#### 1. Variables de Actividad y Flujos Másicos Térmicos")
            data_act = [
                {
                    "Variable": "Población regional",
                    "Símbolo": "P(t)",
                    "Unidades": "Habitantes",
                    "Fuente de Datos": "Eurostat / Banco Mundial (WDI)",
                    "Significado e Interpretación Técnica": "Censo demográfico anual de la región analizada en la estructura de WILIAM."
                },
                {
                    "Variable": "Generación municipal per cápita",
                    "Símbolo": "MSW_pc(t)",
                    "Unidades": "kg / hab / año",
                    "Fuente de Datos": "Eurostat / OECD Environmental Stats",
                    "Significado e Interpretación Técnica": "Tasa anual per cápita de generación de residuos municipales urbanos."
                },
                {
                    "Variable": "Masa total municipal generada",
                    "Símbolo": "M_total(t)",
                    "Unidades": "Gg / año",
                    "Fuente de Datos": "Calculado: P(t) · MSW_pc(t) · 10⁻⁶",
                    "Significado e Interpretación Técnica": "Masa total anual de residuo urbano en base húmeda generada en la región."
                },
                {
                    "Variable": "Fracción a incineración controlada",
                    "Símbolo": "ϕ_inc",
                    "Unidades": "Adimensional (0 - 1)",
                    "Fuente de Datos": "Eurostat / Inventarios Nacionales",
                    "Significado e Interpretación Técnica": "Proporción de residuo destinada a plantas industriales de incineración o valorización energética (WtE)."
                },
                {
                    "Variable": "Fracción a quema abierta",
                    "Símbolo": "ϕ_burn",
                    "Unidades": "Adimensional (0 - 1)",
                    "Fuente de Datos": "Estimaciones de inventario / IPCC",
                    "Significado e Interpretación Técnica": "Proporción de residuo eliminada mediante combustión incontrolada al aire libre sin control de emisiones."
                },
                {
                    "Variable": "Masa incinerada en horno",
                    "Símbolo": "W_inc(t)",
                    "Unidades": "Gg / año",
                    "Fuente de Datos": "Calculado: M_total(t) · ϕ_inc",
                    "Significado e Interpretación Técnica": "Masa anual en base húmeda procesada en incineradores industriales."
                },
                {
                    "Variable": "Masa en quema a cielo abierto",
                    "Símbolo": "W_burn(t)",
                    "Unidades": "Gg / año",
                    "Fuente de Datos": "Calculado: M_total(t) · ϕ_burn",
                    "Significado e Interpretación Técnica": "Masa anual en base húmeda quemada de forma incontrolada en vertederos o zonas rurales."
                },
                {
                    "Variable": "Emisiones históricas observadas",
                    "Símbolo": "E_obs(t)",
                    "Unidades": "Gg CO2, CH4, N2O / año",
                    "Fuente de Datos": "EDGAR v8.0 / IPCC Sector 4_C",
                    "Significado e Interpretación Técnica": "Series temporales de inventario internacional de referencia para la calibración simultánea multigas."
                }
            ]
            st.dataframe(pd.DataFrame(data_act), use_container_width=True, hide_index=True)
            st.caption("Nota metodológica: Las masas se expresan de forma homogénea en Gigagramos (Gg = 1000 t) en base húmeda.")

        with tab_params:
            st.markdown("#### 2. Parámetros Estequiométricos de Combustión y Factores de Emisión (IPCC Vol. 5, Cap. 5)")
            data_params = [
                {
                    "Parámetro": "Fracción de materia seca",
                    "Símbolo": "DM",
                    "Valor por Defecto": "0.80",
                    "Rango de Calibración": "Fijo (0.80)",
                    "Unidades": "Adimensional (0 - 1)",
                    "Naturaleza": "Estático IPCC Tier 1",
                    "Cita Normativa y Justificación Técnica": "Tabla 5.2 IPCC Vol. 5 Cap. 5. Porcentaje medio de materia seca (Dry Matter) sobre residuo húmedo municipal (asume 20% de humedad media ponderada)."
                },
                {
                    "Parámetro": "Fracción de carbono total en materia seca",
                    "Símbolo": "CF",
                    "Valor por Defecto": "0.50",
                    "Rango de Calibración": "Fijo (0.50)",
                    "Unidades": "Adimensional (0 - 1)",
                    "Naturaleza": "Estático IPCC Tier 1",
                    "Cita Normativa y Justificación Técnica": "Tabla 5.2 IPCC Vol. 5 Cap. 5. Contenido elemental de carbono en base seca en la mezcla de residuos sólidos urbanos."
                },
                {
                    "Parámetro": "Fracción de carbono fósil",
                    "Símbolo": "FCF",
                    "Valor por Defecto": "0.90",
                    "Rango de Calibración": "[0.10, 1.00]",
                    "Unidades": "Adimensional (0 - 1)",
                    "Naturaleza": "Calibrado Powell 3D",
                    "Cita Normativa y Justificación Técnica": "Tabla 5.2 IPCC Vol. 5 Cap. 5. Proporción de carbono proveniente de materiales derivados del petróleo (plásticos, neumáticos, sintéticos). El CO2 biogénico es neutro en inventario; solo el carbono fósil (FCF) genera emisiones netas imputables de GEI."
                },
                {
                    "Parámetro": "Factor de oxidación en incineración",
                    "Símbolo": "OF_inc",
                    "Valor por Defecto": "1.00",
                    "Rango de Calibración": "Fijo (1.00)",
                    "Unidades": "Adimensional (0 - 1)",
                    "Naturaleza": "Estático IPCC Tier 1",
                    "Cita Normativa y Justificación Técnica": "Tabla 5.2 IPCC Vol. 5 Cap. 5. Eficiencia de combustión completa (100% oxidado a CO2) en hornos industriales a alta temperatura (>850 °C) con exceso de aire controlado."
                },
                {
                    "Parámetro": "Factor de oxidación en quema abierta",
                    "Símbolo": "OF_burn",
                    "Valor por Defecto": "0.71",
                    "Rango de Calibración": "Fijo (0.71)",
                    "Unidades": "Adimensional (0 - 1)",
                    "Naturaleza": "Estático IPCC Tier 1",
                    "Cita Normativa y Justificación Técnica": "Tabla 5.2 IPCC Vol. 5 Cap. 5. Eficiencia de combustión incompleta a baja temperatura y deficiencia de tiro en quema incontrolada (71% de carbono oxidado a CO2 y 29% inquemado o en cenizas)."
                },
                {
                    "Parámetro": "Razón molecular C -> CO2",
                    "Símbolo": "44/12",
                    "Valor por Defecto": "3.667",
                    "Rango de Calibración": "Fijo",
                    "Unidades": "kg CO2 / kg C",
                    "Naturaleza": "Estático Estequiométrico",
                    "Cita Normativa y Justificación Técnica": "Factor estequiométrico que convierte masa atómica de Carbono (12 g/mol) a masa molecular de Dióxido de Carbono (44 g/mol)."
                },
                {
                    "Parámetro": "Factor de emisión CH4 en incineración",
                    "Símbolo": "EF_CH4,inc",
                    "Valor por Defecto": "5.0 kg/Gg",
                    "Rango de Calibración": "[0.1, 50.0] kg/Gg",
                    "Unidades": "kg CH4 / Gg residuo húmedo",
                    "Naturaleza": "Calibrado Powell 3D",
                    "Cita Normativa y Justificación Técnica": "Tabla 5.3 IPCC Vol. 5 Cap. 5. Gases inquemados procedentes de pirólisis incompleta y zonas de llama con oxígeno subestequiométrico."
                },
                {
                    "Parámetro": "Factor de emisión N2O en incineración",
                    "Símbolo": "EF_N2O,inc",
                    "Valor por Defecto": "0.10 kg/Gg",
                    "Rango de Calibración": "[0.001, 2.00] kg/Gg",
                    "Unidades": "kg N2O / Gg residuo húmedo",
                    "Naturaleza": "Calibrado Powell 3D",
                    "Cita Normativa y Justificación Técnica": "Tabla 5.3 IPCC Vol. 5 Cap. 5. Formación térmica de óxido nitroso en la chimenea a temperaturas intermedias de los gases de escape (750-900 °C)."
                },
                {
                    "Parámetro": "Multiplicador de quema abierta para CH4",
                    "Símbolo": "× 1.3",
                    "Valor por Defecto": "1.30",
                    "Rango de Calibración": "Fijo (1.30)",
                    "Unidades": "Adimensional",
                    "Naturaleza": "Estático IPCC Tier 1",
                    "Cita Normativa y Justificación Técnica": "Tabla 5.3 IPCC Vol. 5 Cap. 5. Factor que incrementa un 30% las emisiones de metano en quema abierta debido a deficiencias de turbulencia y bajas temperaturas de llama."
                },
                {
                    "Parámetro": "Multiplicador de quema abierta para N2O",
                    "Símbolo": "× 1.5",
                    "Valor por Defecto": "1.50",
                    "Rango de Calibración": "Fijo (1.50)",
                    "Unidades": "Adimensional",
                    "Naturaleza": "Estático IPCC Tier 1",
                    "Cita Normativa y Justificación Técnica": "Tabla 5.3 IPCC Vol. 5 Cap. 5. Factor que incrementa un 50% las emisiones de N2O en quema abierta por combustión a baja temperatura de compuestos nitrogenados."
                }
            ]
            st.dataframe(pd.DataFrame(data_params), use_container_width=True, hide_index=True)

        with tab_metr:
            st.markdown("#### 3. Métricas de Evaluación y Salidas del Modelo")
            data_metr = [
                {
                    "Métrica / Salida": "Emisión de CO2 fósil",
                    "Símbolo": "E_CO2,4C(t)",
                    "Formulación Matemática": "[W_inc · OF_inc + W_burn · OF_burn] · DM · CF · FCF · (44/12)",
                    "Unidades": "Gg CO2 / año",
                    "Interpretación Ingenieril": "Emisiones netas de dióxido de carbono procedente de la fracción fósil del residuo (Ecuación 5.1 del IPCC 2006)."
                },
                {
                    "Métrica / Salida": "Emisión de Metano térmico",
                    "Símbolo": "E_CH4,4C(t)",
                    "Formulación Matemática": "[W_inc · EF_CH4,inc + W_burn · (1.3 · EF_CH4,inc)] · 10⁻³",
                    "Unidades": "Gg CH4 / año",
                    "Interpretación Ingenieril": "Emisiones de metano por combustión térmica y quema abierta (Ecuación 5.2 del IPCC 2006)."
                },
                {
                    "Métrica / Salida": "Emisión de Óxido Nitroso térmico",
                    "Símbolo": "E_N2O,4C(t)",
                    "Formulación Matemática": "[W_inc · EF_N2O,inc + W_burn · (1.5 · EF_N2O,inc)] · 10⁻³",
                    "Unidades": "Gg N2O / año",
                    "Interpretación Ingenieril": "Emisiones térmicas de N2O por combustión de compuestos nitrogenados (Ecuación 5.3 del IPCC 2006)."
                },
                {
                    "Métrica / Salida": "Emisiones agregadas en CO2 equivalente",
                    "Símbolo": "E_Total,CO2-eq(t)",
                    "Formulación Matemática": "E_CO2 + 27.9 · E_CH4 + 273 · E_N2O",
                    "Unidades": "Gg CO2-eq / año",
                    "Interpretación Ingenieril": "Balance global estandarizado bajo la métrica AR6 GWP100 para integración en WILIAM."
                },
                {
                    "Métrica / Salida": "Suma de Errores Cuadráticos",
                    "Símbolo": "SSE",
                    "Formulación Matemática": "∑ (CO2_sim - CO2_obs)² + ∑ (CH4_sim - CH4_obs)² + ∑ (N2O_sim - N2O_obs)²",
                    "Unidades": "Gg²",
                    "Interpretación Ingenieril": "Función de coste multigas minimizada en 3D (FCF, EF_CH4, EF_N2O) mediante el algoritmo de Powell."
                },
                {
                    "Métrica / Salida": "Error Cuadrático Medio por Gas",
                    "Símbolo": "MSE",
                    "Formulación Matemática": "(1/N) ∑ (E_sim - E_obs)²",
                    "Unidades": "Gg²",
                    "Interpretación Ingenieril": "Dispersión media residual reportada independientemente para CO2 fósil, CH4 y N2O."
                }
            ]
            st.dataframe(pd.DataFrame(data_metr), use_container_width=True, hide_index=True)

def main():
    """
    Punto de entrada de la aplicación Streamlit para el Sector 4.C.
    Gestiona los controles interactivos de fracciones de tratamiento, la ejecución
    del optimizador numérico de Powell, las curvas temporales y las métricas de ajuste.
    """
    st.title("Simulador y Calibrador IPCC Tier 1 - Sector 4C (2005-2023)")
    st.markdown("Aplicación interactiva para el modelado y calibración mediante el **algoritmo de Powell** (mínimos cuadrados) de las emisiones del sector **4C** (Incineración y Quema a Cielo Abierto de Residuos).")

    render_glosario_4c()

    df_emis, df_pop, df_msw = load_all_data()

    st.sidebar.header("Panel de Control")
    default_idx = LISTA_REGIONES.index('SPAIN') if 'SPAIN' in LISTA_REGIONES else 0
    region_seleccionada = st.sidebar.selectbox("Selecciona Región WILIAM:", LISTA_REGIONES, index=default_idx)
    
    fraccion_incineracion = st.sidebar.slider("Fracción a Incineración", 0.0, 1.0, 0.05, 0.01)
    fraccion_quema = st.sidebar.slider("Fracción a Quema a Cielo Abierto", 0.0, 1.0, 0.02, 0.01)

    total_msw_Gg, pop_array, msw_array = get_msw_mass_Gg(df_pop, df_msw, region_seleccionada)
    M_inc_Gg = total_msw_Gg * fraccion_incineracion
    M_burn_Gg = total_msw_Gg * fraccion_quema

    target_co2 = get_target_4C(df_emis, region_seleccionada, 'CO2')
    target_ch4 = get_target_4C(df_emis, region_seleccionada, 'CH4')
    target_n2o = get_target_4C(df_emis, region_seleccionada, 'N2O')

    if 'fcf_calib' not in st.session_state:
        st.session_state['fcf_calib'] = FCF_DEF
    if 'ef_ch4_calib' not in st.session_state:
        st.session_state['ef_ch4_calib'] = EF_CH4_INC_DEF
    if 'ef_n2o_calib' not in st.session_state:
        st.session_state['ef_n2o_calib'] = EF_N2O_INC_DEF
    if 'optimizado' not in st.session_state:
        st.session_state['optimizado'] = False

    st.sidebar.markdown("---")
    st.sidebar.header("Calibración del Modelo")
    if st.sidebar.button("Ejecutar Optimización (Powell)", type="primary"):
        params_iniciales = [FCF_DEF, EF_CH4_INC_DEF, EF_N2O_INC_DEF]
        bounds = [(0.1, 1.0), (0.1, 50.0), (0.001, 2.0)]
        
        resultado = minimize(
            funcion_objetivo,
            params_iniciales,
            args=(M_inc_Gg, M_burn_Gg, target_co2, target_ch4, target_n2o),
            method='Powell',
            bounds=bounds,
            options={'xtol': 1e-6, 'ftol': 1e-6}
        )
        
        if resultado.success:
            st.session_state['fcf_calib'] = float(resultado.x[0])
            st.session_state['ef_ch4_calib'] = float(resultado.x[1])
            st.session_state['ef_n2o_calib'] = float(resultado.x[2])
            st.session_state['optimizado'] = True
            st.sidebar.success("¡Optimización completada con éxito!")
        else:
            st.sidebar.warning("Optimización finalizada con convergencia parcial.")
            st.session_state['fcf_calib'] = float(resultado.x[0])
            st.session_state['ef_ch4_calib'] = float(resultado.x[1])
            st.session_state['ef_n2o_calib'] = float(resultado.x[2])
            st.session_state['optimizado'] = True

    if st.sidebar.button("Restaurar Valores por Defecto (IPCC)"):
        st.session_state['fcf_calib'] = FCF_DEF
        st.session_state['ef_ch4_calib'] = EF_CH4_INC_DEF
        st.session_state['ef_n2o_calib'] = EF_N2O_INC_DEF
        st.session_state['optimizado'] = False
        st.sidebar.info("Valores restablecidos a los por defecto de IPCC Tier 1.")

    co2_simulado, ch4_simulado, n2o_simulado = simular_4C(
        M_inc_Gg, M_burn_Gg, 
        st.session_state['fcf_calib'], 
        st.session_state['ef_ch4_calib'],
        st.session_state['ef_n2o_calib']
    )

    mse_co2 = np.mean((co2_simulado - target_co2) ** 2)
    mse_ch4 = np.mean((ch4_simulado - target_ch4) ** 2)
    mse_n2o = np.mean((n2o_simulado - target_n2o) ** 2)

    st.markdown(f"### Resultados del Análisis para: **{region_seleccionada}** (2005-2023)")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Fracción C. Fósil (FCF)", f"{st.session_state['fcf_calib']:.3f}", delta="Calibrado" if st.session_state['optimizado'] else "IPCC Default")
    kpi2.metric("EF CH4", f"{st.session_state['ef_ch4_calib']:.3f} kg/Gg", delta="Calibrado" if st.session_state['optimizado'] else "IPCC Default")
    kpi3.metric("MSE CO2", f"{mse_co2:.4f}")
    kpi4.metric("MSE CH4", f"{mse_ch4:.4f}")
    kpi5.metric("MSE N2O", f"{mse_n2o:.4f}")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_co2 = go.Figure()
        fig_co2.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=co2_simulado, mode='lines+markers', name='CO2 Simulado', line=dict(color='#1f77b4', width=2)))
        fig_co2.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=target_co2, mode='lines+markers', name='CO2 Inventario Real', line=dict(color='#d62728', width=2, dash='dash')))
        fig_co2.update_layout(title=f"CO2 (Fósil) - {region_seleccionada}", xaxis_title="Año", yaxis_title="Emisiones (Gg CO2)", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_co2, use_container_width=True)

    with col2:
        fig_ch4 = go.Figure()
        fig_ch4.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=ch4_simulado, mode='lines+markers', name='CH4 Simulado', line=dict(color='#2ca02c', width=2)))
        fig_ch4.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=target_ch4, mode='lines+markers', name='CH4 Inventario Real', line=dict(color='#ff7f0e', width=2, dash='dash')))
        fig_ch4.update_layout(title=f"CH4 - {region_seleccionada}", xaxis_title="Año", yaxis_title="Emisiones (Gg CH4)", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_ch4, use_container_width=True)

    with col3:
        fig_n2o = go.Figure()
        fig_n2o.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=n2o_simulado, mode='lines+markers', name='N2O Simulado', line=dict(color='#9467bd', width=2)))
        fig_n2o.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=target_n2o, mode='lines+markers', name='N2O Inventario Real', line=dict(color='#8c564b', width=2, dash='dash')))
        fig_n2o.update_layout(title=f"N2O - {region_seleccionada}", xaxis_title="Año", yaxis_title="Emisiones (Gg N2O)", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_n2o, use_container_width=True)

    st.markdown("### Tabla de Datos y Auditoría (2005-2023)")
    df_debug = pd.DataFrame({
        'Año': AÑOS_ANALISIS,
        'Población (Hab)': pop_array,
        'Residuos per Cápita (kg)': msw_array,
        'Masa Total Generada (Gg)': total_msw_Gg,
        'Masa Incineración (Gg)': M_inc_Gg,
        'Masa Quema (Gg)': M_burn_Gg,
        'CO2 Simulado (Gg)': co2_simulado,
        'CO2 Real (Gg)': target_co2,
        'CH4 Simulado (Gg)': ch4_simulado,
        'CH4 Real (Gg)': target_ch4,
        'N2O Simulado (Gg)': n2o_simulado,
        'N2O Real (Gg)': target_n2o
    })
    st.dataframe(df_debug.style.format(precision=3), use_container_width=True)

if __name__ == "__main__":
    main()
