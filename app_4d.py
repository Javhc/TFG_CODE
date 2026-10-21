"""
Módulo de Estimación de Emisiones GEI - Sector 4.D: Tratamiento y Descarga de Aguas Residuales
=============================================================================================

Descripción Técnica:
    Herramienta interactiva para la cuantificación, caracterización estequiométrica
    y calibración de las emisiones de Gases de Efecto Invernadero (metano, CH4, y óxido
    nitroso, N2O) derivadas de los sistemas de alcantarillado, depuración y vertido final
    de aguas residuales domésticas e industriales, siguiendo las directrices del IPCC 2006
    (Volumen 5: Residuos, Capítulo 6).

Marco del Proyecto:
    Trabajo de Fin de Grado (TFG) - Grado en Ingeniería Electrónica Industrial y Automática.
    Escuela de Ingenierías Industriales (EII), Universidad de Valladolid (UVa).
    Desarrollado para cerrar el balance de saneamiento y efluentes del sector residuos
    en el modelo de evaluación integrada WILIAM (Horizon 2020 LOCOMOTION, Grant Agreement No. 821105).

Metodología Implementada:
    - Nivel Metodológico: IPCC 2006 Tier 1 estricto con factores de emisión por defecto.
    - Carga Orgánica Degradable (TOW): Basada en Demanda Bioquímica de Oxígeno per cápita (BOD_pc).
    - Capacidad Máxima de Producción de Metano: B0 = 0.6 kg CH4 / kg DBO.
    - Factor de Corrección de Metano (MCF): Representativo de la vía tecnológica y condiciones anaerobias.
    - Emisiones de N2O en Efluentes: Basadas en consumo de proteínas per cápita, factores de residuo
      comercial/industrial y factor de emisión en aguas receptoras.
    - Algoritmo de Calibración: Método de Powell (scipy.optimize.minimize) sobre la suma de
      errores cuadráticos (SSE) frente a series históricas observadas.

Fuentes de Datos Principales:
    - Emisiones históricas: Base de datos EDGAR v8.0 (1970-2023) / Inventarios Level 2 IPCC.
    - Demografía: Censos oficiales Eurostat y Banco Mundial (World Development Indicators).
    - Consumo proteico y alimentos: Balances de disponibilidad alimentaria de FAOSTAT / Food_Regions.

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
# 1. PARÁMETROS DE CONFIGURACIÓN Y CONSTANTES IPCC TIER 1 (SECTOR 4.D)
# =============================================================================
st.set_page_config(
    page_title="Simulador y Calibrador IPCC Tier 1 - Sector 4D (Wastewater)",
    layout="wide",
    initial_sidebar_state="expanded"
)

AÑOS_ANALISIS = np.arange(2005, 2024)
LONGITUD_ANIOS = len(AÑOS_ANALISIS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_EMIS = os.path.join(BASE_DIR, 'data', 'HISTORICAL_GHG_EMISSIONS_REGIONS_IPCC_SECTORS_LEVEL_2.csv')
PATH_POP = os.path.join(BASE_DIR, 'data', 'Population_Regions.csv')
PATH_FOOD = os.path.join(BASE_DIR, 'data', 'Food_Regions.csv')

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

# Parámetros y factores por defecto IPCC 2006 (Vol. 5, Cap. 6, Tablas 6.1 a 6.11)
B0_DEF = 0.6          # Capacidad máxima teórica de producción: 0.6 kg CH4 / kg DBO
MCF_DEF = 0.3         # Factor de Corrección de Metano por defecto (sistemas mixtos / fosas / descargas)
EF_N2O_DEF = 0.005    # Factor de emisión en efluentes acuáticos: 0.005 kg N2O-N / kg N
BOD_PC_DEF = 21.9     # Demanda Bioquímica de Oxígeno per cápita: 21.9 kg DBO / hab / año (~60 g/hab/día)
PROTEIN_PC_DEF = 35.0 # Consumo de proteína per cápita: 35.0 kg proteína / hab / año (~95 g/hab/día)
F_NPR = 0.16           # Fracción de nitrógeno en proteínas: 0.16 kg N / kg proteína
F_NON_CON = 1.04      # Factor corrector por nitrógeno no consumido vertido al desagüe (1.04)
F_IND_COM = 1.25      # Factor de contribución de descargas comerciales e industriales (1.25)


# =============================================================================
# 2. CARGA DE BASES DE DATOS Y EXTRACCIÓN TEMPORAL (2005-2023)
# =============================================================================
@st.cache_data
def load_all_data():
    """
    Carga y almacena en caché los conjuntos de datos históricos estructurados del Sector 4.D.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tupla que contiene:
            - df_emis: Matriz de emisiones históricas EDGAR/IPCC Level 2.
            - df_pop: Series temporales de población regional (2005-2023).
            - df_food: Series de disponibilidad alimentaria y consumo proteico regional.
    """
    df_emis = pd.read_csv(PATH_EMIS, header=None, skiprows=1)
    df_pop = pd.read_csv(PATH_POP)
    df_food = pd.read_csv(PATH_FOOD) if os.path.exists(PATH_FOOD) else pd.DataFrame()
    return df_emis, df_pop, df_food

def get_target_4D(df_emis, region, gas):
    """
    Extrae el vector de emisiones históricas declaradas del inventario para una región y gas.

    Args:
        df_emis (pd.DataFrame): DataFrame con el inventario de emisiones históricas.
        region (str): Código o nombre identificador de la región analizada (e.g., 'SPAIN', 'EU27').
        gas (str): Denominación del gas trazador ('CH4' o 'N2O').

    Returns:
        np.ndarray: Vector de emisiones observadas (Gg gas / año) durante el periodo 2005-2023.
    """
    mask = (df_emis.iloc[:, 0].astype(str).str.strip() == region.strip()) & \
           (df_emis.iloc[:, 1].astype(str).str.strip() == '4_D__Wastewater_Treatment_and_Discharge') & \
           (df_emis.iloc[:, 2].astype(str).str.strip() == gas.strip())
    fila = df_emis[mask]
    if fila.empty:
        return np.zeros(LONGITUD_ANIOS)
    target = fila.iloc[0, 3:3+LONGITUD_ANIOS].values.astype(float)
    return np.nan_to_num(target, nan=0.0)

def get_region_data_4D(df_pop, df_food, region):
    """
    Extrae y sincroniza las series demográficas y de ingesta de proteínas per cápita.

    Fundamento Matemático:
        Para la región agregada 'EU27', agrega la suma demográfica de los Estados Miembros.
        Si existen datos de disponibilidad alimentaria, escala la ingesta de proteínas; en
        caso contrario, adopta el valor por defecto de las directrices IPCC (35.0 kg/hab/año).

    Args:
        df_pop (pd.DataFrame): Matriz con las series demográficas por país/región.
        df_food (pd.DataFrame): Matriz de disponibilidad y consumo de alimentos.
        region (str): Región o país de análisis.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tupla con:
            - pop_array: Serie temporal de población (habitantes).
            - protein_array: Serie temporal de consumo proteico per cápita (kg/hab/año).
    """
    try:
        region_clean = region.strip()
        pop_array = np.zeros(LONGITUD_ANIOS)
        
        # 1. Extracción de población para el periodo 2005-2023
        if region_clean == 'EU27':
            mask_pop = df_pop.iloc[:, 0].astype(str).str.strip().isin(EU27_COUNTRIES)
            pop_df = df_pop[mask_pop]
            if not pop_df.empty:
                for i, year in enumerate(AÑOS_ANALISIS):
                    year_str, year_int = str(year), int(year)
                    suma_pob = 0.0
                    for _, row in pop_df.iterrows():
                        if year_str in row.index: suma_pob += float(row[year_str])
                        elif year_int in row.index: suma_pob += float(row[year_int])
                    pop_array[i] = suma_pob
        else:
            pop_df = df_pop[df_pop.iloc[:, 0].astype(str).str.strip() == region_clean]
            if not pop_df.empty:
                pop_series = pop_df.iloc[0]
                for i, year in enumerate(AÑOS_ANALISIS):
                    year_str, year_int = str(year), int(year)
                    val = float(pop_series[year_str]) if year_str in pop_series.index else (float(pop_series[year_int]) if year_int in pop_series.index else 0.0)
                    pop_array[i] = val

        # 2. Extracción de proteínas per cápita (2005-2023)
        protein_array = np.full(LONGITUD_ANIOS, PROTEIN_PC_DEF)
        if not df_food.empty:
            food_df = df_food[df_food.iloc[:, 0].astype(str).str.strip() == region_clean]
            if not food_df.empty:
                food_series = food_df.iloc[0]
                for i, year in enumerate(AÑOS_ANALISIS):
                    year_str = f"Food_Availability_{year}"
                    if year_str in food_series.index and not pd.isna(food_series[year_str]):
                        val_food = float(food_series[year_str])
                        protein_array[i] = max(20.0, min(120.0, val_food / 1000.0 * 0.15)) if val_food > 500 else PROTEIN_PC_DEF

        return pop_array, protein_array
    except Exception as e:
        st.error(f"Error procesando datos demográficos/alimentarios para {region}: {e}")
        return np.zeros(LONGITUD_ANIOS), np.full(LONGITUD_ANIOS, PROTEIN_PC_DEF)


# =============================================================================
# 3. MOTOR MATEMÁTICO IPCC TIER 1 Y OPTIMIZACIÓN NUMÉRICA (POWELL)
# =============================================================================
def simular_4D(pop_array, protein_array, mcf, ef_n2o_val, bod_pc):
    """
    Ejecuta el balance de masa de emisiones de aguas residuales según IPCC 2006 (Tier 1).

    Fundamento Matemático (IPCC 2006 Vol. 5, Cap. 6, Ec. 6.1 a 6.8):
        Carga orgánica degradable total:
            TOW(t) = Población(t) · BOD_pc  (kg DBO / año)
        Emisiones de metano:
            CH4_sim(t) = [TOW(t) · B0 · MCF - R] · 10⁻⁶  (Gg CH4 / año)
        Nitrógeno en efluentes y emisiones de óxido nitroso:
            N_effluent(t) = Población(t) · Protein_pc(t) · F_NPR · F_NON_CON · F_IND_COM  (kg N / año)
            N2O_sim(t) = N_effluent(t) · EF_N2O · (44 / 28) · 10⁻⁶  (Gg N2O / año)

    Args:
        pop_array (np.ndarray): Serie temporal de población (habitantes).
        protein_array (np.ndarray): Consumo anual de proteína per cápita (kg/hab/año).
        mcf (float): Factor de Corrección de Metano según sistema de tratamiento (adimensional).
        ef_n2o_val (float): Factor de emisión de N2O en efluentes (kg N2O-N / kg N).
        bod_pc (float): Carga de DBO per cápita anual (kg DBO / hab / año).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - ch4_sim: Emisiones simuladas de metano (Gg CH4 / año).
            - n2o_sim: Emisiones simuladas de óxido nitroso (Gg N2O / año).
    """
    # Carga orgánica degradable total en aguas residuales (kg BOD / año)
    ow_total = pop_array * bod_pc
    # Emisiones de CH4 (Gg CH4) = OW · B0 · MCF · 10^-6
    ch4_sim = ow_total * B0_DEF * mcf * 1e-6
    
    # Emisiones de N2O por efluentes acuáticos (Gg N2O)
    # N_effluent = Pop · Protein_pc · F_NPR · F_NON_CON · F_IND_COM (kg N / año)
    n_effluent = pop_array * protein_array * F_NPR * F_NON_CON * F_IND_COM
    n2o_sim = n_effluent * ef_n2o_val * (44.0 / 28.0) * 1e-6
    
    return np.nan_to_num(ch4_sim, nan=0.0), np.nan_to_num(n2o_sim, nan=0.0)

def funcion_objetivo_4D(params, pop_array, protein_array, target_ch4, target_n2o, bod_pc):
    """
    Función de coste escalar que evalúa la suma de errores cuadráticos (SSE) para el Sector 4.D.

    Fundamento Matemático:
        J(MCF, EF_N2O) = ∑ (CH4_sim - CH4_obs)² + ∑ (N2O_sim - N2O_obs)²

    Args:
        params (list[float] | tuple[float, float]): Vector de parámetros [mcf, ef_n2o_val].
        pop_array (np.ndarray): Serie temporal de población (habitantes).
        protein_array (np.ndarray): Consumo proteico per cápita (kg/hab/año).
        target_ch4 (np.ndarray): Serie observada de emisiones de CH4 (Gg CH4 / año).
        target_n2o (np.ndarray): Serie observada de emisiones de N2O (Gg N2O / año).
        bod_pc (float): Carga de DBO per cápita (kg DBO / hab / año).

    Returns:
        float: Error cuadrático escalar combinado (Gg²).
    """
    mcf, ef_n2o_val = params
    ch4_sim, n2o_sim = simular_4D(pop_array, protein_array, mcf, ef_n2o_val, bod_pc)
    error_ch4 = np.sum((ch4_sim - target_ch4) ** 2)
    error_n2o = np.sum((n2o_sim - target_n2o) ** 2)
    return error_ch4 + error_n2o


# =============================================================================
# 4. APLICACIÓN Y PANEL DE CONTROL INTERACTIVO (STREAMLIT)
# =============================================================================
def render_glosario_4d():
    """Renderiza el Glosario Técnico, Parámetros y Fundamento Normativo para el Sector 4.D (Aguas Residuales)."""
    with st.expander("📖 Glosario Técnico, Parámetros y Fundamento Normativo (IPCC 2006)", expanded=False):
        st.markdown(
            """
            ### Marco Metodológico y Normativo — Sector 4.D: Tratamiento y Vertido de Aguas Residuales (IPCC 2006 Vol. 5, Cap. 6)
            Las emisiones del sector saneamiento abarcan la generación de metano ($\text{CH}_4$) por descomposición anaerobia
            de la carga orgánica degradable (**TOW**, basada en la Demanda Bioquímica de Oxígeno $\text{DBO}$ per cápita)
            y las emisiones de óxido nitroso ($\text{N}_2\text{O}$) directas e indirectas por nitrificación/desnitrificación de efluentes ricos en nitrógeno proteico.
            El módulo incluye la optimización bidimensional ($\text{MCF}$ y $\text{EF}_{\text{N}_2\text{O},\text{eff}}$) mediante el **Método de Powell** frente a los inventarios de **EDGAR v8.0**.
            """
        )

        tab_act, tab_params, tab_metr = st.tabs([
            "📊 Variables de Entrada y Actividad",
            "⚙️ Parámetros de Degradación y Calibración (IPCC Vol. 5, Cap. 6)",
            "📈 Métricas de Evaluación y Salidas"
        ])

        with tab_act:
            st.markdown("#### 1. Variables de Actividad y Carga Orgánica Doméstica")
            data_act = [
                {
                    "Variable": "Población regional",
                    "Símbolo": "P(t)",
                    "Unidades": "Habitantes",
                    "Fuente de Datos": "Eurostat / Banco Mundial (WDI)",
                    "Significado e Interpretación Técnica": "Censo demográfico anual de la región analizada bajo la partición territorial de WILIAM."
                },
                {
                    "Variable": "Consumo proteico per cápita",
                    "Símbolo": "Protein_pc(t)",
                    "Unidades": "kg proteína / hab / año",
                    "Fuente de Datos": "FAOSTAT (Hojas de Balance de Alimentos) / Food_Regions",
                    "Significado e Interpretación Técnica": "Ingesta anual media de proteína por habitante (~95 g/hab/día) que determina la excreción de nitrógeno en aguas servidas."
                },
                {
                    "Variable": "Carga DBO per cápita",
                    "Símbolo": "BOD_pc",
                    "Unidades": "kg DBO / hab / año",
                    "Fuente de Datos": "IPCC 2006 por defecto (21.9 kg DBO/hab/año)",
                    "Significado e Interpretación Técnica": "Demanda Bioquímica de Oxígeno doméstica media (~60 g DBO/hab/día) representativa de la carga orgánica individual generada."
                },
                {
                    "Variable": "Carga orgánica total doméstica",
                    "Símbolo": "TOW(t)",
                    "Unidades": "kg DBO / año",
                    "Fuente de Datos": "Calculado: P(t) · BOD_pc",
                    "Significado e Interpretación Técnica": "Total Organics in Wastewater; carga degradable bruta anual antes de los procesos de depuración o vertido."
                },
                {
                    "Variable": "Nitrógeno total en efluentes",
                    "Símbolo": "N_effluent(t)",
                    "Unidades": "kg N / año",
                    "Fuente de Datos": "Calculado conforme a Ecuación 6.8 del IPCC",
                    "Significado e Interpretación Técnica": "Flujo másico anual de nitrógeno orgánico y amoniacal vertido a los cauces y aguas receptoras."
                },
                {
                    "Variable": "Emisiones históricas observadas",
                    "Símbolo": "E_obs(t)",
                    "Unidades": "Gg CH4 / año y Gg N2O / año",
                    "Fuente de Datos": "EDGAR v8.0 / IPCC Sector 4_D",
                    "Significado e Interpretación Técnica": "Series de inventario de referencia para la calibración simultánea de los factores de emisión del sector saneamiento."
                }
            ]
            st.dataframe(pd.DataFrame(data_act), use_container_width=True, hide_index=True)
            st.caption("Nota metodológica: Las cargas orgánicas se calculan en kg DBO/año y se convierten homogéneamente a Gg para el balance final de GEI.")

        with tab_params:
            st.markdown("#### 2. Parámetros de Fermentación Anaerobia y Ciclo del Nitrógeno (IPCC Vol. 5, Cap. 6)")
            data_params = [
                {
                    "Parámetro": "Capacidad máxima de producción de CH4",
                    "Símbolo": "B0",
                    "Valor por Defecto": "0.60",
                    "Rango de Calibración": "Fijo (0.60)",
                    "Unidades": "kg CH4 / kg DBO",
                    "Naturaleza": "Estático Estequiométrico",
                    "Cita Normativa y Justificación Técnica": "Ecuación 6.2 IPCC Vol. 5 Cap. 6. Rendimiento termodinámico máximo teórico de producción de metano a partir del catabolismo anaerobio de materia orgánica."
                },
                {
                    "Parámetro": "Factor de corrección de metano en aguas",
                    "Símbolo": "MCF",
                    "Valor por Defecto": "0.30",
                    "Rango de Calibración": "[0.05, 0.90]",
                    "Unidades": "Adimensional (0 - 1)",
                    "Naturaleza": "Calibrado Powell 2D",
                    "Cita Normativa y Justificación Técnica": "Tabla 6.3 IPCC Vol. 5 Cap. 6. Factor medio ponderado representativo de la fracción de aguas y lodos que fermentan bajo condiciones anaerobias estrictas (sistemas no aireados, fosas sépticas, lagunas anaerobias y vertidos directos)."
                },
                {
                    "Parámetro": "Fracción de nitrógeno en proteínas",
                    "Símbolo": "F_NPR",
                    "Valor por Defecto": "0.16",
                    "Rango de Calibración": "Fijo (0.16)",
                    "Unidades": "kg N / kg proteína",
                    "Naturaleza": "Estático Bioquímico",
                    "Cita Normativa y Justificación Técnica": "Ecuación 6.8 IPCC Vol. 5 Cap. 6. Razón estequiométrica elemental media de nitrógeno presente en las cadenas de aminoácidos (16% en peso)."
                },
                {
                    "Parámetro": "Factor de nitrógeno no consumido",
                    "Símbolo": "F_NON-CON",
                    "Valor por Defecto": "1.04",
                    "Rango de Calibración": "Fijo (1.04)",
                    "Unidades": "Adimensional",
                    "Naturaleza": "Estático IPCC Tier 1",
                    "Cita Normativa y Justificación Técnica": "Tabla 6.11 IPCC Vol. 5 Cap. 6. Factor de corrección que contempla el nitrógeno procedente del desperdicio de alimentos y restos biológicos vertidos por fregaderos a la red de alcantarillado."
                },
                {
                    "Parámetro": "Factor de vertidos comerciales e industriales",
                    "Símbolo": "F_IND-COM",
                    "Valor por Defecto": "1.25",
                    "Rango de Calibración": "Fijo (1.25)",
                    "Unidades": "Adimensional",
                    "Naturaleza": "Estático IPCC Tier 1",
                    "Cita Normativa y Justificación Técnica": "Tabla 6.11 IPCC Vol. 5 Cap. 6. Multiplicador que cuantifica las descargas adicionales de nitrógeno procedentes del sector comercial, hostelería e industrias conectadas a la red pública de saneamiento."
                },
                {
                    "Parámetro": "Razón molecular estequiométrica N2O-N a N2O",
                    "Símbolo": "44/28",
                    "Valor por Defecto": "1.571",
                    "Rango de Calibración": "Fijo",
                    "Unidades": "kg N2O / kg N",
                    "Naturaleza": "Estático Molecular",
                    "Cita Normativa y Justificación Técnica": "Conversión estequiométrica entre la masa de nitrógeno molecular biológico (28 g/mol) y la masa de la molécula de óxido nitroso (44 g/mol)."
                },
                {
                    "Parámetro": "Factor de emisión de N2O en efluentes",
                    "Símbolo": "EF_N2O,eff",
                    "Valor por Defecto": "0.005",
                    "Rango de Calibración": "[0.001, 0.020]",
                    "Unidades": "kg N2O-N / kg N",
                    "Naturaleza": "Calibrado Powell 2D",
                    "Cita Normativa y Justificación Técnica": "Tabla 6.11 IPCC Vol. 5 Cap. 6. Factor de emisión derivado de los procesos de nitrificación y desnitrificación en ríos, estuarios y aguas receptoras que reciben efluentes de depuradora o vertidos brutos."
                },
                {
                    "Parámetro": "Factor de escala dimensional",
                    "Símbolo": "10⁻⁶",
                    "Valor por Defecto": "1.0 × 10⁻⁶",
                    "Rango de Calibración": "Fijo",
                    "Unidades": "Gg / kg",
                    "Naturaleza": "Estático Dimensional",
                    "Cita Normativa y Justificación Técnica": "Factor dimensional que convierte masa de kilogramos (kg) a Gigagramos (Gg)."
                }
            ]
            st.dataframe(pd.DataFrame(data_params), use_container_width=True, hide_index=True)

        with tab_metr:
            st.markdown("#### 3. Métricas de Evaluación y Salidas del Modelo")
            data_metr = [
                {
                    "Métrica / Salida": "Emisión anual de Metano",
                    "Símbolo": "E_CH4,4D(t)",
                    "Formulación Matemática": "TOW(t) · B0 · MCF · 10⁻⁶",
                    "Unidades": "Gg CH4 / año",
                    "Interpretación Ingenieril": "Emisiones directas de metano procedente de fermentación anaerobia de aguas residuales y fangos (Ecuación 6.1 del IPCC 2006)."
                },
                {
                    "Métrica / Salida": "Emisión anual de Óxido Nitroso",
                    "Símbolo": "E_N2O,4D(t)",
                    "Formulación Matemática": "N_effluent(t) · EF_N2O,eff · (44/28) · 10⁻⁶",
                    "Unidades": "Gg N2O / año",
                    "Interpretación Ingenieril": "Emisiones de óxido nitroso producidas en aguas receptoras de efluentes tratados y brutos (Ecuación 6.7 del IPCC 2006)."
                },
                {
                    "Métrica / Salida": "Emisiones agregadas en CO2 equivalente",
                    "Símbolo": "E_Total,CO2-eq(t)",
                    "Formulación Matemática": "27.9 · E_CH4 + 273 · E_N2O",
                    "Unidades": "Gg CO2-eq / año",
                    "Interpretación Ingenieril": "Emisiones conjuntas bajo la métrica AR6 GWP100 estandarizada en WILIAM."
                },
                {
                    "Métrica / Salida": "Suma de Errores Cuadráticos",
                    "Símbolo": "SSE",
                    "Formulación Matemática": "∑ (CH4_sim - CH4_obs)² + ∑ (N2O_sim - N2O_obs)²",
                    "Unidades": "Gg²",
                    "Interpretación Ingenieril": "Función de coste combinada minimizada mediante el algoritmo de optimización de Powell."
                },
                {
                    "Métrica / Salida": "Error Cuadrático Medio por Gas",
                    "Símbolo": "MSE",
                    "Formulación Matemática": "(1/N) ∑ (E_sim - E_obs)²",
                    "Unidades": "Gg²",
                    "Interpretación Ingenieril": "Dispersión media residual evaluada de forma independiente para metano y óxido nitroso."
                }
            ]
            st.dataframe(pd.DataFrame(data_metr), use_container_width=True, hide_index=True)

def main():
    """
    Punto de entrada de la aplicación Streamlit para el Sector 4.D.
    Gestiona los controles de carga orgánica DBO y proteínas, la ejecución del
    algoritmo de Powell, los indicadores de desempeño y las visualizaciones gráficas.
    """
    st.title("Simulador y Calibrador IPCC Tier 1 - Sector 4D (Wastewater Treatment & Discharge)")
    st.markdown("Aplicación interactiva para el modelado y calibración mediante el **algoritmo de Powell** (mínimos cuadrados) de las emisiones del sector **4D** (Aguas Residuales Domésticas e Industriales, 2005-2023).")

    render_glosario_4d()

    df_emis, df_pop, df_food = load_all_data()

    st.sidebar.header("Panel de Control 4D")
    default_idx = LISTA_REGIONES.index('SPAIN') if 'SPAIN' in LISTA_REGIONES else 0
    region_seleccionada = st.sidebar.selectbox("Selecciona Región WILIAM:", LISTA_REGIONES, index=default_idx)
    
    bod_pc = st.sidebar.slider("Carga DBO per cápita (kg BOD/hab/año)", 10.0, 50.0, BOD_PC_DEF, 0.5)
    protein_pc_slider = st.sidebar.slider("Consumo Proteínas per cápita (kg/hab/año)", 20.0, 70.0, PROTEIN_PC_DEF, 0.5)

    pop_array, protein_base = get_region_data_4D(df_pop, df_food, region_seleccionada)
    protein_array = np.full(LONGITUD_ANIOS, protein_pc_slider)

    target_ch4 = get_target_4D(df_emis, region_seleccionada, 'CH4')
    target_n2o = get_target_4D(df_emis, region_seleccionada, 'N2O')

    if 'mcf_calib' not in st.session_state:
        st.session_state['mcf_calib'] = MCF_DEF
    if 'ef_n2o_calib' not in st.session_state:
        st.session_state['ef_n2o_calib'] = EF_N2O_DEF
    if 'optimizado_4d' not in st.session_state:
        st.session_state['optimizado_4d'] = False

    st.sidebar.markdown("---")
    st.sidebar.header("Calibración del Modelo (Powell)")
    if st.sidebar.button("Ejecutar Optimización (Powell)", type="primary"):
        params_iniciales = [MCF_DEF, EF_N2O_DEF]
        bounds = [(0.05, 0.9), (0.001, 0.02)]
        
        resultado = minimize(
            funcion_objetivo_4D,
            params_iniciales,
            args=(pop_array, protein_array, target_ch4, target_n2o, bod_pc),
            method='Powell',
            bounds=bounds,
            options={'xtol': 1e-6, 'ftol': 1e-6}
        )
        
        st.session_state['mcf_calib'] = float(resultado.x[0])
        st.session_state['ef_n2o_calib'] = float(resultado.x[1])
        st.session_state['optimizado_4d'] = True
        st.sidebar.success("¡Optimización completada con éxito!")

    if st.sidebar.button("Restaurar Valores por Defecto (IPCC)"):
        st.session_state['mcf_calib'] = MCF_DEF
        st.session_state['ef_n2o_calib'] = EF_N2O_DEF
        st.session_state['optimizado_4d'] = False
        st.sidebar.info("Valores restablecidos a los por defecto IPCC Tier 1.")

    ch4_simulado, n2o_simulado = simular_4D(
        pop_array, protein_array,
        st.session_state['mcf_calib'],
        st.session_state['ef_n2o_calib'],
        bod_pc
    )

    mse_ch4 = np.mean((ch4_simulado - target_ch4) ** 2)
    mse_n2o = np.mean((n2o_simulado - target_n2o) ** 2)

    st.markdown(f"### Resultados del Análisis para: **{region_seleccionada}** (2005-2023)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("MCF Calibrado", f"{st.session_state['mcf_calib']:.4f}", delta="Calibrado" if st.session_state['optimizado_4d'] else "IPCC Default")
    kpi2.metric("EF N2O Efluentes", f"{st.session_state['ef_n2o_calib']:.5f} kg N2O-N/kg N", delta="Calibrado" if st.session_state['optimizado_4d'] else "IPCC Default")
    kpi3.metric("MSE CH4", f"{mse_ch4:.4f}")
    kpi4.metric("MSE N2O", f"{mse_n2o:.4f}")

    col1, col2 = st.columns(2)
    
    with col1:
        fig_ch4 = go.Figure()
        fig_ch4.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=ch4_simulado, mode='lines+markers', name='CH4 Simulado', line=dict(color='#1f77b4', width=2)))
        fig_ch4.add_trace(go.Scatter(x=AÑOS_ANALISIS, y=target_ch4, mode='lines+markers', name='CH4 Inventario Real', line=dict(color='#d62728', width=2, dash='dash')))
        fig_ch4.update_layout(
            title=f"Emisiones de Metano (CH4) - Aguas Residuales ({region_seleccionada})",
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
            title=f"Emisiones de Óxido Nitroso (N2O) - Efluentes ({region_seleccionada})",
            xaxis_title="Año",
            yaxis_title="Emisiones (Gg N2O)",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_n2o, use_container_width=True)

    st.markdown("### Tabla de Datos y Auditoría Sector 4D (2005-2023)")
    df_debug = pd.DataFrame({
        'Año': AÑOS_ANALISIS,
        'Población (Hab)': pop_array,
        'Proteína per Cápita (kg/hab)': protein_array,
        'CH4 Simulado (Gg)': ch4_simulado,
        'CH4 Real (Gg)': target_ch4,
        'N2O Simulado (Gg)': n2o_simulado,
        'N2O Real (Gg)': target_n2o
    })
    st.dataframe(df_debug.style.format(precision=3), use_container_width=True)

if __name__ == "__main__":
    main()
