"""
Módulo de Simulación y Calibración Numérica - Sector 4.A: Vertido de Residuos Sólidos (FOD)
==========================================================================================

Descripción Técnica:
    Modelo cinético de primer orden (First Order Decay - FOD) multifraccional y multicomponente
    para la estimación de emisiones de metano (CH4) procedentes de vertederos y depósitos
    de residuos sólidos urbanos e industriales (IPCC 2006, Volumen 5: Residuos, Capítulo 3).

Marco del Proyecto:
    Trabajo de Fin de Grado (TFG) - Grado en Ingeniería Electrónica Industrial y Automática.
    Escuela de Ingenierías Industriales (EII), Universidad de Valladolid (UVa).
    Desarrollado como núcleo algorítmico principal para la parametrización regionalizada del
    módulo de residuos del Modelo de Evaluación Integrada WILIAM (Horizon 2020 LOCOMOTION, GA No. 821105).

Metodología Implementada:
    - Enfoque Híbrido: Tier 2 / pseudo-Tier 3 con desagregación en 9 fracciones de degradabilidad
      heterogénea (alimentos, poda, papel, madera, textiles, pañales, lodos, caucho e industrial).
    - Formulación Matemática: Ecuaciones discretas anuales 3.4 y 3.5 del IPCC 2006.
    - Acoplamiento Macroeconómico: Generación de residuos industriales vinculada al Producto Interior Bruto (PIB).
    - Vector de Parámetros Calibrables (12D):
        θ = [k_food, k_garden, k_paper, k_wood, k_textiles, k_nappies,
             k_sludge, k_rubber, k_industrial, H0_MSW, H0_IND, WASTE_PER_GDP]
    - Optimización Numérica: Método libre de derivadas de Powell (scipy.optimize.minimize(method='Powell'))
      con transformación afín al hipercubo unitario [0, 1]¹² y función de coste SSE con barreras de penalización.

Fuentes de Datos Principales:
    - Emisiones históricas observadas: EDGAR v8.0 (1970-2023) / Inventarios Level 2 IPCC (Sector 4_A).
    - Demografía y PIB: Eurostat y Banco Mundial (World Development Indicators).
    - Estadísticas de residuos: OECD / Eurostat (residuos per cápita y porcentaje de vertido).

Autor:
    Javier Hernández Cepeda
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import warnings
import math
from scipy.integrate import solve_ivp

# Inhibición de avisos de advertencia en convergencia numérica
warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURACIÓN DE INTERFAZ Y CONSTANTES ESTRUCTURALES
# =============================================================================
st.set_page_config(page_title="IPCC Tier 4A - Modelo Dinámico", layout="wide")

def render_glosario_4a():
    """Renderiza el Glosario Técnico, Parámetros y Fundamento Normativo para el Sector 4.A (Vertederos)."""
    with st.expander("📖 Glosario Técnico, Parámetros y Fundamento Normativo (IPCC 2006)", expanded=False):
        st.markdown(
            """
            ### Marco Metodológico y Normativo — Sector 4.A: Vertido de Residuos Sólidos (IPCC 2006 Vol. 5, Cap. 3)
            El cálculo de emisiones de metano ($\text{CH}_4$) se fundamenta en el **Modelo de Decaimiento de Primer Orden (First Order Decay - FOD)**.
            La arquitectura implementada en este módulo integra una formulación multicomponente (9 fracciones orgánicas heterogéneas),
            acoplamiento macroeconómico con el PIB regional y calibración numérica de 12 parámetros mediante el **Método de Powell** 
            sobre la serie temporal histórica de los inventarios internacionales **EDGAR v8.0** (1970-2023).
            """
        )

        tab_act, tab_params, tab_metr = st.tabs([
            "📊 Variables de Entrada y Actividad",
            "⚙️ Parámetros del Modelo y Vector 12D (IPCC Vol. 5 & Powell)",
            "📈 Métricas de Evaluación y Salidas"
        ])

        with tab_act:
            st.markdown("#### 1. Variables de Actividad y Señales Dinámicas de Entrada")
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
                    "Significado e Interpretación Técnica": "Tasa bruta anual de generación per cápita de residuos sólidos urbanos municipales (MSW)."
                },
                {
                    "Variable": "Porcentaje de vertido municipal",
                    "Símbolo": "%_disp(t)",
                    "Unidades": "% (0 - 100%)",
                    "Fuente de Datos": "Eurostat / OECD / Serie histórica",
                    "Significado e Interpretación Técnica": "Fracción anual de la masa de residuos municipales que es derivada efectivamente a vertedero o depósito controlado."
                },
                {
                    "Variable": "Masa neta de RSU a vertedero",
                    "Símbolo": "W_MSW(t)",
                    "Unidades": "Mt / año (o Gg / año)",
                    "Fuente de Datos": "Calculado: P(t) · MSW_pc(t) · %_disp(t)",
                    "Significado e Interpretación Técnica": "Flujo másico anual de residuo urbano depositado en el vertedero en base húmeda."
                },
                {
                    "Variable": "Producto Interior Bruto regional",
                    "Símbolo": "GDP(t)",
                    "Unidades": "Miles de M$ (miles de millones USD)",
                    "Fuente de Datos": "Banco Mundial / Módulo Macro WILIAM",
                    "Significado e Interpretación Técnica": "Nivel de actividad económica regional que determina la generación de residuos industriales asimilables a urbanos."
                },
                {
                    "Variable": "Emisiones históricas observadas",
                    "Símbolo": "E_obs(t)",
                    "Unidades": "Mt CH4 / año",
                    "Fuente de Datos": "EDGAR v8.0 / IPCC Sector 4_A",
                    "Significado e Interpretación Técnica": "Serie de emisiones históricas de inventario (1970-2023) utilizada como referencia objetivo para la optimización."
                },
                {
                    "Variable": "Captura y quema de biogás",
                    "Símbolo": "R(t)",
                    "Unidades": "Mt CH4 / año",
                    "Fuente de Datos": "Inventarios Nacionales / EDGAR",
                    "Significado e Interpretación Técnica": "Cantidad de metano capturado mediante extracción activa de biogás, desgasificación y antorcha/valorización."
                }
            ]
            st.dataframe(pd.DataFrame(data_act), use_container_width=True, hide_index=True)
            st.caption("Nota metodológica: Las series temporales abarcan el intervalo 2005-2023 con coherencia dimensional estricta en Mt y Gg.")

        with tab_params:
            st.markdown("#### Bloque A: Parámetros Bioquímicos Estáticos por Fracción (IPCC 2006 Vol. 5, Tablas 2.4 y 2.5)")
            data_frac = [
                {
                    "Fracción de Residuo": "Alimentos (Food Waste)",
                    "DOC (t C / t)": 0.15,
                    "DOC_f (adim)": 0.70,
                    "DOC Biodegradable (t C / t)": 0.105,
                    "Naturaleza": "Estático IPCC Vol. 5",
                    "Justificación Bioquímica y Comportamiento": "Materia orgánica lábil, rica en carbohidratos solubles y proteínas; rápida acidogénesis y fermentación anaerobia activa."
                },
                {
                    "Fracción de Residuo": "Poda y Jardinería (Garden)",
                    "DOC (t C / t)": 0.20,
                    "DOC_f (adim)": 0.70,
                    "DOC Biodegradable (t C / t)": 0.140,
                    "Naturaleza": "Estático IPCC Vol. 5",
                    "Justificación Bioquímica y Comportamiento": "Residuos vegetales lignocelulósicos; velocidad de degradación intermedia-rápida."
                },
                {
                    "Fracción de Residuo": "Papel y Cartón (Paper)",
                    "DOC (t C / t)": 0.40,
                    "DOC_f (adim)": 0.50,
                    "DOC Biodegradable (t C / t)": 0.200,
                    "Naturaleza": "Estático IPCC Vol. 5",
                    "Justificación Bioquímica y Comportamiento": "Celulosa y hemicelulosa con cinética intermedia; aproximadamente el 50% permanece recalcitrante en condiciones anaerobias."
                },
                {
                    "Fracción de Residuo": "Madera y Paja (Wood/Straw)",
                    "DOC (t C / t)": 0.43,
                    "DOC_f (adim)": 0.10,
                    "DOC Biodegradable (t C / t)": 0.043,
                    "Naturaleza": "Estático IPCC Vol. 5",
                    "Justificación Bioquímica y Comportamiento": "Baja biodegradabilidad debida a la barrera estérica de lignina cristalina que impide el acceso enzimático microbiano a la celulosa."
                },
                {
                    "Fracción de Residuo": "Textiles (Textiles)",
                    "DOC (t C / t)": 0.24,
                    "DOC_f (adim)": 0.50,
                    "DOC Biodegradable (t C / t)": 0.120,
                    "Naturaleza": "Estático IPCC Vol. 5",
                    "Justificación Bioquímica y Comportamiento": "Mezcla heterogénea de fibras naturales biodegradables (algodón, lana) y fibras sintéticas no biodegradables."
                },
                {
                    "Fracción de Residuo": "Pañales (Nappies)",
                    "DOC (t C / t)": 0.24,
                    "DOC_f (adim)": 0.50,
                    "DOC Biodegradable (t C / t)": 0.120,
                    "Naturaleza": "Estático IPCC Vol. 5",
                    "Justificación Bioquímica y Comportamiento": "Fracción de pulpa de celulosa absorbente (biodegradable al 50%) combinada con polímeros y películas plásticas inertes."
                },
                {
                    "Fracción de Residuo": "Lodos de EDAR (Sludge)",
                    "DOC (t C / t)": 0.05,
                    "DOC_f (adim)": 0.50,
                    "DOC Biodegradable (t C / t)": 0.025,
                    "Naturaleza": "Estático IPCC Vol. 5",
                    "Justificación Bioquímica y Comportamiento": "Lodos deshidratados de depuración previamente estabilizados; contenido de carbono residual moderado."
                },
                {
                    "Fracción de Residuo": "Caucho y Cuero (Rubber)",
                    "DOC (t C / t)": 0.39,
                    "DOC_f (adim)": 0.00,
                    "DOC Biodegradable (t C / t)": 0.000,
                    "Naturaleza": "Estático IPCC Vol. 5",
                    "Justificación Bioquímica y Comportamiento": "Presencia de enlaces cruzados de vulcanización con azufre químicamente inertes bajo digestión anaerobia (DOC_f = 0.0)."
                },
                {
                    "Fracción de Residuo": "Industrial Asimilable (Industrial)",
                    "DOC (t C / t)": 0.15,
                    "DOC_f (adim)": 0.50,
                    "DOC Biodegradable (t C / t)": 0.075,
                    "Naturaleza": "Estático IPCC Vol. 5",
                    "Justificación Bioquímica y Comportamiento": "Composición agregada de residuos sólidos industriales no peligrosos ligada a la actividad del PIB regional."
                }
            ]
            st.dataframe(pd.DataFrame(data_frac), use_container_width=True, hide_index=True)

            st.markdown("##### Parámetros Estequiométricos y Cinéticos Estáticos (IPCC Vol. 5, Cap. 3)")
            data_static = [
                {
                    "Parámetro": "Fracción molar de CH4 en biogás",
                    "Símbolo": "F",
                    "Valor": "0.50",
                    "Unidades": "Adimensional (0-1)",
                    "Naturaleza": "Estático IPCC",
                    "Justificación Técnica y Cita Normativa": "Razón equimolar canónica del proceso metanogénico (~50% CH4 / 50% CO2 en digestión anaerobia; IPCC Vol. 5, Cap. 3)."
                },
                {
                    "Parámetro": "Razón molecular estequiométrica",
                    "Símbolo": "16/12",
                    "Valor": "1.333",
                    "Unidades": "t CH4 / t C",
                    "Naturaleza": "Estático Estequiométrico",
                    "Justificación Técnica y Cita Normativa": "Conversión estequiométrica entre la masa atómica del carbono (12 g/mol) y la masa molecular del metano (16 g/mol)."
                },
                {
                    "Parámetro": "Factor de corrección de metano (MSW)",
                    "Símbolo": "MCF_MSW",
                    "Valor": "0.71",
                    "Unidades": "Adimensional",
                    "Naturaleza": "Estático Regional",
                    "Justificación Técnica y Cita Normativa": "Pondera el grado de confinamiento anaerobio y profundidad de los vertederos municipales según la serie histórica EDGAR."
                },
                {
                    "Parámetro": "Factor de corrección de metano (IND)",
                    "Símbolo": "MCF_IND",
                    "Valor": "0.72",
                    "Unidades": "Adimensional",
                    "Naturaleza": "Estático Regional",
                    "Justificación Técnica y Cita Normativa": "Nivel medio de compactación y anaerobiosis en depósitos industriales no peligrosos (EDGAR Level 2)."
                },
                {
                    "Parámetro": "Factor de oxidación superficial",
                    "Símbolo": "OX",
                    "Valor": "0.05",
                    "Unidades": "Adimensional (0-1)",
                    "Naturaleza": "Estático IPCC",
                    "Justificación Técnica y Cita Normativa": "Capacidad media de los microorganismos metanótrofos en la capa de cobertura de suelo para oxidar metano a CO2 (IPCC Cap. 3)."
                },
                {
                    "Parámetro": "Retardo temporal de degradación",
                    "Símbolo": "t_delay",
                    "Valor": "1.0",
                    "Unidades": "Año",
                    "Naturaleza": "Estático IPCC (H.8b)",
                    "Justificación Técnica y Cita Normativa": "Convenio de retardo temporal del IPCC: el residuo depositado en el año t inicia su degradación biológica activa el 1 de enero de t+1."
                }
            ]
            st.dataframe(pd.DataFrame(data_static), use_container_width=True, hide_index=True)

            st.markdown("#### Bloque B: Vector de Calibración 12D (Método de Powell en Hipercubo Unitario)")
            data_powell = [
                {
                    "Parámetro": "Constantes cinéticas de degradación (9 fracciones)",
                    "Símbolo": "k_i",
                    "Espacio de Búsqueda": "Cajas bioclimáticas IPCC [k_min, k_max] (Tabla 3.3)",
                    "Unidades": "año⁻¹",
                    "Naturaleza": "Calibrado Powell 12D",
                    "Significado y Justificación Técnica": "Representan la velocidad metabólica de hidrólisis de polímeros. Se optimizan mapeadas en el hipercubo unitario [0, 1]¹² dentro de los rangos admisibles según el régimen bioclimático regional (seco/húmedo y templado/tropical)."
                },
                {
                    "Parámetro": "Stock inicial acumulado municipal",
                    "Símbolo": "H0_MSW",
                    "Espacio de Búsqueda": "[0, 5000]",
                    "Unidades": "Mt C",
                    "Naturaleza": "Calibrado Powell 12D",
                    "Significado e Interpretación Técnica": "Representa el carbono orgánico degradable acumulado en vertedero con anterioridad al año 2005. Modela la memoria biofísica del vertedero y su emisión residual de metano."
                },
                {
                    "Parámetro": "Stock inicial acumulado industrial",
                    "Símbolo": "H0_IND",
                    "Espacio de Búsqueda": "[0, 5000]",
                    "Unidades": "Mt C",
                    "Naturaleza": "Calibrado Powell 12D",
                    "Significado e Interpretación Técnica": "Stock histórico de carbono acumulado de origen industrial anterior a 2005 para estabilizar la curva inicial del modelo."
                },
                {
                    "Parámetro": "Intensidad de generación industrial por PIB",
                    "Símbolo": "WASTE_PER_GDP",
                    "Espacio de Búsqueda": "[0, 100]",
                    "Unidades": "t / millón USD",
                    "Naturaleza": "Calibrado Powell 12D",
                    "Significado e Interpretación Técnica": "Coeficiente macroeconómico que vincula la generación de residuos industriales con el PIB regional en el modelo WILIAM."
                }
            ]
            st.dataframe(pd.DataFrame(data_powell), use_container_width=True, hide_index=True)

        with tab_metr:
            st.markdown("#### Bloque C: Métricas de Ajuste Estadístico y Salidas del Modelo")
            data_metr = [
                {
                    "Métrica": "Suma de Errores Cuadráticos",
                    "Símbolo": "SSE",
                    "Formulación Matemática": "∑ (E_sim(t) - E_obs(t))²",
                    "Unidades": "Mt² CH4",
                    "Interpretación Ingenieril": "Función de coste escalar minimizada numéricamente por el algoritmo de Powell con penalización de barrera exterior."
                },
                {
                    "Métrica": "Error Cuadrático Medio",
                    "Símbolo": "RMSE",
                    "Formulación Matemática": "√[ (1/N) ∑ (E_sim(t) - E_obs(t))² ]",
                    "Unidades": "Mt CH4 / año",
                    "Interpretación Ingenieril": "Dispersión típica del error absoluto de predicción respecto al inventario observado."
                },
                {
                    "Métrica": "RMSE Normalizado",
                    "Símbolo": "nRMSE",
                    "Formulación Matemática": "(RMSE / E_obs_media) · 100",
                    "Unidades": "%",
                    "Interpretación Ingenieril": "Error porcentual estandarizado respecto a la media histórica; permite comparar la bondad del ajuste entre regiones de diversa magnitud."
                },
                {
                    "Métrica": "Eficiencia de Nash-Sutcliffe",
                    "Símbolo": "NSE",
                    "Formulación Matemática": "1 - [ ∑ (E_sim - E_obs)² / ∑ (E_obs - E_obs_media)² ]",
                    "Unidades": "Adimensional (-∞, 1]",
                    "Interpretación Ingenieril": "Eficiencia predictiva 1:1 frente al estimador de la media. Un valor de 1 indica ajuste perfecto; valores > 0.5 representan un ajuste altamente satisfactorio."
                }
            ]
            st.dataframe(pd.DataFrame(data_metr), use_container_width=True, hide_index=True)

            st.info(
                """
                **Interpretación del Índice NSE en Modelos de Emisiones:**
                - En **series históricas planas con captura o desvío estabilizado** (ej. España o USMCA), la varianza observada tiende a cero ($\text{SST} \to 0$). En este escenario límite, cualquier pequeña desviación residual produce un valor de NSE formalmente negativo debido a la patología matemática del denominador, sin que ello denote desajuste físico del modelo (el RMSE se mantiene extremadamente bajo).
                - En **economías emergentes** (ej. China o India), las discontinuidades estadísticas asociadas a planes quinquenales generan un sesgo volumétrico medio residual constante, si bien la morfología dinámica y la aceleración temporal quedan adecuadamente reproducidas ($R^2 > 0.99$).
                """
            )

st.title("Simulador IPCC (Tier 4A) - Modelo Dinámico")
st.markdown("""
Esta aplicación simula el modelo FOD utilizando las **Ecuaciones Discretas Anuales del IPCC** (Eq. 3.4 y 3.5). 
Utiliza un Modelo Dinámico optimizando 9 constantes biológicas $k$, 2 Stocks Iniciales (H0_MSW y H0_IND) y el ratio generacional WASTE_PER_GDP (12 dimensiones) mediante el método libre de gradientes **Powell**, garantizando estabilidad matemática y cumplimiento estricto de las directrices de inventario.
""")

render_glosario_4a()

st.sidebar.markdown("### Configuración del Motor")
if st.sidebar.button("🔄 Recargar Última Calibración"):
    st.cache_data.clear()
    st.sidebar.success("¡Caché limpiada! Los datos se han recargado correctamente.")
    st.rerun()

EU27_REGIONS = [
    'AUSTRIA', 'BELGIUM', 'BULGARIA', 'CROATIA', 'CYPRUS', 'CZECH_REPUBLIC', 
    'DENMARK', 'ESTONIA', 'FINLAND', 'FRANCE', 'GERMANY', 'GREECE', 
    'HUNGARY', 'IRELAND', 'ITALY', 'LATVIA', 'LITHUANIA', 'LUXEMBOURG', 
    'MALTA', 'NETHERLANDS', 'POLAND', 'PORTUGAL', 'ROMANIA', 'SLOVAKIA', 
    'SLOVENIA', 'SPAIN', 'SWEDEN'
]

# Hitos legislativos y directivas de residuos aplicables en la Unión Europea
EVENTOS_GLOBALES_UE = [
    {'año': 2008, 'evento': 'Directiva Marco Residuos UE', 'color': 'purple'},
    {'año': 2015, 'evento': 'Ajuste Metodológico IPCC 2006', 'color': 'gray'}
]

HITOS_PAISES = {
    'AUSTRIA': [{'año': 2004, 'evento': 'Prohibición vertido sin tratar (Reducción DOC)', 'color': 'green'}],
    'BELGIUM': [{'año': 2000, 'evento': 'Prohibición vertido orgánico (Reducción DOC)', 'color': 'green'}],
    'BULGARIA': [{'año': 2007, 'evento': 'Entrada UE (Cierre vertederos no controlados)', 'color': 'blue'}],
    'CROATIA': [
        {'año': 2013, 'evento': 'Entrada UE (Adopción de normativas)', 'color': 'blue'}, 
        {'año': 2015, 'evento': 'Captura LFG y Tratamiento MBT', 'color': 'red'}
    ],
    'CYPRUS': [],
    'CZECH_REPUBLIC': [],
    'DENMARK': [{'año': 1997, 'evento': 'Prohibición vertido orgánico directo', 'color': 'green'}], 
    'ESTONIA': [],
    'FINLAND': [{'año': 2016, 'evento': 'Prohibición total de orgánicos a vertedero', 'color': 'green'}],
    'FRANCE': [
        {'año': 2012, 'evento': 'Aumento tasa vertidos TGAP (Desvío de residuos)', 'color': 'orange'}, 
        {'año': 2015, 'evento': 'Ley Transición (Objetivo reducción vertido 50%)', 'color': 'green'}
    ],
    'GERMANY': [{'año': 2005, 'evento': 'Prohibición vertido sin tratar (MBT/Incineración)', 'color': 'green'}],
    'GREECE': [{'año': 2012, 'evento': 'Nueva Ley Residuos (Fomento desvío vertedero)', 'color': 'orange'}],
    'HUNGARY': [],
    'IRELAND': [{'año': 2011, 'evento': 'Subida drástica Tasa Vertedero', 'color': 'orange'}],
    'ITALY': [{'año': 2003, 'evento': 'Transposición Directiva Vertederos UE', 'color': 'orange'}],
    'LATVIA': [],
    'LITHUANIA': [],
    'LUXEMBOURG': [],
    'MALTA': [],
    'NETHERLANDS': [{'año': 1995, 'evento': 'Prohibición general de vertidos', 'color': 'green'}],
    'POLAND': [{'año': 2012, 'evento': 'Ley Limpieza (Control municipal de vertidos)', 'color': 'orange'}],
    'PORTUGAL': [{'año': 2006, 'evento': 'Plan PERSU II (Cierre basureros y fomento MBT)', 'color': 'orange'}],
    'ROMANIA': [
        {'año': 2007, 'evento': 'Entrada UE (Adopción de normativas)', 'color': 'blue'}, 
        {'año': 2017, 'evento': 'Introducción Impuesto al Vertedero', 'color': 'orange'}
    ],
    'SLOVAKIA': [],
    'SLOVENIA': [],
    'SPAIN': [{'año': 2011, 'evento': 'Ley Residuos 22/2011 (Jerarquía de residuos)', 'color': 'orange'}],
    'SWEDEN': [{'año': 2005, 'evento': 'Prohibición vertido orgánico', 'color': 'green'}],
    'EU27': [],
    'CHINA': [
        {'año': 2011, 'evento': '12º Plan Quinquenal (Control de vertederos)', 'color': 'orange'},
        {'año': 2016, 'evento': '13º Plan Quinquenal (Confinamiento y WtE)', 'color': 'green'},
        {'año': 2020, 'evento': 'Estrategia Nacional Clasificación Residuos', 'color': 'purple'}
    ],
    'INDIA': [
        {'año': 2014, 'evento': 'Swachh Bharat Mission (Higiene urbana)', 'color': 'orange'},
        {'año': 2016, 'evento': 'Solid Waste Management Rules (SWM 2016)', 'color': 'green'}
    ],
    'USMCA': [
        {'año': 2016, 'evento': 'EPA NSPS / Clean Air Act (Control metano LFG)', 'color': 'green'},
        {'año': 2021, 'evento': 'U.S. Methane Emissions Action Plan', 'color': 'blue'}
    ],
    'RUSSIA': [
        {'año': 2017, 'evento': 'Año de la Ecología (Restricción vertido)', 'color': 'orange'},
        {'año': 2019, 'evento': 'Reforma Federal de Residuos (Ley 89-FZ)', 'color': 'green'}
    ]
}

DICT_WASTE_PER_GDP = {
    'AUSTRIA': 1.24164e-07, 'BELGIUM': 0.0, 'BULGARIA': 0.0, 'CROATIA': 0.0,
    'CYPRUS': 2.56224e-08, 'CZECH_REPUBLIC': 2.79194e-07, 'DENMARK': 6.49037e-08,
    'ESTONIA': 0.0, 'FINLAND': 0.0, 'FRANCE': 1.01248e-07, 'GERMANY': 0.0,
    'GREECE': 0.0, 'HUNGARY': 1.09281e-09, 'IRELAND': 0.0, 'ITALY': 1.73472e-18,
    'LATVIA': 0.0, 'LITHUANIA': 1.4336e-05, 'LUXEMBOURG': 0.0, 'MALTA': 0.0,
    'NETHERLANDS': 8.5522e-08, 'POLAND': 1.38778e-17, 'PORTUGAL': 0.0,
    'ROMANIA': 7.28552e-08, 'SLOVAKIA': 0.0, 'SLOVENIA': 0.0, 'SPAIN': 7.2506e-08,
    'SWEDEN': 0.0, 'UK': 0.0, 'CHINA': 3.25879e-07, 'EASOC': 1.10834e-07,
    'INDIA': 0.0, 'LATAM': 6.07484e-08, 'RUSSIA': 8.23608e-07, 'USMCA': 8.83063e-08,
    'LROW': 8.92302e-09
}

# Límites cinéticos de biodegradabilidad k_i (año⁻¹) por zona climática (IPCC 2006 Vol. 5, Tabla 3.3)
limites_k_clima = {
    'TEMPERATE_DRY': {
        'FOOD_WASTE': (0.05, 0.08), 'GARDEN_WASTE': (0.04, 0.06), 'PAPER_WASTE': (0.03, 0.05),
        'WOOD_STRAW_WASTE': (0.01, 0.03), 'TEXTILES_WASTE': (0.03, 0.05), 'DISPOSABLE_NAPPIES_WASTE': (0.03, 0.05),
        'SEWAGE_SLUDGE_WASTE': (0.04, 0.08), 'RUBBER_WASTE': (0.00, 0.01),
        'INDUSTRIAL_WASTE': (0.04, 0.06) 
    },
    'TEMPERATE_WET': {
        'FOOD_WASTE': (0.10, 0.29), 'GARDEN_WASTE': (0.06, 0.18), 'PAPER_WASTE': (0.04, 0.07),
        'WOOD_STRAW_WASTE': (0.02, 0.04), 'TEXTILES_WASTE': (0.04, 0.07), 'DISPOSABLE_NAPPIES_WASTE': (0.04, 0.07),
        'SEWAGE_SLUDGE_WASTE': (0.06, 0.15), 'RUBBER_WASTE': (0.00, 0.01),
        'INDUSTRIAL_WASTE': (0.08, 0.10)
    },
    'TROPICAL_DRY': {
        'FOOD_WASTE': (0.07, 0.10), 'GARDEN_WASTE': (0.05, 0.08), 'PAPER_WASTE': (0.04, 0.06),
        'WOOD_STRAW_WASTE': (0.02, 0.04), 'TEXTILES_WASTE': (0.04, 0.06), 'DISPOSABLE_NAPPIES_WASTE': (0.04, 0.06),
        'SEWAGE_SLUDGE_WASTE': (0.05, 0.10), 'RUBBER_WASTE': (0.00, 0.01),
        'INDUSTRIAL_WASTE': (0.05, 0.08)
    },
    'TROPICAL_WET': {
        'FOOD_WASTE': (0.17, 0.70), 'GARDEN_WASTE': (0.15, 0.20), 'PAPER_WASTE': (0.06, 0.085),
        'WOOD_STRAW_WASTE': (0.03, 0.05), 'TEXTILES_WASTE': (0.06, 0.085), 'DISPOSABLE_NAPPIES_WASTE': (0.06, 0.085),
        'SEWAGE_SLUDGE_WASTE': (0.15, 0.40), 'RUBBER_WASTE': (0.00, 0.01),
        'INDUSTRIAL_WASTE': (0.15, 0.20)
    }
}

# Parámetros bioquímicos estáticos por fracción residual (IPCC 2006 Vol. 5, Tablas 2.4 y 2.5)
params_estaticos = {
    'FOOD_WASTE': {'doc': 0.15, 'docf': 0.7, 'comp_msw': 0.375, 'comp_ind': 0.0},
    'GARDEN_WASTE': {'doc': 0.20, 'docf': 0.7, 'comp_msw': 0.1875, 'comp_ind': 0.0},
    'PAPER_WASTE': {'doc': 0.40, 'docf': 0.5, 'comp_msw': 0.250, 'comp_ind': 0.0},
    'WOOD_STRAW_WASTE': {'doc': 0.43, 'docf': 0.1, 'comp_msw': 0.0625, 'comp_ind': 0.0},
    'TEXTILES_WASTE': {'doc': 0.24, 'docf': 0.5, 'comp_msw': 0.0625, 'comp_ind': 0.0},
    'DISPOSABLE_NAPPIES_WASTE': {'doc': 0.24, 'docf': 0.5, 'comp_msw': 0.0625, 'comp_ind': 0.0},
    'SEWAGE_SLUDGE_WASTE': {'doc': 0.05, 'docf': 0.5, 'comp_msw': 0.0, 'comp_ind': 0.0}, 
    'RUBBER_WASTE': {'doc': 0.39, 'docf': 0.0, 'comp_msw': 0.0, 'comp_ind': 0.0},      
    'INDUSTRIAL_WASTE': {'doc': 0.15, 'docf': 0.5, 'comp_msw': 0.0, 'comp_ind': 1.0}
}

ALL_WASTE_TYPES = list(params_estaticos.keys())
MSW_TYPES = [k for k in ALL_WASTE_TYPES if k != 'INDUSTRIAL_WASTE']

COMPOSICION_2005_DEFAULT = {
    'FOOD_WASTE': 0.404, 'GARDEN_WASTE': 0.026, 'PAPER_WASTE': 0.144,
    'WOOD_STRAW_WASTE': 0.014, 'TEXTILES_WASTE': 0.03, 'DISPOSABLE_NAPPIES_WASTE': 0.006,
    'SEWAGE_SLUDGE_WASTE': 0.0, 'RUBBER_WASTE': 0.003
}

mapa_regiones_clima = {
    'EU27': 'TEMPERATE_WET',
    'BELGIUM': 'TEMPERATE_WET', 'CZECH_REPUBLIC': 'TEMPERATE_WET', 'DENMARK': 'TEMPERATE_WET', 
    'ESTONIA': 'TEMPERATE_WET', 'FINLAND': 'TEMPERATE_WET', 'FRANCE': 'TEMPERATE_WET', 
    'GERMANY': 'TEMPERATE_WET', 'HUNGARY': 'TEMPERATE_WET', 'IRELAND': 'TEMPERATE_WET', 
    'LATVIA': 'TEMPERATE_WET', 'LITHUANIA': 'TEMPERATE_WET', 'LUXEMBOURG': 'TEMPERATE_WET', 
    'NETHERLANDS': 'TEMPERATE_WET', 'POLAND': 'TEMPERATE_WET', 'SLOVAKIA': 'TEMPERATE_WET', 
    'SLOVENIA': 'TEMPERATE_WET', 'SWEDEN': 'TEMPERATE_WET', 'UK': 'TEMPERATE_WET', 
    'USMCA': 'TEMPERATE_WET', 'CHINA': 'TROPICAL_WET',
    'AUSTRIA': 'TEMPERATE_DRY', 'BULGARIA': 'TEMPERATE_DRY', 'CROATIA': 'TEMPERATE_DRY', 
    'CYPRUS': 'TROPICAL_DRY', 'GREECE': 'TROPICAL_DRY', 'ITALY': 'TROPICAL_DRY', 
    'MALTA': 'TROPICAL_DRY', 'PORTUGAL': 'TROPICAL_DRY', 'ROMANIA': 'TEMPERATE_DRY', 
    'SPAIN': 'TROPICAL_DRY', 'RUSSIA': 'TEMPERATE_DRY',
    'INDIA': 'TROPICAL_WET', 'LATAM': 'TROPICAL_WET', 'EASOC': 'TROPICAL_WET',
    'LROW': 'TROPICAL_WET'
}

fracciones_residuos_wiliam = {}
for region, clima in mapa_regiones_clima.items():
    fracciones_residuos_wiliam[region] = {}
    for residuo, params in params_estaticos.items():
        fracciones_residuos_wiliam[region][residuo] = params.copy()
        fracciones_residuos_wiliam[region][residuo]['k_bounds'] = limites_k_clima[clima][residuo]

# Factores empíricos de recuperación de metano R(t) y oxidación en cobertura OX(t)
R_DICT_EMPIRICO = {
    'AUSTRIA': np.array([0.2827, 0.2826, 0.2824, 0.2823, 0.2821, 0.2820, 0.2819, 0.2818, 0.2817, 0.2816, 0.2815, 0.2815, 0.2814, 0.2813, 0.2813, 0.2812, 0.2812, 0.2812, 0.2811]),
    'BELGIUM': np.array([0.2827, 0.2826, 0.2824, 0.2823, 0.2821, 0.2820, 0.2819, 0.2818, 0.2817, 0.2816, 0.2815, 0.2815, 0.2814, 0.2813, 0.2813, 0.2812, 0.2812, 0.2812, 0.2811]),
    'BULGARIA': np.array([0.0966, 0.0965, 0.0964, 0.0964, 0.0964, 0.0963, 0.0963, 0.0962, 0.0962, 0.0962, 0.0962, 0.0961, 0.0961, 0.0961, 0.0961, 0.0961, 0.0960, 0.0960, 0.0960]),
    'CROATIA': np.array([0.0662, 0.0661, 0.0661, 0.0661, 0.0660, 0.0660, 0.0660, 0.0660, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0658, 0.0658, 0.0658, 0.0658, 0.0658]),
    'CYPRUS': np.array([0.0662, 0.0661, 0.0661, 0.0661, 0.0660, 0.0660, 0.0660, 0.0660, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0658, 0.0658, 0.0658, 0.0658, 0.0658]),
    'CZECH_REPUBLIC': np.array([0.0966, 0.0965, 0.0964, 0.0964, 0.0964, 0.0963, 0.0963, 0.0962, 0.0962, 0.0962, 0.0962, 0.0961, 0.0961, 0.0961, 0.0961, 0.0961, 0.0960, 0.0960, 0.0960]),
    'DENMARK': np.array([0.1160, 0.1160, 0.1159, 0.1158, 0.1158, 0.1157, 0.1157, 0.1156, 0.1156, 0.1156, 0.1155, 0.1155, 0.1155, 0.1155, 0.1154, 0.1154, 0.1154, 0.1154, 0.1154]),
    'ESTONIA': np.array([0.1160, 0.1160, 0.1159, 0.1158, 0.1158, 0.1157, 0.1157, 0.1156, 0.1156, 0.1156, 0.1155, 0.1155, 0.1155, 0.1155, 0.1154, 0.1154, 0.1154, 0.1154, 0.1154]),
    'FINLAND': np.array([0.1160, 0.1160, 0.1159, 0.1158, 0.1158, 0.1157, 0.1157, 0.1156, 0.1156, 0.1156, 0.1155, 0.1155, 0.1155, 0.1155, 0.1154, 0.1154, 0.1154, 0.1154, 0.1154]),
    'FRANCE': np.array([0.2827, 0.2826, 0.2824, 0.2823, 0.2821, 0.2820, 0.2819, 0.2818, 0.2817, 0.2816, 0.2815, 0.2815, 0.2814, 0.2813, 0.2813, 0.2812, 0.2812, 0.2812, 0.2811]),
    'GERMANY': np.array([0.2827, 0.2826, 0.2824, 0.2823, 0.2821, 0.2820, 0.2819, 0.2818, 0.2817, 0.2816, 0.2815, 0.2815, 0.2814, 0.2813, 0.2813, 0.2812, 0.2812, 0.2812, 0.2811]),
    'GREECE': np.array([0.0662, 0.0661, 0.0661, 0.0661, 0.0660, 0.0660, 0.0660, 0.0660, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0658, 0.0658, 0.0658, 0.0658, 0.0658]),
    'HUNGARY': np.array([0.0966, 0.0965, 0.0964, 0.0964, 0.0964, 0.0963, 0.0963, 0.0962, 0.0962, 0.0962, 0.0962, 0.0961, 0.0961, 0.0961, 0.0961, 0.0961, 0.0960, 0.0960, 0.0960]),
    'IRELAND': np.array([0.1160, 0.1160, 0.1159, 0.1158, 0.1158, 0.1157, 0.1157, 0.1156, 0.1156, 0.1156, 0.1155, 0.1155, 0.1155, 0.1155, 0.1154, 0.1154, 0.1154, 0.1154, 0.1154]),
    'ITALY': np.array([0.0662, 0.0661, 0.0661, 0.0661, 0.0660, 0.0660, 0.0660, 0.0660, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0658, 0.0658, 0.0658, 0.0658, 0.0658]),
    'LATVIA': np.array([0.1160, 0.1160, 0.1159, 0.1158, 0.1158, 0.1157, 0.1157, 0.1156, 0.1156, 0.1156, 0.1155, 0.1155, 0.1155, 0.1155, 0.1154, 0.1154, 0.1154, 0.1154, 0.1154]),
    'LITHUANIA': np.array([0.1160, 0.1160, 0.1159, 0.1158, 0.1158, 0.1157, 0.1157, 0.1156, 0.1156, 0.1156, 0.1155, 0.1155, 0.1155, 0.1155, 0.1154, 0.1154, 0.1154, 0.1154, 0.1154]),
    'LUXEMBOURG': np.array([0.2827, 0.2826, 0.2824, 0.2823, 0.2821, 0.2820, 0.2819, 0.2818, 0.2817, 0.2816, 0.2815, 0.2815, 0.2814, 0.2813, 0.2813, 0.2812, 0.2812, 0.2812, 0.2811]),
    'MALTA': np.array([0.0662, 0.0661, 0.0661, 0.0661, 0.0660, 0.0660, 0.0660, 0.0660, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0658, 0.0658, 0.0658, 0.0658, 0.0658]),
    'NETHERLANDS': np.array([0.2827, 0.2826, 0.2824, 0.2823, 0.2821, 0.2820, 0.2819, 0.2818, 0.2817, 0.2816, 0.2815, 0.2815, 0.2814, 0.2813, 0.2813, 0.2812, 0.2812, 0.2812, 0.2811]),
    'POLAND': np.array([0.0966, 0.0965, 0.0964, 0.0964, 0.0964, 0.0963, 0.0963, 0.0962, 0.0962, 0.0962, 0.0962, 0.0961, 0.0961, 0.0961, 0.0961, 0.0961, 0.0960, 0.0960, 0.0960]),
    'PORTUGAL': np.array([0.0662, 0.0661, 0.0661, 0.0661, 0.0660, 0.0660, 0.0660, 0.0660, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0658, 0.0658, 0.0658, 0.0658, 0.0658]),
    'ROMANIA': np.array([0.0966, 0.0965, 0.0964, 0.0964, 0.0964, 0.0963, 0.0963, 0.0962, 0.0962, 0.0962, 0.0962, 0.0961, 0.0961, 0.0961, 0.0961, 0.0961, 0.0960, 0.0960, 0.0960]),
    'SLOVAKIA': np.array([0.0966, 0.0965, 0.0964, 0.0964, 0.0964, 0.0963, 0.0963, 0.0962, 0.0962, 0.0962, 0.0962, 0.0961, 0.0961, 0.0961, 0.0961, 0.0961, 0.0960, 0.0960, 0.0960]),
    'SLOVENIA': np.array([0.0662, 0.0661, 0.0661, 0.0661, 0.0660, 0.0660, 0.0660, 0.0660, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0658, 0.0658, 0.0658, 0.0658, 0.0658]),
    'SPAIN': np.array([0.0662, 0.0661, 0.0661, 0.0661, 0.0660, 0.0660, 0.0660, 0.0660, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0659, 0.0658, 0.0658, 0.0658, 0.0658, 0.0658]),
    'SWEDEN': np.array([0.1160, 0.1160, 0.1159, 0.1158, 0.1158, 0.1157, 0.1157, 0.1156, 0.1156, 0.1156, 0.1155, 0.1155, 0.1155, 0.1155, 0.1154, 0.1154, 0.1154, 0.1154, 0.1154]),
    'UK': np.array([0.1160, 0.1160, 0.1159, 0.1158, 0.1158, 0.1157, 0.1157, 0.1156, 0.1156, 0.1156, 0.1155, 0.1155, 0.1155, 0.1155, 0.1154, 0.1154, 0.1154, 0.1154, 0.1154]),
    'CHINA': np.array([0.1773, 0.1772, 0.1771, 0.1770, 0.1770, 0.1769, 0.1768, 0.1768, 0.1767, 0.1766, 0.1766, 0.1765, 0.1765, 0.1765, 0.1764, 0.1764, 0.1764, 0.1764, 0.1763]),
    'EASOC': np.array([0.2687, 0.2685, 0.2684, 0.2683, 0.2681, 0.2680, 0.2679, 0.2678, 0.2677, 0.2677, 0.2676, 0.2675, 0.2675, 0.2674, 0.2673, 0.2673, 0.2673, 0.2672, 0.2672]),
    'INDIA': np.array([0.5634, 0.5631, 0.5628, 0.5625, 0.5622, 0.5620, 0.5618, 0.5616, 0.5614, 0.5612, 0.5611, 0.5609, 0.5608, 0.5607, 0.5606, 0.5605, 0.5604, 0.5603, 0.5602]),
    'LATAM': np.array([0.2316, 0.2315, 0.2314, 0.2312, 0.2311, 0.2310, 0.2309, 0.2309, 0.2308, 0.2307, 0.2306, 0.2306, 0.2305, 0.2305, 0.2304, 0.2304, 0.2304, 0.2303, 0.2303]),
    'RUSSIA': np.array([0.0966, 0.0965, 0.0964, 0.0964, 0.0964, 0.0963, 0.0963, 0.0962, 0.0962, 0.0962, 0.0962, 0.0961, 0.0961, 0.0961, 0.0961, 0.0961, 0.0960, 0.0960, 0.0960]),
    'USMCA': np.array([0.0300, 0.0299, 0.0299, 0.0299, 0.0299, 0.0299, 0.0299, 0.0299, 0.0298, 0.0298, 0.0298, 0.0298, 0.0298, 0.0298, 0.0298, 0.0298, 0.0298, 0.0298, 0.0298]),
    'LROW': np.array([0.2481, 0.2480, 0.2478, 0.2477, 0.2476, 0.2475, 0.2474, 0.2473, 0.2472, 0.2471, 0.2471, 0.2470, 0.2470, 0.2469, 0.2469, 0.2468, 0.2468, 0.2467, 0.2467]),
}

OX_DICT_EMPIRICO = {
    'AUSTRIA': np.array([0.05]*19), 'BELGIUM': np.array([0.05]*19), 'BULGARIA': np.array([0.05]*19),
    'CROATIA': np.array([0.05]*19), 'CYPRUS': np.array([0.05]*19), 'CZECH_REPUBLIC': np.array([0.05]*19),
    'DENMARK': np.array([0.05]*19), 'ESTONIA': np.array([0.05]*19), 'FINLAND': np.array([0.05]*19),
    'FRANCE': np.array([0.05]*19), 'GERMANY': np.array([0.05]*19), 'GREECE': np.array([0.05]*19),
    'HUNGARY': np.array([0.05]*19), 'IRELAND': np.array([0.05]*19), 'ITALY': np.array([0.05]*19),
    'LATVIA': np.array([0.05]*19), 'LITHUANIA': np.array([0.05]*19), 'LUXEMBOURG': np.array([0.05]*19),
    'MALTA': np.array([0.05]*19), 'NETHERLANDS': np.array([0.05]*19), 'POLAND': np.array([0.05]*19),
    'PORTUGAL': np.array([0.05]*19), 'ROMANIA': np.array([0.05]*19), 'SLOVAKIA': np.array([0.05]*19),
    'SLOVENIA': np.array([0.05]*19), 'SPAIN': np.array([0.05]*19), 'SWEDEN': np.array([0.05]*19),
    'UK': np.array([0.05]*19), 'CHINA': np.array([0.05]*19), 'EASOC': np.array([0.05]*19),
    'INDIA': np.array([0.05]*19), 'LATAM': np.array([0.05]*19), 'RUSSIA': np.array([0.05]*19),
    'USMCA': np.array([0.05]*19), 'LROW': np.array([0.05]*19)
}


# =============================================================================
# 2. CARGA DE BASES DE DATOS Y EXTRACCIÓN TEMPORAL
# =============================================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_EMIS = os.path.join(BASE_DIR, 'data', 'HISTORICAL_GHG_EMISSIONS_REGIONS_IPCC_SECTORS_LEVEL_2.csv')
PATH_POP = os.path.join(BASE_DIR, 'data', 'Population_Regions.csv')
PATH_WASTE_PC = os.path.join(BASE_DIR, 'data', 'HISTORICAL_MUNICIPAL_WASTE_PER_CAPITA.csv')
PATH_DISP_PERC = os.path.join(BASE_DIR, 'data', 'HISTORICAL_MUNICIPAL_DISPOSAL_PERCENTAGE.csv')
PATH_GDP = os.path.join(BASE_DIR, 'data', 'GDP_Regions.csv')
PATH_CALIB = os.path.join(BASE_DIR, 'data', 'IPCC_Waste_Model_Reconstructed.csv')
PATH_EVO = os.path.join(BASE_DIR, 'data', 'Evolucion_Porcentajes_Vertederos_2005_2023.csv')

@st.cache_data
def load_heavy_data():
    """
    Carga y almacena en caché las matrices históricas de actividad socioeconómica y emisiones.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - df_emis: Emisiones históricas observadas de metano (EDGAR/IPCC).
            - df_pop: Series temporales de población regional (habitantes).
            - df_waste_pc: Generación municipal de residuos per cápita (kg/hab/año).
            - df_disp_perc: Porcentaje histórico de residuos derivados a vertedero (%).
            - df_gdp: Producto Interior Bruto regionalizado (USD constantes / moneda base).
    """
    try:
        df_emis = pd.read_csv(PATH_EMIS, header=None, skiprows=1)
        df_pop = pd.read_csv(PATH_POP, header=None)
        df_waste_pc = pd.read_csv(PATH_WASTE_PC, header=None)
        df_disp_perc = pd.read_csv(PATH_DISP_PERC, header=None)
        df_gdp = pd.read_csv(PATH_GDP, header=None, skiprows=1)
        return df_emis, df_pop, df_waste_pc, df_disp_perc, df_gdp
    except Exception as e:
        st.error(f"Error crítico al cargar los datos base: {e}")
        df_vacio = pd.DataFrame()
        return df_vacio, df_vacio, df_vacio, df_vacio, df_vacio
    
def load_calibracion():
    """
    Carga la matriz maestra de calibración preexistente con los parámetros regionales óptimos.

    Returns:
        pd.DataFrame: DataFrame con las constantes cinéticas k_i, stocks iniciales H0 y ratios GDP.
    """
    try:
        df = pd.read_csv(PATH_CALIB, encoding='utf-8-sig', sep=None, engine='python')
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.DataFrame()

def load_evolucion_porcentajes():
    """
    Carga la matriz de evolución temporal de composición porcentual de residuos depositados.

    Returns:
        pd.DataFrame: DataFrame con la distribución anual de DDOCma por fracción.
    """
    try:
        df = pd.read_csv(PATH_EVO, encoding='utf-8-sig', sep=None, engine='python')
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.DataFrame()

def extraer_serie(df, reg, años_target, end_year=2023):
    """
    Extrae e interpola la serie temporal numérica de una variable para una región específica.

    Args:
        df (pd.DataFrame): Matriz tabular con datos indexados por región en la primera columna.
        reg (str): Identificador de la región analizada.
        años_target (np.ndarray): Vector de años que definen el horizonte temporal.
        end_year (int, optional): Año final de corte. Por defecto es 2023.

    Returns:
        np.ndarray: Vector numérico continuo de la variable solicitada.
    """
    if df is None or df.empty: 
        return np.zeros(len(años_target))
    try:
        fila = df[df.iloc[:, 0] == reg]
        if fila.empty: 
            return np.zeros(len(años_target))
        
        valores = pd.to_numeric(fila.iloc[0, 1:], errors='coerce').fillna(0.0).values
        if len(valores) >= len(años_target):
            return valores[:len(años_target)]
        else:
            arr = np.zeros(len(años_target))
            arr[:len(valores)] = valores
            return arr
    except Exception:
        return np.zeros(len(años_target))

def get_emisiones_target(df_emis, region, años_target, end_year=2023):
    """
    Extrae el vector de emisiones históricas oficiales de CH4 para el Sector 4.A (Vertederos).

    Fundamento Metrológico:
        Convierte los registros del inventario (reportados en Gg o kt de CH4) a megatoneladas
        (Mt CH4 / año) mediante el factor de escala 10⁻³.

    Args:
        df_emis (pd.DataFrame): Base de datos de emisiones EDGAR/IPCC Level 2.
        region (str): Nombre de la región objetivo.
        años_target (np.ndarray): Intervalo de años de análisis.
        end_year (int, optional): Año límite superior (2023).

    Returns:
        np.ndarray: Vector de emisiones observadas en megatoneladas anuales (Mt CH4 / año).
    """
    if df_emis is None or df_emis.empty:
        return np.zeros(len(años_target))
    try:
        mask = (df_emis.iloc[:, 0] == region) & (df_emis.iloc[:, 1] == '4_A__Solid_Waste_Disposal') & (df_emis.iloc[:, 2] == 'CH4')
        fila = df_emis[mask]
        
        if fila.empty:
            return np.zeros(len(años_target))
            
        valores_edgar_kt = fila.iloc[:, 3:3+len(años_target)].values.flatten().astype(float)
        
        # Conversión de unidades: Gg (kt) a Megatoneladas (Mt = Tg)
        return valores_edgar_kt / 1000.0
        
    except Exception as e:
        print(f"Error extrayendo el target de {region}: {e}")
        return np.zeros(len(años_target))

def get_historical_waste_vectors(region, años_target, df_pop, df_waste_pc, df_disp_perc, df_gdp, *args):
    """
    Reconstruye los vectores anuales de masa de residuos vertidos y magnitudes macroeconómicas.

    Fundamento Matemático:
        Para cada región:
            W_MSW(t) = [Población(t) · Generación_pc(t) · (Tasa_Vertido(t) / 100)] / 10⁹  (Mt/año)
        En el caso de 'EU27', realiza la agregación dinámica país por país de los 27 Estados Miembros
        sumando la masa física enviada a vertedero, la población y el PIB.

    Args:
        region (str): Región o país de análisis.
        años_target (np.ndarray): Vector de años de evaluación.
        df_pop (pd.DataFrame): Matriz demográfica.
        df_waste_pc (pd.DataFrame): Matriz de generación per cápita (kg/hab/año).
        df_disp_perc (pd.DataFrame): Matriz de tasa de depósito en vertedero (%).
        df_gdp (pd.DataFrame): Matriz de Producto Interior Bruto.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - W_MSW: Masa anual de RSU depositada en vertedero (Mt / año).
            - pop_millones: Población en millones de habitantes.
            - disp_perc: Porcentaje agregado de depósito en vertedero (%).
            - gdp: Serie temporal de PIB.
    """
    try:
        n_years = len(años_target)
        
        # Agregación dinámica multifactorial para el bloque comunitario EU27
        if region == 'EU27':
            pop_eu27 = np.zeros(n_years)
            gdp_eu27 = np.zeros(n_years)
            W_MSW_eu27 = np.zeros(n_years)
            
            fila_disp_eu27 = df_disp_perc[df_disp_perc.iloc[:, 0] == 'EU27']
            disp_perc_eu27 = fila_disp_eu27.iloc[:, 16:16+n_years].values.flatten().astype(float) if not fila_disp_eu27.empty else np.zeros(n_years)
            disp_perc_eu27 = np.nan_to_num(disp_perc_eu27)
            
            for pais in EU27_REGIONS:
                fila_pop = df_pop[df_pop.iloc[:, 0] == pais]
                pop_pais = fila_pop.iloc[:, 36:36+n_years].values.flatten().astype(float) if not fila_pop.empty else np.zeros(n_years)
                
                fila_gdp = df_gdp[df_gdp.iloc[:, 0] == pais]
                gdp_pais = fila_gdp.iloc[:, 36:36+n_years].values.flatten().astype(float) if not fila_gdp.empty else np.zeros(n_years)
                
                fila_disp = df_disp_perc[df_disp_perc.iloc[:, 0] == pais]
                disp_perc_pais = fila_disp.iloc[:, 16:16+n_years].values.flatten().astype(float) if not fila_disp.empty else np.zeros(n_years)
                
                fila_wpc = df_waste_pc[df_waste_pc.iloc[:, 0] == pais]
                waste_pc_pais = fila_wpc.iloc[:, 1:1+n_years].values.flatten().astype(float) if not fila_wpc.empty else np.zeros(n_years)
                
                pop_pais = np.nan_to_num(pop_pais)
                gdp_pais = np.nan_to_num(gdp_pais)
                disp_perc_pais = np.nan_to_num(disp_perc_pais)
                waste_pc_pais = np.nan_to_num(waste_pc_pais)
                
                # Conversión de masa: kg a Megatoneladas (factor 10^-9)
                W_MSW_pais = (pop_pais * waste_pc_pais * (disp_perc_pais * 0.01)) / 1e9
                
                pop_eu27 += pop_pais
                gdp_eu27 += gdp_pais
                W_MSW_eu27 += W_MSW_pais
                
            pop_millones_eu27 = pop_eu27 / 1e6
            
            return W_MSW_eu27, pop_millones_eu27, disp_perc_eu27, gdp_eu27

        # Procesamiento estandarizado para regiones individuales
        else:
            fila_pop = df_pop[df_pop.iloc[:, 0] == region]
            pop = fila_pop.iloc[:, 36:36+n_years].values.flatten().astype(float) if not fila_pop.empty else np.zeros(n_years)
            
            fila_gdp = df_gdp[df_gdp.iloc[:, 0] == region]
            gdp = fila_gdp.iloc[:, 36:36+n_years].values.flatten().astype(float) if not fila_gdp.empty else np.zeros(n_years)
            
            fila_disp = df_disp_perc[df_disp_perc.iloc[:, 0] == region]
            disp_perc = fila_disp.iloc[:, 16:16+n_years].values.flatten().astype(float) if not fila_disp.empty else np.zeros(n_years)
            
            fila_wpc = df_waste_pc[df_waste_pc.iloc[:, 0] == region]
            waste_pc = fila_wpc.iloc[:, 1:1+n_years].values.flatten().astype(float) if not fila_wpc.empty else np.zeros(n_years)
            
            pop = np.nan_to_num(pop)
            gdp = np.nan_to_num(gdp)
            disp_perc = np.nan_to_num(disp_perc)
            waste_pc = np.nan_to_num(waste_pc)
            
            # Conversión de masa: kg a Megatoneladas (factor 10^-9)
            W_MSW_array = (pop * waste_pc * (disp_perc * 0.01)) / 1e9
            pop_millones_array = pop / 1e6
            
            return W_MSW_array, pop_millones_array, disp_perc, gdp 
            
    except Exception as e:
        print(f"Error procesando vectores de basura para {region}: {e}")
        n_zeros = len(años_target)
        return np.zeros(n_zeros), np.zeros(n_zeros), np.zeros(n_zeros), np.zeros(n_zeros)


# =============================================================================
# 3. MOTOR DE SIMULACIÓN FIRST ORDER DECAY (FOD) MULTIFRACCIONAL
# =============================================================================
def simular_multiphase_FOD(W_MSW_Mt, W_IND_Mt, k_dict, initial_ddocma_dict, mcf_msw_array, mcf_ind_array, ox_array, R_array, fracciones_region, return_history=False):
    """
    Ejecuta el Modelo de Decaimiento de Primer Orden (FOD) multifraccional del IPCC 2006.

    Fundamento Matemático (IPCC 2006 Vol. 5, Cap. 3, Ec. 3.4 y 3.5):
        Para cada fracción de residuo i con constante de degradación k_i:
            DDOCmd_i(t) = W_i(t) · DOC_i · DOC_f,i · MCF(t)
            DDOCm_decomp_i(t) = DDOCma_i(t-1) · (1 - exp(-k_i))
            CH4_bruto_i(t) = DDOCm_decomp_i(t) · F · (16 / 12)
            DDOCma_i(t) = DDOCmd_i(t) + DDOCma_i(t-1) · exp(-k_i)

        Emisiones netas a la atmósfera:
            CH4_neto(t) = [∑_i CH4_bruto_i(t) - R(t)] · (1 - OX(t))

    Args:
        W_MSW_Mt (np.ndarray): Masa anual de residuos municipales en vertedero (Mt / año).
        W_IND_Mt (np.ndarray): Masa anual de residuos industriales en vertedero (Mt / año).
        k_dict (dict[str, float]): Constantes cinéticas de descomposición (año⁻¹).
        initial_ddocma_dict (dict[str, float]): Stock inicial de carbono reactivo en el vertedero (Mt).
        mcf_msw_array (np.ndarray): Factor de Corrección de Metano para vertederos municipales.
        mcf_ind_array (np.ndarray): Factor de Corrección de Metano para residuos industriales.
        ox_array (np.ndarray): Factor de oxidación en la capa de cobertura del vertedero.
        R_array (np.ndarray): Metano recuperado y extraído (Mt CH4 / año).
        fracciones_region (dict): Parámetros DOC, DOCf y proporciones de mezcla regional.
        return_history (bool, optional): Si es True, retorna el desglose temporal de DDOCma por fracción.

    Returns:
        tuple: Si return_history es False:
            - emisiones_totales_array (np.ndarray): Emisiones netas a la atmósfera (Mt CH4 / año).
            - ch4_generado_array (np.ndarray): Metano bruto generado antes de recuperación (Mt CH4 / año).
            - ddocma_total_array (np.ndarray): Stock acumulado total de carbono reactivo DDOCma (Mt C).
        Si return_history es True:
            - emisiones_totales_array (np.ndarray): Emisiones netas a la atmósfera (Mt CH4 / año).
            - historial_ddocma (dict[str, np.ndarray]): Desglose de DDOCma por cada fracción residual.
    """
    F_CH4 = 0.5
    RATIO_16_12 = 16.0 / 12.0
    
    longitud_anios = len(W_MSW_Mt)
    
    emisiones_totales_array = np.zeros(longitud_anios)
    ch4_generado_array = np.zeros(longitud_anios)
    ddocma_total_array = np.zeros(longitud_anios)
    
    ALL_WASTE_TYPES = list(k_dict.keys())
    MSW_TYPES = [tipo for tipo in ALL_WASTE_TYPES if tipo != 'INDUSTRIAL_WASTE']
    
    ddocma_actual = initial_ddocma_dict.copy()
    
    historial_ddocma = {tipo: np.zeros(longitud_anios) for tipo in ALL_WASTE_TYPES}
    
    for i in range(longitud_anios):
        ch4_generado_total_año = 0.0
        
        # Procesamiento cinético de residuos municipales (MSW)
        for tipo in MSW_TYPES:
            W_tipo = W_MSW_Mt[i] * fracciones_region[tipo].get('comp_msw', 0.0)
            doc = fracciones_region[tipo]['doc']
            docf = fracciones_region[tipo]['docf']
            mcf = mcf_msw_array[i]
            k = k_dict[tipo]
            
            ddocmd = W_tipo * doc * docf * mcf
            ddocm_decomp = ddocma_actual[tipo] * (1.0 - np.exp(-k))
            
            ch4_generado_total_año += ddocm_decomp * F_CH4 * RATIO_16_12
            ddocma_actual[tipo] = ddocmd + (ddocma_actual[tipo] * np.exp(-k))
            
            historial_ddocma[tipo][i] = ddocma_actual[tipo]

        # Procesamiento cinético de residuos industriales asimilables (IND)
        if 'INDUSTRIAL_WASTE' in k_dict:
            W_ind = W_IND_Mt[i]
            doc_ind = fracciones_region['INDUSTRIAL_WASTE']['doc']
            docf_ind = fracciones_region['INDUSTRIAL_WASTE']['docf']
            mcf_ind = mcf_ind_array[i]
            k_ind = k_dict['INDUSTRIAL_WASTE']
            
            ddocmd_ind = W_ind * doc_ind * docf_ind * mcf_ind
            ddocm_decomp_ind = ddocma_actual['INDUSTRIAL_WASTE'] * (1.0 - np.exp(-k_ind))
            
            ch4_generado_total_año += ddocm_decomp_ind * F_CH4 * RATIO_16_12
            ddocma_actual['INDUSTRIAL_WASTE'] = ddocmd_ind + (ddocma_actual['INDUSTRIAL_WASTE'] * np.exp(-k_ind))
            
            historial_ddocma['INDUSTRIAL_WASTE'][i] = ddocma_actual['INDUSTRIAL_WASTE']
            
        # Balance neto descontando recuperación y oxidación
        R_recuperado = R_array[i]
        factor_oxidacion = ox_array[i]
        ch4_neto = (ch4_generado_total_año - R_recuperado) * (1.0 - factor_oxidacion)
        
        emisiones_totales_array[i] = ch4_neto if ch4_neto > 0 else 0.0
        ch4_generado_array[i] = ch4_generado_total_año
        # Acumulación de la masa reactiva total biodegradable en vertedero (Mt C)
        ddocma_total_array[i] = sum(ddocma_actual.values())

    if return_history:
        return emisiones_totales_array, historial_ddocma
        
    return emisiones_totales_array, ch4_generado_array, ddocma_total_array


# =============================================================================
# 4. OPTIMIZADOR NUMÉRICO MULTIVARIABLE (MÉTODO DE POWELL 12D)
# =============================================================================
def optimizar_tier4a_12D(W_MSW_array, GDP_array, RATIO_IND, target_emissions, mcf_msw_array, mcf_ind_array, ox_array, R_array, ALL_WASTE_TYPES, fracciones_region, composicion_2005_dict):
    """
    Optimiza simultáneamente los 12 parámetros del modelo FOD mediante el Método de Powell.

    Vector de Decisión (θ ∈ ℝ¹²):
        θ = [k_1, k_2, ..., k_9, H0_MSW, H0_IND, WASTE_PER_GDP]
        - k_1..k_9: 9 constantes cinéticas acotadas por régimen bioclimático IPCC (año⁻¹).
        - H0_MSW: Stock acumulado de carbono previo de origen municipal (Mt).
        - H0_IND: Stock acumulado de carbono previo de origen industrial (Mt).
        - WASTE_PER_GDP: Factor de intensidad de generación residual por unidad de PIB.

    Normalización Afín y Espacio de Búsqueda:
        Se aplica una transformación lineal θ_norm ∈ [0, 1]¹² para desacoplar las escalas
        físicas y condicionar numéricamente el hessiano implícito en el espacio de parámetros.

    Función Objetivo (SSE):
        J(θ) = ∑ (CH4_sim(t, θ) - CH4_obs(t))² + Barrera_Penalización

    Args:
        W_MSW_array (np.ndarray): Masa de RSU vertido (Mt / año).
        GDP_array (np.ndarray): Serie temporal de PIB.
        RATIO_IND (float): Coeficiente estructural de residuo industrial.
        target_emissions (np.ndarray): Serie de emisiones históricas de referencia (Mt CH4 / año).
        mcf_msw_array (np.ndarray): Serie temporal de MCF municipal.
        mcf_ind_array (np.ndarray): Serie temporal de MCF industrial.
        ox_array (np.ndarray): Factor de oxidación anual.
        R_array (np.ndarray): Metano recuperado anual (Mt CH4 / año).
        ALL_WASTE_TYPES (list[str]): Lista de las 9 categorías residuales modeladas.
        fracciones_region (dict): Límites bioclimáticos y propiedades de DOC y DOCf.
        composicion_2005_dict (dict): Composición gravimétrica relativa de referencia.

    Returns:
        tuple | None: Tupla con (k_opt, H0_MSW_opt, H0_IND_opt, WASTE_PER_GDP_opt,
            W_IND_Mt_opt, best_simulation_Mt, SSE_optimo), o None si los datos son inválidos.
    """
    W_MSW_Mt = np.nan_to_num(np.asarray(W_MSW_array, dtype=np.float64), nan=0.0)
    gdp_t = np.nan_to_num(np.asarray(GDP_array, dtype=np.float64), nan=0.0)
    
    e_t = np.nan_to_num(np.asarray(target_emissions, dtype=np.float64), nan=0.0)
    mcf_msw = np.nan_to_num(np.asarray(mcf_msw_array, dtype=np.float64), nan=0.0)
    mcf_ind = np.nan_to_num(np.asarray(mcf_ind_array, dtype=np.float64), nan=0.0)
    ox = np.nan_to_num(np.asarray(ox_array, dtype=np.float64), nan=0.0)
    R = np.nan_to_num(np.asarray(R_array, dtype=np.float64), nan=0.0)

    if np.all(e_t == 0) or len(e_t) == 0: 
        return None

    lb_phys, ub_phys = [], []
    for w in ALL_WASTE_TYPES:
        k_min, k_max = fracciones_region[w]['k_bounds']
        lb_phys.append(k_min)
        ub_phys.append(k_max)
        
    # Límites físicos para variables macroeconómicas y stocks históricos: H0_MSW, H0_IND y WASTE_PER_GDP
    lb_phys.extend([0.0, 0.0, 0.0])
    ub_phys.extend([5000.0, 5000.0, 100.0]) 
    
    lb_phys = np.array(lb_phys, dtype=np.float64)
    ub_phys = np.array(ub_phys, dtype=np.float64)
    scale = ub_phys - lb_phys

    num_wastes = len(ALL_WASTE_TYPES)
    num_vars = num_wastes + 3

    def desnormalizar(p_norm):
        return lb_phys + p_norm * scale

    def objective_function(vars_opt):
        p_safe = np.clip(vars_opt, 0.0, 1.0)
        p = desnormalizar(p_safe)
        
        k_dict = {w: p[i] for i, w in enumerate(ALL_WASTE_TYPES)}
        
        H0_MSW_Mt = p[-3]
        H0_IND_Mt = p[-2]
        WASTE_PER_GDP_opt = p[-1]
        
        # Generación macroeconómica del residuo industrial acoplada al PIB
        W_IND_Mt = (gdp_t * WASTE_PER_GDP_opt * RATIO_IND) / 1000.0
        
        total_msw_weight = sum([composicion_2005_dict.get(w, 0.0) for w in ALL_WASTE_TYPES if w != 'INDUSTRIAL_WASTE'])
        
        initial_ddocma_dict = {}
        for w in ALL_WASTE_TYPES:
            if w == 'INDUSTRIAL_WASTE':
                initial_ddocma_dict[w] = H0_IND_Mt
            else:
                peso_relativo = composicion_2005_dict.get(w, 0.0) / total_msw_weight if total_msw_weight > 0 else 0.0
                initial_ddocma_dict[w] = H0_MSW_Mt * peso_relativo
        
        simulated_Mt, _, _ = simular_multiphase_FOD(W_MSW_Mt, W_IND_Mt, k_dict, initial_ddocma_dict, mcf_msw, mcf_ind, ox, R, fracciones_region)
        
        if np.any(simulated_Mt < 0) or np.isnan(simulated_Mt).any() or np.isinf(simulated_Mt).any():
            return 1e9 + np.sum(np.abs(simulated_Mt))
            
        return np.sum((simulated_Mt - e_t)**2)

    x0_norm = np.full(num_vars, 0.5, dtype=np.float64)
    x0_norm[-3:] = 0.05
    bnds = [(0.0, 1.0)] * num_vars

    res = minimize(objective_function, x0_norm, bounds=bnds, method='Powell', options={'maxiter': 2000})
    
    if hasattr(res, 'x'):
        res.x = np.clip(res.x, 0.0, 1.0)
        p_opt = desnormalizar(res.x)
        
        k_opt = {w: p_opt[i] for i, w in enumerate(ALL_WASTE_TYPES)}
        H0_MSW_opt = p_opt[-3]
        H0_IND_opt = p_opt[-2]
        WASTE_PER_GDP_opt = p_opt[-1]
        
        W_IND_Mt_opt = (gdp_t * WASTE_PER_GDP_opt * RATIO_IND) / 1000.0
        
        total_msw_weight = sum([composicion_2005_dict.get(w, 0.0) for w in ALL_WASTE_TYPES if w != 'INDUSTRIAL_WASTE'])
        initial_ddocma_dict = {}
        for w in ALL_WASTE_TYPES:
            if w == 'INDUSTRIAL_WASTE':
                initial_ddocma_dict[w] = H0_IND_opt
            else:
                peso_relativo = composicion_2005_dict.get(w, 0.0) / total_msw_weight if total_msw_weight > 0 else 0.0
                initial_ddocma_dict[w] = H0_MSW_opt * peso_relativo
                
        best_simulation_Mt, _, _ = simular_multiphase_FOD(W_MSW_Mt, W_IND_Mt_opt, k_opt, initial_ddocma_dict, mcf_msw, mcf_ind, ox, R, fracciones_region)
        
        return k_opt, H0_MSW_opt, H0_IND_opt, WASTE_PER_GDP_opt, W_IND_Mt_opt, best_simulation_Mt, res.fun
        
    return None


# =============================================================================
# 5. PROCEDIMIENTOS GLOBALES Y GENERACIÓN DE MATRICES WILIAM
# =============================================================================
def calcular_matriz_global(df_calibracion_local):
    """
    Ejecuta el ciclo de calibración automatizado para todas las regiones del modelo WILIAM.

    Procesa secuencialmente cada región geográfica válida, ajusta el vector de 12 parámetros
    mediante el optimizador de Powell, extrae las dinámicas de composición temporal de DDOCma
    y estructura los DataFrames consolidados para su exportación a WILIAM.

    Args:
        df_calibracion_local (pd.DataFrame): Matriz base de calibración histórica.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - df_tabla: Resumen de parámetros óptimos y error cuadrático por región.
            - df_calibracion: Matriz maestra de coeficientes compatible con WILIAM.
            - df_evolucion: Evolución temporal del porcentaje de masa reactiva por fracción.
    """
    resultados_tabla = []
    resultados_calibracion = []
    resultados_evolucion = []  
    
    regiones_validas = lista_regiones.copy()
    años_analisis = np.arange(2005, 2024)
    longitud_anios = len(años_analisis)
  
    for reg in regiones_validas:
        e_target = get_emisiones_target(df_emis, reg, años_analisis)
        if e_target is None or np.all(e_target == 0): continue
        
        W_MSW, _, _, GDP_array = get_historical_waste_vectors(reg, años_analisis, df_pop, df_waste_pc, df_disp_perc, df_gdp)
        RATIO_IND = 1.0  
        
        mcf_msw_array = np.full(longitud_anios, 0.71)
        mcf_ind_array = np.full(longitud_anios, 0.72)
    
        R_array = R_DICT_EMPIRICO.get(reg, np.zeros(longitud_anios))
        ox_array = OX_DICT_EMPIRICO.get(reg, np.zeros(longitud_anios))
        
        min_len = min(len(e_target), len(W_MSW), len(GDP_array), len(mcf_msw_array), len(mcf_ind_array), len(ox_array), len(R_array))
        
        e_target = e_target[:min_len]
        W_MSW = W_MSW[:min_len]
        GDP_array = GDP_array[:min_len]
        mcf_msw_array = mcf_msw_array[:min_len]
        mcf_ind_array = mcf_ind_array[:min_len]
        ox_array = ox_array[:min_len]
        R_array = R_array[:min_len]
        
        res_tupla = optimizar_tier4a_12D(W_MSW, GDP_array, RATIO_IND, e_target, mcf_msw_array, mcf_ind_array, ox_array, R_array, ALL_WASTE_TYPES, fracciones_residuos_wiliam[reg], COMPOSICION_2005_DEFAULT)
        
        if res_tupla is not None:
            k_optimos, H0_MSW_Mt, H0_IND_Mt, waste_per_gdp_opt, W_IND_Mt_opt, e_sim_Mt, error_cuadratico = res_tupla
            
            fila = {
                'Región': reg,
                'Error Cuadrático': round(error_cuadratico, 2),
                'H0_MSW_Total (Mt)': round(H0_MSW_Mt, 1),
                'H0_IND_Total (Mt)': round(H0_IND_Mt, 1),
                'WASTE_PER_GDP_Optimo': waste_per_gdp_opt
            }
            for tipo in ALL_WASTE_TYPES:
                fila[f'k_{tipo}'] = round(k_optimos[tipo], 4)
            resultados_tabla.append(fila)
            
            fila_calib = {
                'Region': reg,
                'k_FOOD': k_optimos.get('FOOD_WASTE', 0), 
                'k_GARDEN': k_optimos.get('GARDEN_WASTE', 0),
                'k_PAPER': k_optimos.get('PAPER_WASTE', 0), 
                'k_WOOD': k_optimos.get('WOOD_STRAW_WASTE', 0),
                'k_TEXTILES': k_optimos.get('TEXTILES_WASTE', 0), 
                'k_NAPPIES': k_optimos.get('DISPOSABLE_NAPPIES_WASTE', 0),
                'k_SEWAGE_SLUDGE': k_optimos.get('SEWAGE_SLUDGE_WASTE', 0), 
                'k_RUBBER': k_optimos.get('RUBBER_WASTE', 0),
                'k_INDUSTRIAL': k_optimos.get('INDUSTRIAL_WASTE', 0),
                'H0_MSW': H0_MSW_Mt,
                'H0_IND': H0_IND_Mt,
                'WASTE_PER_GDP': waste_per_gdp_opt
            }
            resultados_calibracion.append(fila_calib)
            
            ddocm_inicial_optimo = {}
            
            # Asignación de stock inicial de carbono para residuos municipales
            for tipo in MSW_TYPES:
                W_tipo_2005 = H0_MSW_Mt * COMPOSICION_2005_DEFAULT.get(tipo, 0)
                doc_tipo = params_estaticos[tipo]['doc']
                docf_tipo = params_estaticos[tipo]['docf']
                ddocm_inicial_optimo[tipo] = W_tipo_2005 * doc_tipo * docf_tipo * mcf_msw_array[ 0 ]
            
            # Asignación de stock inicial de carbono para residuos industriales
            doc_ind = params_estaticos['INDUSTRIAL_WASTE']['doc']
            docf_ind = params_estaticos['INDUSTRIAL_WASTE']['docf']
            ddocm_inicial_optimo['INDUSTRIAL_WASTE'] = H0_IND_Mt * doc_ind * docf_ind * mcf_ind_array[ 0 ]
            
            _, historial_ddocma = simular_multiphase_FOD(
                W_MSW, W_IND_Mt_opt, k_optimos, ddocm_inicial_optimo, 
                mcf_msw_array, mcf_ind_array, ox_array, R_array, fracciones_residuos_wiliam[reg],
                return_history=True
            )
            
            for i, año in enumerate(años_analisis[:min_len]):
                ddocma_total_año = sum(historial_ddocma[tipo][i] for tipo in MSW_TYPES)
                
                fila_evo = {
                    'Region': reg,
                    'Year': año,
                    'DDOCma_Total': round(ddocma_total_año, 4)
                }
                
                for tipo in MSW_TYPES:
                    porcentaje = (historial_ddocma[tipo][i] / ddocma_total_año * 100) if ddocma_total_año > 0 else 0
                    fila_evo[f'%_{tipo}'] = round(porcentaje, 2)
                
                resultados_evolucion.append(fila_evo)
                
    return pd.DataFrame(resultados_tabla), pd.DataFrame(resultados_calibracion), pd.DataFrame(resultados_evolucion)


# =============================================================================
# 6. ENTORNO REACTIVO Y FLUJO PRINCIPAL DE STREAMLIT
# =============================================================================
try:
    df_emis, df_pop, df_waste_pc, df_disp_perc, df_gdp = load_heavy_data()
    df_calibracion = load_calibracion()
    
    if df_emis is not None and not df_emis.empty:
        lista_regiones = sorted(df_emis[0].dropna().unique())
    else:
        lista_regiones = EU27_REGIONS
    if 'EU27' not in lista_regiones: lista_regiones.insert(0, 'EU27')
except Exception as e:
    st.error(f"Error crítico cargando datos: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Selección de Región")
default_idx = lista_regiones.index('SPAIN') if 'SPAIN' in lista_regiones else 0
seleccion_reg = st.sidebar.selectbox("Elige la Región WILIAM para analizar:", lista_regiones, index=default_idx)

clima_region = mapa_regiones_clima.get(seleccion_reg, 'TEMPERATE_DRY')
st.sidebar.markdown(f"**Régimen Bioclimático:** `{clima_region}`")

btn_calibrar = st.sidebar.button("⚡ Ejecutar Calibración (Powell 12D)", type="primary")

st.markdown(f"### Panel de Simulación y Calibración — Región: **{seleccion_reg}**")
st.divider()

tab_region, tab_global = st.tabs(["Análisis por Región", "Resumen Global"])

with tab_region:
    años_analisis = np.arange(2005, 2024)
    longitud_anios = len(años_analisis)
    
    W_MSW, pop_millones_array, disp_perc_array, GDP_array = get_historical_waste_vectors(seleccion_reg, años_analisis, df_pop, df_waste_pc, df_disp_perc, df_gdp)
    RATIO_IND = 1.0
    
    mcf_msw_array = np.full(longitud_anios, 0.71)
    mcf_ind_array = np.full(longitud_anios, 0.72)
    e_target = get_emisiones_target(df_emis, seleccion_reg, años_analisis)

    R_array = R_DICT_EMPIRICO.get(seleccion_reg, np.zeros(longitud_anios))
    ox_array = OX_DICT_EMPIRICO.get(seleccion_reg, np.zeros(longitud_anios))

    min_len = min(len(e_target), len(W_MSW), len(GDP_array), len(mcf_msw_array), len(mcf_ind_array), len(ox_array), len(R_array), len(pop_millones_array), len(disp_perc_array))
    
    e_target = e_target[:min_len]
    W_MSW = W_MSW[:min_len]
    GDP_array = GDP_array[:min_len]
    mcf_msw_array = mcf_msw_array[:min_len]
    mcf_ind_array = mcf_ind_array[:min_len]
    ox_array = ox_array[:min_len]
    R_array = R_array[:min_len]
    pop_millones_array = pop_millones_array[:min_len]
    disp_perc_array = disp_perc_array[:min_len]
    años_analisis = años_analisis[:min_len]

    # Claves de estado de sesión para persistencia por región
    state_key_res = f"calib_res_{seleccion_reg}"
    state_key_calib = f"es_calibrado_{seleccion_reg}"

    # Ejecución bajo demanda al accionar el botón de calibración en la barra lateral
    if btn_calibrar:
        st.info("⚙️ Ejecutando Calibración Dinámica en Vivo (Motor Powell - Arquitectura 12D)")
        with st.spinner(f"Optimizando parámetros cinéticos y stocks iniciales para {seleccion_reg} mediante el Método de Powell..."):
            resultado_tupla = optimizar_tier4a_12D(
                W_MSW, GDP_array, RATIO_IND, e_target, mcf_msw_array, mcf_ind_array, ox_array, R_array, ALL_WASTE_TYPES, fracciones_residuos_wiliam[seleccion_reg], COMPOSICION_2005_DEFAULT
            )
            if resultado_tupla is not None:
                st.session_state[state_key_res] = resultado_tupla
                st.session_state[state_key_calib] = True

    # Si ya se calibró en la sesión, recuperar de st.session_state
    if state_key_res in st.session_state:
        resultado_tupla = st.session_state[state_key_res]
        es_en_vivo = st.session_state.get(state_key_calib, False)
    else:
        # Modo por defecto: Cargar parámetros precalibrados desde la matriz maestra (sub-segundo)
        es_en_vivo = False
        reg_row = df_calibracion[df_calibracion['Region'] == seleccion_reg] if df_calibracion is not None and not df_calibracion.empty and 'Region' in df_calibracion.columns else pd.DataFrame()
        
        if not reg_row.empty:
            row = reg_row.iloc[0]
            k_dict_opt = {
                'FOOD_WASTE': float(row.get('k_FOOD', 0.0)),
                'GARDEN_WASTE': float(row.get('k_GARDEN', 0.0)),
                'PAPER_WASTE': float(row.get('k_PAPER', 0.0)),
                'WOOD_STRAW_WASTE': float(row.get('k_WOOD', 0.0)),
                'TEXTILES_WASTE': float(row.get('k_TEXTILES', 0.0)),
                'DISPOSABLE_NAPPIES_WASTE': float(row.get('k_NAPPIES', 0.0)),
                'SEWAGE_SLUDGE_WASTE': float(row.get('k_SEWAGE_SLUDGE', 0.0)),
                'RUBBER_WASTE': float(row.get('k_RUBBER', 0.0)),
                'INDUSTRIAL_WASTE': float(row.get('k_INDUSTRIAL', 0.0))
            }
            H0_MSW_Mt = float(row.get('H0_MSW', 0.0))
            H0_IND_Mt = float(row.get('H0_IND', 0.0))
            waste_per_gdp_opt = float(row.get('WASTE_PER_GDP', 0.0))
            W_IND_Mt_opt = (GDP_array * waste_per_gdp_opt * RATIO_IND) / 1000.0

            total_msw_weight = sum([COMPOSICION_2005_DEFAULT.get(w, 0.0) for w in ALL_WASTE_TYPES if w != 'INDUSTRIAL_WASTE'])
            initial_ddocma_dict = {}
            for w in ALL_WASTE_TYPES:
                if w == 'INDUSTRIAL_WASTE':
                    initial_ddocma_dict[w] = H0_IND_Mt
                else:
                    peso_relativo = COMPOSICION_2005_DEFAULT.get(w, 0.0) / total_msw_weight if total_msw_weight > 0 else 0.0
                    initial_ddocma_dict[w] = H0_MSW_Mt * peso_relativo

            best_simulation_Mt, _, _ = simular_multiphase_FOD(
                W_MSW, W_IND_Mt_opt, k_dict_opt, initial_ddocma_dict,
                mcf_msw_array, mcf_ind_array, ox_array, R_array,
                fracciones_residuos_wiliam[seleccion_reg]
            )
            error_cuadratico = float(np.sum((best_simulation_Mt - e_target)**2))
            resultado_tupla = (k_dict_opt, H0_MSW_Mt, H0_IND_Mt, waste_per_gdp_opt, W_IND_Mt_opt, best_simulation_Mt, error_cuadratico)
            st.session_state[state_key_res] = resultado_tupla
            st.session_state[state_key_calib] = False
        else:
            # Fallback en caso de que la región no esté presente en la matriz precalibrada
            resultado_tupla = optimizar_tier4a_12D(
                W_MSW, GDP_array, RATIO_IND, e_target, mcf_msw_array, mcf_ind_array, ox_array, R_array, ALL_WASTE_TYPES, fracciones_residuos_wiliam[seleccion_reg], COMPOSICION_2005_DEFAULT
            )
            if resultado_tupla is not None:
                st.session_state[state_key_res] = resultado_tupla
                st.session_state[state_key_calib] = True

    # --- CÁLCULO DE LA LÍNEA DE BASE TIER 1 (IPCC DEFAULT) ---
    k_base = {
        w: (fracciones_residuos_wiliam[seleccion_reg][w]['k_bounds'][0] + fracciones_residuos_wiliam[seleccion_reg][w]['k_bounds'][1]) / 2.0
        for w in ALL_WASTE_TYPES
    }
    initial_ddocma_base = {w: 0.0 for w in ALL_WASTE_TYPES}
    W_IND_base = np.zeros_like(W_MSW)
    e_base_Mt, _, _ = simular_multiphase_FOD(
        W_MSW, W_IND_base, k_base, initial_ddocma_base,
        mcf_msw_array, mcf_ind_array, ox_array, R_array,
        fracciones_residuos_wiliam[seleccion_reg]
    )
    sse_base = float(np.sum((e_base_Mt - e_target)**2))
    rmse_base = float(np.sqrt(np.mean((e_base_Mt - e_target)**2)))

    if resultado_tupla is not None:
        k_dict_opt, H0_MSW_Mt, H0_IND_Mt, waste_per_gdp_opt, W_IND_Mt_opt, e_sim_Mt, error_cuadratico = resultado_tupla
        
        # Evaluación estadística formal: SSE, RMSE, nRMSE y mejora vs Baseline
        sse_val = float(error_cuadratico)
        rmse_val = float(np.sqrt(np.mean((e_sim_Mt - e_target)**2)))
        mean_obs = float(np.mean(e_target))
        nrmse_val = float((rmse_val / mean_obs) * 100.0) if mean_obs > 0 else 0.0
        delta_rmse = float(((rmse_base - rmse_val) / rmse_base) * 100.0) if rmse_base > 0 else 0.0

        if es_en_vivo:
            st.success(f"⚡ Calibración numérica Powell 12D ejecutada en vivo (SSE = {sse_val:.4f} Mt²).")
        else:
            st.info(f"📁 Parámetros oficiales precalibrados cargados desde IPCC_Waste_Model_Reconstructed.csv (SSE = {sse_val:.4f} Mt²). Pulse 'Ejecutar Calibración' en la barra lateral para recalibrar en vivo.")

        st.markdown("#### Métricas Estadísticas de Calibración (FOD vs EDGAR)")
        cols_eval = st.columns(4)
        cols_eval[0].metric("SSE Calibrado (Mt² CH4)", f"{sse_val:.4f}")
        cols_eval[1].metric("RMSE Calibrado (Mt CH4)", f"{rmse_val:.4f}", delta=f"{delta_rmse:+.2f}% vs Baseline", delta_color="normal")
        cols_eval[2].metric("nRMSE (%)", f"{nrmse_val:.2f}%")
        cols_eval[3].metric("RMSE Baseline Tier 1 (Mt)", f"{rmse_base:.4f}")

        st.markdown("#### Parámetros Óptimos Encontrados")
        cols_metric = st.columns(3)
        cols_metric[0].metric("WASTE_PER_GDP", f"{waste_per_gdp_opt:.4e}")
        cols_metric[1].metric("H0_MSW (Mt)", f"{H0_MSW_Mt:.2f}")
        cols_metric[2].metric("H0_IND (Mt)", f"{H0_IND_Mt:.2f}")

        st.write("**Constantes de degradación biológica (k) vs Rango Permitido:**")
        
        dict_optimos = {w: f"{k_dict_opt[w]:.4f}" for w in ALL_WASTE_TYPES}
        dict_optimos['Métrica'] = 'Valor Óptimo'
        
        dict_limites = {w: f"[{fracciones_residuos_wiliam[seleccion_reg][w]['k_bounds'][0]}, {fracciones_residuos_wiliam[seleccion_reg][w]['k_bounds'][1]}]" for w in ALL_WASTE_TYPES}
        dict_limites['Métrica'] = 'Rango Permitido'
        
        cols = ['Métrica'] + ALL_WASTE_TYPES
        df_k = pd.DataFrame([dict_optimos, dict_limites], columns=cols)
        
        st.dataframe(df_k, use_container_width=True, hide_index=True)
        
        W_MSW_Mt = np.nan_to_num(W_MSW, nan=0.0)

        MSW_TYPES = [t for t in ALL_WASTE_TYPES if t != 'INDUSTRIAL_WASTE']
        
        ddocm_inicial_optimo_reg = {}
        for tipo in MSW_TYPES:
            W_tipo_2005 = H0_MSW_Mt * COMPOSICION_2005_DEFAULT.get(tipo, 0)
            doc_tipo = params_estaticos[tipo]['doc']
            docf_tipo = params_estaticos[tipo]['docf']
            ddocm_inicial_optimo_reg[tipo] = W_tipo_2005 * doc_tipo * docf_tipo * mcf_msw_array[0]
        
        doc_ind = params_estaticos['INDUSTRIAL_WASTE']['doc']
        docf_ind = params_estaticos['INDUSTRIAL_WASTE']['docf']
        ddocm_inicial_optimo_reg['INDUSTRIAL_WASTE'] = H0_IND_Mt * doc_ind * docf_ind * mcf_ind_array[0]
    
        _, historial_ddocma_reg = simular_multiphase_FOD(
            W_MSW, W_IND_Mt_opt, k_dict_opt, ddocm_inicial_optimo_reg, 
            mcf_msw_array, mcf_ind_array, ox_array, R_array, fracciones_residuos_wiliam[seleccion_reg],
            return_history=True
        )
        
        datos_grafico = []
        for i, año in enumerate(años_analisis[:min_len]):
            fila_grafico = {'Year': año}
            for tipo in MSW_TYPES:
                fila_grafico[tipo] = historial_ddocma_reg[tipo][i]
            datos_grafico.append(fila_grafico)
            
        df_grafico = pd.DataFrame(datos_grafico)
        df_grafico.set_index('Year', inplace=True) 
        df_porcentajes = df_grafico.div(df_grafico.sum(axis=1), axis=0) * 100

        st.markdown("### 📊 Galería de Resultados y Dinámica de Descomposición")
        tab_calib, tab_stock, tab_heatmap = st.tabs(["📈 Curva de Calibración", "📊 Stock DDOCma por Fracción", "🗺️ Mapa de Calor (ki)"])

        with tab_calib:
            mostrar_hitos = st.checkbox("🚩 Mostrar Hitos Históricos y Cambios Estructurales", value=True)
            fig_calib = go.Figure()
            fig_calib.add_trace(go.Scatter(x=años_analisis, y=e_target, mode='lines+markers', name='Dato Histórico EDGAR (Mt)', line=dict(color='black', dash='dot', width=2)))
            fig_calib.add_trace(go.Scatter(x=años_analisis, y=e_sim_Mt, mode='lines', name='Simulación Powell Dinámica (Mt)', line=dict(color='#1f77b4', width=3)))
            fig_calib.add_trace(go.Scatter(x=años_analisis, y=e_base_Mt, mode='lines', name='Línea de Base Tier 1 (IPCC Default)', line=dict(color='#d62728', dash='dash', width=2)))
            
            if mostrar_hitos:
                lista_eventos_reg = HITOS_PAISES.get(seleccion_reg, []).copy()
                if seleccion_reg in EU27_REGIONS or seleccion_reg == 'EU27':
                    lista_eventos_reg.extend(EVENTOS_GLOBALES_UE)
                    
                if lista_eventos_reg:
                    ymax = max(max(e_target), max(e_sim_Mt), max(e_base_Mt))
                    desfase_y = 0 
                    for hito in lista_eventos_reg:
                        if hito['año'] in años_analisis:
                            fig_calib.add_vline(x=hito['año'], line_width=1.5, line_dash="dash", line_color=hito['color'])
                            altura_etiqueta = ymax * (0.95 - (desfase_y * 0.12))
                            fig_calib.add_annotation(
                                x=hito['año'], 
                                y=altura_etiqueta,
                                text=f"<b>{hito['evento']}</b>",
                                showarrow=True, 
                                arrowhead=2,
                                arrowsize=1.5,
                                arrowwidth=2,
                                arrowcolor=hito['color'],
                                ax=-60,
                                ay=-40,
                                bgcolor="rgba(255, 255, 255, 0.95)",
                                bordercolor=hito['color'],
                                borderwidth=2,
                                font=dict(size=13, color=hito['color'])
                            )
                            desfase_y = (desfase_y + 1) % 3

            fig_calib.update_layout(
                title=f"Ajuste de Emisiones CH4 (FOD Powell vs EDGAR) — Región: {seleccion_reg}",
                xaxis_title="Año",
                yaxis_title="Emisiones de Metano (Mt CH4 / año)",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_calib, use_container_width=True)

        with tab_stock:
            tipo_grafico = st.radio(
                "Estilo de visualización del Stock DDOCma:",
                options=["Áreas Apiladas (Porcentaje %)", "Líneas (Porcentaje %)"],
                horizontal=True
            )
            if tipo_grafico == "Áreas Apiladas (Porcentaje %)":
                fig_stock = go.Figure()
                for tipo in MSW_TYPES:
                    fig_stock.add_trace(go.Scatter(
                        x=df_grafico.index,
                        y=df_grafico[tipo],
                        mode='none',           
                        stackgroup='one',      
                        groupnorm='percent',
                        name=tipo
                    ))
                fig_stock.update_layout(
                    title=f"Evolución Porcentual del Carbono Acumulado DDOCma — {seleccion_reg}",
                    xaxis_title="Año",
                    yaxis_title="Porcentaje de DDOCma (%)",
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis=dict(type='linear', range=[0, 100], ticksuffix='%'),
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig_stock, use_container_width=True)
            else:
                fig_lines = go.Figure()
                for tipo in MSW_TYPES:
                    fig_lines.add_trace(go.Scatter(
                        x=df_porcentajes.index,
                        y=df_porcentajes[tipo],
                        mode='lines+markers',
                        name=tipo
                    ))
                fig_lines.update_layout(
                    title=f"Trayectorias Porcentuales de DDOCma por Fracción — {seleccion_reg}",
                    xaxis_title="Año",
                    yaxis_title="Porcentaje de DDOCma (%)",
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis=dict(ticksuffix='%'),
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig_lines, use_container_width=True)

        with tab_heatmap:
            st.subheader(f"Mapa de Calor de Constantes Cinéticas ki — {seleccion_reg} ({clima_region})")
            labels_k = [w.replace('_WASTE', '') for w in ALL_WASTE_TYPES]
            z_heatmap = [
                [fracciones_residuos_wiliam[seleccion_reg][w]['k_bounds'][0] for w in ALL_WASTE_TYPES],
                [k_dict_opt[w] for w in ALL_WASTE_TYPES],
                [fracciones_residuos_wiliam[seleccion_reg][w]['k_bounds'][1] for w in ALL_WASTE_TYPES]
            ]
            text_heatmap = [[f"{val:.4f}" for val in row] for row in z_heatmap]
            fig_hm = go.Figure(data=go.Heatmap(
                z=z_heatmap,
                x=labels_k,
                y=['Cota Mínima IPCC', 'Valor Óptimo (Powell)', 'Cota Máxima IPCC'],
                text=text_heatmap,
                texttemplate="%{text}",
                textfont={"size": 13, "color": "white"},
                colorscale='Viridis',
                colorbar=dict(title='k (año⁻¹)')
            ))
            fig_hm.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Fracción de Residuo Sólido",
                yaxis_title="Nivel de Parametrización"
            )
            st.plotly_chart(fig_hm, use_container_width=True)


        st.markdown("### 📊 Panel de Depuración Anual (Inputs y Outputs del Simulador)")
        total_msw_weight = sum([COMPOSICION_2005_DEFAULT.get(w, 0.0) for w in ALL_WASTE_TYPES if w != 'INDUSTRIAL_WASTE'])
        initial_ddocma_dict = {}
        for w in ALL_WASTE_TYPES:
            if w == 'INDUSTRIAL_WASTE':
                initial_ddocma_dict[w] = H0_IND_Mt
            else:
                peso_relativo = COMPOSICION_2005_DEFAULT.get(w, 0.0) / total_msw_weight if total_msw_weight > 0 else 0.0
                initial_ddocma_dict[w] = H0_MSW_Mt * peso_relativo
                
        _, ddocma_total, metano_bruto_total = simular_multiphase_FOD(
            W_MSW_Mt, W_IND_Mt_opt, k_dict_opt, initial_ddocma_dict, 
            mcf_msw_array, mcf_ind_array, ox_array, R_array, fracciones_residuos_wiliam[seleccion_reg]
        )
        
        df_debug = pd.DataFrame({
            'Año': años_analisis[:min_len],
            'Población (M)': np.round(pop_millones_array[:min_len], 2),
            'PIB (GDP)': np.round(GDP_array[:min_len], 2),
            '% MSW a Vertedero': np.round(disp_perc_array[:min_len], 2),
            'Residuos MSW Entrantes (Mt)': np.round(W_MSW_Mt[:min_len], 4),
            'Residuos IND Entrantes (Mt)': np.round(W_IND_Mt_opt[:min_len], 4),
            'DDOCma Acumulado (Mt)': np.round(ddocma_total[:min_len], 4),
            'Metano Bruto Generado (Mt)': np.round(metano_bruto_total[:min_len], 4),
            'Factor MCF MSW': np.round(mcf_msw_array[:min_len], 2),
            'Factor MCF IND': np.round(mcf_ind_array[:min_len], 2),
            'Factor Oxidación (OX)': np.round(ox_array[:min_len], 2),
            'Metano Recuperado (R)': np.round(R_array[:min_len], 4),
            'Emisiones Simuladas (Mt)': np.round(e_sim_Mt[:min_len], 4),
            'Emisiones Históricas (Mt)': np.round(e_target[:min_len], 4)
        })
        st.dataframe(df_debug, use_container_width=True, hide_index=True)

    else:
        st.error("Error crítico procesando la simulación. Revisa la integridad de los datos históricos de esta región.")

with tab_global:
    st.subheader("Matriz de Resultados Global")
    
    # Botón para ejecutar el recálculo masivo global bajo demanda
    recalc_solicitado = st.button("⚡ Recalcular Matriz Global Completa (36 Regiones - Bucle Powell)", key="btn_recalc_global", help="Ejecuta la optimización Powell 12D para las 36 macrorregiones WILIAM de forma secuencial (tiempo estimado: 3 a 5 minutos).")
    
    if recalc_solicitado:
        with st.spinner("Procesando matriz global con el Modelo Dinámico (12 Dimensiones) para todas las regiones..."):
            df_glob_calc, df_calib_calc, df_evo_calc = calcular_matriz_global(df_calibracion)
            if not df_glob_calc.empty:
                st.session_state['df_global_recalc'] = df_glob_calc
                st.session_state['df_calib_recalc'] = df_calib_calc
                st.session_state['df_evo_recalc'] = df_evo_calc
                st.success("Recálculo global completado exitosamente.")
    
    # Determinar qué matrices presentar (recalculadas o precargadas)
    if 'df_global_recalc' in st.session_state:
        df_mostrar_global = st.session_state['df_global_recalc']
        df_mostrar_calib = st.session_state['df_calib_recalc']
        df_mostrar_evo = st.session_state['df_evo_recalc']
    else:
        df_mostrar_global = pd.DataFrame()
        df_mostrar_calib = df_calibracion
        df_mostrar_evo = load_evolucion_porcentajes()
    
    if not df_mostrar_global.empty:
        st.markdown("#### Matriz Global de Optimización Recalculada")
        st.dataframe(df_mostrar_global, use_container_width=True, hide_index=True)
        st.divider()

    if not df_mostrar_calib.empty:
        st.markdown("#### Matriz Maestra de Calibración (IPCC_Waste_Model_Reconstructed.csv)")
        st.dataframe(df_mostrar_calib, use_container_width=True, hide_index=True)
    
        csv_export = '\ufeff' + df_mostrar_calib.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Matriz Maestra (IPCC_Waste_Model_Reconstructed.csv)",
            data=csv_export.encode('utf-8'),
            file_name="IPCC_Waste_Model_Reconstructed.csv",
            mime="text/csv"
        )
        st.divider()

    if not df_mostrar_evo.empty:
        st.markdown("#### Evolución Temporal de Composición en Vertederos (DDOCma)")
        st.dataframe(df_mostrar_evo, use_container_width=True, hide_index=True)
            
        csv_evo = '\ufeff' + df_mostrar_evo.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Evolución Temporal de Porcentajes",
            data=csv_evo.encode('utf-8'),
            file_name="Evolucion_Porcentajes_Vertederos_2005_2023.csv",
            mime="text/csv"
        )
    else:
        st.info("No se han podido cargar los datos de evolución temporal. Asegúrate de que el archivo CSV está en la misma carpeta que este script.")