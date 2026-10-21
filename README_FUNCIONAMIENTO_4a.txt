================================================================================
MANUAL DE FUNCIONAMIENTO Y ARQUITECTURA DE DATOS
CATEGORÍA 4.A: TRATAMIENTO Y VERTIDO DE RESIDUOS SÓLIDOS (SOLID WASTE DISPOSAL)
Herramienta: app_4a.py
Metodología: IPCC 2006 (Volumen 5, Capítulo 3) — Régimen Híbrido Tier 2 / pseudo-Tier 3
Proyecto: Módulo Satélite de Residuos para el Modelo Integrado WILIAM (LOCOMOTION H2020)
Autor: Javier Hernández Cepeda
================================================================================

1. PROPÓSITO DE LA HERRAMIENTA
------------------------------
La herramienta app_4a.py constituye el núcleo algorítmico principal del Trabajo
de Fin de Grado (TFG). Modela la dinámica temporal de descomposición anaerobia
de los residuos sólidos depositados en vertederos mediante el Modelo de Decaimiento
de Primer Orden (First Order Decay - FOD) y calibra numéricamente sus constantes
cinéticas e inercias históricas frente a inventarios empíricos de emisiones (EDGAR v8.0).

El objetivo es cerrar el déficit metodológico del Sector 4 (Residuos) en la
arquitectura del modelo de evaluación integrada WILIAM (proyecto europeo LOCOMOTION),
proporcionando matrices estructuradas de parámetros y tensores de emisión para
36 macrorregiones mundiales en el periodo histórico 2005-2023.

2. ARQUITECTURA DE ENTRADA DE DATOS (CARPETA "data/")
----------------------------------------------------------------
La aplicación consume seis bases de datos en formato CSV estructurado ubicadas
en la subcarpeta relativa "data/":

1. HISTORICAL_GHG_EMISSIONS_REGIONS_IPCC_SECTORS_LEVEL_2.csv:
   Emisiones históricas oficiales de gases de efecto invernadero (base de datos
   EDGAR v8.0 / inventarios Level 2 del IPCC para la categoría 4.A, CH4).
2. Population_Regions.csv:
   Población total por región y año (habitantes; censos Eurostat / Banco Mundial).
3. HISTORICAL_MUNICIPAL_WASTE_PER_CAPITA.csv:
   Tasa anual de generación per cápita de residuos municipales (kg/hab/año).
4. HISTORICAL_MUNICIPAL_DISPOSAL_PERCENTAGE.csv:
   Porcentaje de residuos municipales destinados a vertedero (%).
5. GDP_Regions.csv:
   Producto Interior Bruto regional a precios constantes (miles de millones de $,
   miles de M$, base macroeconómica WILIAM).
6. IPCC_Waste_Model_Reconstructed.csv:
   Matriz maestra consolidada que almacena los parámetros calibrados (ki*, H0*,
   WASTE_PER_GDP*) y el error cuadrático resultante para las 36 regiones.

3. FUNDAMENTO MATEMÁTICO: MODELO FOD MULTIFRACCIONAL DISCRETO
--------------------------------------------------------------
El simulador implementa el modelo discreto anual con retardo temporal de un año
completo (tdelay = 1 año, Hipótesis H.8b; Ecuaciones 3.4 y 3.5 del IPCC 2006).

El flujo masivo de residuos se desagrega en 9 fracciones de degradación independiente:
- Alimentos (FOOD_WASTE)
- Restos vegetales y jardín (GARDEN_WASTE)
- Papel y cartón (PAPER_WASTE)
- Madera y paja (WOOD_STRAW_WASTE)
- Textiles (TEXTILES_WASTE)
- Pañales desechables (DISPOSABLE_NAPPIES_WASTE)
- Lodos de depuradora (SEWAGE_SLUDGE_WASTE)
- Caucho y cuero (RUBBER_WASTE)
- Residuos industriales asimilables (INDUSTRIAL_WASTE)

Ecuaciones de Estado:
1. Carbono Orgánico Degradable depositado anualmente en vertedero (Mt C):
      DDOCmd_i(t) = W_i(t) * DOC_i * DOCf_i * MCF(t)
2. Descomposición anual a partir del stock remanente del ejercicio precedente (Mt C):
      DDOCm_decomp_i(t) = DDOCma_i(t-1) * (1 - exp(-k_i))
3. Generación bruta estequiométrica de metano (Mt CH4 / año):
      CH4_gen(t) = SUM_i [ DDOCm_decomp_i(t) * F * (16 / 12) ]
4. Actualización del stock reactivo confinado al cierre del año (Mt C):
      DDOCma_i(t) = DDOCmd_i(t) + DDOCma_i(t-1) * exp(-k_i)
5. Emisiones netas a la atmósfera con salvaguarda física de no negatividad (Mt CH4 / año):
      E_CH4(t) = max(0.0, [CH4_gen(t) - R(t)] * [1 - OX(t)])

Donde:
- k_i: Constante cinética de descomposición biológica de la fracción i (año⁻¹).
- DOC_i: Contenido intrínseco de carbono orgánico degradable (t C / t residuo).
- DOCf_i: Fracción de DOC disimilada en condiciones anaerobias (0.5 - 0.7; 0.1 en madera; 0.0 en caucho).
- MCF(t): Factor de corrección de metano por tipología de vertedero (0.71 MSW, 0.72 IND).
- F: Fracción molar de metano en el biogás generado (0.5 por defecto IPCC).
- 16/12: Razón estequiométrica de masas molares (M_CH4 / M_C).
- R(t): Metano recuperado y valorizado o antorchado (Mt CH4 / año).
- OX(t): Factor de oxidación superficial en la capa de cobertura (0.05).

4. ALGORITMO DE CALIBRACIÓN: MÉTODO DE POWELL 12D
--------------------------------------------------
Para ajustar las curvas simuladas a las observaciones históricas de EDGAR v8.0,
el sistema formula un problema inverso optimizando simultáneamente un vector 12D:
   θ = [k_food, k_garden, k_paper, k_wood, k_textiles, k_nappies,
        k_sludge, k_rubber, k_industrial, H0_MSW, H0_IND, WASTE_PER_GDP]

- H0_MSW y H0_IND: Stocks basales históricos de carbono acumulados antes de 2005 (Mt C).
- WASTE_PER_GDP: Intensidad generacional de residuos industriales respecto al PIB (t/M$).
- Generación industrial: W_IND(t) = GDP(t) * WASTE_PER_GDP * 1.0e-3 (Mt/año).
- Algoritmo numérico: Método de Direcciones Conjugadas de Powell (SciPy). Opera libre
  de derivadas mediante búsquedas lineales unidimensionales con el algoritmo de Brent,
  evitando inestabilidades ante discontinuidades y la convolución paso-bajo del FOD.
- Normalización afín: Mapeo del dominio físico acotado al hipercubo unitario [0, 1]¹²
  para equilibrar órdenes de magnitud dispares entre k_i (~10⁻² a 10⁻¹) y H0 (~10¹ a 10³ Mt).
- Función objetivo: Suma de Errores Cuadráticos (SSE, Mt² CH4) penalizada con una función
  de barrera exterior de 1.0e9 ante simulaciones físicamente inválidas (valores negativos o NaN).
- Criterio de parada: maxiter = 2000 iteraciones completas.

5. NAVEGACIÓN Y PANELES EN LA INTERFAZ STREAMLIT
------------------------------------------------
1. PANEL LATERAL (CONFIGURACIÓN):
   - Selector de región WILIAM (36 regiones disponibles; e.g., SPAIN, GERMANY, CHINA, USMCA).
   - Régimen bioclimático asociado (TEMPERATE_WET, TEMPERATE_DRY, TROPICAL_WET, TROPICAL_DRY).
   - Límites bioclimáticos admisibles para cada fracción según IPCC 2006.
   - Botón "Ejecutar Calibración (Powell 12D)" para optimización en vivo.
2. PESTAÑA "ANÁLISIS POR REGIÓN":
   - Carga instantánea por defecto de parámetros óptimos precalculados.
   - Cuadro de métricas cuantitativas: SSE Calibrado (Mt²), RMSE Calibrado (Mt),
     nRMSE relativo (%) y RMSE de la línea de base Tier 1 por defecto del IPCC.
   - Tabla de parámetros convergidos: k_i* vs. rango físico normativo, H0 y ratio PIB.
   - Galería gráfica interactiva (Plotly):
     * Curva de Calibración: simulación Powell vs. inventario EDGAR con hitos regulatorios.
     * Stock DDOCma: evolución porcentual de la composición interna de carbono activo.
     * Mapa de Calor: posición relativa de las cinéticas calibradas en sus rangos climáticos.
   - Tabla de auditoría paso a paso con los inputs y outputs anuales del simulador.
3. PESTAÑA "RESUMEN GLOBAL":
   - Botón de recálculo masivo global secuencial para las 36 regiones WILIAM.
   - Tabla maestra consolidada de parámetros y errores.
   - Descarga directa de las matrices persistentes para WILIAM:
     * IPCC_Waste_Model_Reconstructed.csv (vector paramétrico regional 12D).
     * Evolucion_Porcentajes_Vertederos_2005_2023.csv (dinámica del stock DDOCma).
================================================================================
