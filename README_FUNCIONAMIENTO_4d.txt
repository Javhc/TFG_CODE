================================================================================
MANUAL DE FUNCIONAMIENTO Y FLUJO DE DATOS
CATEGORÍA 4.D: TRATAMIENTO Y VERTIDO DE AGUAS RESIDUALES
Herramienta: app_4d.py
Metodología: IPCC 2006 (Volumen 5, Capítulo 6) — Nivel Tier 1 con Calibración Powell 2D
Rol en WILIAM: Balance de metano y óxido nitroso por saneamiento y efluentes
================================================================================

1. PROPÓSITO DE LA HERRAMIENTA
------------------------------
La herramienta app_4d.py cuantifica y calibra las emisiones de GEI procedentes
del tratamiento y descarga de aguas residuales domésticas e industriales:
  a) Metano (CH4): Catabolismo anaerobio de la carga orgánica disuelta (DBO).
  b) Óxido nitroso (N2O): Nitrificación y desnitrificación de efluentes ricos
     en nitrógeno proteico vertidos a masas receptoras acuáticas.

2. ARQUITECTURA DE ENTRADA DE DATOS (CARPETA "data/")
----------------------------------------------------------------
1. HISTORICAL_GHG_EMISSIONS_REGIONS_IPCC_SECTORS_LEVEL_2.csv:
   Emisiones históricas oficiales para "4_D__Wastewater_Treatment_and_Discharge"
   (CH4 y N2O, periodo 2005-2023, base EDGAR v8.0).
2. Population_Regions.csv: Población histórica regional.
3. Food_Regions.csv: Disponibilidad y consumo proteico per cápita (FAOSTAT).

3. FORMULACIÓN MATEMÁTICA IPCC TIER 1
-------------------------------------
1. Carga orgánica degradable total doméstica (Ecuaciones 6.1 a 6.3 del IPCC):
     TOW(t) = Población(t) * BOD_pc   [kg DBO / año]
   Donde BOD_pc = 21.9 kg DBO / hab / año (estándar europeo de 60 g DBO/hab/día).

2. Emisiones directas de Metano:
     E_CH4,4D(t) = TOW(t) * B0 * MCF * 10^-6   [Gg CH4]
   Donde B0 = 0.60 kg CH4 / kg DBO (capacidad estequiométrica máxima teórica) y
   MCF = 0.30 (factor de corrección de metano por defecto Tier 1 no desagregado).

3. Carga de nitrógeno en efluentes y emisiones de Óxido Nitroso (Eq. 6.7 y 6.8):
     N_effluent(t) = Población(t) * Protein_pc(t) * F_NPR * F_NON-CON * F_IND-COM   [kg N / año]
     E_N2O,4D(t) = N_effluent(t) * EF_N2O,eff * (44 / 28) * 10^-6   [Gg N2O]
   Donde:
   - Protein_pc = 35.0 kg proteína / hab / año (consumo medio de ~95 g/hab/día).
   - F_NPR = 0.16 kg N / kg proteína (fracción elemental de nitrógeno en proteínas).
   - F_NON-CON = 1.04 (factor corrector por desperdicio arrojado al desagüe).
   - F_IND-COM = 1.25 (factor de descargas comerciales e industriales conectadas).
   - EF_N2O,eff = 0.005 kg N2O-N / kg N (factor de emisión en aguas receptoras).
   - 44/28 es la razón estequiométrica molar entre N2O (44 g/mol) y N2 (28 g/mol).

4. Potencial de Calentamiento Global (IPCC AR6 GWP100):
     E_Total,CO2-eq(t) = 27.9 * E_CH4(t) + 273 * E_N2O(t)   [Gg CO2-eq]

4. CALIBRACIÓN NUMÉRICA POWELL 2D
---------------------------------
Optimiza regionalmente el vector [MCF, EF_N2O,eff] en los intervalos admisibles
[(0.05, 0.90), (0.001, 0.020)] minimizando el error multigas:
   SSE = ∑ (CH4_sim - CH4_obs)² + ∑ (N2O_sim - N2O_obs)²
con tolerancias xtol = 1e-6, ftol = 1e-6.
================================================================================
