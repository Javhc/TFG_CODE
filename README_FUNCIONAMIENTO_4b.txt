================================================================================
MANUAL DE FUNCIONAMIENTO Y FLUJO DE DATOS
CATEGORÍA 4.B: TRATAMIENTO BIOLÓGICO DE RESIDUOS SÓLIDOS
Herramienta: app_4b.py
Metodología: IPCC 2006 (Volumen 5, Capítulo 4) — Nivel Tier 1 con Calibración Powell 2D
Rol en WILIAM: Cierre del balance global de emisiones de residuos biológicos
================================================================================

1. PROPÓSITO DE LA HERRAMIENTA
------------------------------
La herramienta app_4b.py calcula y calibra las emisiones de Gases de Efecto
Invernadero (GEI) procedentes del tratamiento biológico de residuos sólidos
municipales en base húmeda, diferenciando dos vías tecnológicas:
  a) Compostaje aerobio (pilas y túneles; emisiones de CH4 y N2O).
  b) Digestión anaerobia hermética (biometanización; fugas técnicas de CH4).

2. ARQUITECTURA DE ENTRADA DE DATOS (CARPETA "data/")
----------------------------------------------------------------
1. HISTORICAL_GHG_EMISSIONS_REGIONS_IPCC_SECTORS_LEVEL_2.csv:
   Emisiones históricas oficiales declaradas para "4_B__Biological_Treatment_of_Solid_Waste"
   (CH4 y N2O, periodo 2005-2023, base EDGAR v8.0).
2. Population_Regions.csv:
   Población histórica por región (censos Eurostat y Banco Mundial).
3. HISTORICAL_MUNICIPAL_WASTE_PER_CAPITA.csv:
   Generación anual de residuos municipales per cápita (kg/hab/año).

3. FORMULACIÓN MATEMÁTICA IPCC TIER 1
-------------------------------------
1. Masa total municipal generada en base húmeda (Gg / año):
     M_total(t) = Población(t) * MSW_pc(t) * 10^-6

2. Desacoplamiento másico por vía de tratamiento:
     M_compost(t) = M_total(t) * ϕ_comp
     M_anaerobic(t) = M_total(t) * ϕ_ana

3. Emisiones directas de Metano y Óxido Nitroso (Ecuaciones 4.1 y 4.2 del IPCC):
     E_CH4(t) = [ M_compost(t) * EF_CH4,comp + M_anaerobic(t) * EF_CH4,ana ] * 10^-3   [Gg CH4]
     E_N2O(t) = [ M_compost(t) * EF_N2O,comp + M_anaerobic(t) * EF_N2O,ana ] * 10^-3   [Gg N2O]

4. Potencial de Calentamiento Global (IPCC AR6 GWP100):
     E_Total,CO2-eq(t) = 27.9 * E_CH4(t) + 273 * E_N2O(t)   [Gg CO2-eq]

4. PARÁMETROS POR DEFECTO Y CALIBRACIÓN NUMÉRICA POWELL 2D
-----------------------------------------------------------
- Valores por defecto IPCC 2006 (Vol. 5, Cap. 4, Tabla 4.1):
  * Compostaje: EF_CH4,comp = 4.0 g/kg; EF_N2O,comp = 0.24 g/kg.
  * Digestión anaerobia: EF_CH4,ana = 0.8 g/kg; EF_N2O,ana = 0.0 g/kg (estáticos).
- Optimización Powell 2D:
  Ajusta regionalmente el vector [EF_CH4,comp, EF_N2O,comp] en los intervalos
  admisibles [(0.5, 15.0), (0.01, 1.5)] g/kg minimizando el error cuadrático multigas:
     SSE = ∑ (CH4_sim - CH4_obs)² + ∑ (N2O_sim - N2O_obs)²
  con tolerancias xtol = 1e-6, ftol = 1e-6.
================================================================================
