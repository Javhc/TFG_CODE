================================================================================
MANUAL DE FUNCIONAMIENTO Y FLUJO DE DATOS
CATEGORÍA 4.C: INCINERACIÓN Y QUEMA A CIELO ABIERTO DE RESIDUOS
Herramienta: app_4c.py
Metodología: IPCC 2006 (Volumen 5, Capítulo 5) — Nivel Tier 1 con Calibración Powell 3D
Rol en WILIAM: Cuantificación de emisiones de CO2 fósil, CH4 y N2O por combustión
================================================================================

1. PROPÓSITO DE LA HERRAMIENTA
------------------------------
La herramienta app_4c.py modela las emisiones de GEI producidas por la combustión
térmica de residuos sólidos municipales, discriminando entre:
  a) Incineración controlada en plantas industriales (con o sin WtE).
  b) Quema incontrolada a cielo abierto (open burning).

Bajo la convención internacional del IPCC, el CO2 biogénico es de ciclo cerrado
(neutro); por tanto, ÚNICAMENTE el carbono de origen fósil (plásticos, sintéticos)
computa como emisión neta en los balances de inventario.

2. ARQUITECTURA DE ENTRADA DE DATOS (CARPETA "data/")
----------------------------------------------------------------
1. HISTORICAL_GHG_EMISSIONS_REGIONS_IPCC_SECTORS_LEVEL_2.csv:
   Emisiones históricas oficiales para "4_C__Incineration_and_Open_Burning_of_Waste"
   (CO2 fósil, CH4, N2O, periodo 2005-2023, base EDGAR v8.0).
2. Population_Regions.csv: Población histórica regional.
3. HISTORICAL_MUNICIPAL_WASTE_PER_CAPITA.csv: Generación per cápita (kg/hab/año).

3. FORMULACIÓN MATEMÁTICA IPCC TIER 1
-------------------------------------
1. Masa generada y flujos de combustión en base húmeda (Gg / año):
     M_total(t) = Población(t) * MSW_pc(t) * 10^-6
     W_inc(t) = M_total(t) * ϕ_inc
     W_burn(t) = M_total(t) * ϕ_burn

2. Balance estequiométrico de CO2 fósil (Ecuación 5.1 del IPCC):
     E_CO2,4C(t) = [ W_inc(t) * OF_inc + W_burn(t) * OF_burn ] * DM * CF * FCF * (44 / 12)
   Donde DM = 0.80 (materia seca), CF = 0.50 (carbono total en materia seca),
   FCF = 0.90 (fracción de carbono fósil), OF_inc = 1.0 (oxidación completa),
   OF_burn = 0.71 (oxidación incompleta) y 44/12 es la relación estequiométrica molar.

3. Emisiones directas de gases de combustión incompleta (Ecuaciones 5.2 y 5.3):
     E_CH4,4C(t) = [ W_inc(t) * EF_CH4,inc + W_burn(t) * 1.3 * EF_CH4,inc ] * 10^-3   [Gg CH4]
     E_N2O,4C(t) = [ W_inc(t) * EF_N2O,inc + W_burn(t) * 1.5 * EF_N2O,inc ] * 10^-3   [Gg N2O]
   Los factores multiplicativos 1.3 y 1.5 cuantifican el exceso de emisiones por
   baja temperatura y turbulencia deficiente en quema abierta.

4. Potencial de Calentamiento Global (IPCC AR6 GWP100):
     E_Total,CO2-eq(t) = E_CO2(t) + 27.9 * E_CH4(t) + 273 * E_N2O(t)   [Gg CO2-eq]

4. CALIBRACIÓN NUMÉRICA POWELL 3D
---------------------------------
Optimiza el vector θ = [FCF, EF_CH4,inc, EF_N2O,inc] en los rangos
[(0.10, 1.00), (0.1, 50.0), (0.001, 2.00)] minimizando el error multigas:
   SSE = ∑ (CO2_sim - CO2_obs)² + ∑ (CH4_sim - CH4_obs)² + ∑ (N2O_sim - N2O_obs)²
con tolerancias xtol = 1e-6, ftol = 1e-6.
================================================================================
