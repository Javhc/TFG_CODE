# Simulación y Calibración de Emisiones de GEI del Sector Residuos (IPCC 2006) para WILIAM

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![SciPy](https://img.shields.io/badge/SciPy-Powell%20Optimization-8CAAE6)
![License](https://img.shields.io/badge/License-MIT-green)

Este repositorio contiene el código fuente y las herramientas interactivas desarrolladas en **Python (Streamlit)** para el cálculo, visualización y calibración numérica de las emisiones de gases de efecto invernadero (GEI) del **Sector 4 (Residuos)** según las directrices del **IPCC 2006** y su refinamiento de 2019.

El proyecto ha sido desarrollado como Trabajo de Fin de Grado (TFG) en la **Escuela de Ingenierías Industriales de la Universidad de Valladolid (UVa)**, en colaboración con el **Grupo de Energía, Economía y Dinámica de Sistemas (GEEDS)**, con el objetivo de cubrir el vacío metodológico del sector residuos en el modelo de evaluación integrada **WILIAM** (*Within Limits Integrated Assessment Model*, proyecto europeo Horizon 2020 LOCOMOTION).

---

## Descripción general del proyecto

El sistema modela las emisiones históricas y calibra los parámetros biofísicos y operativos para las **36 macrorregiones** del modelo WILIAM (los 27 Estados miembros de la Unión Europea de forma individual más 9 grandes bloques internacionales, además del agregado comunitario `EU27`) a lo largo del horizonte histórico **2005–2023** ($N_t = 19$ años).

El ajuste paramétrico se realiza resolviendo un problema de optimización inversa libre de derivadas mediante el **método de direcciones conjugadas de Powell** (`scipy.optimize.minimize`), contrastando las salidas de los modelos frente a las series empíricas del inventario global **EDGAR v8.0** (*Joint Research Centre*, Comisión Europea).

### Herramientas incluidas

Cada una de las cuatro categorías del Sector 4 del IPCC se implementa en una aplicación web interactiva e independiente:

| Archivo | Categoría IPCC 2006 | Gases modelados | Metodología y calibración |
| :--- | :--- | :---: | :--- |
| **`app_4a.py`** | **4.A** – Eliminación de residuos sólidos en vertederos (*Solid Waste Disposal*) | $\text{CH}_4$ | Modelo dinámico multifraccional de decaimiento de primer orden (**FOD**) con 9 fracciones + calibración Powell en 12D ($\tilde{\mathbf{p}} \in [0, 1]^{12}$: neuf constantes $k_i^*$, stocks iniciales $H_{0,\text{MSW}}^*$, $H_{0,\text{IND}}^*$ y ratio industrial $\omega_{\text{IND}}^*$). |
| **`app_4b.py`** | **4.B** – Tratamiento biológico de residuos sólidos (*Biological Treatment*) | $\text{CH}_4$, $\text{N}_2\text{O}$ | Balance estático **Tier 1** (compostaje aerobio y digestión anaerobia) + ajuste regional con Powell de $\text{EF}_{\text{comp},\text{CH}_4}^*$ y $\text{EF}_{\text{comp},\text{N}_2\text{O}}^*$. |
| **`app_4c.py`** | **4.C** – Incineración y quema a cielo abierto (*Incineration and Open Burning*) | $\text{CO}_2$, $\text{CH}_4$, $\text{N}_2\text{O}$ | Balance estequiométrico **Tier 1** de carbono fósil y gases traza + ajuste regional con Powell de $\text{FCF}^*$, $\text{EF}_{\text{CH}_4,\text{inc}}^*$ y $\text{EF}_{\text{N}_2\text{O},\text{inc}}^*$. |
| **`app_4d.py`** | **4.D** – Tratamiento y descarga de aguas residuales (*Wastewater Treatment*) | $\text{CH}_4$, $\text{N}_2\text{O}$ | Modelo de carga orgánica ($\text{DBO}$) y nitrógeno efluente **Tier 1** + ajuste regional con Powell de $\text{MCF}^*$ y $\text{EF}_{\text{N}_2\text{O},\text{eff}}^*$. |

---

## Estructura del repositorio y datos de entrada

Para ejecutar correctamente las aplicaciones, los scripts y los ficheros tabulares de entrada (`.csv`) deben ubicarse en el directorio raíz del proyecto:

```text
├── data/                                    # Carpeta con los repositorios CSV de entrada (EDGAR, Eurostat, FAO, WILIAM)
├── app_4a.py                                # Herramienta interactiva - Categoría 4.A (Vertederos - FOD y Powell)
├── app_4b.py                                # Herramienta interactiva - Categoría 4.B (Tratamiento biológico)
├── app_4c.py                                # Herramienta interactiva - Categoría 4.C (Incineración y quema abierta)
├── app_4d.py                                # Herramienta interactiva - Categoría 4.D (Aguas residuales)
├── IPCC_Waste_Model_Reconstructed.csv       # Matriz consolidada de parámetros y emisiones exportada para WILIAM
├── README_FUNCIONAMIENTO_4a.txt             # Guía operativa y técnica específica de la herramienta 4.A
├── README_FUNCIONAMIENTO_4b.txt             # Guía operativa y técnica específica de la herramienta 4.B
├── README_FUNCIONAMIENTO_4c.txt             # Guía operativa y técnica específica de la herramienta 4.C
├── README_FUNCIONAMIENTO_4d.txt             # Guía operativa y técnica específica de la herramienta 4.D
├── requirements.txt                         # Dependencias de Python necesarias para el proyecto
└── README.md                                # Documentación principal del repositorio
```

### Fuentes de datos utilizadas

1. **Emisiones históricas de referencia:** Base de datos **EDGAR v8.0** (*Emissions Database for Global Atmospheric Research*), elaborada por el *Joint Research Centre* (JRC) de la Comisión Europea.
2. **Actividad de residuos municipales:** Series históricas de generación per cápita y tasas de tratamiento procedentes de **Eurostat** y de las bases de datos del proyecto **WILIAM / LOCOMOTION**.
3. **Datos demográficos, macroeconómicos y dietéticos:** Series de población y PIB de **WILIAM**, y balances alimentarios de **FAOSTAT** (Organización de las Naciones Unidas para la Alimentación y la Agricultura).
4. **Parámetros biofísicos y límites climáticos:** Volumen 5 (*Waste*) de las *2006 IPCC Guidelines for National Greenhouse Gas Inventories* y potenciales de calentamiento global a 100 años ($\text{GWP}_{100}$) del Sexto Informe de Evaluación del IPCC (**AR6**): $\text{CO}_2 = 1$, $\text{CH}_4 = 27.9$, $\text{N}_2\text{O} = 273$.

---

## Instalación y puesta en marcha

### 1. Requisitos previos
Se requiere **Python 3.9** o superior. Se recomienda crear un entorno virtual limpio antes de instalar las dependencias:

```bash
# Clonar el repositorio
git clone https://github.com/Javhc/TFG_CODE
cd TFG_CODE

# Crear y activar un entorno virtual (opcional pero recomendado)
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux / macOS:
source venv/bin/activate
```

### 2. Instalación de dependencias
Instala las bibliotecas científico-numéricas y de visualización necesarias (`numpy`, `pandas`, `scipy`, `plotly` y `streamlit`):

```bash
pip install -r requirements.txt
```
Alternativa:
```bash
pip install numpy pandas scipy plotly streamlit
```

### 3. Ejecución de las herramientas
Cada aplicación se lanza localmente en el navegador web mediante el comando `streamlit run`:

```bash
# Ejecutar la herramienta de Vertederos (Categoría 4.A - Modelo FOD)
streamlit run app_4a.py

# Ejecutar la herramienta de Tratamiento Biológico (Categoría 4.B)
streamlit run app_4b.py

# Ejecutar la herramienta de Incineración y Quema Abierta (Categoría 4.C)
streamlit run app_4c.py

# Ejecutar la herramienta de Aguas Residuales (Categoría 4.D)
streamlit run app_4d.py
```

---

## Salidas generadas para su integración en WILIAM

Las herramientas permiten exportar directamente desde la interfaz gráfica (o mediante sus rutinas de cálculo global) las tablas de consulta (*lookup tables*) en formato `.csv` listas para su integración en los módulos de forzamiento climático de WILIAM:

* **`IPCC_Waste_Model_Reconstructed.csv`**: Matriz consolidada para las 36 macrorregiones con los 12 parámetros óptimos calibrados del modelo FOD ($k_i^*$, $H_{0,\text{MSW}}^*$, $H_{0,\text{IND}}^*$, $\omega_{\text{IND}}^*$), el error cuadrático residual ($\text{SSE}$) y las trayectorias anuales de emisiones simuladas (2005–2023).
* **`Evolucion_Porcentajes_Vertederos_2005_2023.csv`**: Evolución dinámica de la composición porcentual interna del stock de carbono orgánico degradable acumulado ($\text{DDOC}_m^a(t)$) por fracción de residuo y región.
* **Tablas multicategoría (4.B, 4.C y 4.D)**: Vectores de factores de emisión regionales calibrados y series de emisiones desglosadas por gas ($\text{CH}_4$, $\text{CO}_2$, $\text{N}_2\text{O}$) y agregadas en $\text{CO}_2\text{-eq}$.

---

## Autoría y créditos académicos

* **Autor:** Javier Hernández Cepeda
* **Titulación:** Grado en Ingeniería Electrónica Industrial y Automática
* **Tutores:**
  * Luis Javier Miguel González (*Departamento de Ingeniería de Sistemas y Automática, Escuela de Ingenierías Industriales, Universidad de Valladolid*).
  * Gonzalo Parrado Hernando (*Grupo de Energía, Economía y Dinámica de Sistemas - GEEDS, Universidad de Valladolid / School of Ecology, Environment and Resources, Guangdong University of Technology*).
* **Institución:** Escuela de Ingenierías Industriales – Universidad de Valladolid (UVa), 2026.

---

## Licencia

Este proyecto se distribuye bajo la **Licencia MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
