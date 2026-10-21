================================================================================
GUIA UNIVERSAL DE INSTALACION Y PRIMERA EJECUCION
Herramienta: app_4x.py (Streamlit)
================================================================================

Esta guia esta disenada para que cualquier usuario, en cualquier ordenador y
sistema operativo (Windows, macOS o Linux), pueda poner en marcha la aplicacion
desde cero de forma sencilla y garantizada.

Antes de comenzar crear una carpeta en el directorio donde están todos los archivos y llamarla "data"
En esta se deben introducir todos los .csv

--------------------------------------------------------------------------------
1. REQUISITOS PREVIOS DEL SISTEMA
--------------------------------------------------------------------------------
- Sistema Operativo: Windows 10/11, macOS o distribuciones Linux (Ubuntu, Debian, Fedora, etc.).
- Python: Version 3.9, 3.10, 3.11 o 3.12 instalada.
- Navegador Web: Google Chrome, Microsoft Edge, Mozilla Firefox o Safari.

Como verificar si Python esta instalado:
   Abra una terminal o consola de comandos y escriba:
      python --version
   (o en Linux/macOS):
      python3 --version

   Si muestra una version como "Python 3.10.x", puede continuar.
   Si no esta instalado:
      - Descarguelo gratuitamente de la pagina oficial: https://www.python.org/downloads/
      - IMPORTANTE EN WINDOWS: Durante el instalador, marque obligatoriamente la
        casilla "Add python.exe to PATH" en la primera pantalla antes de pulsar Instalar.

--------------------------------------------------------------------------------
2. PASO 1: ABRIR LA TERMINAL EN LA CARPETA DE ESTE ARCHIVO
--------------------------------------------------------------------------------
Debe situar la consola de comandos en la carpeta donde ha extraido o descargado
este modulo (donde se encuentra este archivo y "app_4a.py").

Forma rapida en Windows:
   1. Abra el Explorador de Archivos y entre en la carpeta de este modulo ("TFG_CODE").
   2. Haga clic en la barra de direcciones superior (donde pone la ruta de la carpeta).
   3. Escriba "cmd" o "powershell" y pulse la tecla Enter.
   4. Se abrira directamente la consola situada en esta carpeta.

Forma tradicional (cualquier sistema operativo):
   Abra la terminal (PowerShell, CMD, Terminal de Mac o Bash) y navegue hasta
   la carpeta usando el comando "cd" seguido de la ruta donde tenga guardada la carpeta:
      cd ".../TFG_CODE"

--------------------------------------------------------------------------------
3. PASO 2: CREAR Y ACTIVAR UN ENTORNO VIRTUAL (RECOMENDADO)
--------------------------------------------------------------------------------
El entorno virtual crea un espacio aislado para instalar las librerias del proyecto
sin interferir con otros programas de su ordenador.

A) Crear el entorno virtual (solo se hace la primera vez):
   En Windows:
      python -m venv venv
   En macOS / Linux:
      python3 -m venv venv

B) Activar el entorno virtual:
   - En Windows con PowerShell:
      .\venv\Scripts\Activate.ps1
      (Nota de seguridad en Windows: Si PowerShell muestra un error de directiva
       de ejecucion de scripts, ejecute primero esta orden y vuelva a activar:
       Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process)

   - En Windows con Simbolo del Sistema (CMD):
      venv\Scripts\activate.bat

   - En macOS / Linux:
      source venv/bin/activate

   Al activarse correctamente, vera que aparece "(venv)" al principio de la linea
   de comandos.

--------------------------------------------------------------------------------
4. PASO 3: INSTALAR LAS LIBRERIAS NECESARIAS
--------------------------------------------------------------------------------
Con el entorno virtual activado, instale las dependencias mediante el archivo
"requirements.txt" incluido en esta carpeta:

   pip install -r requirements.txt

(Opcion alternativa sin archivo):
   pip install streamlit pandas numpy scipy plotly

Espere a que finalice la descarga e instalacion de los paquetes (Streamlit,
Pandas, NumPy, SciPy y Plotly).

--------------------------------------------------------------------------------
5. PASO 4: EJECUTAR LA APLICACION INTERACTIVA
--------------------------------------------------------------------------------
Para arrancar la aplicacion web de la Categoria 4.A, ejecute en la terminal:

   streamlit run app_4X.py (app_4a.py, app_4b.py, app_4c.py, app_4d.py)

Si desea asignar de forma explicita el puerto predeterminado de la suite (8501):

   streamlit run app_4x.py --server.port 8501

Al arrancar, la terminal mostrara un mensaje confirmando el servidor local:
   Local URL: http://localhost:8501

Su navegador web se abrira de forma automatica en esa direccion.
Si por alguna razon no se abre solo, abra manualmente su navegador web favorito
(Chrome, Edge, Firefox, etc.) e introduzca en la barra de direcciones:
   http://localhost:8501

--------------------------------------------------------------------------------
6. PASO 5: COMO DETENER EL PROGRAMA
--------------------------------------------------------------------------------
Para detener el servidor de la aplicacion en cualquier momento:
   1. Vaya a la ventana de la terminal donde esta corriendo el proceso.
   2. Presione la combinacion de teclas: Ctrl + C
   3. Para salir del entorno virtual cuando termine su sesion de trabajo, escriba:
      deactivate

--------------------------------------------------------------------------------
7. RESOLUCION DE PROBLEMAS COMUNES (TROUBLESHOOTING)
--------------------------------------------------------------------------------
- Caso A: "python o pip no se reconoce como un comando interno o externo"
  Causa: Python no se agrego a las variables de entorno PATH durante la instalacion.
  Solucion: Vuelva a instalar Python marcando la casilla "Add python.exe to PATH"
  o intente ejecutar con el comando "py -m streamlit run app_4x.py".

- Caso B: "No module named streamlit"
  Causa: El entorno virtual no esta activado o la instalacion de dependencias fallo.
  Solucion: Asegurese de que la terminal muestra "(venv)" a la izquierda y repita:
  pip install -r requirements.txt

- Caso C: "Port 8501 is already in use"
  Causa: Hay otra aplicacion o ventana previa ocupando ese puerto de red.
  Solucion: Puede arrancar la aplicacion en otro puerto libre ejecutando:
  streamlit run app_4x.py --server.port 8505

- Caso D: "FileNotFoundError / No se encuentran los archivos CSV"
  Causa: La subcarpeta de datos se ha movido o renombrado.
  Solucion: Verifique que la subcarpeta "data" permanece intacta dentro
  de la misma carpeta donde se encuentra "app_4x.py". La herramienta busca
  los datos de forma relativa a su propia ubicacion.
================================================================================
