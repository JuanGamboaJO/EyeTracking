# EYEIA

## Descripción

Este repositorio contiene la base de datos y los archivos necesarios para correr el script que se encarga de predecir la posición donde estes viendo en la pantalla (Resolución de 1600x900).

## Estructura del Proyecto

- **subjects/**
  - Contiene las imágenes correspondientes a los ojos de cada sujeto, separadas por ojo derecho e izquierdo.

- **ojoderecho/**
  - Contiene las imágenes del ojo derecho que se usarán para el entrenamiento de la red neuronal.

- **ojoizquierdo/**
  - Contiene las imágenes del ojo izquierdo que se usarán para el entrenamiento de la red neuronal.

- **testeyesderecho/**
  - Contiene las imágenes del ojo derecho que se usarán para el test de la red neuronal.

- **testeyesizquierdo/**
  - Contiene las imágenes del ojo izquierdo que se usarán para el test de la red neuronal.

- **xModels/**
  - Contiene los modelos entrenados.

## Instrucciones de Uso

1. **Clonar el repositorio**
    ```sh
    git clone https://JuanGamboaJO/EyeTracking.git
    ```

2. **Estructura de las Carpetas**
    - Asegúrate de que las carpetas mencionadas anteriormente contengan las imágenes necesarias antes de ejecutar el programa.

3. **Entrenamiento y Test de la Red Neuronal**
    - Usa las imágenes en `ojoderecho` y `ojoizquierdo` para el entrenamiento.
    - Usa las imágenes en `testeyesderecho` y `testeyesizquierdo` para el test.
    - Los modelos entrenados se guardarán en la carpeta `xModels`.

4. **Ejecutable**
    - "Para usar el programa sin la necesidad de instalar librerías o Python, puedes descargar el ejecutable desde este link.. (https://drive.google.com/file/d/1NeGz-CNXEafsPqvC8eph8F-J2UJk-D3m/view?usp=sharing)