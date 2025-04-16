# Detección de Neumonía en Rayos X con CNN

Proyecto de clasificación de imágenes médicas usando redes neuronales convolucionales para identificar neumonía en radiografías de tórax.

## Descripción

El objetivo de este proyecto es desarrollar un modelo de clasificación de imágenes médicas que permita identificar la presencia de neumonía en radiografías de tórax. Para ello, se ha utilizado un dataset de imágenes de rayos X de tórax de pacientes con neumonía y sin neumonía. Usaremos redes neuronales convolucionales (CNN) con tecnologías como TensorFlow y Keras para entrenar el modelo.

## Instalación de dependencias

Antes de nada hay que instalar las dependencias que se encuentran en el archivo `requirements.txt:`

```
pip install -r requirements.txt
```

## Entrenamiento del modelo

Basta con ejecutar el archivo `train_model.py`:

```
python3 train_model.py
```

## Probar el modelo

Al ejecutar el archivo `test_model.py` se probará el modelo con el dataset de test y generará un archivo `predictions.csv` que contendrá para cada imagen analizada la valoración del modelo, siendo 0 sano y 1 con neumonía.

```
python3 test_model.py
```
## Iniciar interfaz gráfica

Para iniciar la interfaz es necesario poner en la terminal

```
streamlit run app.py
```
