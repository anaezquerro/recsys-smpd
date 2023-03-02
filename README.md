# *Iteración 0: Implementación de modelo basado en popularidad*

Equipo:
- Ana Xiangning Pereira Ezquerro ([ana.ezquerro@udc.es](mailto:ana.ezquerro@udc.es)).
- Pedro Souza López ([pedro.souza@udc.es](mailto:pedro.souza@udc.es)).

## Explicación breve del código 

El código adjunto a este README se organiza en tres archivos `.py`:

- `baseline.py`: Contiene la implementación del modelo basado en popularidad. La clase
 `BaselineModel` permite _parsear_ el dataset, realizar el conteo de _tracks_ y
  crear una _submission_ basada en el conteo realizado la cual, opcionalmente, puede ser exportada.
- `evaluation.py`: La clase `Evaluator` soporta la lectura de una _submission_ así como
del _ground truth_ para realizar la evaluación basada en las métricas del 
[challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#evaluation);
RPrediction, NDCG y clicks.
- `main.py`: Estructura el flujo de ejecución permitiendo ejecutar solamente los pasos deseados.
Además, permite mostar los tiempos de ejecución de los distintos pasos.

## Manual de uso

Para utilizar las distintas funcionalidades del código se debe ejecutar el archivo `main.py`
 por línea de comandos, seguido de algunos argumentos que indican qué se va a 
 ejecutar y si se muestran o no los distintos tiempos de ejecución. 
 
 - Si se añade el argumento `-pred` se ejecutará únicamente la lectura del `.json`,
 el conteo de las _tracks_, la creación de las predicciones del modelo de poularidad 
 y la generación del archivo `baseline.csv.gz` que guarda la predicción. 