# *Iteración 0: Implementación de modelo basado en popularidad*

Equipo:
- Ana Xiangning Pereira Ezquerro ([ana.ezquerro@udc.es](mailto:ana.ezquerro@udc.es)).
- Pedro Souza López ([pedro.souza@udc.es](mailto:pedro.souza@udc.es)).

## Explicación breve del código 

El código adjunto con este README se organiza en tres archivos `.py`:

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

Para llevar a cabo la ejecución del código 