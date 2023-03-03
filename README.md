# *Iteración 0: Implementación de modelo basado en popularidad*

Equipo:
- Ana Xiangning Pereira Ezquerro ([ana.ezquerro@udc.es](mailto:ana.ezquerro@udc.es)).
- Pedro Souza López ([pedro.souza@udc.es](mailto:pedro.souza@udc.es)).

## Explicación breve del código 

El código adjunto a este README se organiza en tres archivos `.py`:

- `baseline.py`: Contiene la implementación del modelo basado en popularidad. La clase
 `BaselineModel` permite _parsear_ el dataset, realizar el conteo de _tracks_ en paralelo,
devolver recomendaciones para el conjunto de test (`spotify_test_playlists/`) y, de forma opcional, 
exportar dichas recomendaciones en un archivo `csv.gz` en el 
[formato del challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#submission-format)
- `evaluation.py`: La clase `Evaluator` soporta la lectura de una _submission_ así como
del _ground truth_ (en formato JSON) para realizar la evaluación basada en las 
[métricas del challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#evaluation);
RPrediction, NDCG y clicks.
- `main.py`: Estructura el flujo de ejecución permitiendo ejecutar solamente los pasos deseados.
Además, permite mostar los tiempos de ejecución de los distintos pasos.

## Manual de uso

Para utilizar las distintas funcionalidades del código se debe ejecutar el archivo `main.py`
 por línea de comandos, seguido de algunos argumentos que indican qué se va a 
 ejecutar y si se muestran o no los distintos tiempos de ejecución. 

Por defecto, en caso de no añadir ningún argumento, se realizan los siguientes pasos:

1. Conteo de los _tracks_ más populares del _dataset_ (`BaselineModel.train()`).
2. Recomendación de los _tracks_ más populares que no se encuentren en las playlists de test
   (`BaselineModel.predict()`).
3. Almacenamiento del fichero `baseline.csv.gz` en el formato de la _submission_.
4. Evaluación del modelo a través de la lectura de `baseline.csv.gz` y del archivo JSON con el _ground thruth_ del 
conjunto de test con las métricas mencionadas anteriormente.

De forma opcional se pueden añadir los siguientes argumentos:

- `-pred`: Se ejecutarán los pasos 1, 2 y 3.
- `-eval`: Se ejecutará solamente el paso 4. Para que funcione es necesario haber generado previamente el archivo de la _submission_.

Además, a todas estas combinaciones de parámetros se le puede añadir:

- `-t`: Se mostrarán los tiempos de ejecución de los pasos que se ejecuten.
- `-out {file}`: Se guardará el archivo `csv.gz` en la ruta especificada.

Por ejemplo, para ejecutar todos los pasos (1-4) mostrando los tiempos de ejecución y guardando la _submission_ en 
el fichero `iteration0.csv.gz` se debe ejecutar:

```
python3 main.py -t -out iteration0.csv.gz
```