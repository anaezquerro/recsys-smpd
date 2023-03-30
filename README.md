# Modelos de Recomendación para el Spotify Million Playlist Dataset Challenge


Equipo:
- Ana Xiangning Pereira Ezquerro ([ana.ezquerro@udc.es](mailto:ana.ezquerro@udc.es)).
- Pedro Souza López ([pedro.souza@udc.es](mailto:pedro.souza@udc.es)).

## Modelos implementados

- Modelo basado en popularidad (`baseline.py`). 
- Modelo basado en vecindarios (`neighbour.py`) de playlists o de tracks.

## Manual de uso

Para realizar las recomendaciones y evaluar los modelos se puede ejecutar directamente el archivo 
`main.py` por línea de comandos, seguido de los siguientes argumentos:

- `model`: Nombre del modelo que se quiere ejecutar (`popularity`, `user`, `item`).
- `action`: Modo de ejecución: recomendación (`pred`) y evaluación (`eval`). En caso de no 
no especificar ninguna se ejecutarán ambas.

A mayores se pueden añadir forma opcional los siguientes parámetros:

- `submit_path`: Ruta en la que se exportará la recomnedación en el 
[formato del challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/).
- `num_threads`: Número de hilos que se utilizarán para paralelizar en todas las fases 
predicción y recomendación. Por defecto el número de procesos de la máquina.
- `num_threads_rec`: Número de hilos que se utilizarán para paralelizar la fase de 
recomendación.
- `batch_size`: Para el modelo de vecindario, el número de filas de la matriz con las 
que se paraleliza el cálculo de la similitud. Por defecto 500.
- `k`: Número de vecinos a considerar la similitud en el modelo de vecindarios. Por defecto 100.
- `time`: Valor booleano que indica si see quiere imprimir por pantalla o no los tiempos 
de ejecución.

## Ejemplos de ejecuciones

