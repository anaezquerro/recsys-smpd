# Modelos de Recomendación para el Spotify Million Playlist Dataset Challenge


Equipo:
- Ana Xiangning Pereira Ezquerro ([ana.ezquerro@udc.es](mailto:ana.ezquerro@udc.es)).
- Pedro Souza López ([pedro.souza@udc.es](mailto:pedro.souza@udc.es)).

## Modelos implementados

- Modelo basado en popularidad ([baseline.py](models/baseline.py)). 
- Modelo basado en vecindarios ([neighbour.py](models/neighbour.py)) de playlists (_user-based_) o de tracks (_item-based_).

Para mayor información acerca de la implementación de los modelos se recomienda leer [models/README.md](models/README.md).

## Manual de uso

Para realizar las recomendaciones y evaluar los modelos se puede ejecutar directamente el archivo 
[main.py](main.py) por línea de comandos, seguido de los siguientes argumentos:

- `model`: Nombre del modelo que se quiere ejecutar (`base`, `user`, `item`).
- `-eval`: Añadir el _flag_ para evaluar la _submission generada_.

A mayores se pueden añadir forma opcional los siguientes parámetros:

- `submit_path`: Ruta en la que se exportará la recomendación en el 
[formato del challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/). Por defecto
  - En el `BaselineModel` se guardará como `submissions/base.csv.gz`.
  - En el `NeighourModel` se guardará como `submissions/user.csv.gz` o `submissions/item.csv.gz`.
- `num_threads`: Número de hilos que se utilizarán para paralelizar en todas las fases 
predicción y recomendación. Por defecto:
  - En el `BaselineModel` es el número de procesadores de la máquina
  - En el `NeighourModel` para `sparsify` es el número de procesadores de la máquina, y para `recommend` es 8 al realizar los productos matriciales y el número de procesadores de la máquina para realizar la recomendación.
- `time`: Valor booleano que indica si see quiere imprimir por pantalla o no los tiempos de ejecución. Por defecto `True`.
- `verbose`: Valor booleano que indica si se quiere imprimir por pantalla un _trace_ de la ejecución. Por defecto `True`.

Para `NeighbourModel` se añaden parámetros adicionales:
- `action`: Para el modelo de vecindarios se ha separado la ejecución en dos fases: `sparsify` (para generar las matrices 
_sparse_ del dataset y guardarlas) y `recommend` (para generar recomendacioes en base a un vecindario). Por defecto, `recommend`.
- `batch_size`: Número de filas de la matriz con las que se paraleliza el cálculo de la similitud. Por defecto 500.
- `k`: Número de vecinos a considerar la similitud en el modelo de vecindarios. Por defecto 100.
- `train_path`: Ruta donde se guarda la matriz _sparse_ de entrenamiento.
- `test_path`: Ruta donde se guarda la matriz _sparse_ de test.
- `trackmap_path`: Ruta donde se guarda el diccionario que mapea _track_ URIs a columnas de las matrices _sparse_.
- `matrix_path`: Ruta donde se guardará la matriz de _ratings_ estimados.
- `load`: Valor booleano que indica si se debe cargar o no la matriz de _ratings_ estimados.

## Ejemplos de ejecuciones

Para probar los dos modelos implementados se sugiere no alterar los parámetros que vienen por defecto (a excepción del 
número de hilos y _batch_size_). A continuación se propone un ejemplo de ejecución por línea de comandos para probar 
los dos modelos desde cero.

Para probar el `BaselineModel`:

```shell
python3 main.py base -eval -t -v
```

Para probar el `NeighbourModel`, primero generar las matrices _sparse_:

```shell
python3 main.py user --action sparsify -t -v --train_path=data/Rtrain.npz
```

Y después se pueden probar los modelos _user_ e _item_ _based_:

```shell
python3 main.py user -eval --action recommend -t -v --k=100
```

```shell
python3 main.py item -eval --action recommend -t -v --k=20 --batch_size=15000
```

Para el `NeighbourModel` se comprobó la configuración de _batch_size_ y _num_threads_ en una máquina de 32GB de RAM y 20 hilos. 
Se recomienda realizar previamente una estimación de la configuración a utilizar dependiendo de las capacidades de la máquina.


## Resultados 

En el siguiente [enlace](https://udcgal-my.sharepoint.com/:f:/g/personal/ana_ezquerro_udc_es/EuDyme7p-uFPpVomMjwWkmgBhpUUz3clxkTMELy2J0BZjA?e=FOFokB) 
se puede acceder a los archivos `csv.gz` de las _submissions_ de cada modelo.

| Modelo          | R-Precision | nDCG | CRT   |
|-----------------|-------------|------|-------|
| popularity      | 0.02        | 0.09 | 17.33 |
| user (k=100)    | 0.16        | 0.32 | 4.74  |
| item (k=20)     | 0.13        | 0.23 | 7.44  |
| puresvd (k=10)  |             |      |       |  
| puresvd (k=50)  |             |      |       |  
| puresvd (k=1-0) |             |      |       |  



