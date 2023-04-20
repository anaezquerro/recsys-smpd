# Modelos de Recomendación para el Spotify Million Playlist Dataset Challenge


Equipo:
- Ana Xiangning Pereira Ezquerro ([ana.ezquerro@udc.es](mailto:ana.ezquerro@udc.es)).
- Pedro Souza López ([pedro.souza@udc.es](mailto:pedro.souza@udc.es)).

## Modelos implementados

- Modelo basado en popularidad ([baseline.py](models/baseline.py)). 
- Modelo basado en vecindarios ([neighbour.py](models/neighbour.py)) de playlists (_user-based_) o de tracks (_item-based_).
- Modelo basado en SVD puro ([puresvd.py](models/puresvd.py)).

Para mayor información acerca de la implementación de los modelos se recomienda leer [models/README.md](models/README.md).

## Manual de uso

Para realizar las recomendaciones y evaluar los modelos se puede ejecutar directamente el archivo 
[main.py](main.py) por línea de comandos, seguido de los siguientes argumentos:

- `model`: Nombre del modelo que se quiere ejecutar. Las opciones son `base`, `neighbour`, `puresvd` y `eval`. En caso 
de escoger `eval` se lanza el [Evaluator](utils/evaluation.py) y se debe especificar obligatoriamente el `submit_path`.
- `-eval`: Sólo funcional cuando se especifica un modelo de recomendación. Al añadir el _flag_ se realiza la evaluación de la _submission_ generada (especificada en el argument
`submit_path`).

A mayores se pueden añadir forma opcional los siguientes parámetros a los modelos:


- `submit_path`: Ruta en la que se exportará la recomendación en el 
[formato del challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/). 
- `num_threads`: Número de hilos que se utilizarán para paralelizar en todas las fases 
predicción y recomendación.
- `time`: *Flag* que indica si see quiere imprimir por pantalla o no los tiempos de ejecución. 
- `verbose`: *Flag* que indica si se quiere imprimir por pantalla un _trace_ de la ejecución. 

En la siguiente tabla podemos ver el recopilatorio de los parámetros que se utilizan y sus valores por defecto dependiendo 
del modelo. Nótese que:

- Los *flags* van precedidos por un guión (-).
- Las celdas en blanco indican que el parámetro no se usa para ese modelo en concreto.

| Parámetro       | `base`                    | `neighbour`                 | `puresvd`                               |
|-----------------|---------------------------|-----------------------------|-----------------------------------------|
| `submit_path`   | `submissions/base.csv.gz` | `submissions/{hood}.csv.gz` | `submissions/puresvd{ftest}-{h}.csv.gz` |
| `num_threads`   | `#proc`                   | `(8, #proc)`                | `#proc`                                 |
| `action`        |                           | `recommend`                 | `recommend`                             |
| `batch_size`    |                           | `100`                       | `100`                                   |
| `k`             |                           | `100`                       |                                         |
| `matrix_path`   |                           | `data/Rest.npz`             |                                         |
| `-load`         |                           | `False`                     |                                         |
| `train_path`    |                           | `data/Rtrain.npz`           | `data/Rtrain.npz`                       |
| `test_path`     |                           | `data/Rtest.npz`            | `data/Rtest.npz`                        |
| `trackmap_path` |                           | `data/trackmap.pickle`      | `data/trackmap.pickle`                  |
| `h`             |                           |                             | `10`                                    |
| `-ftest`        |                           |                             | `False`                                 |     
| `U_path`        |                           |                             | `data/U.npy`                            |
| `V_path`        |                           |                             | `data/V.npy`                            |
| `S_path`        |                           |                             | `data/S.npy`                            |

---
## Ejemplos de ejecuciones

Para probar los dos modelos implementados se sugiere no alterar los parámetros que vienen por defecto (a excepción del 
número de hilos y _batch_size_). A continuación se propone un ejemplo de ejecución por línea de comandos para probar 
los dos modelos desde cero.


### [`BaselineModel`](models/baseline.py)


```shell
python3 main.py base -t -v && python3 main.py eval -t -v --submit_path=submissions/base.csv.gz
```
### [`NeighbourModel`](models/neighbour.py)

Primero generar las matrices _sparse_:

```shell
python3 main.py neighbour user --action sparsify -t -v
```

Y después se pueden probar los modelos _user_ e _item_ _based_:

```shell
python3 main.py neighbour user -eval --action recommend -t -v --k=100
```

```shell
python3 main.py neighbour item -eval --action recommend -t -v --k=20 --batch_size=15000
```

Para el `NeighbourModel` se comprobó la configuración de _batch_size_ y _num_threads_ en una máquina de 32GB de RAM y 20 hilos. 
Se recomienda realizar previamente una estimación de la configuración a utilizar dependiendo de las capacidades de la máquina.

### [`PureSVDModel`](models/puresvd.py)

Sin utilizar la matriz de test para factorizar con 10 factores latentes:
```shell
python3 main.py puresvd -eval --action recommend -t -v --h=10 --batch_size=50 --num_threads=10
```

Utilizando la matriz de test para factorizar con 10 factores latentes:

```shell
python3 main.py puresvd -ftest -eval --action recommend -t -v --h=10 --batch_size=100 --num_threads=10
```


## Resultados 

En el siguiente [enlace](https://udcgal-my.sharepoint.com/:f:/g/personal/ana_ezquerro_udc_es/EuDyme7p-uFPpVomMjwWkmgBhpUUz3clxkTMELy2J0BZjA?e=FOFokB) 
se puede acceder a los archivos `csv.gz` de las _submissions_ de cada modelo.


| Modelo          | R-Precision | nDCG         | clicks |
|-----------------|-------------|--------------|--------|
| popularity      | 0.02        | 0.09         | 17.33  |
| user (k=100)    | 0.16        | 0.32         | 4.75    |
| item (k=20)     | 0.13        | 0.23         | 7.44   |
| puresvd (h=10)  | 0.07        | 0.20         | 10.24  |  
| puresvd' (h=10) | 0.08        | 0.21         | 10.28  |
| puresvd (h=50)  | 0.11        | 0.27         | 6.46   |
| puresvd' (h=50) | 0.12        | 0.27         | 6.55   |

**Nota**: En el modelo `puresvd` el apóstrofe indica que se utilizó la matriz de test para factorizar.