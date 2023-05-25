# Recommender Systems for the [Spotify Million Playlist Dataset Challenge ](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) <img  class="lazyloaded" src="spotify.svg" height="30px">

Welcome to our first Recommender Systems repo! :wave: This work was developed as a college assignment in our last college year. We were encouraged to implement classical approaches in Recommender Systems to tackle the popular [Spotify Million Playlist Dataset Challenge <img  class="lazyloaded" src="spotify.svg" height="15px"> (SMPD)](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge). This challenge was released in the [ACM RecSys Conference 2018](https://www.recsyschallenge.com/2018/) and all data is available in [Spotify website](https://research.atspotify.com/datasets/). Researches were intended to make accurate recommendations for a huge dataset of 1M playlists with 2M unique tracks of 300k different artists. 

Our approaches are highly updated from the state-of-the-art in Recommender Systems since they were developed for academic purposes. However, it might be interesting to take them into account as baselines of new proposals, so we decided to release all our code in order to help the research community. In our college assignment we were expected to implement in local machines (with limited resources) efficient parsers for the dataset to compute naive recommendations with classical techniques. In our lecture sessions we followed [Ricci's Handbook](https://link.springer.com/book/10.1007/978-0-387-85820-3) to learn some basis, so if you are interested in check the theoretical aspect of our implemented models we suggest you to consult this fantastic manuscript :v:.



## Implemented models

Following the [submission format](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#submission-format) of the challenge, we have to recommend 500 tracks with a given limitation: it is not permitted to suggest some track that it is *already included* in a test playlist.

The developed models are the following:

- Baseline model based on popularity ([models/baseline.py](models/baseline.py)):

Firstly, we were asked to compute a *baseline* model based on popularity. We copmute the most popular tracks counting the number of times a track is included in some playlist.








## Team builders :construction_worker:

- Ana Ezquerro ([ana.ezquerro@udc.es](mailto:ana.ezquerro@udc.es), [GitHub](https://github.com/anaezquerro)).
- Pedro Souza ([pedro.souza@udc.es](mailto:pedro.souza@udc.es), [GitHub](https://github.com/pedrosouzaa1)).

## Models

- *Baseline*: Popularity.
- Neighbor-based Model with two variants: user-based and item-based.
- 
- Modelo basado en popularidad ([baseline.py](models/baseline.py)). 
- Modelo basado en vecindarios ([neighbour.py](models/neighbour.py)) de playlists (_user-based_) o de tracks (_item-based_).
- Modelo basado en SVD puro ([puresvd.py](models/puresvd.py)).
- Modelo basado en vecindarios de tracks utilizando embeddings ([track2vec.py](models/track2vec.py)).

Para mayor información acerca de la implementación de los modelos se recomienda leer [models/README.md](models/README.md).

## Manual de uso

Para realizar las recomendaciones y evaluar los modelos se puede ejecutar directamente el archivo 
[main.py](main.py) por línea de comandos, seguido de los siguientes argumentos:

- `model`: Nombre del modelo que se quiere ejecutar. Las opciones son `base`, `neighbour`, `puresvd`, `track2vec` y `eval`. En caso 
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

| Parámetro       | `base`                    | `neighbour`                 | `puresvd`                               | `track2vec`                    |
|-----------------|---------------------------|-----------------------------|-----------------------------------------|--------------------------------|
| `submit_path`   | `submissions/base.csv.gz` | `submissions/{hood}.csv.gz` | `submissions/puresvd{ftest}-{h}.csv.gz` | `submissions/track2vec.csv.gz` |
| `num_threads`   | `#proc`                   | `(8, #proc)`                | `#proc`                                 | `#proc`                        |
| `action`        |                           | `recommend`                 | `recommend`                             | `recommend`                    | 
| `batch_size`    |                           | `100`                       | `100`                                   | `100`                          | 
| `k`             |                           | `100`                       |                                         | `10`                           |
| `matrix_path`   |                           | `data/Rest.npz`             |                                         |                                |
| `-load`         |                           | `False`                     |                                         |                                |
| `train_path`    |                           | `data/Rtrain.npz`           | `data/Rtrain.npz`                       | `data/Rtrain.npz`              |
| `test_path`     |                           | `data/Rtest.npz`            | `data/Rtest.npz`                        | `data/Rtest.npz`               |
| `trackmap_path` |                           | `data/trackmap.pickle`      | `data/trackmap.pickle`                  | `data/trackmap.pickle`         | 
| `h`             |                           |                             | `10`                                    |                                |
| `-ftest`        |                           |                             | `False`                                 |                                |     
| `U_path`        |                           |                             | `data/U.npy`                            |                                |
| `V_path`        |                           |                             | `data/V.npy`                            |                                |
| `S_path`        |                           |                             | `data/S.npy`                            | `data/S-track2vec.npz`         |
| `embded_dim`    |                           |                             |                                         | `50`                           |
| `context_size`  |                           |                             |                                         | `10`                           |
| `model_path`    |                           |                             |                                         | `data/track2vec`               |
| `num_epochs`    |                           |                             |                                         | `50`                           |
| `granularity`   |                           |                             |                                         | `10e3`                         |

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


### [`Track2Vec`](models/track2vec.py)

Para entrenar el modelo [gensim.Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) con un tamaño de 
*embedding* `embed_dim=100` y 100 *epochs*:

```shell
python3 main.py track2vec train -v -t --embed_dim=100 --num_epochs=100
```

Para realizar recomendaciones aproximando el algoritmo kNN con [gensim.AnnoyIndexer](https://radimrehurek.com/gensim/auto_examples/tutorials/run_annoy.html) 
con `k=20`:

```shell
python3 main.py track2vec recommend -eval -v -t --num_threads 1 10 --k=20 --annoy --num_trees=100
```

Para realizar recomendaciones computando la matriz de similitudes (muy costoso):

```shell
python3 main.py track2vec recommend -eval -v -t --num_threads 5 10 --k=20
```


## Resultados 

En el siguiente [enlace](https://udcgal-my.sharepoint.com/:f:/g/personal/ana_ezquerro_udc_es/EuDyme7p-uFPpVomMjwWkmgBhpUUz3clxkTMELy2J0BZjA?e=FOFokB) 
se puede acceder a los archivos `csv.gz` de las _submissions_ de cada modelo.


| Modelo                  | R-Precision | nDCG | clicks |
|-------------------------|-------------|------|--------|
| popularity              | 0.02        | 0.09 | 17.33  |
| user (k=100)            | 0.16        | 0.32 | 4.74   |
| item (k=20)             | 0.13        | 0.25 | 5.98   |
| puresvd (h=10)          | 0.07        | 0.20 | 10.24  |  
| puresvd' (h=10)         | 0.08        | 0.21 | 10.28  |
| puresvd (h=50)          | 0.11        | 0.27 | 6.46   |
| puresvd' (h=50)         | 0.12        | 0.27 | 6.55   |
| track2vec (d=100, k=20) | 0.02        | 0.07 | 14.36  |

**Nota**: En el modelo `puresvd` el apóstrofe indica que se utilizó la matriz de test para factorizar.