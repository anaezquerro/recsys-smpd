# Explicación de los modelos implementados

En este documento se explican brevemente los modelos implementados para la práctica 
y el preporcesamiento necesario para ejecutar cada uno de ellos sobre el 
[SPMD](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).

## Modelo basado en popularidad ([baseline.py](baseline.py))

Para el modelo basado en popularidad el único preprocesamiento necesario para 
generar una _submission_ de recomendación es contar la ocurrencia de los _tracks_ sobre 
todo el conjunto de entrenamiento. Una vez realizado el conteo, se recomiendan los top N
_tracks_ más populares para cada _playlist_ de test que no han sido añádidos a la misma.

Consideraciones de la implementación:

- Para realizar el conteo de _tracks_ no se tienen en cuenta la frecuencia dentro de 
cada _playlist_. Es decir, si un _track_ aparece múltiples veces en una _playlist_
sólo se le tendrá en cuenta una ocurrencia.

Fases de la ejecución del modelo:

1. `preprocess`: _Parsing_ del conjunto de entrenamiento para realizar el conteo de los _tracks_.
2. `recommend`: Generación de la recomendación.

## Modelo basado en vecindarios ([neighbour.py](neighbour.py))

En el modelo basado en vecindarios, para calcular las similaridades entre _playlists_ 
y _tracks_ fue necesario crear una matriz _sparse_ del conjunto de entrenamiento y otra 
del conjunto de test. 
La clase `Sparse` ([sparse.py](../utils/sparse.py)) realiza este proceso en paralelo: primero recoge todas 
las _playlists_ y todos los _tracks_ del dataset y crea las matrices _sparse_ y las 
almacena.

Notación del código:


| Variable  | Significado                           | Dimensiones     |
|-----------|---------------------------------------|-----------------|
| `train`   | Conjunto de entrenamiento             |                 |
| `test`    | Conjunto de test                      |                 |
| `m_train` | Número de _playlists_ en `train`      |                 |
| `m_test`  | Número de _playlists_ en `test`       |                 |
| `n`       | Número de _tracks_ en todo el dataset |                 | 
| `Rtrain`  | Matriz de _ratings_ de `train`        | [`m_train`, `n`] |
| `Rtest`   | Matriz de _ratings_ de `test`         | [`m_test`, `n`]  |
| `Rest`    | Matriz de _ratings_ estimada          | [`m_test`, `n`]  |

Una vez con las matrices almacenadas se procede a computar la matriz de similaridades 
y la estimación de los nuevos _ratings_ para test:

- _User-based_ (`Suser`): La matriz es de dimensiones [`m_test`, `m_train`] y se puede computar 
como `Rtest @ t(Rtrain)`. Después el `Rest` se calcula como `Suser @ Rtrain`. 
- _Item-based_ (`Sitem`): La matriz es de dimensiones [`n`, `n`] y se puede computar 
como `t(Rtrain) @ Rtrain`. Después el `Rest` se calcula como `Rtest @ Sitem`.

Finalmente se procesa la matriz `Rtest` de forma paralela para tomar los _tracks_ correspondientes 
a los _ratings_ más altos (excluyendo aquellos que ya están en `test`).


Consideraciones de la implementación:

- Para implementar este modelo se usó esencialmente la clase [`csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
de la librería [Scipy](https://scipy.org/).
- Para aquellas _playlists_ de test que no tienen ningun _track_ añadido no 
se le pueden realizar recomendaciones, pues su vector de _ratings_ no tiene ningún valor. 
Para estas *playlists* se les recomendará _tracks_ según el modelo basado en popularidad.
- Para el modelo _item-based_, cuando se computa la matriz de similaridades se obtiene un _warning_ de `ZeroDivision`. 
Esto se debe a que algunas columnas de `Rtrain` tienen norma igual a 0 (se corresponden a _tracks_ que no aparecen en el 
conjunto de entrenamiento). Se debe ignorar este _warning_, pues de forma automática [Numpy](https://numpy.org/) no 
devuelve infinitos (como debería ser formalmente), sino que devuelve ceros. 

Fases de la ejecución del modelo:

1. `sparsify`: Generación de las matrices _sparse_.
2. `recommend`: Generación de la recomendación _user_ o _item_ _based_.





