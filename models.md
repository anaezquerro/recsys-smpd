# Explicación de los modelos implementados

En este documento se explican brevemente los modelos implementados para la práctica 
y el preporcesamiento necesario para ejecutar cada uno de ellos sobre el 
[SPMD](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).

## Modelo basado en popularidad (`baseline.py`)

Para el modelo basado en popularidad el único preprocesamiento necesario para 
generar una _submission_ de recomendación es contar la ocurrencia de los _tracks_ sobre 
todo el conjunto de entrenamiento. Una vez realizado el conteo, se recomiendan los top N
_tracks_ más populares para cada _playlist_ de test que no han sido añádidos a la misma.

Consideraciones de la implementación:

- Para realizar el conteo de _tracks_ no se tienen en cuenta la frecuencia dentro de cada _playlist_. 


