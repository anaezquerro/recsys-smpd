# Recommender Systems for the [Spotify Million Playlist Dataset Challenge ](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) <img  class="lazyloaded" src="spotify.svg" height="30px">

Welcome to our first Recommender Systems repo! :wave: This work was developed as a college assignment in our last year. We were encouraged to implement classical approaches in Recommender Systems to tackle the popular [Spotify Million Playlist Dataset Challenge <img  class="lazyloaded" src="spotify.svg" height="15px"> (SMPD)](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge). This challenge was released in the [ACM RecSys Conference 2018](https://www.recsyschallenge.com/2018/) and all data is available in [Spotify website](https://research.atspotify.com/datasets/). Researches were intended to make accurate recommendations for a huge dataset of 1M playlists with 2M unique tracks of 300k different artists. 

Our approaches are highly outdated from the state-of-the-art in Recommender Systems since they were developed for academic purposes. However, it might be interesting to take them into account as baselines of new proposals, so we decided to release all our code in order to help the research community. In our college assignment we were expected to implement in local machines (with limited resources) efficient parsers for the dataset to compute naive recommendations with classical techniques. In our lecture sessions we followed [Ricci's Handbook](https://link.springer.com/book/10.1007/978-0-387-85820-3) to learn some basis, so if you are interested in check the theoretical aspect of our implemented models we suggest you to consult this fantastic manuscript :v:.



## Implemented models

Following the [submission format](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#submission-format) 
of the challenge, we recommend 500 tracks based on a model with a given limitation: it is not permitted to suggest some 
track that it is *already included* in a test playlist. You can find a detailed description of 
each model in the [paper](recsys-smpd.pdf) attached to this repository.

### Popularity baseline

We compute the most popular tracks counting the number of times a track is included in some playlist. We do not take 
into account those tracks that are included more than once in each playlist. 

### Neighborhood-based model 

We implemented two variants of the neighborhood model: one where elements are playlists (user-based version) and other where elements are tracks (item-based version). In each version, we compute the cosine similarity between elements (either tracks/items or playlists/users) and estimate the ratings of each test playlist given the following formula:

```math
 \hat{r}_{u,i} = \sum_{v\in V_u} s_{u,v} r_{v,i} \quad  \text{(user-based)}
 ```

```math
\hat{r}_{u,i} = \sum_{j\in\mathcal{J}_i} s_{i,j} r_{u,j} \quad \text{(item-based)} 
```


where $V_u$ and $\mathcal{J}_i$ represent the neighborhood of user $u$ and item $i$ of size $k$, respectively and $s$ is the cosine similarity. The size $k$ is given as an hyper-parameter in the implementation. Then, the top 500 tracks not rated with highest score are recommended to each test playlist.

### Pure Singular Value Decomposition

We compute the binary sparse matrix of train test playlists wrt. to all tracks and use SVD to project each playlist into a lower dimension:

$$R=U\times \Sigma \times V^t \approx \tilde{U} \times \tilde{\Sigma} \times \tilde{V}^t$$

In the first variation we use both the train and test sparse matrices to compute factorization ($R=R_\text{train}|R_\text{test}$). In the second variation we calculate the factorization using only training data ($R=R_\text{train}$) and then project each test playlist vector ($\vec{r}_{m+1}$) to the latent space with:

```math
\vec{u}_{m+1} = \vec{r}_{m+1} \tilde{V}
```

Scores for each pair (playlist, track) is computed by the dot product of their corresponding latent vectors:

$$ \hat{r}_{u,i} = \vec{u}_u \times \tilde{\Sigma} \times \vec{v}_i^t $$

### CBOW model: track2vec

In the last iteration, we trained [word2vec model](https://arxiv.org/abs/1301.3781) to contextualize tracks in different playlists as they were words in sentences. The result is a contextualized embedding per track with a lower dimension. Then, we use the item-based rating estimation of the Neighborhood-based model to compute the rating using cosine similarity.



## Data preparation

You can download the complete dataset by registering a profile in the [official website](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge). Place the `spotify_million_playlist_dataset/` in the parent folder of the cloned repository and run the script [split-test.py](split-test.py) to extract 10k playlists as the evaluation set. 

```shell
python3 split-test.py
```

The result should be:
```
recsys-spmd/
    spotify_million_playlist_dataset/
        data/
        train/
        gold.json
        input.json
        ...
    ...
```

The subfolder `train/`contains the original JSON slices with some playlists removed and stored in  `input.json` and `gold.json` files. `input.json` contains some tracks of each test playlist and `gold.json` contains the expected tracks to be recommended of each test playlist.

## User manual

In order to make recommendations and evaluate each model, you can directly run the `main.py` script in terminal with the following arguments:

- `model`: Name of the model to run. Choices are `base`, `neighbor`, `puresvd`, `track2vec` y `eval`. In case of `eval`, the [Evaluator](utils/evaluation.py) class will be called and it is mandatory to specify a `submit_path` to compute evaluation metrics.
- `-eval`: Will only have effect if a model is specified with `model`. When this flag is added, the submission generated by the called model will be evaluated. 

Optionally, these parameters can be added to all models:

- `submit_path`: Path where recommendations will be saved in the 
[challenge submission format](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/). 
- `num_threads`: Number of processes used to parallelize all running phases (prediction and recommendation). 
- `time`: *Flag* to indicate if execution time is traced in terminal. 
- `verbose`: *Flag* to indicated if execution process is _traced_ in terminal.

In the following table we provide a summary of the parameters used and default values per model. Note that flags are preceded by a hyphen (-) and a blank cell means that parameter is not used by the model.

| Parameter       | `base`                    | `neighbor`                 | `puresvd`                               | `track2vec`                    |
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


## Execution examples

In order to test all implemented models, we suggest to not change default parameters (except number of processes and batch size). Here we provide execution examples in terminal to test models:



[`BaselineModel`](models/baseline.py):


```shell
python3 main.py base -t -v && python3 main.py eval -t -v --submit_path=submissions/base.csv.gz
```
[`NeighbourModel`](models/neighbour.py):

1. Generate sparse matrices:

```shell
python3 main.py neighbor user --action sparsify -t -v
```

2. Test user-based and item-based variantes:

```shell
python3 main.py neighbor user -eval --action recommend -t -v --k=100
```

```shell
python3 main.py neighbor item -eval --action recommend -t -v --k=20 --batch_size=15000
```


[`PureSVDModel`](models/puresvd.py):

1. Using only the train matrix to factorize with 10 latent factors:

```shell
python3 main.py puresvd -eval --action recommend -t -v --h=50 --batch_size=30 --num_threads=10
```

2. Using the train and test matrix to factorize with 10 latent factors:

```shell
python3 main.py puresvd -ftest -eval --action recommend -t -v --h=50 --batch_size=50 --num_threads=10
```


[`Track2Vec`](models/track2vec.py):

1. First we need to train [word2vec](https://arxiv.org/abs/1301.3781) model with [gensim library](https://radimrehurek.com/gensim/models/word2vec.htm) with an embedding dimension of $d=100$ during 100 epochs:

```shell
python3 main.py track2vec train -v -t --embed_dim=100 --num_epochs=100
```
2. To make recommendations using the kNN approximation of the [gensim.AnnoyIndexer](https://radimrehurek.com/gensim/auto_examples/tutorials/run_annoy.html) with $k=20$:

```shell
python3 main.py track2vec recommend -eval -v -t --num_threads 1 10 --k=20 --annoy --num_trees=100
```

3. To make recommendations using exact cosine similarity (this is computationally expensive):

```shell
python3 main.py track2vec recommend -eval -v -t --num_threads 5 10 --k=20
```


## Results

| Model                  | R-Precision | nDCG | clicks |
|-------------------------|-------------|------|--------|
| popularity              | 0.02        | 0.08 | 20.8   |
| user (k=100)            | 0.14        | 0.27 | 5.08   |
| item (k=20)             | 0.10        | 0.19 | 7.44   |
| puresvd (h=50)          | 0.09        | 0.22 | 9.2    |
| puresvd' (h=50)         | 0.09        | 0.22 | 9.14   |
| track2vec (d=100, k=20) | 0.02        | 0.07 | 14.36  |

**Note**: In `puresvd` the ' means that test matrix was used for factorization.


## Team builders :construction_worker:

- Ana Ezquerro ([ana.ezquerro@udc.es](mailto:ana.ezquerro@udc.es), [GitHub](https://github.com/anaezquerro)).
- Pedro Souza ([pedro.souza@udc.es](mailto:pedro.souza@udc.es), [GitHub](https://github.com/pedrosouzaa1)).

