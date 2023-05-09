from gensim.models import Word2Vec
from scipy.sparse import csr_matrix, load_npz, save_npz, vstack
from utils.tools import tolist, load_pickle, read_json, pop_empty, coalesce
from typing import List, Tuple, Dict
from utils.constants import TEST_FILE, N_RECS, MAX_THREADS, INFO_ROW
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from gensim.similarities.annoy import AnnoyIndexer
from models.neighbour import recommend, csr_argsort
class Track2VecModel:
    def __init__(self, embed_dim: int, context_size: int, k: int, model_path: str, train_path: str, test_path: str,
                 trackmap_path: str, S_path: str):
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.model_path = model_path
        self.train_path = train_path
        self.test_path = test_path
        self.S_path = S_path
        self.trackmap_path = trackmap_path
        self.k = k


    def load(self):
        self.model = Word2Vec.load(self.model_path)

    def train(self, num_epochs: int, num_threads: int, verbose: bool):
        if verbose:
            print('Training word2vec model with train playlists')
        Rtrain = load_npz(self.train_path)
        n_tracks = Rtrain.shape[1]
        train_playlists = tolist(Rtrain)
        del Rtrain

        self.model = Word2Vec(sentences=[range(n_tracks)], vector_size=self.embed_dim, window=self.context_size, min_count=1, workers=num_threads)
        self.model.train(train_playlists, total_examples=len(train_playlists), epochs=num_epochs)
        self.model.save(self.model_path)


    def pure_similarity(self, batch_size: int, num_threads: int, verbose: int) -> csr_matrix:

        if verbose:
            print('Computing pure item similarity for all tracks')

        embeds = np.array([self.model.wv[track] for track in range(len(self.model.wv))])
        self.model = None
        track_norm = np.array([np.sqrt(np.sum(embeds[track]**2)) for track in range(embeds.shape[0])])

        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            futures = list()

            for i in range(0, embeds.shape[0], batch_size):
                futures.append(
                    pool.submit(pure_similarity, i, batch_size, embeds, self.k, track_norm, verbose)
                )

            S = csr_matrix((embeds.shape[0], embeds.shape[0]))
            for _ in range(len(futures)):
                S_partial = futures.pop(0)
                S[i:(i + S_partial.shape[0]), :] = S_partial
                S[:, i:(i + S_partial.shape[0])] = S_partial.transpose()

                # now get the top K similarities
                cols, values = csr_argsort(S, topK=self.k)
                cols, values = map(lambda x: x.flatten().tolist(), (cols, values))
                rows = np.arange(S_partial.shape[0])
                S = csr_matrix((values, (rows, cols)), shape=(embeds.shape[0], embeds.shape[0]))

        save_npz(file=self.S_path, matrix=S)
        return S

    def annoy_similarity(self, num_threads, num_trees: int, verbose: int) -> csr_matrix:
        if verbose:
            print('Computing item similarity for all tracks with Annoy Indexer')

        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            futures = list()
            indexes = coalesce(len(self.model.wv), num_threads)
            for i in range(num_threads):
                start, end = indexes[i], indexes[i+1]
                futures.append(
                    pool.submit(annoy_similarity, start, end, self.model, self.k, num_trees, verbose)
                )

            S = futures.pop(0).result()
            for _ in range(len(futures)):
                S = vstack([S, futures.pop(0).result()])

        save_npz(file=self.S_path, matrix=S)
        return S


    def recommend(self, submit_path: str, num_threads: int, batch_size: int, annoy: bool, num_trees: int, verbose: bool):
        # first we must compute the similarity matrix
        if annoy:
            S = self.annoy_similarity(num_threads, num_trees, verbose)
        else:
            S = self.pure_similarity(batch_size, num_threads, verbose)

        # now obtain the estimated ratings
        Rtest = load_npz(self.test_path)
        Rest = Rtest @ S

        # create popularity vector
        Rtrain = load_npz(self.train_path)
        popular = np.copy(np.asarray(-(Rtrain.sum(axis=0))).argsort().ravel()).tolist()[:N_RECS]
        del Rtest, S, Rtrain

        # load additional maps
        test = read_json(TEST_FILE)
        trackmap = load_pickle(self.trackmap_path)
        pidmap = load_pickle(self.test_path.replace('.npz', '.pickle'))
        test = {pid: list(map(trackmap.get, tracks)) for pid, tracks in test.items()}
        test_empty = pop_empty(test)


        # compute parallel track recommendation
        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            pids, tracks = zip(*test.items())
            futures = list()
            indexes = coalesce(len(pids), num_threads)

            for i in range(num_threads):
                start, end = indexes[i], min(indexes[i + 1], len(self.model.wv))
                futures.append(
                    pool.submit(
                        recommend,
                        Rest, dict(zip(pids[start:end], tracks[start:end])), pidmap, popular, verbose)
                )

            playlists = futures.pop(0).result()
            for _ in range(len(futures)):
                playlists |= futures.pop(0).result()

        # convert trackmap from id -> track_uri
        trackmap = {value: key for key, value in trackmap.items()}

        # write results in the submission
        with open(submit_path, 'w', encoding='utf8') as file:
            file.write(INFO_ROW + '\n')
            for pid, tracks in playlists.items():
                file.write(f'{pid},' + ','.join(list(map(trackmap.get, tracks))) + '\n')
            for pid in test_empty:
                file.write(f'{pid},' + ','.join(list(map(trackmap.get, popular))) + '\n')


def pure_similarity(i: int, batch_size: int, embeds: np.ndarray, k: int, track_norm: np.ndarray, verbose: bool):
    if verbose:
        print(f'Computing similarity for track {i}/{embeds.shape[0]}')
    v = embeds[i:(i+batch_size)]
    b = len(v)
    S = (v @ embeds[i:].T)
    S /= track_norm[i:]
    S /= track_norm[i:(i+b)].reshape(b, 1)
    S[range(b), range(b)] = 0
    cols = np.argsort(-S, axis=1)[:, :k]
    values = S[[[i] for i in range(len(v))], cols.tolist()]
    rows = np.repeat(np.arange(b), k).flatten().tolist()
    S = csr_matrix((values.flatten().tolist(), (rows, (cols.flatten() + i).tolist())), shape=(b, embeds.shape[0]), dtype=np.float32)
    return S


def annoy_similarity(start: int, end: int, model: Word2Vec, k: int, num_trees: int, verbose: bool):
    info = lambda i: print(f'Computing similarity for track {i}/{len(model.wv)}') if verbose and (i%1000 == 0) else None
    indexer = AnnoyIndexer(model, num_trees=num_trees)
    rows, cols, values = list(), list(), list()
    tracks = list(range(start, end))
    for i, track in enumerate(tracks):
        info(track)
        neighbours, sims = map(list, zip(*model.wv.most_similar(model.wv[i], topn=k+1, indexer=indexer)))
        if track in neighbours:
            index = neighbours.index(track)
            neighbours.pop(index)
            sims.pop(index)
        else:
            neighbours.pop(-1)
            sims.pop(-1)
        rows += ([i]*k)
        cols += neighbours
        values += sims
    S = csr_matrix((values, (rows, cols)), shape=(len(tracks), len(model.wv)))
    return S






