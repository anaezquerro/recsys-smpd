from gensim.models import Word2Vec
from scipy.sparse import csr_matrix, load_npz, vstack
from utils.tools import tolist, load_pickle, read_json, pop_empty, coalesce
from typing import List, Tuple, Dict
from utils.constants import TEST_FILE, N_RECS, MAX_THREADS, INFO_ROW
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from gensim.similarities.annoy import AnnoyIndexer
from models.neighbour import recommend, csr_argsort
class Track2VecModel:
    def __init__(self, embed_dim: int, context_size: int, k: int, model_path: str, train_path: str, test_path: str, trackmap_path: str):
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.model_path = model_path
        self.train_path = train_path
        self.test_path = test_path
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


    def item_similarity(self, batch_size: int, num_threads: int, verbose: int) -> csr_matrix:

        if verbose:
            print('Computing item similarity for all tracks')

        embeds = np.array([self.model.wv[track] for track in range(len(self.model.wv))])
        self.model = None
        track_norm = np.array([np.sqrt(np.sum(embeds[track]**2)) for track in range(embeds.shape[0])])

        S = csr_matrix((embeds.shape[0], embeds.shape[0]))
        for i in range(0, batch_size*num_threads, batch_size):
            S_partial = similarity(i, batch_size, embeds, self.k, track_norm, verbose)
            S[i:(i+S_partial.shape[0]), :] = S_partial
            S[:, i:(i+S_partial.shape[0])] = S_partial.transpose()

        # with ProcessPoolExecutor(max_workers=num_threads) as pool:
        #     futures = list()
        #
        #     for i in range(0, batch_size*num_threads, batch_size):
        #         futures.append(
        #             pool.submit(similarity, i, batch_size, embeds, self.k, track_norm, verbose)
        #         )
        #
        #     S = csr_matrix((embeds.shape[0], embeds.shape[0]))
        #     for _ in range(len(futures)):
        #         S = vstack([S, futures.pop(0).result()])

        return S

    def recommend(self, submit_path: str, num_threads: int, batch_size: int, verbose: bool):
        # first we must compute the similarity matrix
        S = self.item_similarity(batch_size, num_threads, verbose)

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
                start, end = indexes[i], indexes[i + 1]
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

#
#
# def similarity(i: int, batch_size: int, embeds: np.ndarray, k: int, track_norm: np.ndarray, granularity: int, verbose: bool):
#     if verbose:
#         print(f'Computing similarity for track {i}/{embeds.shape[0]}')
#     v = embeds[i:(i+batch_size)]
#     b = v.shape[0]
#     S = csr_matrix((b, embeds.shape[0]))
#
#     min_values = np.repeat(-np.Inf, b).reshape(b, 1)
#     for j in range(0, embeds.shape[0], granularity):
#         if verbose:
#             print(f'Computing similarity between tracks [{i}, {i+b}] and tracks [{j}, {j+granularity}]')
#         start, end = j, min(j+granularity, embeds.shape[0])
#         sim = v @ embeds[start:end].T
#         sim /= track_norm[start:end]
#         sim /= track_norm[i:(i+b)].reshape(b, 1)
#         sim[sim < min_values] = 0
#         S[:, start:end] = sim
#         S.eliminate_zeros()
#
#         # construct again the sparse matrix
#         cols, values = csr_argsort(S, k, remov_diag=True)
#         rows = np.repeat(np.arange(b), k).tolist()
#         S = csr_matrix((values.flatten().tolist(), (rows, cols.flatten().tolist())), shape=(b, embeds.shape[0]))
#         min_values = S.min(axis=1).toarray()
#
#     return S


def similarity(i: int, batch_size: int, embeds: np.ndarray, k: int, track_norm: np.ndarray, verbose: bool):
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
    print(S.shape)
    return S






if __name__ == '__main__':
    model = Track2VecModel(embed_dim=100, context_size=20, model_path='data/track2vec',
                           train_path='data/Rtrain.npz', test_path='data/Rtest.npz', trackmap_path='data/trackmap.pickle',
                           k=10)
    model.load()
    start = time.time()
    model.recommend('submissions/embed.csv.gz', batch_size=500, num_threads=10, verbose=True)
    end = time.time()


