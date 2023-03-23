from typing import List, Dict, Tuple
from typing import Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os, pickle, time
from tools import coalesce, read_json, flatten
from scipy.sparse import load_npz, save_npz, csr_matrix, hstack
from scipy.sparse.linalg import norm

TRAIN_FOLDER = 'spotify_train_dataset/data/'
MAX_THREADS = os.cpu_count()
TEST_FILE = 'spotify_test_playlists/test_input_playlists.json'


class NeighbourModel:
    def __init__(
            self,
            k: int,
            top: int = 20,
            matrix_path: str = 'data/R.npz',
            train_path: str = 'data/Rtrain.npz',
            test_path: str = 'data/Rtest.npz',
            trackmap_path: str = 'data/track-map.pickle',
            popular_path: str = 'data/most-popular.npz',
            batch_size: int = 100,
            num_threads: int = MAX_THREADS,
    ):

        # n: número de playlists, m: número de tracks
        self.n_playlists = None
        self.n_tracks = None
        test = read_json(TEST_FILE)
        self.test_pids = test.keys()
        self.empty = list(filter(lambda pid: len(test[pid]) == 0 , test.keys()))
        self.Rtrain, self.Rtest, self.popular = None, None, None
        self.Ntrain = None

        self.k = k
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.matrix_path = matrix_path
        self.train_path = train_path
        self.test_path = test_path
        self.popular_path = popular_path
        self.trackmap_path = trackmap_path

    def predict(self):
        self.Rtrain = load_npz(self.train_path)
        self.Rtest = load_npz(self.test_path)
        self.popular = load_npz(self.popular_path)

        self.Ntrain = norm(self.Rtrain, axis=0)

        # S = self.similarity()
        S = list()
        for start in range(0, 200, self.batch_size):
            S_partial = self._similarity(start)
            S.append(S_partial)

        S = sparse.hstack(S)
        Rest = S.dot(self.Rtrain)

        return Rest

    def user_similarity(self):
        self.Rtrain = load_npz(self.train_path)
        self.Rtest = load_npz(self.test_path)
        self.popular = load_npz(self.popular_path)

        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()
            for start in range(0, self.Rtest.shape[0], self.batch_size):
                futures.append(
                    pool.submit(self._user_similarity, start)
                )

            S = hstack([f.result() for f in futures])
        return S

    def item_similarity(self):
        self.Rtrain = load_npz(self.train_path).transpose()



    def _user_similarity(self, start: int):
        # v ~ [batch_size, n]
        v = self.Rtest[start:(start+self.batch_size), :]
        b = v.shape[0]

        # get v*t(Rtrain)
        sim = v.dot(self.Rtrain.transpose())

        # get matrix of Norms
        Ntest = norm(v, axis=1)

        # normalize
        sim = sim.multiply(1/self.Ntrain)
        sim = sim.multiply(1/Ntest.reshape(b, 1)).toarray()
        sim_k = (-sim.toarray()).argsort(axis=1)[:, -self.k:]

        rows = sim_k.flatten()
        cols = np.repeat(np.arange(b), self.k)
        data = sim.toarray()[:, sim_k].flatten()

        sim_sparse = csr_matrix((data, (rows, cols)), shape=(b, self.n_tracks))

        return sim_sparse

    def _item_similarity(self, start: int):
        # v ~ [batch_size, n]
        v = self.Rtrain[start:(start+self.batch_size),:]
        b = v.shape[0]

        # get v*t(Rtrain)
        sim = v.dot(self.Rtrain.transpose())

        # get matrix of Norms
        Ntest = norm(v, axis=1)








