from typing import List, Dict, Tuple
from typing import Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os, pickle, time
from tools import coalesce, read_json, flatten
from scipy.sparse import load_npz, save_npz, csr_matrix, vstack
from scipy.sparse.linalg import norm


TRAIN_FOLDER = 'spotify_train_dataset/data/'
MAX_THREADS = os.cpu_count()
TEST_FILE = 'spotify_test_playlists/test_input_playlists.json'


def csr_argsort(X: csr_matrix, topK: int, remov_diag: int = None):
    indptr = X.indptr
    cols = X.indices
    data = X.data
    sorted_indices = np.zeros((X.shape[0], topK), dtype=int)
    sorted_values = np.zeros((X.shape[0], topK), dtype=float)

    row, i = 0, indptr[0]
    for j in indptr[1:]:
        relcols = cols[i:j]
        values = data[i:j]
        if isinstance(remov_diag, int):
            values = values[relcols != (row+remov_diag)]
            relcols = relcols[relcols != (row+remov_diag)]
        sorted_row = relcols[(-values).argsort().tolist()[:topK]]
        sorted_indices[row, :len(sorted_row)] = sorted_row
        sorted_values[row, :len(sorted_row)] = np.sort(values)[::-1].tolist()[:topK]
        i = j
        row +=1
    return sorted_indices, sorted_values




def user_similarity(i: int, batch_size: int, Rtrain: csr_matrix, Rtest: csr_matrix,
               Ntrain: np.ndarray, Ntest: np.ndarray, k: int):
    # print(f'User similarity of user {i}/{Rtest.shape[0]}')

    # v ~ [batch_size, n_tracks]
    v = Rtest[i:(i + batch_size), :]
    b = v.shape[0]

    # compute v * t(Rtrain) ~ [batch_size, m_train]
    sim = v.dot(Rtrain.transpose())

    # normalize with the norm2
    sim = sim.multiply(1 / Ntrain)
    sim = sim.multiply(1 / Ntest[i:(i + batch_size)].reshape(b, 1))

    # store indices of the K similarities
    top_k, data = csr_argsort(csr_matrix(sim), k)

    rows = np.repeat(np.arange(b), k).tolist()
    cols = top_k.flatten().tolist()
    data = data.flatten().tolist()
    return csr_matrix((data, (rows, cols)), shape=(b, Rtrain.shape[0]))

def item_similarity(i: int, batch_size: int, Rtrain: csr_matrix, Ntracks: np.ndarray, k: int):

    # v ~ [batch_size, n_tracks]
    v = Rtrain.transpose()[i:(i + batch_size), :]
    b = v.shape[0]


    # compute v * Rtrain ~ [batch_size, n_tracks]
    sim = v.dot(Rtrain)

    # normalize with the norm2
    sim = sim.multiply(1 / Ntracks)
    sim = sim.multiply(1 / Ntracks[i:(i + batch_size)].reshape(b, 1))

    # store indices of the K similarities
    sim = csr_matrix(sim)
    top_k, data = csr_argsort(sim, k, remov_diag=i)

    rows = np.repeat(np.arange(b), k).tolist()
    cols = top_k.flatten().tolist()
    data = data.flatten().tolist()
    return csr_matrix((data, (rows, cols)), shape=(b, Rtrain.shape[1]))



class NeighbourModel:
    def __init__(
            self,
            k: int,
            top: int = 20,
            matrix_path: str = 'data/R.npz',
            train_path: str = 'data/Rtrain.npz',
            test_path: str = 'data/Rtest.npz',
            trackmap_path: str = 'data/track-map.pickle',
            batch_size: int = 200,
            num_threads: int = MAX_THREADS,
    ):

        # n: número de playlists, m: número de tracks
        self.n_tracks = None
        test = read_json(TEST_FILE)
        self.test_pids = test.keys()
        self.empty = list(filter(lambda pid: len(test[pid]) == 0, test.keys()))
        self.Rtrain, self.Rtest, self.popular = None, None, None
        self.Ntrain = None

        self.k = k
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.matrix_path = matrix_path
        self.train_path = train_path
        self.test_path = test_path
        self.trackmap_path = trackmap_path



    def user_based(self):
        self.Rtest = load_npz(self.test_path)
        Ntrain = norm(self.Rtrain, axis=1)
        Ntest = norm(self.Rtest, axis=1)

        start = time.time()

        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()

            for i in range(0, self.Rtest.shape[0], self.batch_size):
                futures.append(
                    pool.submit(
                        user_similarity,
                        i, self.batch_size, self.Rtrain, self.Rtest, Ntrain, Ntest, self.k)
                )

            S = futures.pop(0).result()
            for _ in range(len(futures)):
                S = vstack([S, futures.pop(0).result()])

        #
        # S = None
        # for i in range(0, 500, self.batch_size):
        #     if S is None:
        #         S = user_similarity(i, self.batch_size, self.Rtrain, self.Rtest, Ntrain, Ntest, self.k)
        #     else:
        #         S = vstack([S,
        #                     user_similarity(i, self.batch_size, self.Rtrain, self.Rtest, Ntrain, Ntest, self.k)
        #                     ])


        end = time.time()
        print(f'Predicting time: {end-start}')

        Rest = S.dot(self.Rtrain)
        return Rest


    def item_based(self):
        Ntracks = norm(self.Rtrain, axis=0)

        start = time.time()

        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()

            for i in range(0, self.Rtrain.shape[1], self.batch_size):
                futures.append(
                    pool.submit(
                        item_similarity,
                        i, self.batch_size, self.Rtrain, Ntracks, self.k)
                )

            S = futures.pop(0).result()
            for _ in range(len(futures)):
                S = vstack([S, futures.pop(0).result()])

        # S = None
        # for i in range(0, 500, self.batch_size):
        #     if S is None:
        #         S = item_similarity(i, self.batch_size, self.Rtrain, Ntracks, self.k)
        #     else:
        #         S = vstack([S,
        #                     item_similarity(i, self.batch_size, self.Rtrain, Ntracks, self.k)
        #                     ])

        end = time.time()
        print(f'Predicting time: {end - start}')

        Rest = S.dot(self.Rtrain.transpose())
        return Rest


    def predict(self, mode='user', submission_path: str = 'submissions/p1.csv.gz'):
        self.Rtrain = load_npz(self.train_path)

        if mode == 'user':
            Rest = self.user_based()
        elif mode == 'item':
            Rest = self.item_based()
        else:
            raise NotImplementedError

        # # open track map
        # with open(self.trackmap_path, 'rb') as file:
        #     trackmap = pickle.load(file)
        #
        # # open pid map
        # with open(self.test_path.replace('.npz', '.pickle'), 'rb') as file:
        #     pidmap = pickle.load(file)
        #
        # # for pid, pidrow in pidmap.items():
        # #     track_ids =






if __name__ == '__main__':
    model = NeighbourModel(10, batch_size=1000, num_threads=8)
    model.predict('item')
    # row = np.array([0, 0, 1, 2, 2, 2])
    # col = np.array([0, 2, 3, 0, 1, 3])
    # data = np.array([3, 1, -4, 5, -3, 2])
    # m = csr_matrix((data, (row, col)), shape=(3, 4))
    # print(m.toarray())
    # print(csr_argsort(m, 3))


















