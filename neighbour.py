from typing import List, Dict, Tuple
from typing import Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os, pickle, time, sys
from tools import coalesce, read_json, flatten, submit
from scipy.sparse import load_npz, save_npz, csr_matrix, vstack
from scipy.sparse.linalg import norm
from tools import INFO_ROW
N_RECS = 500

TRAIN_FOLDER = 'spotify_train_dataset/data/'
MAX_THREADS = os.cpu_count()
TEST_FILE = 'spotify_test_playlists/test_input_playlists.json'
SUBMISSION_FOLDER = 'submissions/'

if not os.path.exists(SUBMISSION_FOLDER):
    os.makedirs(SUBMISSION_FOLDER)

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
    print(f'User similarity of user {i}/{Rtest.shape[0]}')

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
    print(f'Item similarity of item {i}/{Rtrain.shape[1]}')

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

def recommend(Rest: csr_matrix, test: Dict[int, List[int]], pidmap: Dict[int, int]):
    playlists = dict()
    for i, pid in enumerate(test.keys()):
        row = pidmap[pid]
        tracks = Rest.indices[Rest.indptr[row]: Rest.indptr[row+1]]
        similarities = Rest.data[Rest.indptr[row]: Rest.indptr[row+1]]
        tracks = tracks[(-similarities).argsort().tolist()]
        tracks = list(filter(lambda track: track not in test[pid], tracks))
        playlists[pid] = tracks[:N_RECS]
    return playlists



class NeighbourModel:
    def __init__(
            self,
            k: int,
            train_path: str = 'data/Rtrain.npz',
            test_path: str = 'data/Rtest.npz',
            trackmap_path: str = 'data/track-map.pickle',
            batch_size: int = 200,
            num_threads: int = MAX_THREADS,
    ):

        # n: número de playlists, m: número de tracks
        self.n_tracks = None
        self.Rtrain, self.Rtest = None, None
        self.Ntrain = None

        self.k = k
        self.batch_size = batch_size
        self.num_threads = num_threads
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
        # for i in range(0, 3000, self.batch_size):
        #     if S is None:
        #         S = item_similarity(i, self.batch_size, self.Rtrain, Ntracks, self.k)
        #     else:
        #         S = vstack([S,
        #                     item_similarity(i, self.batch_size, self.Rtrain, Ntracks, self.k)
        #                     ])

        end = time.time()
        print(f'Predicting time: {end - start}')

        self.Rtest = load_npz(self.test_path)
        Rest = self.Rtest.dot(S)
        return Rest

    def predict(self, mode='user', submit_path: str = None, save_matrix: str = None, load: bool = False):
        self.Rtrain = load_npz(self.train_path)

        if load:
            Rest = load_npz(save_matrix)
        else:
            if mode == 'user':
                Rest = self.user_based()
            elif mode == 'item':
                Rest = self.item_based()
            else:
                raise NotImplementedError
            if save_matrix:
                save_npz(save_matrix, Rest)

        # compute most popular tracks

        popular = np.copy(np.asarray(-(self.Rtrain.sum(axis=0))).argsort().ravel()).tolist()[:N_RECS]
        del self.Rtrain, self.Rtest

        # open pidmap, trackmap and test json
        with open(self.test_path.replace('.npz', '.pickle'), 'rb') as file:
            pidmap = pickle.load(file)
        with open(self.trackmap_path, 'rb') as file:
            trackmap = pickle.load(file)

        test = read_json(TEST_FILE)
        test = {pid: list(map(trackmap.get, tracks)) for pid, tracks in test.items()}

        # invert trackmap
        trackmap = {v: k for k, v in trackmap.items()}

        tstart = time.time()

        file = open(submit_path, 'w', encoding='utf8')
        file.write(INFO_ROW + '\n')

        pids = list(pidmap.keys())
        indexes = coalesce(len(pids), self.num_threads)
        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()

            for i in range(self.num_threads):
                start, end = indexes[i], indexes[i+1]
                futures.append(
                    pool.submit(recommend, Rest, {pid: test[pid] for pid in pids[start:end]}, pidmap)
                )

            for _ in range(len(futures)):
                playlists = futures.pop(0).result()
                for pid, tracks in playlists.items():
                    file.write(f'{pid},' + ','.join(list(map(trackmap.get, tracks))) + '\n')

        tend = time.time()
        print(f'Recommendation time: {tend-tstart}')

        # add pids of the empty set
        popular = list(map(trackmap.get, popular))
        for pid in (test.keys() - pidmap.keys()):
            file.write(f'{pid},' + ','.join(popular) + '\n')

        file.close()



if __name__ == '__main__':
    # model = NeighbourModel(10, batch_size=int(5e2), num_threads=8)
    # model.predict('user', submit_path=f'{SUBMISSION_FOLDER}/user-based.csv.gz', save_matrix='data/Rest-user.npz', load=True)

    model = NeighbourModel(10, batch_size=int(20e3), num_threads=8)
    model.predict('item', submit_path=f'{SUBMISSION_FOLDER}/item-based.csv.gz', save_matrix='data/Rest-item.npz')
















