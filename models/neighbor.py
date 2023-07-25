from typing import List, Dict, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import sys, time
from utils.tools import coalesce, read_json, load_pickle
from scipy.sparse import load_npz, save_npz, csr_matrix, vstack
from scipy.sparse.linalg import norm
from utils.constants import N_RECS, INFO_ROW, INPUT_FILE

class NeighborModel:

    NAME = 'NeighborModel'

    def __init__(self, neighbor: str, k: int, train_path: str, test_path: str, trackmap_path):
        super().__init__()
        self.n_tracks = None
        self.Rtrain, self.Rtest = None, None
        self.Ntrain = None
        self.verbose, self.num_threads, self.batch_size = None, None, None

        self.neighbor = neighbor
        self.k = k
        self.train_path = train_path
        self.test_path = test_path
        self.trackmap_path = trackmap_path


    def recommend(self, submit_path: str, batch_size: int, num_threads: Tuple[int], matrix_path: str, load: bool, verbose: bool):
        """
        Makes recommendation based on the neighborhood selected (user or item).
        :param batch_size: Number of rows that will be assumed by each thread to compute cosine similarity.
        :param num_threads: Tuple of integers. First integer is the number of threads used for matrix computation and
        second integer is the number of threads used for recommendation.
        :param submit_path: Path where submission will be stored.
        :param matrix_path: Path where estimated ratings matrix will be stored.
        :param load: Boolean value to load the estimated ratings matrix in matrix_path.
        """
        self.verbose = verbose
        self.num_threads = num_threads[0]
        self.batch_size = batch_size

        # compute the matrix of estimated ratings
        self.Rtrain = load_npz(self.train_path)
        if load:
            Rest = load_npz(matrix_path)
        else:
            Rest = self.compute_rating(matrix_path)

        # compute most popular tracks and remove Rtrain and Rtest matrices
        popular = np.copy(np.asarray(-(self.Rtrain.sum(axis=0))).argsort().ravel()).tolist()[:N_RECS]
        del self.Rtrain, self.Rtest

        # read test file, trackmap (track_uri -> col) and pidmap (pid -> row)
        test = read_json(INPUT_FILE)
        trackmap = load_pickle(self.trackmap_path)
        pidmap = load_pickle(self.test_path.replace('.npz', '.pickle'))

        # remove empty playlists from the test set
        test = {pid: list(map(trackmap.get, tracks)) for pid, tracks in test.items()}
        test_empty = list()
        for pid in set(test.keys()):
            if len(test[pid]) == 0:
                test_empty.append(pid)
                test.pop(pid)

        # compute parallel track recommendation
        self.num_threads = num_threads[1]
        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            pids, tracks = zip(*test.items())
            futures = list()
            indexes = coalesce(len(pids), self.num_threads)

            for i in range(self.num_threads):
                start, end = indexes[i], indexes[i+1]
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

    def compute_rating(self, save_matrix: str) -> csr_matrix:
        if self.neighbor == 'user':
            Rest = self.user_based()
        elif self.neighbor == 'item':
            Rest = self.item_based()
        else:
            raise NotImplementedError

        if save_matrix:
            save_npz(save_matrix, Rest)

        return Rest
    def user_based(self) -> csr_matrix:
        if self.verbose:
            print(f'Computing user-based similarity')

        self.Rtest = load_npz(self.test_path)
        Ntrain = norm(self.Rtrain, axis=1)
        Ntest = norm(self.Rtest, axis=1)

        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()
            for i in range(0, self.Rtest.shape[0], self.batch_size):
                futures.append(
                    pool.submit(
                        user_similarity,
                        i, self.batch_size, self.Rtrain, self.Rtest, Ntrain, Ntest, self.k, self.verbose)
                )

            S = futures.pop(0).result()
            for _ in range(len(futures)):
                S = vstack([S, futures.pop(0).result()])

        Rest = S.dot(self.Rtrain)
        return Rest

    def item_based(self) -> csr_matrix:
        Ntracks = norm(self.Rtrain, axis=0)

        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()

            for i in range(0, self.Rtrain.shape[1], self.batch_size):
                futures.append(
                    pool.submit(
                        item_similarity,
                        i, self.batch_size, self.Rtrain, Ntracks, self.k, self.verbose)
                )

            S = futures.pop(0).result()
            for _ in range(len(futures)):
                S = vstack([S, futures.pop(0).result()])

        self.Rtest = load_npz(self.test_path)
        Rest = self.Rtest.dot(S)
        return Rest


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
               Ntrain: np.ndarray, Ntest: np.ndarray, k: int, verbose: bool):
    if verbose:
        print(f'User similarity of user {i}/{Rtest.shape[0]}')

    # v ~ [batch_size, n_tracks]
    v = Rtest[i:(i + batch_size), :]
    b = v.shape[0]

    # compute v * t(Rtrain) ~ [batch_size, m_train]
    sim = v.dot(Rtrain.T.tocsr())

    # normalize with the norm2
    # sim = sim.multiply(1 / Ntrain)
    sim = sim.multiply(1/Ntrain)
    sim = sim.multiply(1 / Ntest[i:(i + batch_size)].reshape(b, 1))


    # store indices of the K similarities
    top_k, data = csr_argsort(csr_matrix(sim), k)

    rows = np.repeat(np.arange(b), k).tolist()
    cols = top_k.flatten().tolist()
    data = data.flatten().tolist()
    return csr_matrix((data, (rows, cols)), shape=(b, Rtrain.shape[0]), dtype=np.float32)

def item_similarity(i: int, batch_size: int, Rtrain: csr_matrix, Ntracks: np.ndarray, k: int, verbose: bool):
    if verbose:
        print(f'Item similarity of item {i}/{Rtrain.shape[1]}')

    # v ~ [batch_size, n_tracks]
    v = Rtrain.T.tocsr()[i:(i + batch_size), :]
    b = v.shape[0]

    # compute v * Rtrain ~ [batch_size, n_tracks]
    sim = v.dot(Rtrain)

    # normalize with the norm2
    sim = sim.multiply(1 / Ntracks)
    sim = sim.multiply(1 / Ntracks[i:(i + batch_size)].reshape(b, 1))

    # store indices of the K similarities
    sim = csr_matrix(sim, dtype=np.float32)
    top_k, data = csr_argsort(sim, k, remov_diag=i)

    rows = np.repeat(np.arange(b), k).tolist()
    cols = top_k.flatten().tolist()
    data = data.flatten().tolist()
    return csr_matrix((data, (rows, cols)), shape=(b, Rtrain.shape[1]), dtype=np.float32)

def recommend(Rest: csr_matrix, test: Dict[int, List[int]], pidmap: Dict[int, int], popular: np.array, verbose=True) -> Dict[int, List[int]]:
    playlists = dict()
    info = lambda i: print(f'Computing recommendation for playlist {i}/{len(test)}') if (i%100==0) and verbose else None
    for i, pid in enumerate(test.keys()):
        info(i)
        ratings = Rest.getrow(pidmap[pid])
        ratings[:, test[pid]] = 0
        ratings.eliminate_zeros()
        ratings, cols = ratings.data, ratings.indices
        playlists[pid] = cols[(-ratings).argsort().tolist()[:N_RECS]].tolist()
        if len(playlists[pid]) < N_RECS:
            news = np.setdiff1d(popular, np.array(playlists[pid] + test[pid]))[:(N_RECS-len(playlists[pid]))]
            playlists[pid] += news.tolist()
    return playlists


if __name__ == '__main__':
    train_path, test_path, trackmap_path = 'data/Rtrain.npz', 'data/Rtest.npz', 'data/trackmap.pickle'

    if sys.argv[1] == 'user':
        model = NeighborModel('user', 100, train_path, test_path, trackmap_path)
        batch_size = int(5e2)
    elif sys.argv[1] == 'item':
        model = NeighborModel('item', 50, train_path, test_path, trackmap_path)
        batch_size = int(10e3)
    else:
        raise NotImplementedError


    start = time.time()
    model.recommend(f'submissions/{sys.argv[1]}-based.csv.gz', batch_size=batch_size, num_threads=(8, 12),
                    matrix_path=f'data/Rest.npz', load=True, verbose=True)
    end = time.time()
    print(f'Recommendation time: {end-start}')
