from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from typing import Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
MAX_THREADS = os.cpu_count()


class NeighbourModel:
    def __init__(
            self,
            k: int,
            batch_size: int = 100,
            num_threads: int = MAX_THREADS,
            train_path: str = 'R_train.npz'
    ):
        """
        Initialization of the neighbour user-based recommender system.
        :param k: Size of the neighbourhood.
        :param batch_size: Number of rows of the test matrix that will be used
        to compute similarity at each submit of the pool.
        :param num_threads: Number of threads of the pool.
        :param train_path: Path where train sparse matrix has been previously
        stored.
        """

        # n: número de playlists, m: número de tracks
        self.Rtrain = sparse.load_npz(train_path)   # ~ (n, m)
        self.Ntrain = norm(self.Rtrain, axis=1)
        self.k = k
        self.n, self.m = self.Rtrain.shape

        self.batch_size = batch_size
        self.num_threads = num_threads

    def predict(self, test_path: str):
        self.Rtest = sparse.load_npz(test_path)
        # S = self.similarity()
        S = list()
        for start in range(0, 200, self.batch_size):
            S_partial = self._similarity(start)
            S.append(S_partial)

        S = sparse.hstack(S)
        Rest = S.dot(self.Rtrain)

        return Rest


    def similarity(self):
        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()
            for start in range(0, self.n, self.batch_size):
                futures.append(
                    pool.submit(self._similarity, start)
                )

            S = sparse.hstack([f.result() for f in futures])
        return S


    def _similarity(self, start: int):

        # extract v (a slice of R matrix)
        v = self.Rtest[start:(start+self.batch_size), :]       # ~ (b=end-start, m)
        b = v.shape[0]

        # get dot product between v and R
        sim = csr_matrix.transpose(self.Rtrain.dot(csr_matrix.transpose(v)))   # ~ (b, n)

        Ntest = norm(v, axis=1)

        # normalize
        sim = sim.multiply(1/self.Ntrain)
        sim = sim.multiply(Ntest.reshape(b, 1)).toarray()
        sim_k = (-sim.toarray()).argsort(axis=1)[:, -self.k:]

        rows = sim_k.flatten()
        cols = np.repeat(np.arange(b), self.k)
        data = sim.toarray()[:, sim_k].flatten()

        sim_sparse = csr_matrix((data, (rows, cols)), shape=(b, self.n))

        return sim_sparse




if __name__ == '__main__':
    model = NeighbourModel(100)
    model.predict('R_test.npz')







        









