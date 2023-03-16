from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from typing import Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor



class NeighbourModel:
    def __init__(self, matrix_path: str, k: int, batch_size: int, num_threads: int):
        # n: número de playlists, m: número de tracks
        self.Rp = sparse.load_npz(matrix_path)   # ~ (n, m)
        self.k = k
        self.n, self.m = self.Rp.shape

        self.batch_size = batch_size
        self.num_threads = num_threads


    def similarity(self):
        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()
            for start in range(0, self.n, self.batch_size):
                futures.append(
                    pool.submit(self._similarity, start, start+self.batch_size)
                )

            S = sparse.hstack([f.result() for f in futures])


    def _similarity(self, indexes: Tuple[int]):
        start, end = indexes

        # extract v (a slice of R matrix)
        v = csr_matrix.transpose(self.Rp[start:end, :])         # ~ (b=end-start, m)

        # get dot product between v and R
        sim = csr_matrix.transpose(self.Rp.dot(csr_matrix.transpose(v)))   # ~ (b, n)

        # get norm of v
        norms = norm(v, axis=0) # ~ b

        # normalize
        sim = sim/norms

        # get indices of the top k similarities
        sim_k = sim.argsort(axis=0)[:, -self.k:]      # ~ [b, k]


        sim = self.Rp.dot(v)  # ~ (n, end-start)
        norms = norm(v, axis=0)     # ~ (end-start)
        sim = sim/norms
        sim_k = sim.argsort(axis=0)[-self.k:]

        rows = sim_k.flatten()
        cols = np.repeat(np.arange(end-start), self.k)
        data = sim[:, sim_k].flatten()

        sim_sparse = csr_matrix((data, (rows, cols)), shape=(end-start, self.m))

        return sim_sparse













        









