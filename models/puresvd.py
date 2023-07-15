from sparsesvd import sparsesvd
from scipy.sparse import load_npz, vstack, csr_matrix
import numpy as np
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor
from utils.constants import N_RECS, INPUT_FILE, INFO_ROW
from utils.tools import coalesce, read_json, load_pickle, pop_empty
import time

class PureSVDModel:

    NAME = 'PureSVD'

    def __init__(
            self,
            h: int,
            use_test: bool,
            train_path: str,
            test_path: str,
            trackmap_path: str
    ):
        """
        Initialization of the PureSVD model.
        :param h: Dimension of the latent space.
        :param use_test: Boolean parameter to use the test matrix or not.
        :param train_path: Path where sparse train matrix is stored.
        :param test_path: Path where sparse test matrix is stored.
        :param trackmap_path: Path where the map from track URIs to columns in the matrix is stored in.
        """
        self.h = h
        self.use_test = use_test
        self.train_path = train_path
        self.test_path = test_path
        self.trackmap_path = trackmap_path

        self.Utest, self.S, self.V = None, None, None

    def factorize(self, U_path: str, S_path: str, V_path: str, verbose: bool):
        """
        Applies matrix factorization to obtain U, S and V matrices.
        :param U_path: Path to store U matrix (for test playlists).
        :param S_path: Path to store S matrix.
        :param V_path: Path to store V matrix.
        """
        if verbose:
            print('Factorizing sparse matrices...')
        # load Rtrain and Rtest
        Rtrain = load_npz(self.train_path)
        Rtest = load_npz(self.test_path)

        # create R matrix using Rtest or not
        m_test = Rtest.shape[0]
        R = vstack([Rtrain, Rtest]).tocsc() if self.use_test else Rtrain.tocsc()
        del Rtrain, Rtest

        # R ~ [m, n]
        # U ~ [m, h]
        # S ~ [h, h]
        # V ~ [n, h]
        Ut, S, Vt = sparsesvd(R, self.h)

        # now obtain Utest ~ [m_test, h]
        if self.use_test:
            Utest = Ut.T[-m_test:]
            del Ut
        else:
            # project vectors
            Rtest = load_npz(self.test_path)
            Utest = Rtest @ Vt.T
            del Rtest

        self.Utest = Utest
        self.S = S
        self.V = Vt.T

        if U_path:
            np.save(U_path, Utest)
        if S_path:
            np.save(S_path, S)
        if V_path:
            np.save(V_path, self.V)


    def recommend(self, submit_path: str, batch_size: int, num_threads: int, verbose: bool):
        """
        Compute recommendations.
        :param submit_path: Path to store the submissions.
        :param batch_size: Batch size to distribute test matrix rows.
        :param num_threads: Number of threads to parallelize the dot product.
        :param verbose: Boolean parameter to display the trace.
        """
        # compute most popular tracks and remove Rtrain and Rtest matrices
        Rtrain = load_npz(self.train_path)
        popular = np.copy(np.asarray(-(Rtrain.sum(axis=0))).argsort().ravel()).tolist()[:N_RECS]
        del Rtrain

        # read test file, trackmap (track_uri -> col) and pidmap (pid -> row)
        test = read_json(INPUT_FILE)
        trackmap = load_pickle(self.trackmap_path)
        pidmap = load_pickle(self.test_path.replace('.npz', '.pickle'))
        pidmap = {row: pid for pid, row in pidmap.items()}      # invert pidmap

        # remove empty playlists from the test set
        test_empty = pop_empty(test)
        test = {pid: list(map(trackmap.get, tracks)) for pid, tracks in test.items()}

        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            futures = list()

            for i in range(0, len(test), batch_size):
                futures.append(
                    pool.submit(recommend, i, batch_size, self.Utest, self.S, self.V, test, pidmap, popular, verbose)
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



def recommend(i: int, batch_size: int,
              Utest: np.ndarray, S: np.ndarray, V: np.ndarray, test: Dict[int, List[int]],
              pidmap: Dict[int, int], popular: np.ndarray, verbose: bool):
    playlists = dict()
    if verbose:
        print(f'Computing recommendation for playlist {i}/{Utest.shape[0]}')
    u = Utest[i:(i + batch_size)]
    slice = u @ (np.diag(S) @ V.T)
    slice[np.isclose(slice, 0, atol=1e-8)] = 0
    slice = csr_matrix(slice, dtype=np.float32)

    for j in range(i, i + u.shape[0]):
        pid = pidmap.pop(j)
        included = test.pop(pid)
        ratings = slice.getrow(j - i)
        ratings[:, included] = 0
        ratings.eliminate_zeros()
        cols, values = ratings.indices, ratings.data
        playlists[pid] = cols[(-values).argsort().tolist()[:N_RECS]].tolist()
        if len(playlists[pid]) < N_RECS:
            news = (np.setdiff1d(popular, np.array(playlists[pid]+included)))[:(N_RECS-len(playlists[pid]))]
            playlists[pid] += news.tolist()

    return playlists













