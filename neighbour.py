from typing import List, Dict, Tuple
from typing import Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os, pickle, time
from tools import coalesce, read_json, flatten
from scipy.sparse import load_npz, save_npz, csr_matrix
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
        self.n_playlists = None
        self.n_tracks = None
        self.test_pids = read_json(TEST_FILE).keys()
        self.Rtrain, self.Rtest, self.popular = None, None, None

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




    # parsing functions to obtain sparse matrix

    def preprocess(self):
        start = time.time()
        train_paths = list(map(lambda x: f'{TRAIN_FOLDER}/{x}', os.listdir(TRAIN_FOLDER)[:1000]))

        _, relations = self.parse_tracks(train_paths + [TEST_FILE])

        self.save_sparse(relations, self.matrix_path)

        test_relations = dict()
        for i, pid in enumerate(self.test_pids):
            test_relations[i] = relations.pop(pid)

        self.save_sparse(relations, self.train_path, 'most-popular.npz')
        self.save_sparse(test_relations, self.test_path)
        end = time.time()

        print(f'Preprocessing time (compute and store sparse matrix: {end-start}')

    def save_sparse(self, relations: Dict[int, Tuple[int]], path: str, col_sum: bool | str = False):
        from scipy.sparse import csr_matrix, save_npz
        rows, cols = list(), list()
        for pid, tracks in relations.items():
            rows += [pid]*len(tracks)
            cols += tracks
        (rows, cols), data = map(np.array, (rows, cols)), np.ones(len(rows))
        matrix = csr_matrix((data, (rows, cols)), shape=(max(relations.keys()) + 1, self.n_tracks))
        save_npz(path, matrix)

        if col_sum:
            save_npz(col_sum, norm(matrix, axis=0))


    def parse_tracks(
            self,
            paths: List[str],
    ) -> Tuple[Dict[str, int], Dict[int, Tuple[int]]]:

        track_map = dict()      # stores track_uri -> id
        relations = dict()      # stores pid -> List[id]

        indexes = coalesce(len(paths), self.num_threads)

        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()

            for i in range(self.num_threads):
                start, end = indexes[i], indexes[i+1]
                futures.append(
                    pool.submit(_collect_tracks, paths[start:end])
                )

            for _ in range(len(futures)):
                partial_track_map, partial_relations = futures.pop(0).result()

                # add track_map new values that are not in the global track_map
                new_tracks = partial_track_map.keys() - track_map.keys()
                track_map |= dict(zip(new_tracks, range(len(track_map), len(track_map)+len(new_tracks))))

                # create a dictionary to map those tracks repeated to the global track ids
                partial2global = {partial_track_map[track]: track_map[track] for track in partial_track_map.keys()}

                # map indices of partial_relations to the global ids
                new_relations = {pid: tuple(map(partial2global.get, tracks)) for pid, tracks in partial_relations.items()}
                relations |= new_relations

        # save track_map
        with open(self.trackmap_path, 'wb') as file:
            pickle.dump(track_map, file)
        self.n_tracks = len(track_map)
        return track_map, relations


def _collect_tracks(paths: List[str]) -> Tuple[Dict[str, int], Dict[int, Tuple[int]]]:
    track_map = dict()   # partial thread view of track_uri -> id
    relations = dict()   # partial thread view of pid -> List[id]

    for path in paths:
        playlists = read_json(path, lambda x: tuple(set(x)))
        new_tracks = set(flatten(playlists.values(), levels=1)) - track_map.keys()
        track_map |= dict(zip(new_tracks, range(len(track_map), len(track_map)+len(new_tracks))))

        # map track_uri -> id
        playlists = {pid: tuple(map(track_map.get, tracks)) for pid, tracks in playlists.items()}
        relations |= playlists

    return track_map, relations




if __name__ == '__main__':
    model = NeighbourModel(100)
    model.preprocess()







        









