import time, os, pickle
from typing import Dict, List, Tuple
from tools import coalesce, read_json, flatten
import numpy as np
from concurrent.futures import ProcessPoolExecutor

TRAIN_FOLDER = 'spotify_train_dataset/data/'
MAX_THREADS = os.cpu_count()
TEST_FILE = 'spotify_test_playlists/test_input_playlists.json'


def collect_tracks(paths: List[str]) -> Tuple[Dict[str, int], Dict[int, Tuple[int]]]:
    track_map = dict()  # partial thread view of track_uri -> id
    relations = dict()  # partial thread view of pid -> List[id]

    for path in paths:
        playlists = read_json(path, lambda x: tuple(set(x)))
        new_tracks = set(flatten(playlists.values(), levels=1)) - track_map.keys()
        track_map |= dict(zip(new_tracks, range(len(track_map), len(track_map) + len(new_tracks))))

        # map track_uri -> id
        playlists = {pid: tuple(map(track_map.get, tracks)) for pid, tracks in playlists.items()}
        relations |= playlists

    return track_map, relations


class Sparse:
    def __init__(
            self,
            train_path: str = 'data/Rtrain.npz',
            test_path: str = 'data/Rtest.npz',
            trackmap_path: str = 'data/trackmap.pickle',
            num_threads: int = MAX_THREADS,
    ):

        self.n_playlists = None
        self.n_tracks = None
        self.num_threads = num_threads
        self.train_path = train_path
        self.test_path = test_path
        self.trackmap_path = trackmap_path

        self.test_pids = read_json(TEST_FILE).keys()

    def preprocess(self):
        """
        Makes preprocessing of all dataset. The result should be:
        - Stored train sparse matrix.
        :return:
        """
        start = time.time()
        train_paths = list(map(lambda x: f'{TRAIN_FOLDER}/{x}', os.listdir(TRAIN_FOLDER)[:1000]))

        _, relations = self.parse_tracks(train_paths + [TEST_FILE])

        test_relations = dict()
        for i, pid in enumerate(self.test_pids):
            test_relations[pid] = relations.pop(pid)

        self.save_sparse(relations, self.train_path)
        self.save_sparse(test_relations, self.test_path, rowmap_path=self.test_path.replace('.npz', '.pickle'))
        end = time.time()

        print(f'Preprocessing time (compute and store sparse matrix: {end - start}')

    def parse_tracks(
            self,
            paths: List[str],
    ) -> Tuple[Dict[str, int], Dict[int, Tuple[int]]]:

        track_map = dict()  # stores track_uri -> id
        relations = dict()  # stores pid -> List[id]

        indexes = coalesce(len(paths), self.num_threads)

        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()

            for i in range(self.num_threads):
                start, end = indexes[i], indexes[i + 1]
                futures.append(
                    pool.submit(collect_tracks, paths[start:end])
                )

            for _ in range(len(futures)):
                partial_track_map, partial_relations = futures.pop(0).result()

                # add track_map new values that are not in the global track_map
                new_tracks = partial_track_map.keys() - track_map.keys()
                track_map |= dict(zip(new_tracks, range(len(track_map), len(track_map) + len(new_tracks))))

                # create a dictionary to map those tracks repeated to the global track ids
                partial2global = {partial_track_map[track]: track_map[track] for track in partial_track_map.keys()}

                # map indices of partial_relations to the global ids
                new_relations = {pid: tuple(map(partial2global.get, tracks)) for pid, tracks in
                                 partial_relations.items()}
                relations |= new_relations

        # save track_map
        with open(self.trackmap_path, 'wb') as file:
            pickle.dump(track_map, file)
        self.n_tracks = len(track_map)
        return track_map, relations

    def save_sparse(self, relations: Dict[int, Tuple[int]], path: str, rowmap_path: str = None):
        from scipy.sparse import csr_matrix, save_npz

        rows, cols = list(), list()
        relations = {pid: tracks for pid, tracks in relations.items() if len(tracks) > 0}

        for i, (pid, tracks) in enumerate(relations.items()):
            rows += [i] * len(tracks)
            cols += tracks

        # save mapping of pid -> row
        if rowmap_path:
            with open(rowmap_path, 'wb') as file:
                pickle.dump(dict(zip(relations.keys(), range(len(relations)))), file)

        # save sparse matrix
        (rows, cols), data = map(np.array, (rows, cols)), np.ones(len(rows))
        matrix = csr_matrix((data, (rows, cols)), shape=(len(relations), self.n_tracks))
        save_npz(path, matrix)


if __name__ == '__main__':
    sparse = Sparse()
    sparse.preprocess()