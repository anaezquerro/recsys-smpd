import time, os, pickle
from typing import Dict, List, Tuple
from utils.tools import coalesce, read_json, flatten, save_pickle
from utils.constants import TEST_FILE, TRAIN_FOLDER
import numpy as np
from concurrent.futures import ProcessPoolExecutor




def collect_tracks(paths: List[str], verbose) -> Tuple[Dict[str, int], Dict[int, Tuple[int]]]:
    trackmap = dict()  # partial thread view of track_uri -> id
    relations = dict()  # partial thread view of pid -> List[id]
    info = lambda i: print(f'Collecting tracks from file {i}/{len(paths)}') if (i%10 == 0) and verbose else None

    for i, path in enumerate(paths):
        info(i)
        playlists = read_json(path, lambda x: tuple(set(x)))
        new_tracks = set(flatten(playlists.values(), levels=1)) - trackmap.keys()
        trackmap |= dict(zip(new_tracks, range(len(trackmap), len(trackmap) + len(new_tracks))))

        # map track_uri -> id
        playlists = {pid: tuple(map(trackmap.get, tracks)) for pid, tracks in playlists.items()}
        relations |= playlists

    return trackmap, relations


class Sparse:
    def __init__(self, train_path: str, test_path: str, trackmap_path: str):
        self.n_playlists = None
        self.n_tracks = None
        self.num_threads, self.verbose = None, None
        self.train_path = train_path
        self.test_path = test_path
        self.trackmap_path = trackmap_path

        self.test_pids = read_json(TEST_FILE).keys()

    def preprocess(self, num_threads: int, verbose: bool):
        """
        Makes preprocessing of all dataset. The result should be:
        1) train sparse matrix stored in train_path.
        2) test sparse matrix stored in test_path
        3) trackmap to store relations between track_uri -> column of the matrix.
        4) pidmap to store relations between test pid -> row of the test sparse matrix
        """
        self.num_threads = num_threads
        self.verbose = verbose
        train_paths = list(map(lambda x: f'{TRAIN_FOLDER}/{x}', os.listdir(TRAIN_FOLDER)[:1000]))

        _, relations = self.parse_tracks(train_paths + [TEST_FILE])

        test_relations = dict()
        for i, pid in enumerate(self.test_pids):
            test_relations[pid] = relations.pop(pid)

        self.save_sparse(relations, self.train_path)
        self.save_sparse(test_relations, self.test_path, rowmap_path=self.test_path.replace('.npz', '.pickle'))

    def parse_tracks(self, paths: List[str]) -> Tuple[Dict[str, int], Dict[int, Tuple[int]]]:
        trackmap = dict()  # stores track_uri -> id
        relations = dict()  # stores pid -> List[id]

        if self.verbose:
            print(f'Collecting tracks from the train and test set')

        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()
            indexes = coalesce(len(paths), self.num_threads)

            for i in range(self.num_threads):
                start, end = indexes[i], indexes[i + 1]
                futures.append(
                    pool.submit(collect_tracks, paths[start:end], self.verbose)
                )

            for _ in range(len(futures)):
                partial_trackmap, partial_relations = futures.pop(0).result()

                # add trackmap new values that are not in the global trackmap
                new_tracks = partial_trackmap.keys() - trackmap.keys()
                trackmap |= dict(zip(new_tracks, range(len(trackmap), len(trackmap) + len(new_tracks))))

                # create a dictionary to map those tracks repeated to the global track ids
                partial2global = {partial_trackmap[track]: trackmap[track] for track in partial_trackmap.keys()}

                # map indices of partial_relations to the global ids
                new_relations = {pid: tuple(map(partial2global.get, tracks)) for pid, tracks in
                                 partial_relations.items()}
                relations |= new_relations

        # save trackmap
        save_pickle(trackmap, self.trackmap_path)
        self.n_tracks = len(trackmap)
        return trackmap, relations

    def save_sparse(self, relations: Dict[int, Tuple[int]], path: str, rowmap_path: str = None):
        from scipy.sparse import csr_matrix, save_npz

        if self.verbose:
            print(f'Saving sparse matrices')

        rows, cols = list(), list()
        relations = {pid: tracks for pid, tracks in relations.items() if len(tracks) > 0}

        for i, (pid, tracks) in enumerate(relations.items()):
            rows += [i] * len(tracks)
            cols += tracks

        # save mapping of pid -> row
        if rowmap_path:
            save_pickle(dict(zip(relations.keys(), range(len(relations)))), rowmap_path)

        # save sparse matrix
        (rows, cols), data = map(np.array, (rows, cols)), np.ones(len(rows))
        matrix = csr_matrix((data, (rows, cols)), shape=(len(relations), self.n_tracks), dtype=np.float32)
        save_npz(path, matrix)


if __name__ == '__main__':
    sparse = Sparse(train_path='data/Rtrain.npz', test_path='data/Rtest.npz', trackmap_path='data/trackmap.pickle')
    sparse.preprocess(num_threads=12, verbose=True)