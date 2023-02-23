import os, json, time
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
from structs import Track, Reader
TRAIN_FOLDER = 'spotify_train_dataset/data/'
PREDICT_FILE = 'spotify_test_playlists/test_input_playlists.json'
N_RECS = 500
INFO_ROW = 'sr-assignments, pedro-ana, ana.ezquerro@udc.es'


def coalesce(N: int, num_threads: int) -> list:
    n_tasks = N//num_threads
    rem_tasks = N % num_threads

    indexes = [0]
    for i in range(1, N+1):
        if i <= rem_tasks:
            indexes.append(indexes[i-1] + n_tasks + 1)
        else:
            indexes.append(indexes[i-1] + n_tasks)
    return indexes


class BaselineModel:
    def __init__(self, num_threads: int = 9):
        self.num_threads = num_threads
        self.tracks = dict()
        self.trained = False


    def train(self):
        files = os.listdir(TRAIN_FOLDER)
        indexes = coalesce(len(files), self.num_threads)
        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()
            for i in range(self.num_threads):
                start, end = indexes[i], indexes[i+1]
                futures.append(pool.submit(self._count, files[start:end]))
            for f in futures:
                tracks = f.result().values()
                for track in tracks:
                    if track.track_uri in self.tracks.keys():
                        self.tracks[track.track_uri] += track
                    else:
                        self.tracks[track.track_uri] = track

        self.tracks = sorted(self.tracks.values(), key=lambda track: track.occurrences)
        self.trained = True

    def _count(self, paths: List[str]):
        tracks_counted = dict()
        for path in paths:
            playlists = Reader(f'{TRAIN_FOLDER}/{path}').read()
            for pid, tracks in playlists.items():
                for track in tracks:
                    try:
                        track = tracks_counted[track.track_uri]  # check if track_uri is already stored
                    except KeyError:
                        pass
                    track.add_occurrence()
                    tracks_counted[track.track_uri] = track
        return tracks_counted

    def predict(self, save: str = None) -> Dict[int, List[str]]:
        assert self.trained, 'This model has not been trained with any data'

        # extract playlists for evaluation
        pids, tracks = zip(*Reader(PREDICT_FILE, only_uri=True).read().items())
        playlists = dict()

        # parallelize prediction
        indexes = coalesce(len(pids), self.num_threads)
        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = list()
            for i in range(self.num_threads):
                start, end = indexes[i], indexes[i+1]
                futures.append(
                    pool.submit(self._predict, dict(zip(pids[start:end], tracks[start:end])))
                )

            for _ in range(len(futures)):
                result = futures.pop(0).result()
                playlists = playlists | result

        if save:
            self.submit(save, playlists)

        return playlists


    def _predict(self, playlists: Dict[int, List[str]]) -> Dict[int, List[str]]:
        for pid, tracks in playlists.items():
            tracks = tracks.copy()
            playlists[pid] = []
            i = 0
            while len(playlists[pid]) < N_RECS:
                if not self.tracks[i].track_uri in tracks:
                    playlists[pid].append(self.tracks[i].track_uri)
                i += 1
        return playlists


    def submit(self, path: str, playlists: Dict[int, List[str]]):
        file = open(path, 'w', encoding='utf8')
        file.write(INFO_ROW + '\n')
        for pid, tracks in playlists.items():
            file.write(f'{pid},' + ','.join(tracks) + '\n')
        file.close()


if __name__ == '__main__':
    model = BaselineModel(8)

    # training (counting track occurrences)
    start = time.time()
    model.train()
    end = time.time()
    print('Training time:', (end-start))

    # prediction
    start = time.time()
    model.predict('submissions/baseline.csv.gz')
    end = time.time()
    print('Prediction time:', (end-start))










