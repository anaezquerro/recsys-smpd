import os, time, json
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Callable
from utils.tools import submit, coalesce, read_json
from utils.constants import MAX_THREADS, TRAIN_FOLDER, N_RECS, INPUT_FILE

class BaselineModel:

    NAME = 'BaselineModel'

    def __init__(self, binfreq: bool = True):
        super().__init__()
        self.popular = None
        self.binfreq = binfreq

    def preprocess(self, num_threads: int, verbose: bool) -> List[str]:
        paths = os.listdir(TRAIN_FOLDER)


        if verbose:
            print('Initializing track counting')

        # parallelize track counting
        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            indexes = coalesce(len(paths), num_threads)
            futures = list()

            for i in range(num_threads):
                start, end = indexes[i], indexes[i + 1]
                futures.append(
                    pool.submit(count, paths[start:end], set if self.binfreq else tuple, verbose)
                )

            counter = futures.pop(0).result()
            for f in futures:
                new_counter = f.result()

                # append those tracks that are in counter but are not in counter
                counter |= {key: new_counter[key] for key in new_counter.keys() - counter.keys()}

                # add those tracks that are both in counter and counter
                counter |= {key: new_counter[key] + counter[key] for key in new_counter.keys() & counter.keys()}

        # sort dictionary by value (since value stores occurrences)
        self.popular = list(sorted(counter.keys(), key=counter.get, reverse=True))

        return self.popular

    def recommend(self, submit_path: str, num_threads: int, verbose: bool) -> Dict[int, List[str]]:
        # extract playlists for evaluation
        pids, tracks = zip(*read_json(INPUT_FILE).items())

        # parallelize prediction
        if verbose:
            print(f'Initializing recommendation')
        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            indexes = coalesce(len(pids), num_threads)
            futures = list()
            for i in range(num_threads):
                start, end = indexes[i], indexes[i+1]
                futures.append(
                    pool.submit(get_popular, dict(zip(pids[start:end], tracks[start:end])), self.popular, verbose)
                )

            playlists = futures.pop(0).result()
            for _ in range(len(futures)):
                playlists |= futures.pop(0).result()

        submit(submit_path, playlists)
        return playlists



def count(paths: List[str], func: Callable, verbose: bool) -> Dict[int, List[str]]:
    counter = dict()
    info = lambda i: print(f'Counting tracks from file {i}/{len(paths)}') if verbose and (i%10==0) else None
    for i, path in enumerate(paths):
        info(i)
        playlists = read_json(f'{TRAIN_FOLDER}/{path}', func)
        for pid, tracks in playlists.items():
            for track in tracks:
                try:
                    counter[track] += 1
                except KeyError:
                    counter[track] = 1
    return counter


def get_popular(playlists: Dict[int, List[str]], popular: List[int], verbose: bool) -> Dict[int, List[str]]:
    info = lambda i: print(f'Getting popularity for playlist {i}/{len(playlists)}') if verbose and (i%100 == 0) else None

    for i, (pid, tracks) in enumerate(playlists.items()):
        info(i)
        tracks = tracks.copy()
        playlists[pid] = []
        j = 0
        while len(playlists[pid]) < N_RECS:
            if not popular[j] in tracks:
                playlists[pid].append(popular[j])
            j += 1
    return playlists



if __name__ == '__main__':
    model = BaselineModel()

    # training (counting track occurrences)
    start = time.time()
    tracks = model.preprocess(MAX_THREADS, True)
    end = time.time()
    print('Preprocessing time:', (end-start))
    print('-'*80)

    # prediction
    start = time.time()
    model.recommend('submissions/baseline.csv.gz', MAX_THREADS, True)
    end = time.time()
    print('Prediction time:', (end-start))










