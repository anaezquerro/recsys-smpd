import numpy as np
import pandas as pd
from tools import *
import os, pickle
from typing import Dict, Tuple, Set
from concurrent.futures import ProcessPoolExecutor
TRAIN_FOLDER = 'spotify_train_dataset/data/'
MAX_THREADS = os.cpu_count()
TEST_FILE = 'spotify_test_playlists/test_input_playlists.json'


def _collect_tracks(paths: List[str]):
    relations = dict()
    track_map = dict()
    for i, path in enumerate(paths):
        print(f'Processing path {i}/{len(paths)}')
        playlists = json.load(open(TRAIN_FOLDER + '/' + path, 'r', encoding='utf8'))['playlists']
        for playlist in playlists:
            tracks = set(map(lambda track: track['track_uri'], playlist['tracks']))
            track_map |= dict(zip(tracks - track_map.keys(), range(len(tracks))))
            relations[int(playlist['pid'])] = list(map(track_map.get, tracks))
    return relations, track_map

def train_sparse(
        matrix_path: str = 'R_train.npz',
        trackmap_path: str = 'track-map.pickle',
        num_threads: int = 8
    ):
    paths = os.listdir(TRAIN_FOLDER)[:10]

    # store global results of the mapping track -> id
    track_map = dict()
    relations = dict()
    indexes = coalesce(len(paths), num_threads)

    with ProcessPoolExecutor(max_workers=num_threads) as pool:
        futures = list()
        for i in range(num_threads):
            start, end = indexes[i], indexes[i+1]
            futures.append(
                pool.submit(_collect_tracks, paths[start:end])
            )
        for future in futures:
            new_relations, new_track_map = future.result()

            # añadimos la parte de new_track_map que no está en track_map
            not_repeated = new_track_map.keys() - track_map.keys()
            track_map |= dict(zip(not_repeated, range(len(track_map), len(track_map)+len(not_repeated))))

            # creamos un diccionario que realice el mapeado de new_track_map[track] -> track_map[track]
            new2old = {new_track_map[track]: track_map[track] for track in new_track_map.keys()}

            # mapeamos los índices de new_relations
            new_relations = {pid: tuple(map(new2old.get, tracks)) for pid, tracks in new_relations.items()}
            relations |= new_relations

    n_playlists, n_tracks = max(relations.keys()) + 1, len(track_map)

    # save track map
    with open(trackmap_path, 'wb') as trackmap_file:
        pickle.dump(track_map, trackmap_file)
    del track_map

    # create sparse matrix
    from scipy.sparse import csr_matrix, save_npz
    rows, cols = list(), list()
    for pid, tracks in relations.items():
        rows += [pid]*len(tracks)
        cols += tracks
    rows, cols = map(np.array, (rows, cols))
    data = np.ones(len(rows))
    matrix = csr_matrix((data, (rows, cols)), shape=(n_playlists, n_tracks))
    save_npz(matrix_path, matrix)


#
#
# def toCSV(N: int = 1000):
#     paths = os.listdir(TRAIN_FOLDER)[:N]
#
#     relations = dict()
#     indexes = open('indexes.csv', 'w')
#     rel_csv = open('relations.csv', 'w')
#     for path in paths:
#         playlists = json.load(open(TRAIN_FOLDER + '/' + path, 'r', encoding='utf8'))['playlists']
#         for playlist in playlists:
#             tracks = set(map(lambda track: track['track_uri'], playlist['tracks']))
#             for track in tracks:
#                 try:
#                     relations[track]
#                 except KeyError:
#                     relations[track] = len(relations)
#                     rel_csv.write(f"{relations[track]},{track}\n")
#                 indexes.write(f"{playlist['pid']},{relations[track]}\n")
#     rel_csv.close()
#     indexes.close()
#
#     return relations
#
#
#
# def csv2sparse(csv_path):
#     df = pd.read_csv(csv_path, names=['playlist', 'track'])
#
#     row = np.array(df.playlist)
#     col = np.array(df.track)
#     data = np.ones(len(df.playlist))
#
#     matrix = sparse.csr_matrix((data, (row, col)))
#
#     return matrix
#
#
# def test2sparse(relations: Dict[str, int]):
#     playlists = json.load(open(TEST_FILE, encoding='utf8'))['playlists']
#     rows = list()
#     cols = list()
#     for i, playlist in enumerate(playlists):
#         tracks_indices = list()
#         tracks = set(map(lambda track: track['track_uri'], playlist['tracks']))
#         for track in tracks:
#             try:
#                 tracks_indices.append(relations[track])
#             except KeyError:
#                 continue
#         cols += tracks_indices
#         rows += [i] * len(tracks_indices)
#
#     rows, cols = np.array(rows), np.array(cols)
#     data = np.ones(len(rows))
#     matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(playlists), len(relations)))
#     return matrix


if __name__ == '__main__':
    train_sparse()


