import csv
from scipy import sparse
import numpy as np
import pandas as pd
import time
from tools import *
from typing import List, Dict
import os
from typing import Dict
TRAIN_FOLDER = 'spotify_train_dataset/data/'
MAX_THREADS = os.cpu_count()
TEST_FILE = 'spotify_test_playlists/test_input_playlists.json'






def toCSV(N: int = 1000):
    paths = os.listdir(TRAIN_FOLDER)[:N]

    relations = dict()
    indexes = open('indexes.csv', 'w')
    rel_csv = open('relations.csv', 'w')
    for path in paths:
        playlists = json.load(open(TRAIN_FOLDER + '/' + path, 'r', encoding='utf8'))['playlists']
        for playlist in playlists:
            tracks = set(map(lambda track: track['track_uri'], playlist['tracks']))
            for track in tracks:
                try:
                    relations[track]
                except KeyError:
                    relations[track] = len(relations)
                    rel_csv.write(f"{relations[track]},{track}\n")
                indexes.write(f"{playlist['pid']},{relations[track]}\n")
    rel_csv.close()
    indexes.close()

    return relations



def csv2sparse(csv_path):
    df = pd.read_csv(csv_path, names=['playlist', 'track'])

    row = np.array(df.playlist)
    col = np.array(df.track)
    data = np.ones(len(df.playlist))

    matrix = sparse.csr_matrix((data, (row, col)))

    return matrix


def test2sparse(relations: Dict[str, int]):
    playlists = json.load(open(TEST_FILE, encoding='utf8'))['playlists']
    rows = list()
    cols = list()
    for i, playlist in enumerate(playlists):
        tracks_indices = list()
        tracks = set(map(lambda track: track['track_uri'], playlist['tracks']))
        for track in tracks:
            try:
                tracks_indices.append(relations[track])
            except KeyError:
                continue
        cols += tracks_indices
        rows += [i] * len(tracks_indices)

    rows, cols = np.array(rows), np.array(cols)
    data = np.ones(len(rows))
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(playlists), len(relations)))
    return matrix


if __name__ == '__main__':
    relations = toCSV()
    matrix_train = csv2sparse('indexes.csv')
    sparse.save_npz('R_train.npz', matrix_train)
    matrix_test = test2sparse(relations)
    sparse.save_npz('R_test.npz', matrix_test)



