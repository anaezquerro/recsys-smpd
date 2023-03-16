import csv
from scipy import sparse
import numpy as np
import pandas as pd
import time
from tools import *
from typing import List, Dict
import os
TRAIN_FOLDER = 'spotify_train_dataset/data/'
MAX_THREADS = os.cpu_count()


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

def CSVtoMatrix(csv_path):
    df = pd.read_csv(csv_path, names=['playlist', 'track'])

    row = np.array(df.playlist)
    col = np.array(df.track)
    data = np.ones(len(df.playlist))

    matrix = sparse.csr_matrix((data, (row, col)))

    return matrix


if __name__ == '__main__':
    toCSV()
    matrix = CSVtoMatrix('indexes.csv')
    sparse.save_npz('matrix.npz', matrix)

