import csv
from scipy import sparse
import numpy as np
import pandas as pd
import time

start = time.time()

with open('uri.csv', 'r') as rel:
    tmp = csv.reader(rel)
    for i in tmp:
        relation = i[:-1]

df = pd.read_csv('train.csv', names=['playlist', 'track']).drop_duplicates()

playlists = list(df.playlist)
tracks = list(df.track)


row = np.array(playlists)
col = np.array(tracks)
data = np.ones(len(playlists))
matrix = sparse.coo_matrix((data, (row, col)))

counter = np.array(matrix.sum(axis=0))[0]

popularity = {relation[i]: counter[i] for i in range(len(relation))}

order = sorted(popularity, key=popularity.get, reverse=True)

end = time.time()
print(end-start)
