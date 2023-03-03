from typing import List, Dict
import json

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

def read_json(path: str):
    data = json.load(open(path, 'r', encoding='utf8'))['playlists']
    playlists = dict()
    for playlist in data:
        playlists[int(playlist['pid'])] = list(map(lambda track: track['track_uri'], playlist['tracks']))
    return playlists