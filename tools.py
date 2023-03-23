from typing import List, Dict, Callable, Iterable
import json

def coalesce(N: int, num_threads: int) -> list:
    n_tasks = N//num_threads
    rem_tasks = N % num_threads

    indexes = [0]
    for i in range(1, num_threads+1):
        if i <= rem_tasks:
            indexes.append(indexes[i-1] + n_tasks + 1)
        else:
            indexes.append(indexes[i-1] + n_tasks)
    return indexes

def read_json(path: str, funct: Callable = lambda x: x) -> Dict[int, List[str]]:
    data = json.load(open(path, 'r', encoding='utf8'))['playlists']
    playlists = dict()
    for playlist in data:
        playlists[int(playlist['pid'])] = funct(map(lambda track: track['track_uri'], playlist['tracks']))
    return playlists


def flatten(list_of_lists, levels=None):
    items = list()
    for l in list_of_lists:
        if (isinstance(l, Iterable)) and (levels is None or levels != 0):
            items += flatten(l, levels if levels is None else levels-1)
        else:
            items.append(l)
    return items