import os
from typing import List, Dict, Callable, Iterable
import json, pickle
from utils.constants import INFO_ROW

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

def read_json(path: str, funct: Callable = set) -> Dict[int, List[str]]:
    data = json.load(open(path, 'r', encoding='utf8'))['playlists']
    playlists = dict()
    for playlist in data:
        playlists[int(playlist['pid'])] = funct(map(lambda track: track['track_uri'], playlist['tracks']))
    return playlists

def pop_empty(playlists: Dict[int, List[str]]):
    test_empty = list()
    for pid in set(playlists.keys()):
        if len(playlists[pid]) == 0:
            test_empty.append(pid)
            playlists.pop(pid)
    return test_empty



def flatten(list_of_lists, levels=None):
    items = list()
    for l in list_of_lists:
        if (isinstance(l, Iterable)) and (levels is None or levels != 0):
            items += flatten(l, levels if levels is None else levels-1)
        else:
            items.append(l)
    return items



def submit(path: str, playlists: Dict[int, List[str]], fill: List[str] = None):
    with open(path, 'w', encoding='utf8') as file:
        file.write(INFO_ROW + '\n')
        for pid, tracks in playlists.items():
            file.write(f'{pid},' + ','.join(tracks if len(tracks) > 0 else fill) + '\n')


def load_pickle(path: str):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def save_pickle(obj, path: str):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def create_folder(path: str):
    folder = '/'.join(path.split('/')[:-1])
    if len(folder) > 0 and not os.path.exists(folder):
        os.makedirs(folder)