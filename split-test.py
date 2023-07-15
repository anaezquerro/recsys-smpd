import numpy as np 
import json, os
from concurrent.futures import ProcessPoolExecutor
from typing import List

DATASET_FOLDER = 'spotify_million_playlist_dataset/'
SOURCE_FOLDER = f'{DATASET_FOLDER}/data/'
TRAIN_FOLDER = f'{DATASET_FOLDER}/train/'
if not os.path.exists(TRAIN_FOLDER):
    os.makedirs(TRAIN_FOLDER)
INPUT_FILE = f'{DATASET_FOLDER}/input.json'
GOLD_FILE = f'{DATASET_FOLDER}/gold.json'
CHALLENGE_FILE = 'spotify_million_playlist_dataset_challenge/challenge_set.json'

m = int(1e6)
meval = int(1e4)
scenarios = 10
seed = 1234
step = int(1e3)
num_threads = os.cpu_count()


def scenario(playlist: dict, scenario_type: int):
    cuts = dict(zip(range(1, 11), [0, 1, 5, 5, 10, 10, 25, 25, 100, 100]))

    # delete repeated tracks 
    uris = set()
    tracks = list()
    for track in playlist['tracks']:
        if track['track_uri'] not in uris:
            tracks.append(track)
            uris.add(track['track_uri'])
    playlist['tracks'] = tracks 

    if scenario_type in [8, 10]:
        np.random.shuffle(playlist['tracks'])


    input_playlist = playlist.copy()
    input_playlist['tracks'] = input_playlist['tracks'][: min(cuts[scenario_type], len(playlist['tracks'])//3)]
    playlist['tracks'] = playlist['tracks'][len(input_playlist['tracks']):]
    return input_playlist, playlist

def extract_test(start: int, end: int, test_pids: np.ndarray):
    test_pids = (test_pids[(test_pids >= start) & (test_pids < end)]).tolist()
    test_playlists = list()
    for i in range(start, end, step):
        file = f'mpd.slice.{i}-{i+step-1}.json'
        print(f'Processing file {(i-start)//step}/{(end-start)//step}: {file}')
        playlists = json.load(open(f'{SOURCE_FOLDER}/{file}', 'r'))
        train_playlists = dict(zip(range(step), playlists['playlists']))
        while test_pids[0] < i:
            pid = test_pids.pop(0)
            test_playlists.append(playlists['playlists'][pid%step])
            train_playlists.pop(pid%step)
        playlists['playlists'] = [train_playlists[pid] for pid in sorted(train_playlists.keys())]
        with open(f'{TRAIN_FOLDER}/{file}', 'w') as handle:
            json.dump(playlists, handle, indent=4)
    return test_playlists


if __name__ == '__main__':
    np.random.seed(seed)
    test_pids = np.array(sorted(np.random.choice(m, size=meval, replace=False)))
    test_playlists = list()

    # read playlists from source folder
    with ProcessPoolExecutor(max_workers=num_threads) as pool:
        futures = list()
        thread_step = (m//step//num_threads)*step

        for i in range(0, m, thread_step):
            start, end = i, i + thread_step
            futures.append(pool.submit(extract_test, start, end, test_pids))

        for _ in range(len(futures)):
            test_playlists += futures.pop(0).result()

    challenge = json.load(open(CHALLENGE_FILE, 'r'))
    challenge['playlists'] = test_playlists

    
    # shuffle test playlists
    np.random.shuffle(test_playlists)
    
    # mask some results given each scenario
    input_playlists, test_playlists = map(lambda x: sorted(x, key=lambda x: x['pid']), zip(*map(lambda x: scenario(x[1], (x[0]//(meval//scenarios))+1), enumerate(sorted(test_playlists, key=len)))))
    
    challenge['playlists'] = test_playlists
    with open(GOLD_FILE, 'w') as handle:
        json.dump(challenge, handle, indent=4)
    challenge['playlists'] = input_playlists
    with open(INPUT_FILE, 'w') as handle:
        json.dump(challenge, handle, indent=4)






        
    


    
    


