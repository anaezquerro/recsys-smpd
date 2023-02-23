import json
import glob
import os
import time

path = 'spotify_train_dataset/data'

rel, counter = dict(), dict()

jsons = glob.glob(os.path.join(path, '*.json'))[0:2]

train = open('train.csv', 'w')
relation = open('uri.csv', 'w')
for route in jsons:
    with open(route, "r") as j:
        batch = json.load(j)
        for pl in batch['playlists']:
            for track in pl['tracks']:
                uri = track['track_uri']
                try:
                    counter[uri] += 1
                except KeyError:
                    counter[uri] = 1
                    print(uri, end=',', file=relation)
                    rel[uri] = len(rel)
                print(pl['pid'], rel[uri], sep=',', file=train)

train.close()
relation.close()

order = sorted(counter, key=counter.get, reverse=True)

test = open()
