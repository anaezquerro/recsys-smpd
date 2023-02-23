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

test = open('spotify_test_dataset/test_input_playlists.json', 'w')
test = json.load(test)

playlists = test['playlists']

sub = open('submission.csv.gz', 'w')
for pl in playlists:
    tracks = list(map(lambda x: x['track_uri'], pl['tracks']))
    submit = list()
    i = 0
    while len(submit) < 500:
        if order[i] not in tracks:
            submit.append(order[i])
        i += 1
    sub.write(f"{pl['pid']}, {', '.join(submit)}\n")


sub.close()
test.close()