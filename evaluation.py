from structs import Reader
from typing import Dict, List
import re
import numpy as np
GOLD_FILE = 'spotify_test_playlists/test_eval_playlists.json'


class Evaluator:
    def __init__(self, predicted_file: str):
        self.predicted_file = predicted_file

        self.golds = Reader(GOLD_FILE, only_uri=True).read()
        self.preds = self.read_submission()
        self.relevants = {
            pid: np.array([track in self.golds[pid] for track in self.preds[pid]])
                  for pid in self.golds.keys()
        }


    def read_submission(self) -> Dict[int, List[str]]:
        raw_data = open(self.predicted_file, 'r', encoding='utf8').read()

        # get lines, remove empty break lines and strip
        raw_lines = list(map(lambda x: x.strip(), re.split('\n+', raw_data)))
        raw_lines = list(filter(lambda x: len(x) > 0, raw_lines))

        # remove comments and info row
        raw_lines = list(filter(lambda x: x[0] != '#', raw_lines))[1:]

        # clean those lines which do not start with pid
        raw_playlists = list(filter(lambda x: x.split(',')[0].isnumeric(), raw_lines))

        # return predictions
        playlists = dict()
        for line in raw_playlists:
            items = list(map(lambda x: x.strip(), line.split(',')))
            playlists[int(items[0])] = items[1:]
        return playlists

    def RPrecision(self):
        values = list()
        for pid in self.golds.keys():
            n_relevants = self.relevants[pid].sum()
            values.append(n_relevants/len(self.golds[pid]))
        return np.mean(values)

    def NDCG(self):
        values = list()
        for pid in self.golds.keys():
            # compute DCG
            golds, preds, relevants = self.golds[pid], self.preds[pid], self.relevants[pid]

            dcg = relevants[0] + (relevants[1:]/np.log2(np.arange(2, len(relevants) + 1))).sum()
            idcg = 1 + (1/np.log2(np.arange(2, len(golds)))).sum()

            values.append(dcg/idcg)
        return np.mean(values)

    def clicks(self):
        values = list()
        for j, pid in enumerate(self.golds.keys()):
            golds, preds = self.golds[pid], self.preds[pid]

            for i, track in enumerate(preds):
                if track in golds:
                    if j == 4:
                        print(golds)
                        print(track)
                    values.append(i//10)
                    break
        return values



if __name__ == '__main__':
    evaluator = Evaluator('submissions/baseline.csv.gz')
    preds = evaluator.read_submission()
    print(evaluator.clicks())
