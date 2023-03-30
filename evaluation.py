from tools import *
from typing import Dict, List
import re
import numpy as np
GOLD_FILE = 'spotify_test_playlists/test_eval_playlists.json'


class Evaluator:
    def __init__(self, predicted_file: str):
        self.predicted_file = predicted_file
        self.golds = read_json(GOLD_FILE)
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
            n_relevants = self.relevants[pid][:len(self.golds[pid])].sum()
            values.append(n_relevants/len(self.golds[pid]))
        return np.mean(values)

    def NDCG(self):
        values = list()
        for pid in self.golds.keys():
            # compute DCG
            golds, preds, relevants = self.golds[pid], self.preds[pid], self.relevants[pid]

            dcg = sum(relevants/np.log2(np.arange(2, len(relevants)+2)))
            idcg = sum(1 / np.log2(np.arange(2, len(golds) + 2)))

            values.append(dcg/idcg)

        return np.mean(values)

    def clicks(self):
        values = list()
        for j, pid in enumerate(self.golds.keys()):
            golds, preds = self.golds[pid], self.preds[pid]
            clicked = False

            for i, track in enumerate(preds):
                if track in golds:
                    values.append(i//10)
                    clicked = True
                    break
            if not clicked:
                values.append(51)

        return np.mean(values)

if __name__ == '__main__':
    evaluator = Evaluator('submissions/item-based.csv.gz')
    print(evaluator.RPrecision())
    print(evaluator.NDCG())
    print(evaluator.clicks())