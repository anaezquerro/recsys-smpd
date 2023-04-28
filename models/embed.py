from gensim.models import Word2Vec
from scipy.sparse import csr_matrix, load_npz, vstack
from utils.tools import tolist, load_pickle, read_json, pop_empty, coalesce
from typing import List, Tuple, Dict
from utils.constants import TEST_FILE, N_RECS, MAX_THREADS, INFO_ROW
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from gensim.similarities.annoy import AnnoyIndexer
from models.neighbour import recommend

class EmbedModel:
    def __init__(self, embed_dim: int, context_size: int, k: int, model_path: str, train_path: str, test_path: str, trackmap_path: str, load: bool):
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.model_path = model_path
        self.train_path = train_path
        self.test_path = test_path
        self.trackmap_path = trackmap_path
        self.k = k
        if load:
            self.model = Word2Vec.load(model_path)


    def train(self, num_epochs: int, num_threads: int, verbose: bool):
        if verbose:
            print('Training word2vec model with train playlists')
        Rtrain = load_npz(self.train_path)
        n_tracks = Rtrain.shape[1]
        train_playlists = tolist(Rtrain)
        del Rtrain

        self.model = Word2Vec(sentences=[range(n_tracks)], vector_size=self.embed_dim, window=self.context_size, min_count=1, workers=num_threads)
        self.model.train(train_playlists, total_examples=len(train_playlists), epochs=num_epochs)
        self.model.save(self.model_path)


    def item_similarity(self, num_trees: int, num_threads: int, verbose: int) -> csr_matrix:
        n_tracks = load_npz(self.train_path).shape[1]

        if verbose:
            print('Computing item similarity for all tracks')

        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            futures = list()
            indexes = coalesce(n_tracks, num_threads)

            for i in range(num_threads):
                start, end = indexes[i], min(indexes[i+1], n_tracks)
                futures.append(
                    pool.submit(similarity, list(range(start, end)), self.model, num_trees, self.k, n_tracks, verbose)
                )

            S = futures.pop(0).result()
            for _ in range(len(futures)):
                S = vstack([S, futures.pop(0).result()])

        return S

    def recommend(self, submit_path: str, num_threads: int, num_trees: int, verbose: bool):
        # first we must compute the similarity matrix
        S = self.item_similarity(num_trees, num_threads, verbose)

        # now obtain the estimated ratings
        Rtest = load_npz(self.test_path)
        Rest = Rtest @ S

        # create popularity vector
        Rtrain = load_npz(self.train_path)
        popular = np.copy(np.asarray(-(Rtrain.sum(axis=0))).argsort().ravel()).tolist()[:N_RECS]
        del Rtest, S, Rtrain

        # load additional maps
        test = read_json(TEST_FILE)
        trackmap = load_pickle(self.trackmap_path)
        pidmap = load_pickle(self.test_path.replace('.npz', '.pickle'))
        test = {pid: list(map(trackmap.get, tracks)) for pid, tracks in test.items()}
        test_empty = pop_empty(test)

        # compute parallel track recommendation
        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            pids, tracks = zip(*test.items())
            futures = list()
            indexes = coalesce(len(pids), num_threads)

            for i in range(num_threads):
                start, end = indexes[i], indexes[i + 1]
                futures.append(
                    pool.submit(
                        recommend,
                        Rest, dict(zip(pids[start:end], tracks[start:end])), pidmap, popular, verbose)
                )

            playlists = futures.pop(0).result()
            for _ in range(len(futures)):
                playlists |= futures.pop(0).result()

        # convert trackmap from id -> track_uri
        trackmap = {value: key for key, value in trackmap.items()}

        # write results in the submission
        with open(submit_path, 'w', encoding='utf8') as file:
            file.write(INFO_ROW + '\n')
            for pid, tracks in playlists.items():
                file.write(f'{pid},' + ','.join(list(map(trackmap.get, tracks))) + '\n')
            for pid in test_empty:
                file.write(f'{pid},' + ','.join(list(map(trackmap.get, popular))) + '\n')



def similarity(tracks: List[int], model: Word2Vec, num_trees: int, k: int, n_tracks: int, verbose: bool):
    annoy_index = AnnoyIndexer(model, num_trees)
    rows, cols, data = list(), list(), list()
    if verbose:
        print(f'Computing similarity for track {tracks[0]}/{n_tracks}')
    sim = list(map(lambda track: zip(*model.wv.most_similar([model.wv[track]], topn=k, indexer=annoy_index)), tracks))
    for i, (indices, values) in enumerate(sim):
        rows += [i]*len(values)
        cols += indices
        data += values
    S = csr_matrix((data, (rows, cols)), shape=(len(tracks), n_tracks))
    return S




if __name__ == '__main__':
    model = EmbedModel(embed_dim=100, context_size=20, model_path='data/track2vec',
                       train_path='data/Rtrain.npz', test_path='data/Rtest.npz', trackmap_path='data/trackmap.pickle',
                       k=10, load=False)
    model.train(num_epochs=100, num_threads=10, verbose=True)
    model.recommend('submissions/embed.csv.gz', num_threads=10, num_trees=100, verbose=True)

