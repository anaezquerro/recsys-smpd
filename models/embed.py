from gensim.models import Word2Vec
from scipy.sparse import csr_matrix, load_npz, vstack
from utils.tools import tolist, load_pickle, read_json, pop_empty, coalesce
from typing import List, Tuple, Dict
from utils.constants import TEST_FILE, N_RECS, MAX_THREADS
import numpy as np
from concurrent.futures import ProcessPoolExecutor

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


    def item_similarity(self, batch_size: int, num_threads: int, verbose: int) -> csr_matrix:
        n_tracks = load_npz(self.train_path).shape[1]
        if verbose:
            print('Computing item similarity for all tracks')

        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            futures = list()

            for i in range(0, n_tracks, batch_size):
                futures.append(
                    pool.submit(similarity, list(range(i, i+batch_size)), self.model, self.k, n_tracks, verbose)
                )

            S = futures.pop(0).result()
            for _ in range(len(futures)):
                S = vstack([S, futures.pop(0)])

        return S


def similarity(tracks: List[int], model: Word2Vec, k: int, n_tracks: int, verbose: bool):
    S = csr_matrix((len(tracks), n_tracks))
    if verbose:
        print(f'Computing similarity for track {tracks[0]}/{n_tracks}')
    data = list(map(lambda track: zip(*model.wv.most_similar(positive=[track], topn=k)), tracks))
    for i, (cols, values) in enumerate(data):
        S[i, list(cols)] = values
    return S




if __name__ == '__main__':
    model = EmbedModel(embed_dim=10, context_size=10, model_path='data/track2vec',
                       train_path='data/Rtrain.npz', test_path='data/Rtest.npz', trackmap_path='data/trackmap.pickle', k=10, load=True)
    model.item_similarity(batch_size=100, num_threads=MAX_THREADS, verbose=True)

