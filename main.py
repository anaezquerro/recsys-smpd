import os.path
from argparse import ArgumentParser
from utils.constants import SUBMISSION_FOLDER, MAX_THREADS
from utils.evaluation import Evaluator
from utils.tools import create_folder
import time
from utils.sparse import Sparse
from models import BaselineModel, NeighbourModel, PureSVDModel, Track2VecModel


def evaluator(args):
    evaluator = Evaluator(args.submit_path)

    start = time.time()
    evaluator.read_submission()
    print(f'RPrecision: {evaluator.RPrecision()}')
    print(f'NDCG: {evaluator.NDCG()}')
    print(f'Clicks: {evaluator.clicks()}')
    end = time.time()

    if args.time:
        print('Evaluation time:', (end - start))

def base(args):
    # preprocess some arguments
    args.submit_path = args.submit_path if args.submit_path else f'submissions/base.csv.gz'
    args.num_threads = max(args.num_threads)

    model = BaselineModel()

    start = time.time()
    model.preprocess(args.num_threads, args.verbose)
    end = time.time()
    if args.time:
        print('Counting tracks time:', end - start)
        print('-' * 80)

    start = time.time()
    model.recommend(args.submit_path, args.num_threads, args.verbose)
    end = time.time()
    if args.time:
        print('Recommending time:', end - start)
        print('-' * 80)



def neighbour(args):
    # preprocess some arguments
    args.submit_path = args.submit_path if args.submit_path else f'submissions/{args.hood}.csv.gz'
    args.num_threads = (args.num_threads, args.num_threads) if len(args.num_threads) == 1 else args.num_threads[:2]
    [create_folder(path) for path in [args.matrix_path, args.train_path, args.test_path, args.trackmap_path]]

    if 'sparsify' in args.action:
        start = time.time()
        sparse = Sparse(train_path=args.train_path, test_path=args.test_path, trackmap_path=args.trackmap_path)
        sparse.preprocess(max(args.num_threads), args.verbose)
        end = time.time()
        if args.time:
            print('Preprocessing time:', end - start)
            print('-' * 80)

    if 'recommend' in args.action:
        start = time.time()
        model = NeighbourModel(args.hood, args.k,
                               train_path=args.train_path, test_path=args.test_path, trackmap_path=args.trackmap_path)
        model.recommend(submit_path=args.submit_path, num_threads=args.num_threads, batch_size=args.batch_size,
                        matrix_path=args.matrix_path, load=args.load, verbose=args.verbose)
        end = time.time()
        if args.time:
            print('Recommending time:', end - start)
            print('-' * 80)

def puresvd(args):
    # preprocess some arguments
    if not args.submit_path:
        args.submit_path = f'submissions/puresvd{int(args.ftest)}-{args.h}.csv.gz'
    args.num_threads = max(args.num_threads)
    for path in [args.train_path, args.test_path, args.trackmap_path, args.U_path, args.S_path, args.V_path]:
        create_folder(path)

    if 'sparsify' in args.action:
        start = time.time()
        sparse = Sparse(train_path=args.train_path, test_path=args.test_path, trackmap_path=args.trackmap_path)
        sparse.preprocess(max(args.num_threads), args.verbose)
        end = time.time()
        if args.time:
            print('Preprocessing time:', end - start)
            print('-' * 80)

    if 'recommend' in args.action:
        start = time.time()
        model = PureSVDModel(h=args.h, use_test=args.ftest,
                             train_path=args.train_path, test_path=args.test_path, trackmap_path=args.trackmap_path)
        model.factorize(U_path=args.U_path, S_path=args.S_path, V_path=args.V_path, verbose=args.verbose)
        model.recommend(submit_path=args.submit_path, num_threads=args.num_threads, batch_size=args.batch_size,
                        verbose=args.verbose)
        end = time.time()
        if args.time:
            print('Recommending time:', end - start)
            print('-' * 80)


def track2vec(args):
    # preprocess some arguments
    if not args.submit_path:
        args.submit_path = f'submissions/track2vec.csv.gz'
    for path in [args.train_path, args.test_path, args.trackmap_path, args.model_path]:
        create_folder(path)

    model = Track2VecModel(embed_dim=args.embed_dim, context_size=args.context_size, k=args.k,
                           model_path=args.model_path, train_path=args.train_path, test_path=args.test_path,
                           trackmap_path=args.trackmap_path, S_path=args.S_path)

    i = 0
    if 'train' in args.action:
        start = time.time()
        model.train(num_epochs=args.num_epochs, num_threads=args.num_threads[i], verbose=args.verbose)
        end = time.time()
        if args.time:
            print('Training time:', end-start)
            print('-' * 80)
        i += 1
    else:
        model.load()

    if 'recommend' in args.action:
        start = time.time()
        model.recommend(submit_path=args.submit_path, num_threads=args.num_threads[i],
                        batch_size=args.batch_size, verbose=args.verbose)
        end = time.time()
        if args.time:
            print('Recommending time:', end-start)
            print('-' * 80)



def add_global(subparser):
    subparser.add_argument('-eval', action='store_true', default=False, help='whether to output the evaluation')
    subparser.add_argument('-t', '--time', action='store_true', default=False, help='whether to display execution times')
    subparser.add_argument('-v', '--verbose', action='store_true', default=False, help='whether to display the trace of recommendation process')
    subparser.add_argument('-p', '--submit_path', default=None, help='where to store the submission')
    subparser.add_argument('-n', '--num_threads', type=int, nargs='*', default=(8, MAX_THREADS), help='number of threads to parallelize the execution')

def add_paths(subparser):
    subparser.add_argument('--train_path', type=str, default='data/Rtrain.npz', help='Path to store sparse train matrix')
    subparser.add_argument('--test_path', type=str, default='data/Rtest.npz', help='Path to store sparse test matrix')
    subparser.add_argument('--trackmap_path', type=str, default='data/trackmap.pickle', help='Path to store map from track URIs to indices')


if __name__ == '__main__':
    parser = ArgumentParser(description='Test our recommender systems for the Spotify Million Playlist Dataset')
    # global arguments (these arguments are used for all models)
    parser.add_argument('-eval', action='store_true', default=False, help='whether to output the evaluation')
    parser.add_argument('-t', '--time', action='store_true', default=False, help='whether to display execution times')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='whether to display the trace of recommendation process')
    parser.add_argument('-p', '--submit_path', default=None, help='where to store the submission')
    parser.add_argument('-n', '--num_threads', type=int, nargs='*', default=(8, MAX_THREADS), help='number of threads to parallelize the execution')

    subparsers = parser.add_subparsers(title='Models', dest='model')

    baseline_parser = subparsers.add_parser('base', help='Baseline model based on popularity')
    eval_parser = subparsers.add_parser('eval', help='Evaluator model')

    # add neighbour model arguments
    neighbour_parser = subparsers.add_parser('neighbour', help='Neighbour model based on user and item similarity')
    neighbour_parser.add_argument('hood', choices=['user', 'item'], default='user', type=str, help='neighbourhood to user')
    neighbour_parser.add_argument('--action', choices=['sparsify', 'recommend'], type=str, nargs='*', default=['recommend'])
    neighbour_parser.add_argument('--k', type=int, default=100)
    neighbour_parser.add_argument('--batch_size', type=int, default=500)
    neighbour_parser.add_argument('--matrix_path', type=str, default='data/Rest.npz')
    neighbour_parser.add_argument('--load', action='store_true', default=False)
    add_paths(neighbour_parser)

    # add puresvd arguments
    puresvd_parser = subparsers.add_parser('puresvd', help='PureSVD model')
    puresvd_parser.add_argument('--action', choices=['sparsify', 'recommend'], type=str, nargs='*', default=['recommend'])
    puresvd_parser.add_argument('--h', type=int, default=10)
    puresvd_parser.add_argument('-ftest', action='store_true', default=False, help='whether to factorize using test sparse matrix')
    puresvd_parser.add_argument('--batch_size', type=int, default=100)
    puresvd_parser.add_argument('--U_path', type=str, default='data/U.npy')
    puresvd_parser.add_argument('--V_path', type=str, default='data/V.npy')
    puresvd_parser.add_argument('--S_path', type=str, default='data/S.npy')
    add_paths(puresvd_parser)

    # add track2vec arguments
    track2vec_parser = subparsers.add_parser('track2vec', help='Track2Vec model')
    track2vec_parser.add_argument('action', choices=['train', 'recommend'], default='recommend', type=str, nargs='*', help='Action to perform')
    track2vec_parser.add_argument('--embed_dim', type=int, default=50, help='Vector size used in the space model')
    track2vec_parser.add_argument('--context_size', type=int, default=10, help='Context size used for training vector weights')
    track2vec_parser.add_argument('--k', type=int, default=10, help='Neighbourhood size')
    track2vec_parser.add_argument('--model_path', type=str, default='data/track2vec', help='Path to store Gensim model')
    track2vec_parser.add_argument('--S_path', type=str, default='data/S-track2vec.npz')
    track2vec_parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs in training')
    track2vec_parser.add_argument('--granularity', type=int, default=int(10e3), help='Number of trees for the Annoy Index')
    track2vec_parser.add_argument('--batch_size', type=int, default=100)
    add_paths(track2vec_parser)


    for subparser in [eval_parser, baseline_parser, neighbour_parser, puresvd_parser, track2vec_parser]:
        add_global(subparser)

    args, unknown = parser.parse_known_args()

    if args.model == 'base':
        base(args)
    elif args.model == 'neighbour':
        neighbour(args)
    elif args.model == 'puresvd':
        puresvd(args)
    elif args.model == 'track2vec':
        track2vec(args)
    elif args.model == 'eval':
        evaluator(args)

    if args.eval:
        evaluator(args)















