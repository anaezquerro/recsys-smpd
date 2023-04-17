import os.path
from argparse import ArgumentParser
from utils.constants import SUBMISSION_FOLDER, MAX_THREADS
from utils.evaluation import Evaluator
from utils.tools import create_folder
import time
from models.neighbour import NeighbourModel
from models.baseline import BaselineModel
from utils.sparse import Sparse


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
    args.num_threads = max(args.num_threads)

    from models.baseline import BaselineModel
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
        model = NeighbourModel(args.model, args.k,
                               train_path=args.train_path, test_path=args.test_path, trackmap_path=args.trackmap_path)
        model.recommend(submit_path=args.submit_path, num_threads=args.num_threads, batch_size=args.batch_size,
                        matrix_path=args.matrix_path, load=args.load, verbose=args.verbose)
        end = time.time()
        if args.time:
            print('Recommending time:', end - start)
            print('-' * 80)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model', choices=['base', 'user', 'item'], default=None, nargs='?')
    parser.add_argument('-eval', action='store_true', default=False)

    # global arguments
    parser.add_argument('-t', '--time', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-p', '--submit_path', default=None)
    parser.add_argument('-n', '--num_threads', type=int, nargs='*', default=(8, MAX_THREADS))

    # neighbour arguments
    parser.add_argument('--action', choices=['sparsify', 'recommend'], type=str, nargs='*', default=['recommend'])
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--train_path', type=str, default='data/Rtrain.npz')
    parser.add_argument('--test_path', type=str, default='data/Rtest.npz')
    parser.add_argument('--trackmap_path', type=str, default='data/trackmap.pickle')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--matrix_path', type=str, default='data/Rest.npz')
    parser.add_argument('--load', action='store_true', default=False)

    args = parser.parse_args()
    args.submit_path = f'{SUBMISSION_FOLDER}/{args.model}.csv.gz' if not args.submit_path else args.submit_path
    create_folder(args.submit_path)

    if args.model == 'base':
        base(args)
    elif args.model in ['item', 'user']:
        neighbour(args)
    if args.eval:
        evaluator(args)







