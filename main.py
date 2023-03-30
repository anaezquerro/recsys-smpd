import argparse
from evaluation import *
from baseline import *
from neighbour import *

SUBMISSION_PATH = 'submissions.csv.gz'

def prediction(args):

    if args.model == 'base':
        model = BaselineModel(num_threads=args.num_threads_pred)

        # training (counting track occurrences)
        print("Counting track ocurrences...")
        start = time.time()
        model.train()
        end = time.time()
        train_time = start - end

        # prediction
        print("Creating predictions...")
        start = time.time()
        model.predict(args.path)
        end = time.time()

        if args.time:
            print('Training time:', train_time)
            print('Prediction time:', (end - start), '\n\n')

    else:
        model = NeighbourModel(k=args.k, batch_size=args.batch_size, num_threads=args.num_threads_pred)

        start = time.time()
        Rest, popular = model.predict(args.model)
        model.recommend(Rest, popular, submit_path=args.path)
        end = time.time()

        if args.time:
            print('Prediction time:', (end - start), '\n\n')


def evaluation(args):
    print("Evaluating model...")
    evaluator = Evaluator(args.path)

    start = time.time()
    evaluator.read_submission()
    print(f'RPrecision: {evaluator.RPrecision()}')
    print(f'NDCG: {evaluator.NDCG()}')
    print(f'Clicks: {evaluator.clicks()}')
    end = time.time()

    if args.time:
        print('\nEvaluation time:', (end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['base', 'user', 'item'], default='base')
    parser.add_argument('-a', '--action', choices=['pred', 'eval'], default='both')
    parser.add_argument('-t', '--time', action='store_true')
    parser.add_argument('-k', '--k', type=int, default=100)
    parser.add_argument('-p', '--path', default=SUBMISSION_PATH)
    parser.add_argument('-b', '--batch_size', type=int, default=500)
    parser.add_argument('-n', '--num_threads_pred', type=int, default=MAX_THREADS)
    parser.add_argument('--num_threads_rec', type=int, default=MAX_THREADS)

    args = parser.parse_args()

    if args.action == 'pred':
        prediction(args)
    elif args.action == 'eval':
        evaluation(args)
    else:
        prediction(args)
        evaluation(args)



