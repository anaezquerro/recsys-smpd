import sys
from evaluation import *
from baseline import *

SUBMISSION_PATH = 'baseline.csv.gz'

def prediction(submission_path: str, print_time: bool):
    model = BaselineModel()

    # training (counting track occurrences)
    print("Counting track ocurrences...")
    start = time.time()
    model.train()
    end = time.time()
    train_time = end - start

    # prediction
    print("Creating predictions...")
    start = time.time()
    model.predict(submission_path)
    end = time.time()

    if print_time:
        print('Training time:', train_time)
        print('Prediction time:', (end - start), '\n\n')

def evaluation(submission_path: str, print_time: bool):
    print("Evaluating popularity model...")
    evaluator = Evaluator(submission_path)

    start = time.time()
    evaluator.read_submission()
    print(f'RPrecision with popularity model: {evaluator.RPrecision()}')
    print(f'NDCG with popularity model: {evaluator.NDCG()}')
    print(f'Clicks with popularity model: {evaluator.clicks()}')
    end = time.time()

    if print_time:
        print('\nEvaluation time:', (end - start))

if __name__ == '__main__':
    print_time = '-t' in sys.argv
    submission_path = SUBMISSION_PATH if not '-out' in sys.argv else sys.argv[sys.argv.index('-out') + 1]

    if '-pred' in sys.argv:
        prediction(submission_path, print_time)
    elif '-eval' in sys.argv:
        evaluation(submission_path, print_time)
    else:
        prediction(submission_path, print_time)
        evaluation(submission_path, print_time)



