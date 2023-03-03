import sys
from evaluation import *
from baseline import *

SUBMISSION_PATH = 'baseline.csv.gz'

def prediction(submission_path: str, print_time: bool):
    model = BaselineModel()

    # training (counting track occurrences)
    print("Counting track ocurrences")
    start = time.time()
    model.train()
    end = time.time()
    train_time = end - start

    # prediction
    print("Creating predictions")
    start = time.time()
    model.predict('submissions/baseline.csv.gz')
    end = time.time()

    if print_time:
        print('Training time:', train_time)
        print('Prediction time:', (end - start))

def evaluation(submission_path: str, print_time: bool):
    print("Evaluating popularity model")
    evaluator = Evaluator('submissions/baseline.csv.gz')

    start = time.time()
    evaluator.read_submission()
    print(f'RPrecision with popularity model {evaluator.RPrecision()}')
    print(f'NDCG with popularity model {evaluator.RPrecision()}')
    print(f'Clicks with popularity model {evaluator.clicks()}')
    end = time.time()

    if print_time:
        print('Evaluating time:', (end - start))

if __name__ == '__main__':
    print_time = 't' in sys.argv
    submission_path = SUBMISSION_PATH if not '-out' in sys.argv else sys.argv[sys.argv.index('-out') + 1]

    if '-pred' or '-both' in sys.argv:
        prediction(submission_path, print_time)
    if '-eval' or '-both' in sys.argv:
        evaluation(submission_path, print_time)

    if any([token in sys.argv for token in ['-pred', '-eval', '-both']]):
        prediction(submission_path, print_time)
        evaluation(submission_path, print_time)



