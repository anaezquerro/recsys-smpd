import sys
from evaluation import *
from baseline import *

def prediction(print_time=True):
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

def evaluation(print_time=True):
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

    if '-both' in sys.argv:
        prediction('-t' in sys.argv)
        evaluation('-t' in sys.argv)
    elif '-pred' in sys.argv:
        prediction('-t' in sys.argv)
    elif '-eval' in sys.argv:
        evaluation('-t' in sys.argv)
    else:
        prediction()
        evaluation()


