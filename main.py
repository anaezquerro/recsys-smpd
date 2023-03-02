import sys
from evaluation import *
from baseline import *

if __name__ == '__main__':
    args = sys.argv[1:]

    if '-pred' or '-both' in args:
        model = BaselineModel()

        # training (counting track occurrences)
        start = time.time()
        tracks = model.train()
        end = time.time()

        # prediction
        start = time.time()
        model.predict('submissions/baseline.csv.gz')
        end = time.time()

        if '-t' in args:
            print('Training time:', (end - start))
            print('Prediction time:', (end-start))

    if '-eval' or '-both' in args:
        evaluator = Evaluator('submissions/baseline.csv.gz')

        evaluator.read_submission()

        print(f'RPrecision with popularity model {evaluator.RPrecision()}')
        print(f'NDCG with popularity model {evaluator.RPrecision()}')
        print(f'Clicks with popularity model {evaluator.clicks()}')


