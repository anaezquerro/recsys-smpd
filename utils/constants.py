import os

INFO_ROW = 'team_info, PedroAna, ana.ezquerro@udc.es'
TRAIN_FOLDER = 'spotify_train_dataset/data/'
MAX_THREADS = os.cpu_count()
TEST_FILE = 'spotify_test_playlists/test_input_playlists.json'
GOLD_FILE = 'spotify_test_playlists/test_eval_playlists.json'
N_RECS = 500
SUBMISSION_FOLDER = 'submissions/'
DEFAULT_SUBMISSION = f'{SUBMISSION_FOLDER}/submission.csv.gz'