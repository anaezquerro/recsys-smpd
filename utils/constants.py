import os

INFO_ROW = 'team_info, PedroAna, ana.ezquerro@udc.es'
TRAIN_FOLDER = 'spotify_million_playlist_dataset/train/'
MAX_THREADS = os.cpu_count()
INPUT_FILE = 'spotify_million_playlist_dataset/input.json'
GOLD_FILE = 'spotify_million_playlist_dataset/gold.json'
N_RECS = 500
SUBMISSION_FOLDER = 'submissions/'
DEFAULT_SUBMISSION = f'{SUBMISSION_FOLDER}/submission.csv.gz'