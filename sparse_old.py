def toCSV(N: int = 1000, save_list: bool = True):
    pids_list = list() if save_list else None

    paths = os.listdir(TRAIN_FOLDER)[:N]

    relations = dict()
    indexes = open('indexes.csv', 'w')
    rel_csv = open('relations.csv', 'w')
    for path in paths:
        playlists = json.load(open(TRAIN_FOLDER + '/' + path, 'r', encoding='utf8'))['playlists']
        for i, playlist in enumerate(playlists):
            tracks = set(map(lambda track: track['track_uri'], playlist['tracks']))
            for track in tracks:
                try:
                    relations[track]
                except KeyError:
                    relations[track] = len(relations)
                    rel_csv.write(f"{relations[track]},{track}\n")
                indexes.write(f"{i},{relations[track]}\n")
            if save_list:
                pids_list.append(playlist['pid'])

    rel_csv.close()
    indexes.close()

    return relations, pids_list


def csv2sparse(csv_path):
    df = pd.read_csv(csv_path, names=['playlist', 'track'])

    row = np.array(df.playlist)
    col = np.array(df.track)
    data = np.ones(len(df.playlist))

    matrix = sparse.csr_matrix((data, (row, col)))

    return matrix


def test2sparse(relations: Dict[str, int]):
    playlists = json.load(open(TEST_FILE, encoding='utf8'))['playlists']
    rows = list()
    cols = list()
    for i, playlist in enumerate(playlists):
        tracks_indices = list()
        tracks = set(map(lambda track: track['track_uri'], playlist['tracks']))
        for track in tracks:
            try:
                tracks_indices.append(relations[track])
            except KeyError:
                continue
        cols += tracks_indices
        rows += [i] * len(tracks_indices)

    rows, cols = np.array(rows), np.array(cols)
    data = np.ones(len(rows))
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(playlists), len(relations)))
    return matrix

