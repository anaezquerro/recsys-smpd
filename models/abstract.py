


class Model:
    def __init__(self):
        pass

    def preprocess(self, num_threads: int, verbose: bool):
        raise NotImplementedError

    def recommend(self, submit_path: str, num_threads: int, verbose: bool):
        raise NotImplementedError

