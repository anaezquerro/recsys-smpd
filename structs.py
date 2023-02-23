from typing import List, Dict
import json

class Track:
    def __init__(self, track_uri: str):
        self.track_uri = track_uri
        self.occurrences = 0

    def add_occurrence(self):
        self.occurrences += 1

    def __eq__(self, other):
        return self.track_uri == other.track_uri

    def __add__(self, other):
        new_track = Track(self.track_uri)
        new_track.occurrences = self.occurrences + other.occurrences
        return new_track

    def __repr__(self):
        return self.track_uri

    @classmethod
    def from_json(cls, djson: dict):
        return Track(djson['track_uri'])


class Playlist:
    def __init__(self, pid: int, n_tracks: int):
        self.pid = pid
        self.n_tracks = n_tracks



class Reader:
    def __init__(self, path: str, only_uri: bool = False):
        self.path = path
        if only_uri:
            self.func = lambda track: track['track_uri']
        else:
            self.func = lambda track: Track.from_json(track)

    def read(self) -> Dict[int, List[Track] | str]:
        data = json.load(open(self.path, 'r', encoding='utf8'))['playlists']
        playlists = dict()
        for playlist in data:
            playlists[int(playlist['pid'])] = list(map(self.func, playlist['tracks']))
        return playlists

