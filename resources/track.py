import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import typing
import os
import json

from resources.states import TrackState, Level, TrackInfo
from resources.coordinatesystem import Heading, Position
from resources.actions import Action


class Track:
    def __init__(self, track_map: np.ndarray, start_position: Position = None, start_heading: Heading = None,
                 correct_turns=None, straight_lengths=None, on_track_points=None, drs_points=None):

        self.map = track_map
        self.start_position = start_position if start_position is not None else Position(0, 0)
        self.start_heading = start_heading if start_heading is not None else Heading(0, 1)
        self.on_track_points = on_track_points
        self.correct_turns = correct_turns
        self.straight_lengths = np.array(straight_lengths) if straight_lengths is not None else None
        # 2D array of row, column co-ordinates of DRS activation points
        self.drs_points = np.array(drs_points) if drs_points is not None else None

    @property
    def num_rows(self):
        return self.map.shape[0]

    @property
    def num_cols(self):
        return self.map.shape[1]

    def get_start_position_and_heading(self):
        return self.start_position, self.start_heading

    def get_state_for_position(self, position: Position, heading: Heading):
        row = min(position.row, self.num_rows - 1)
        col = min(position.column, self.num_cols - 1)

        def find_distance(map_slice, edge_distance):
            distance = np.nonzero(map_slice)[0]
            if len(distance) == 0:
                distance = edge_distance
            else:
                distance = distance[0]
            return distance

        # Get distances to nearest walls in 4 directions in track co-ordinates
        track_right = find_distance(self.map[row, col+1:], self.num_cols - 1 - col)
        track_left = 0 if 0 == col else find_distance(self.map[row, col-1::-1], col)
        track_down = find_distance(self.map[row+1:, col], self.num_rows - 1 - row)
        track_up = 0 if 0 == row else find_distance(self.map[row-1::-1, col], row)

        # Rotate into car heading. "heading" is a unit vector in (row, column)
        ahead, left, behind, right = heading.rotate_from_track_to_car(track_left=track_left, track_right=track_right,
                                                                      track_up=track_up, track_down=track_down)

        # DRS
        if self.drs_points is not None:
            drs_available = np.any(np.all(np.array([position.row, position.column]) == self.drs_points, axis=1))
        else:
            drs_available = False

        return TrackState(distance_ahead=ahead, distance_behind=behind, distance_left=left, distance_right=right,
                          position=position, drs_available=drs_available)

    def get_new_position(self, start_position: Position, speed, heading: Heading):
        crashed = False
        if 0 == speed:
            new_position = start_position
        else:
            try:
                new_position = start_position.get_new_position(heading, 1)
            except ValueError:
                crashed = True

        # Check if crashed
        if crashed or not self.on_track(new_position) or self.map[new_position.row, new_position.column] != 0:
            crashed = True
            new_position = start_position
        else:
            crashed = False

        # Check if finished
        finished = self.is_finished(new_position)

        return new_position, crashed, finished

    def plot_track(self, ax=None, position=None, show_drs=True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        else:
            ax.clear()

        image = ax.imshow(self.map, cmap='binary')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(np.arange(0.5, self.num_cols), minor=True)
        ax.set_yticks(np.arange(0.5, self.num_rows), minor=True)
        ax.tick_params(axis='both', which='minor', bottom=False, left=False, top=False, right=False)
        ax.grid(which='minor', alpha=0.3)

        ax.plot(self.start_position.column, self.start_position.row, 'bs')
        ax.plot(self.num_cols - 1, self.num_rows - 1, 'g*')
        if self.drs_points is not None and show_drs:
            ax.plot(self.drs_points[:, 1], self.drs_points[:, 0], 'c^')

        if position is not None:
            ax.plot(position.column, position.row, 'r.', ms=20)
        return ax, image

    def on_track(self, position: Position):
        return (0 <= position.row < self.num_rows) and (0 <= position.column < self.num_cols)

    def is_finished(self, position: Position):
        return position.row == (self.num_rows - 1) and position.column == (self.num_cols - 1)

    def get_track_info(self):
        if self.straight_lengths is None:
            return TrackInfo(length=0, number_of_straights=0, shortest_straight=0, longest_straight=0,
                             average_straight=0.0)

        return TrackInfo(length=int(np.sum(self.straight_lengths)), number_of_straights=len(self.straight_lengths),
                         shortest_straight=int(np.min(self.straight_lengths)),
                         longest_straight=int(np.max(self.straight_lengths)),
                         average_straight=float(np.mean(self.straight_lengths)))

    def to_json_compatible(self):
        return {
            'map': self.map.tolist(),
            'start_position': self.start_position.to_json_compatible(),
            'start_heading': self.start_heading.to_json_compatible(),
            'correct_turns': [(position.to_json_compatible(), action.name)
                              for position, action in self.correct_turns.items()]
            if self.correct_turns is not None else None,
            'straight_lengths': [int(l) for l in self.straight_lengths] if self.straight_lengths is not None else None,
            'on_track_points': self.on_track_points.tolist() if self.on_track_points is not None else None,
            'drs_points': self.drs_points.tolist() if self.drs_points is not None else None
        }

    @staticmethod
    def from_json_compatible(dict_from_json, level: Level):
        # Apply adjustments for different levels
        if level in [Level.Learner, Level.Young]:
            dict_from_json['drs_points'] = None

        return Track(
            track_map=np.array(dict_from_json['map']),
            start_position=Position.from_json_compatible(dict_from_json['start_position']),
            start_heading=Heading.from_json_compatible(dict_from_json['start_heading']),
            correct_turns={Position.from_json_compatible(p): Action.from_name(a)
                           for p, a in dict_from_json['correct_turns']}
                          if dict_from_json['correct_turns'] is not None else None,
            straight_lengths=dict_from_json.get('straight_lengths', None),
            on_track_points=np.array(dict_from_json.get('on_track_points', None)),
            drs_points=dict_from_json.get('drs_points', None)
        )


class TrackStore:
    store_filename = os.path.join(os.path.dirname(__file__), 'track_store.json')
    level_to_key = {Level.Learner: Level.Learner.name, Level.Young: Level.Young.name, Level.Rookie: Level.Young.name,
                    Level.Pro: Level.Young.name}

    @classmethod
    def read_store(cls):
        # Read in any existing tracks, keep them in their JSON dict form
        if os.path.isfile(cls.store_filename):
            with open(cls.store_filename, 'r') as f:
                dict_of_tracks_json = json.load(f)
        else:
            print('Warning: Did not find track store json')
            dict_of_tracks_json = {}

        return dict_of_tracks_json

    @classmethod
    def save_track(cls, track: Track, level: Level):
        # Load from cache
        dict_of_tracks_json = cls.read_store()
        key = cls.level_to_key[level]

        # Check if level is in store and create empty list if not
        if key not in dict_of_tracks_json:
            dict_of_tracks_json[key] = []

        # Append new track
        dict_of_tracks_json[key].append(track.to_json_compatible())

        # Write back to file
        with open(cls.store_filename, 'w') as f:
            json.dump(dict_of_tracks_json, f)

    @classmethod
    def load_all_tracks(cls, level: Level) -> typing.List[Track]:
        # Load from store
        dict_of_tracks_json = cls.read_store()
        key = cls.level_to_key[level]

        return [Track.from_json_compatible(json_dict, level) for json_dict in dict_of_tracks_json.get(key, [])]

    @classmethod
    def load_track(cls, level: Level, index=None) -> Track:
        # Load from store
        dict_of_tracks_json = cls.read_store()
        key = cls.level_to_key[level]
        list_of_tracks_json = dict_of_tracks_json.get(key, [])

        if index is None:
            track_json = np.random.choice(list_of_tracks_json)
        else:
            track_json = list_of_tracks_json[index - 1]

        return Track.from_json_compatible(track_json, level)

    @classmethod
    def clear_cache(cls, level):
        dict_of_tracks_json = cls.read_store()
        key = cls.level_to_key[level]
        dict_of_tracks_json[key] = []

        # Write back to file
        with open(cls.store_filename, 'w') as f:
            json.dump(dict_of_tracks_json, f)

    @classmethod
    def get_number_of_tracks(cls, level: Level):
        # Load from store
        dict_of_tracks_json = cls.read_store()
        key = cls.level_to_key[level]

        return len(dict_of_tracks_json.get(key, []))

    @classmethod
    def plot_all_tracks(cls, level):
        tracks = cls.load_all_tracks(level)

        fig = plt.figure(figsize=(9, 5))
        for i in range(len(tracks)):
            ax = fig.add_subplot(4, 6, i + 1)
            tracks[i].plot_track(ax=ax)
        fig.tight_layout()


if __name__ == '__main__':
    TrackStore.plot_all_tracks(Level.Rookie)
