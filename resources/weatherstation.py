import numpy as np
import json
import os
from resources.states import WeatherState
from resources.rng import rng


data_filename = os.path.join(os.path.dirname(__file__), 'weather_data.json')
with open(data_filename, 'r') as f:
    weather_data_cache = {key: np.array(data) for key, data in json.load(f).items()}
track_water_level = weather_data_cache.pop('track_water_level')
track_grip_cache = 1 - 0.8 * track_water_level / 100


class WeatherStation:

    def __init__(self):
        self.weather_data = weather_data_cache
        self.track_grip = track_grip_cache
        self.current_index = 0

    def prepare_for_race(self):
        self.current_index = rng().choice(np.where(np.array(self.track_grip)[:-1500] == 1)[0], 1)[0]

    def get_state(self):
        state = WeatherState(**{key: value[self.current_index] for key, value in self.weather_data.items()})
        return state

    def get_track_grip(self):
        return self.track_grip[self.current_index]

    def update(self):
        self.current_index += 1
        if self.current_index >= len(self.track_grip):
            self.current_index = 0