import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, Union

from resources.coordinatesystem import *
from resources.actions import *
from resources.states import *
from resources.rng import driver_rng


class Driver:
    def __init__(self, name, print_info=False):
        self._name = name
        self.print_info = print_info

    @property
    def name(self):
        return self._name

    def prepare_for_race(self):
        raise NotImplementedError

    def choose_tyres(self, track_info):         # for level 4 - Pro Driver
        return None

    def choose_aero(self, track_info):          # for level 4 - Pro Driver
        return None

    def make_a_move(self, car_state: CarState, track_state: TrackState) -> Action:
        raise NotImplementedError

    def update_with_action_results(self, *args, **kwargs):
        raise NotImplementedError

    def update_after_race(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_driver_class_for_level(level: Level):
        if Level.Learner == level:
            from drivers.learnerdriver import LearnerDriver
            return LearnerDriver
        elif Level.Young == level:
            from drivers.youngdriver import YoungDriver
            return YoungDriver
        else:
            raise ValueError(f'Driver not defined for level {level.name}')
