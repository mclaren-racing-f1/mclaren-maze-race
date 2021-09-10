from dataclasses import dataclass
from enum import Enum
import numpy as np
from resources.coordinatesystem import Position, Heading
from resources.actions import TyreChoice


@dataclass
class State:
    def to_json_compatible(self):
        json_dict = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_json_compatible'):
                value = value.to_json_compatible()
            if isinstance(value, (np.int64, np.int32)):
                value = int(value)
            elif isinstance(value, np.bool_):
                value = bool(value)

            json_dict[key] = value
        return json_dict

    @classmethod
    def from_json_compatible(cls, json_dict: dict):
        if 'position' in json_dict:
            json_dict['position'] = Position.from_json_compatible(json_dict['position'])
        if 'heading' in json_dict:
            json_dict['heading'] = Heading.from_json_compatible(json_dict['heading'])
        return cls(**json_dict)


@dataclass
class TrackState(State):
    distance_ahead: int
    distance_left: int
    distance_right: int
    distance_behind: int
    position: Position
    drs_available: bool = False
    safety_car_active: bool = False


@dataclass
class CarState(State):
    speed: float
    heading: tuple
    drs_active: bool = False
    tyre_choice: TyreChoice = TyreChoice.Learner
    tyre_grip: float = 1.0
    tyre_age: int = 0


@dataclass
class WeatherState(State):
    air_temperature: float = 0.0
    track_temperature: float = 0.0
    humidity: float = 0.0
    rain_intensity: float = 0.0


@dataclass
class ActionResult(State):
    turned_ok: bool
    crashed: bool
    spun: bool
    finished: bool
    safety_car_speed_exceeded: bool = False
    safety_car_penalty_level: int = 0


class Level(Enum):
    Learner = 1
    Young = 2
    Rookie = 3
    Pro = 4
    Champion = 5


@dataclass()
class TrackInfo(State):
    length: int
    number_of_straights: int
    shortest_straight: int
    longest_straight: int
    average_straight: float
