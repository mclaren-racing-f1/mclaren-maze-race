from enum import Enum


class BaseEnum(Enum):
    @classmethod
    def from_name(cls, name):
        if name in cls.__members__:
            return getattr(cls, name)
        else:
            return None

    def to_json_compatible(self):
        return self.name


class Action(BaseEnum):
    Continue = 0
    LightThrottle = 1
    FullThrottle = 2
    LightBrake = -1
    HeavyBrake = -2
    TurnLeft = 3
    TurnRight = 4
    OpenDRS = 5
    ChangeTyres = 6

    @staticmethod
    def get_all_actions():
        return list(Action.__members__.values())

    @staticmethod
    def get_sl_actions():
        return [action for action in Action.__members__.values()
                if action not in [Action.TurnLeft, Action.TurnRight, Action.OpenDRS, Action.ChangeTyres]]


class TyreChoice(BaseEnum):
    Soft = 'soft'
    Medium = 'medium'
    Hard = 'hard'
    Learner = 'learner'             # only for use in levels 1-3

    @staticmethod
    def get_choices():
        return [TyreChoice.Soft, TyreChoice.Medium, TyreChoice.Hard]


class AeroSetup(BaseEnum):
    HighDownforce = 'high_downforce'
    Balanced = 'balanced'
    LowDrag = 'low_drag'
