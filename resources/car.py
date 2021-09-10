import numpy as np
from scipy.interpolate import interp1d
import os
import json
from time import time as time_fn

from resources.actions import *
from resources.states import CarState, Level
from resources.coordinatesystem import Position, Heading
from resources.rng import rng

CREATED_DYNAMICS = {}


class CarDynamicsModel:
    def __init__(self, top_speed=300, tyre_multiplier=1.0, downforce_multiplier=1.0):
        self._top_speed = top_speed

        # Cornering
        self.tyre_multiplier = tyre_multiplier
        self.downforce_multiplier = downforce_multiplier
        self.base_max_cornering_speed = (140 + 20 * rng().rand())

        # DRS
        self.drs_multiplier = 1.15 + 0.05*np.random.rand()

    def top_speed(self, current_speed, drs_active=False, drag_multiplier=1.0):
        if drs_active:
            top_speed = self._top_speed * self.drs_multiplier
        else:
            top_speed = self._top_speed

        return max(current_speed, top_speed / drag_multiplier)

    def max_cornering_speed(self, grip_multiplier=1.0):
        return self.base_max_cornering_speed * grip_multiplier

    def full_throttle(self, current_speed):
        raise NotImplementedError

    def light_throttle(self, current_speed):
        raise NotImplementedError

    def light_brake(self, current_speed):
        raise NotImplementedError

    def heavy_brake(self, current_speed):
        raise NotImplementedError

    def can_turn(self, current_speed, grip_multiplier=1.0):
        return current_speed < self.max_cornering_speed(grip_multiplier)


class CarDynamicsModel1(CarDynamicsModel):
    def __init__(self, top_speed=300, tyre_multiplier=1.0, downforce_multiplier=1.0):
        super().__init__(top_speed, tyre_multiplier, downforce_multiplier)

        # Full throttle - piecewise linear
        self.ft_delta_speed_at_0 = 5 + 10 * rng().rand()                     # 5 to 15 kph - wheel spin at low speed
        self.ft_input_speed_at_peak = 150 + 50 * rng().rand()            # peak acceleration between 150 & 200kph
        self.ft_peak_delta = 60 + 20 * rng().rand()

        # Light throttle - simple linear
        self.lt_delta_speed_at_0 = 40 + 10 * rng().rand()                   # 40 to 50 kph - avoid wheel spin

        # Heavy braking - piecewise linear
        self.hb_input_speed_at_peak = 200 + 50*rng().rand()                 # 200 to 250kph
        self.hb_peak_delta = 75 + 40*rng().rand()
        self.hb_full_lock_speed = 295 + 10*rng().rand()                     # heavy braking does nothing above this

        # Light braking - proportional
        self.lb_multiplier = 0.8 + 0.1*rng().rand()

    def full_throttle(self, current_speed, drs_active=False, grip_multiplier=1.0, drag_multiplier=1.0):
        top_speed = self.top_speed(current_speed, drs_active, drag_multiplier)
        if current_speed > self.ft_input_speed_at_peak:
            abs_gradient = self.ft_peak_delta / (top_speed - self.ft_input_speed_at_peak)
            delta = self.ft_peak_delta - abs_gradient * (current_speed - self.ft_input_speed_at_peak)
        else:
            abs_gradient = (self.ft_peak_delta - self.ft_delta_speed_at_0) / self.ft_input_speed_at_peak
            delta = self.ft_delta_speed_at_0 + abs_gradient * current_speed
        delta *= grip_multiplier

        return min(current_speed + delta, top_speed)

    def light_throttle(self, current_speed, drs_active=False, grip_multiplier=1.0, drag_multiplier=1.0):
        top_speed = self.top_speed(current_speed, drs_active, drag_multiplier)
        abs_gradient = self.lt_delta_speed_at_0 / top_speed
        delta = self.lt_delta_speed_at_0 - current_speed * abs_gradient
        delta *= grip_multiplier

        return min(current_speed + delta, top_speed)

    def heavy_brake(self, current_speed, drs_active=False, grip_multiplier=1.0, drag_multiplier=1.0):
        if current_speed > self.hb_full_lock_speed:
            delta = 0
        elif current_speed > self.hb_input_speed_at_peak:
            abs_gradient = self.hb_peak_delta / (self.hb_full_lock_speed - self.hb_input_speed_at_peak)
            delta = self.hb_peak_delta - (current_speed - self.hb_input_speed_at_peak) * abs_gradient
        else:
            delta = self.hb_peak_delta
        delta *= grip_multiplier

        return max(current_speed - delta, 0)

    def light_brake(self, current_speed, grip_multiplier=1):
        delta = current_speed * (1 - self.lb_multiplier)
        return max(current_speed  - delta*grip_multiplier, 0)


# --------------------------------------------------------------------------------------------------------------
# --------------------------------   Tyres
tyre_data_filename = os.path.join(os.path.dirname(__file__), 'tyres.json')
with open(tyre_data_filename, 'r') as f:
    tyre_data_from_file = json.load(f)

tyre_data_cache = {TyreChoice.from_name(tyre_choice_name): data_list
                  for tyre_choice_name, data_list in tyre_data_from_file.items()}


class TyreModel:
    def __init__(self):
        self.tyre_data = tyre_data_cache

        self.current_set_numbers = {tyre_choice: rng().choice(len(self.tyre_data[tyre_choice]))
                                    for tyre_choice in self.tyre_data}
        self.current_grip_fn = None
        self.current_tyre_choice = None

    def _grip_fn(self, tyre_choice, set_number):
        data = np.array(self.tyre_data[tyre_choice][set_number])
        return interp1d(data[:, 0], data[:, 1], kind='linear', fill_value=(data[0, 1], data[-1, 1]), bounds_error=False)

    def new_tyres_please(self, tyre_choice: TyreChoice, set_number=None):
        # Store choice, used mostly for debugging
        self.current_tyre_choice = tyre_choice

        if TyreChoice.Learner == tyre_choice:
            self.current_grip_fn = lambda age: 1
            return

        # Increment the set number
        if set_number is None:
            self.current_set_numbers[tyre_choice] += 1
        else:
            self.current_set_numbers[tyre_choice] = set_number
        if self.current_set_numbers[tyre_choice] >= len(self.tyre_data[tyre_choice]):
            self.current_set_numbers[tyre_choice] = 0

        # New tyre curve
        self.current_grip_fn = self._grip_fn(tyre_choice, self.current_set_numbers[tyre_choice])

    def get_grip(self, age: int):
        if self.current_grip_fn is None:
            raise ValueError('No tyres fitted to car. Must call new_tyres_please first.')
        else:
            grip = self.current_grip_fn(age)
            if not isinstance(age, np.ndarray):
                return float(grip)
            else:
                return grip


class AeroModel:
    aero_multipliers = {AeroSetup.HighDownforce: 1.2, AeroSetup.Balanced: 1.0, AeroSetup.LowDrag: 0.8}

    @staticmethod
    def get_grip(aero_setup: AeroSetup):
        return AeroModel.aero_multipliers[aero_setup]


class Car:
    def __init__(self, dynamics_model: CarDynamicsModel, level: Level = None):
        self.speed = None
        self.heading = None
        self.dynamics_model = dynamics_model
        self.drs_active = False

        # Tyres
        if level is None or level == Level.Pro:
            self.tyre_model = TyreModel()
        else:
            self.tyre_model = None
        self.tyre_choice = None
        self.tyre_age = 0

        self.aero_setup = None

    def prepare_for_race(self, start_heading, tyre_choice: TyreChoice = TyreChoice.Learner,
                         aero_setup: AeroSetup = AeroSetup.Balanced):
        self.speed = 0
        self.heading = start_heading
        self.drs_active = False
        self.tyre_choice = tyre_choice
        self.tyre_age = 0
        if self.tyre_model is not None:
            self.tyre_model.new_tyres_please(tyre_choice)
        self.aero_setup = aero_setup

    @property
    def tyre_multiplier(self):
        if self.tyre_model is None:
            return 1
        else:
            return self.tyre_model.get_grip(self.tyre_age)

    def apply_action(self, action, increment_tyre_age=True, tyre_choice: TyreChoice = None,
                     track_grip: float = 1.0):
        spun = False
        if action == Action.FullThrottle:
            self.speed = self.dynamics_model.full_throttle(self.speed, self.drs_active,
                                                           self.grip_multiplier(track_grip), self.drag_multiplier)

        elif action == Action.LightThrottle:
            self.speed = self.dynamics_model.light_throttle(self.speed, self.drs_active,
                                                            self.grip_multiplier(track_grip), self.drag_multiplier)

        elif action == Action.LightBrake:
            self.speed = self.dynamics_model.light_brake(self.speed, self.grip_multiplier(track_grip))

        elif action == Action.HeavyBrake:
            self.speed = self.dynamics_model.heavy_brake(self.speed, self.drs_active,
                                                         self.grip_multiplier(track_grip), self.drag_multiplier)

        elif action in (Action.TurnLeft, Action.TurnRight):
            can_turn = self.dynamics_model.can_turn(self.speed, self.grip_multiplier(track_grip))

            if can_turn:
                # Co-ordinate system is (row, column) positive = (down, right)
                if action == Action.TurnLeft:
                    self.heading = self.heading.get_left_heading()
                else:
                    self.heading = self.heading.get_right_heading()

            else:                           # SPIN!!!
                spun = True

        elif action == Action.OpenDRS:
            self.drs_active = True

        elif action == Action.ChangeTyres:      # caller also needs to provide the tyre_choice argument
            if tyre_choice is None:
                raise ValueError('tyre_choice cannot be None when changing tyres')
            self.tyre_choice = tyre_choice
            self.tyre_age = -1 if increment_tyre_age else 0      # about to increment it
            self.tyre_model.new_tyres_please(tyre_choice)

        # Ensure speed is above 0 and max speed
        self.speed = max(self.speed, 0)

        # Close DRS automatically if we are braking or turning
        if action in [Action.LightBrake, Action.HeavyBrake, Action.TurnLeft, Action.TurnRight]:
            self.drs_active = False

        # Increment tyre age
        if increment_tyre_age:
            self.tyre_age += 1

        return spun

    def get_state(self):
        return CarState(speed=self.speed, heading=self.heading, drs_active=self.drs_active, tyre_choice=self.tyre_choice,
                        tyre_grip=self.tyre_multiplier, tyre_age=self.tyre_age)

    def crashed(self):
        self.speed = 0

    def grip_multiplier(self, track_grip=1.0):
        return self.aero_multiplier * self.tyre_multiplier * track_grip

    @property
    def aero_multiplier(self):
        return AeroModel.get_grip(self.aero_setup)

    @property
    def drag_multiplier(self):
        return 1 / self.aero_multiplier

    @staticmethod
    def get_car_for_level(level: Level, new=False):
        if new or level not in CREATED_DYNAMICS:
            if level in [Level.Learner, Level.Young, Level.Rookie, Level.Pro]:
                CREATED_DYNAMICS[level] = CarDynamicsModel1()
            else:
                raise ValueError(f'Not created cars for level {level.name} yet')

        # Create car with the cached dynamics
        return Car(CREATED_DYNAMICS[level], level)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    speed_in = np.linspace(0, 300, 1000)
    r = 1
    speed_out_full_throttle = np.zeros((speed_in.size, r))
    speed_out_light_throttle = np.zeros((speed_in.size, r))
    speed_out_heavy_braking = np.zeros((speed_in.size, r))
    speed_out_light_braking = np.zeros((speed_in.size, r))
    speed_time = np.zeros((15, r))

    for i in range(r):
        dynamics = CarDynamicsModel1(300)

        speed_out_full_throttle[:, i] = np.array([dynamics.full_throttle(s) for s in speed_in])
        speed_out_light_throttle[:, i] = np.array([dynamics.light_throttle(s) for s in speed_in])
        speed_out_heavy_braking[:, i] = np.array([dynamics.heavy_brake(s) for s in speed_in])
        speed_out_light_braking[:, i] = np.array([dynamics.light_brake(s) for s in speed_in])

        for j in range(1, speed_time.shape[0]):
            speed_time[j, i] = dynamics.full_throttle(speed_time[j - 1, i])

    fig = plt.figure(figsize=(14, 10))
    ax_thr = fig.add_subplot(2, 3, 1)
    ft_lines = ax_thr.plot(speed_in, speed_out_full_throttle, 'r')
    lt_lines = ax_thr.plot(speed_in, speed_out_light_throttle, 'b')
    ax_thr.set_xlabel('Speed In')
    ax_thr.set_ylabel('Speed Out')
    ax_thr.set_title('Throttle')
    ax_thr.legend([ft_lines[0], lt_lines[0]], ['Full throttle', 'Light throttle'])

    ax_thr = fig.add_subplot(2, 3, 4)
    ft_lines = ax_thr.plot(speed_in, speed_out_full_throttle - speed_in[:, None], 'r')
    lt_lines = ax_thr.plot(speed_in, speed_out_light_throttle - speed_in[:, None], 'b')
    ax_thr.set_xlabel('Speed In')
    ax_thr.set_ylabel('Delta Speed')
    ax_thr.set_title('Throttle')
    ax_thr.legend([ft_lines[0], lt_lines[0]], ['Full throttle', 'Light throttle'])

    ax_time = fig.add_subplot(2, 3, 2)
    ax_time.plot(speed_time)
    ax_time.set_xlabel('Time')
    ax_time.set_ylabel('Speed')

    ax_brake = fig.add_subplot(2, 3, 3)
    lb_lines = ax_brake.plot(speed_in, speed_out_light_braking, 'b')
    hb_lines = ax_brake.plot(speed_in, speed_out_heavy_braking, 'r')
    ax_brake.set_xlabel('Speed in')
    ax_brake.set_ylabel('Speed after braking')
    ax_brake.set_title('Braking')
    ax_brake.legend([hb_lines[0], lb_lines[0]], ['Heavy braking', 'Light braking'])

    ax_brake = fig.add_subplot(2, 3, 6)
    lb_lines = ax_brake.plot(speed_in, speed_out_light_braking - speed_in[:, None], 'b')
    hb_lines = ax_brake.plot(speed_in, speed_out_heavy_braking - speed_in[:, None], 'r')
    ax_brake.set_xlabel('Speed in')
    ax_brake.set_ylabel('Delta Speed')
    ax_brake.set_title('Braking')
    ax_brake.legend([hb_lines[0], lb_lines[0]], ['Heavy braking', 'Light braking'])

    fig.tight_layout()
    plt.show()
