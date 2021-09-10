from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import minimize
import scipy.linalg
from time import time as time_fn
from sklearn.linear_model import LinearRegression
import warnings
from matplotlib.ticker import MaxNLocator

from drivers.driver import *
from drivers.rookiedriver import RookieDriver


class ProDriver(RookieDriver):
    def __init__(self, name, weather_on=True, random_action_probability=0.5, random_action_decay=0.96,
                 min_random_action_probability=0.0, allow_pitstops=True, grip_fig=None, *args, **kwargs):

        super().__init__(name, random_action_probability=random_action_probability,
                         random_action_decay=random_action_decay,
                         min_random_action_probability=min_random_action_probability, *args, **kwargs)

        self.tyre_data = {tyre_choice: np.empty((1000, 0)) for tyre_choice in TyreChoice.get_choices()}
        self.current_tyre_choice = TyreChoice.Medium            # hard coded for now, you can improve this!
        self.current_tyre_age = 0
        self.current_base_tyre_model = None
        self.current_tyre_parameters = None
        self.at_turn = False
        self.move_number = 0
        self.allow_pitstops = allow_pitstops
        self.pit_loss = 3.0
        self.timings = defaultdict(lambda: 0)

        # Plotting of grip data
        self.grip_fig = grip_fig
        self.grip_fig_axes = {}
        self.grip_fig_obj = {}
        self.tyre_data_plot = None
        self.tyre_model_plot = None
        self.target_plot = None

        self.track_info = None
        self.straight_ends = []
        self.completed_straights = []
        self.box_box_box = False
        self.last_action_was_random = False
        self.target_speed_grips = None

        # Weather
        self.weather_on = weather_on
        self.weather_data = []
        self.track_grips = []
        self.last_raining_move = -1000
        self.current_weather_state = None
        self.track_grip_model_x = None
        self.track_grip_model_y = None
        self.num_previous_steps = None
        self.num_future_steps = None

    def choose_tyres(self, track_info: TrackInfo) -> TyreChoice:
        # This method is called at the start of the race and whenever the driver chooses to make a pitstop. It needs to
        # return a TyreChoice enum

        # TODO: make an informed choice here!
        # self.current_tyre_choice = ...

        self.track_info = track_info
        self.fit_base_tyre_model()
        return self.current_tyre_choice

    def prepare_for_race(self):
        self.move_number = 0
        self.straight_ends = []
        self.completed_straights = []
        self.weather_data.append([])
        self.track_grips.append([])
        self.track_grip_model_y = None  # will need to refit these as won't have historic data for this race
        self.track_grip_model_x = None

        if self.grip_fig is not None:
            self.grip_fig.clear()
            self.grip_fig_axes = {}
            self.grip_fig_obj = {}
            self.tyre_ax = None
            self.target_ax = None

    def choose_aero(self, track_info):
        return AeroSetup.Balanced

    def make_a_move(self, car_state: CarState, track_state: TrackState, weather_state: WeatherState) -> Action:
        self.move_number += 1
        if 1 == self.move_number:
            self.update_target_speeds(grips_up_straight=np.tile(car_state.tyre_grip, track_state.distance_ahead))

        # Store the tyre data
        if 0 == car_state.tyre_age:         # new tyre, add a column of nans ready for data
            n = self.tyre_data[car_state.tyre_choice].shape[0]
            self.tyre_data[car_state.tyre_choice] = np.hstack([self.tyre_data[car_state.tyre_choice],
                                                               np.full((n, 1), np.nan)])
        self.tyre_data[car_state.tyre_choice][car_state.tyre_age, -1] = car_state.tyre_grip
        self.current_tyre_age = car_state.tyre_age
        if car_state.tyre_choice != self.current_tyre_choice:
            self.current_tyre_choice = car_state.tyre_choice
            self.fit_base_tyre_model()

        # Weather
        if weather_state.rain_intensity > 0:
            self.last_raining_move = self.move_number
        self.current_weather_state = weather_state
        if weather_state.rain_intensity > 0:
            self.fit_track_grip()
            self.update_target_speeds(track_state.distance_ahead, car_state, weather_state)

        # If we are at the end of the straight and it is not a dead end then choose the turn direction
        if track_state.distance_ahead == 0 and not (track_state.distance_left == 0 and track_state.distance_right == 0
                                                    and car_state.speed > 0):
            # Store this straight
            if track_state.position not in self.straight_ends and track_state.distance_behind > 0:
                self.straight_ends.append(track_state.position)
                self.completed_straights.append(track_state.distance_behind)

            self.at_turn = True
            return self._choose_turn_direction(track_state)

        # Update the target speeds at the start of a straight, to take tyre degradation into account. We could do this
        # every move but limit it to avoid slowing code down too much
        elif track_state.distance_ahead > 0 and self.at_turn:
            if weather_state.rain_intensity == 0:           # we will already have updated them otherwise
                t0 = time_fn()
                self.update_target_speeds(track_state.distance_ahead, car_state, weather_state)
                # print(f'\tUpdating target speeds took {time_fn() - t0: .2f} seconds')
            t0 = time_fn()
            self.box_box_box = self.should_we_change_tyres()
            # print(f'\tTesting pit stop {time_fn() - t0: .2f} seconds')
            if self.box_box_box and self.print_info:
                print('Box! Box! Box!')

            self.drs_was_active = False         # start of new straight, reset log

        self.at_turn = False

        # Are we changing tyres?
        if self.box_box_box and car_state.speed == 0:
            self.box_box_box = False
            return Action.ChangeTyres

        # Get the target speed
        target_speed = self._get_target_speed(track_state.distance_ahead, track_state.safety_car_active)

        # Get the current grip level
        current_grip = self.get_grip(car_state, turns_ahead=0, weather_state=weather_state)

        # Choose action that gets us closest to target, or choose randomly
        if driver_rng().rand() > self.random_action_probability:
            action = self._choose_move_from_models(car_state.speed, target_speed, car_state.drs_active,
                                                   grip_multiplier=current_grip)
            self.last_action_was_random = False
        else:
            action = self._choose_randomly(Action.get_sl_actions())
            self.last_action_was_random = True

        # If DRS is available then need to decide whether to open DRS or not.
        if track_state.drs_available and not car_state.drs_active and track_state.distance_ahead > 0:
            # Simulate the straight with and without DRS and check which we think will be faster
            time_no_drs, targets_broken_no_drs, *_ = self.simulate_straight(car_state.speed,
                                                                            track_state.distance_ahead,
                                                                            drs_active=False,
                                                                            safety_car_active=track_state.safety_car_active,
                                                                            weather_state=weather_state)
            time_drs, targets_broken_drs, *_ = self.simulate_straight(car_state.speed, track_state.distance_ahead - 1,
                                                                      drs_active=True,
                                                                      safety_car_active=track_state.safety_car_active,
                                                                      weather_state=weather_state)
            targets_broken_drs |= car_state.speed > self.target_speeds[track_state.distance_ahead - 1]
            time_drs = (1 / (car_state.speed + 1)) + time_drs

            if (time_drs < time_no_drs or driver_rng().rand() < self.random_action_probability
                or any(len(data) < 10 for data in self.drs_data.values())) and not targets_broken_drs:
                action = Action.OpenDRS
                self.drs_was_active = True
                if self.print_info:
                    print('Opening DRS')
            elif self.print_info:
                print('Chose not to open DRS')

        self.random_action_probability = max(self.random_action_probability * self.random_action_decay,
                                             self.min_random_action_probability)

        return action

    def _get_target_speed(self, distance_ahead, safety_car_active, target_speeds=None):
        if target_speeds is None:
            target_speeds = self.target_speeds

        if distance_ahead == 0 or self.box_box_box:
            target_speed = 0                                            # dead end - need to stop!!
        else:
            target_speed = target_speeds[distance_ahead - 1]       # target for next step

        if safety_car_active:
            target_speed = min(target_speed, self.safety_car_speed)

        return target_speed

    def estimate_next_speed(self, action: Action, speed, drs_active: bool, grip_multiplier: float = 1.0):
        data = np.array(self.get_data(action, drs_active))
        if data.shape[0] < 2:
            return speed
        interp = interp1d(data[:, 0], data[:, 1], fill_value='extrapolate', assume_sorted=False)
        return speed + interp(speed) * grip_multiplier      # predict delta

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
                                   action: Action, new_car_state: CarState, new_track_state: TrackState,
                                   result: ActionResult, previous_weather_state: WeatherState):

        if previous_track_state.safety_car_active:
            self._update_safety_car(previous_car_state, result)

        if previous_track_state.distance_ahead == 0:          # end of straight
            if result.crashed or result.spun:
                grip = self.get_grip(previous_car_state, weather_state=previous_weather_state)

                if self.print_info:
                    if self.last_action_was_random:
                        print('\tCrashed! Last action was random though')

                    elif previous_car_state.speed > self.target_speeds[0] + 1:
                        if action in self.sl_data:
                            est_speed = self.estimate_next_speed(action, previous_car_state.speed,
                                                                 previous_car_state.drs_active, grip)
                        else:
                            est_speed = previous_car_state.speed
                        print(f'\tCrashed! We targeted {self.target_speeds[0]:.0f} speed '
                              f'but were going {previous_car_state.speed: .0f}. '
                              f'We thought we would be going {est_speed :.0f} using a grip of {grip:.2f}.'
                              f'DRS was {"" if self.drs_was_active else "not "}active this straight.')

                    else:
                        print(f'\tCrashed! We targeted {self.target_speeds[0]:.0f} speed '
                              f'and were going {previous_car_state.speed: .0f}. '
                              f'EoS speed unmodified is {self.end_of_straight_speed: .0f}. '
                              f'We used a grip of {grip:.2f} which gives {grip * self.end_of_straight_speed: .0f}')

                self.end_of_straight_speed = min(self.end_of_straight_speed,
                                                 previous_car_state.speed / grip - 10)
                if previous_track_state.distance_left > 0 or previous_track_state.distance_right > 0:
                    self.lowest_crash_speed = min(previous_car_state.speed / grip, self.lowest_crash_speed)
            else:
                previous_grip = self.get_grip(previous_car_state, weather_state=previous_weather_state)
                self.end_of_straight_speed = min(max(self.end_of_straight_speed,
                                                 (previous_car_state.speed / previous_grip) + 1),
                                                 self.lowest_crash_speed)

            # Refit tyre model now we have more data
            t0 = time_fn()
            self.fit_tyre_model()
            # print(f'\tFitting tyre model took {time_fn() - t0: .2f} seconds')
            t0 = time_fn()
            self.fit_track_grip()
            # print(f'\tFitting track grip model took {time_fn() - t0: .2f} seconds')

        # record the change in speed resulting from the action we took
        elif action in self.sl_data and self.move_number > self.last_raining_move + 30:
            # Remove the grip effect from the delta to get the true dynamics
            if Action.HeavyBrake == action and 0 == new_car_state.speed:
                # Heavy braking delta can take the car "below" zero, which is then capped at 0. When this happens we
                # don't see the effect of the grip multiplier so we don't want to normalise the delta
                normalised_delta = (new_car_state.speed - previous_car_state.speed)
            else:
                normalised_delta = (new_car_state.speed - previous_car_state.speed) / self.get_grip(previous_car_state, exclude_track=True)
            normalised_delta = max(normalised_delta, -previous_car_state.speed)   # can't go below zero

            # Record the point if it is not on top of another point (interpolation doesn't like points too close
            # together in x, plus it is also a bit unnecessary) and we are below 200 points (just to keep code
            # performance up)
            current_data = self.get_data(action, previous_car_state.drs_active)
            if 0 == len(current_data):
                closest_distance = 1000
            else:
                closest_distance = np.min(np.abs(np.array(current_data)[:, 0] - previous_car_state.speed))
            if closest_distance > 1 and len(current_data) < 200 and previous_car_state.speed + normalised_delta > 0:
                new_data = [previous_car_state.speed, normalised_delta]
                current_data.append(new_data)

        # Record the track grip and weather data
        if action in self.sl_data:
            expected_delta = self.estimate_next_speed(action, previous_car_state.speed,
                                                      previous_car_state.drs_active,
                                                      self.get_grip(previous_car_state, exclude_track=True)) \
                              - previous_car_state.speed
        else:
            expected_delta = 0

        if expected_delta > 0:
            estimated_track_grip = (new_car_state.speed - previous_car_state.speed) / expected_delta
            estimated_track_grip = max(min(estimated_track_grip, 1), 0.1)
            self.track_grips[-1].append(estimated_track_grip)
            self.weather_data[-1].append(self.weather_state_to_list(previous_weather_state))

        elif len(self.track_grips[-1]) > 0:         # don't start with nans
            self.track_grips[-1].append(np.nan)
            self.weather_data[-1].append(self.weather_state_to_list(previous_weather_state))

    def update_target_speeds(self, distance_ahead=None, car_state=None, weather_state=None, grips_up_straight=None,
                             assign_to_self=True):
        """ Either need to specify distance_ahead, car_state, and weather_state or a custom set of grips_up_straight """
        t0 = time_fn()

        if grips_up_straight is None:
            grips = self.get_grip(car_state, distance_ahead, weather_state=weather_state)  # from start to end of straight
            grips *= 0.95                                       # add a little safety margin to our prediction
            grips_up_straight = np.atleast_1d(grips)[::-1]

        previous_targets = np.copy(self.target_speeds)
        target_speeds = np.zeros_like(self.target_speeds)
        speed = self.end_of_straight_speed * grips_up_straight[0]       # modify by expected grip at end of straight

        # Pre compute the changes in speed for quicker look up. As the grip changes down the straight we compute deltas
        # with grip = 1 and then convert to next speeds later
        test_input_speeds = np.linspace(0, 350, 351)
        test_speed_deltas = {action: self.estimate_next_speed(action, test_input_speeds, False, grip_multiplier=1)
                                     - test_input_speeds
                             for action in self.sl_data}

        for i in range(len(self.target_speeds)):
            if np.all(target_speeds[i:] == speed):
                break                   # there won't be any further changes so save some time
            target_speeds[i] = speed
            previous_grip = grips_up_straight[min(i + 1, len(grips_up_straight) - 1)]
            speed = np.nanmax([self.estimate_previous_speed(test_input_speeds,
                                                            test_input_speeds + test_speed_deltas[action]*previous_grip,
                                                            speed)
                               for action in self.sl_data])
            speed = max(speed, 10)

        if assign_to_self:
            self.target_speeds = target_speeds
            self.target_speed_grips = grips_up_straight

        if self.print_info and not np.array_equal(previous_targets, self.target_speeds):
            # print(f'New target speeds: mid-straight->{np.array2string(self.target_speeds[5::-1], precision=0)}<-end. '
            #       f'Forecasted grips are {grips_up_straight[5::-1]}')
            self.plot_grip()

        self.timings['update_target_speeds'] += time_fn() - t0

        return target_speeds

    def simulate_straight(self, speed, distance_ahead, drs_active, safety_car_active, weather_state=None, grips=None):
        t0 = time_fn()
        if grips is None:
            grips = self.get_grip(car_state=None, turns_ahead=distance_ahead, tyre_grip=self.current_tyre_grip,
                                  tyre_age=self.current_tyre_age, weather_state=weather_state)
            target_speeds = self.target_speeds
        else:
            # Custom grips provided, need to get custom target speeds to match them
            target_speeds = self.update_target_speeds(grips_up_straight=grips[::-1], assign_to_self=False)

        speeds = np.zeros(distance_ahead)
        break_target_speed = False
        for d in range(distance_ahead):
            target_speed = self._get_target_speed(distance_ahead - d, safety_car_active, target_speeds)
            action = self._choose_move_from_models(speed, target_speed, drs_active, grip_multiplier=grips[d])
            speeds[d] = self.estimate_next_speed(action, speed, drs_active, grip_multiplier=grips[d])
            speed = speeds[d]
            break_target_speed |= speed > target_speed
        time = np.sum(1 / (speeds + 1))

        self.timings['simulate_straight'] += time_fn() - t0

        return time, break_target_speed, speeds, grips

    @property
    def current_tyre_grip(self):
        return self.get_measured_tyre_grip(self.current_tyre_age)

    def get_measured_tyre_grip(self, age, tyre_choice=None):
        if tyre_choice is None:
            tyre_choice = self.current_tyre_choice

        if self.tyre_data[tyre_choice].shape[1] == 0:
            return np.nan
        else:
            return self.tyre_data[tyre_choice][age, -1]

    def get_grip(self, car_state: Union[CarState, None] = None, turns_ahead=0, tyre_grip=None, tyre_age=None,
                 weather_state=None, exclude_track=False):
        if car_state is not None:
            tyre_grip = car_state.tyre_grip
            tyre_age = car_state.tyre_age
        if tyre_age is None:
            tyre_age = self.current_tyre_age
        if tyre_grip is None:
            tyre_grip = self.get_measured_tyre_grip(tyre_age)

        if 0 == turns_ahead:
            grip = max(tyre_grip, 0.1)
        else:
            ages = tyre_age + np.arange(1, turns_ahead + 1)
            future_grips = self.forecast_tyre_grip(ages)
            grip = np.maximum(np.concatenate([[tyre_grip], future_grips]), 0.1)

        # Track grip from rain
        if not exclude_track and self.weather_on:
            if weather_state is None:
                weather_state = self.current_weather_state      # used cached value
            track_grip = self.forecast_track_grip(weather_state, turns_ahead)
            if 0 == turns_ahead:
                track_grip = float(track_grip)
            grip *= track_grip

        return np.maximum(grip, 0.1)             # if predicted grip drops too low we can get stuck

    def forecast_tyre_grip(self, tyre_ages, parameters=None):
        if parameters is None:
            if self.current_tyre_parameters is None:
                self.fit_tyre_model()
            parameters = self.current_tyre_parameters
        x_offset, x_scale, y_offset, y_scale = parameters
        return y_offset + y_scale * self.current_base_tyre_model(x_offset + x_scale*tyre_ages)

    @staticmethod
    def logistic(p, t):
        return p[0] + p[1] / (1 + np.exp(-p[2] + t / p[3]))

    def fit_base_tyre_model(self):
        # Fit initial model
        data = self.tyre_data[self.current_tyre_choice]
        t = np.arange(data.shape[0])
        if 0 == data.shape[1] or np.sum(~np.isnan(data[:, -1])) < 2:
            # Have no data so use a logistic as an initial guess. It will be offset and stretched to fit the observed
            # data as it comes in
            if self.current_tyre_choice == TyreChoice.Soft:
                p = 0.15, 1.05, 8, 16
            elif self.current_tyre_choice == TyreChoice.Medium:
                p = [0.1, 0.9, 15, 20]
            else:
                p = [0.15, 0.65, 8, 50]
            self.current_base_tyre_model = lambda t: self.logistic(p, t)

        else:
            b_no_nan = np.any(~np.isnan(data), axis=1)
            m = np.nanmean(data[b_no_nan, :], axis=1)
            t = t[b_no_nan]
            self.current_base_tyre_model = interp1d(t, m, kind='linear', fill_value='extrapolate')

    def fit_tyre_model(self):
        t0 = time_fn()
        # Fit the base tyre model to the new data coming in but translating it and scaling it
        ages = np.arange(self.tyre_data[self.current_tyre_choice].shape[0])
        latest_data = self.tyre_data[self.current_tyre_choice][:, -1]
        b_no_nan = ~np.isnan(latest_data)
        latest_data = latest_data[b_no_nan]
        ages = ages[b_no_nan]

        def obj_fun(p):
            error = latest_data - self.forecast_tyre_grip(ages, p)
            weight = np.ones_like(error)
            weight[error < 0] = 10
            return np.mean(weight * error**2)

        p0 = [0, 1, 0, 1] if self.current_tyre_parameters is None else self.current_tyre_parameters
        res = minimize(obj_fun, p0, bounds=[(-100, 100), (0.8, 1.5), (-0.2, 0.2), (0.8, 1.3)], method='Powell')
        self.current_tyre_parameters = res.x

        self.timings['fit_tyre_model'] += time_fn() - t0

    def should_we_change_tyres(self):
        if not self.allow_pitstops or np.nanmin(self.tyre_data[self.current_tyre_choice]) > 0.6:
            return False

        t0 = time_fn()

        time_current_tyres = self.simulate_to_end_of_race(self.current_tyre_age)
        time_new_tyres = self.simulate_to_end_of_race(0)
        start_grip = self.get_measured_tyre_grip(0)
        lost_grip_fraction = (start_grip - self.current_tyre_grip) / start_grip

        if self.print_info:
            if self.grip_fig:
                if 'pitstop_prediction' not in self.grip_fig_obj:
                    self.grip_fig_obj['pitstop_prediction'] = \
                        self.grip_fig_axes['tyres'].text(0.05, 0.05, '',
                                                         transform=self.grip_fig_axes['tyres'].transAxes)
                self.grip_fig_obj['pitstop_prediction'].set_text(f'Predicted race time\ncurrent tyres: '
                                                                 f'{time_current_tyres: .03f}, new tyres:'
                                                                 f' {time_new_tyres: .03f}')
                self.grip_fig.canvas.draw()

        self.timings['should_we_change_tyres'] += time_fn() - t0

        return time_new_tyres < time_current_tyres - self.pit_loss and lost_grip_fraction > 0.1

    def simulate_to_end_of_race(self, start_tyre_age):
        straights_remaining = max(self.track_info.number_of_straights - len(self.straight_ends), 1)
        straight_length = int(self.track_info.average_straight)
        total_length = straights_remaining * (straight_length + 1)        # +1 as we take move turning each straight
        #            |- total # squares to move through -|   |- total # of turns --|
        num_moves = straights_remaining * straight_length + (straights_remaining - 1)

        # grips[i] is grip at time point i used to move from speed[i] to speed[i+1]
        grips = self.get_grip(turns_ahead=num_moves, tyre_age=start_tyre_age, exclude_track=True)
        speeds = 500 * np.ones(num_moves + 1)     # add fake corner in at the end as this driver brakes for the finish
        speeds[0] = self.end_of_straight_speed * grips[0]       # should really be previous grip but close enough
        # speed of turns
        speeds[straight_length::(straight_length+1)] = self.end_of_straight_speed * grips[straight_length::(
                straight_length+1)]
        # Speed starting next straight same as turn
        speeds[straight_length+1::(straight_length+1)] = speeds[straight_length:-2:(straight_length+1)]

        # Pre compute the changes in speed for quicker look up. As the grip changes down the straight we compute deltas
        # with grip = 1 and then convert to next speeds later
        test_input_speeds = np.linspace(0, 350, 351)
        actions = list(self.sl_data.keys())
        test_speed_deltas = np.zeros((test_input_speeds.size, len(actions)))
        for i, action in enumerate(actions):
            test_speed_deltas[:, i] = self.estimate_next_speed(action, test_input_speeds, False, grip_multiplier=1) \
                                      - test_input_speeds

        # First the backwards pass to compute the maximum safe speeds
        for i in range(total_length - 1):
            possible_next_speeds = np.maximum(test_input_speeds[:, None] + test_speed_deltas* grips[-i-2], 0)
            safe_next_speeds = test_input_speeds[np.any(possible_next_speeds <= speeds[-i-1], axis=1)]
            max_safe_input_speed = np.max(safe_next_speeds)
            speeds[-i-2] = np.minimum(max_safe_input_speed, speeds[-i-2])

        # Next the forward pass to work out what we can actually reach
        for i in range(num_moves):
            next_speeds = np.maximum(speeds[i] + grips[i] * test_speed_deltas[int(np.round(speeds[i])), :], 0)
            if np.any(next_speeds < speeds[i+1]):
                speeds[i+1] = np.max(next_speeds * (next_speeds < speeds[i+1]))
            else:
                speeds[i+1] = np.min(next_speeds)

        # Compute time
        time = np.sum(1 / (1 + speeds[:-1]))
        return time

    def plot_grip(self):
        if self.grip_fig is not None and self.print_info:
            # ------ Create axes ------
            # Top left
            if self.grip_fig_axes.get('grips', None) is None:
                self.grip_fig_axes['grips'] = self.grip_fig.add_subplot(2, 2, 1)
                self.grip_fig_axes['grips'].set_ylabel('Total grip for example straight', fontsize=12)
                self.grip_fig_axes['grips'].set_xlabel('Distance to end of straight', fontsize=12)
                self.grip_fig_axes['grips'].set_title('Forecasted grip for straight')
                self.grip_fig_axes['grips'].xaxis.set_major_locator(MaxNLocator(integer=True))
                self.grip_fig_obj['grips'] = []

            # Bottom left
            if self.grip_fig_axes.get('tyres', None) is None:
                self.grip_fig_axes['tyres'] = self.grip_fig.add_subplot(2, 2, 3)
                self.grip_fig_axes['tyres'].set_ylabel('Tyre grip', fontsize=12)
                self.grip_fig_axes['tyres'].set_xlabel('Move number', fontsize=12)
                self.grip_fig_axes['tyres'].set_title('Forecasted tyre degradation')
                self.grip_fig_obj['tyre_data'] = self.grip_fig_axes['tyres'].plot(np.nan, np.nan, 'b+', zorder=3)[0]
                self.grip_fig_obj['tyre_forecast'] = []

            # Top right
            if self.grip_fig_axes.get('targets', None) is None:
                self.grip_fig_axes['targets'] = self.grip_fig.add_subplot(2, 2, 2)
                self.grip_fig_axes['targets'].set_ylabel('Target speeds', fontsize=12)
                self.grip_fig_axes['targets'].set_xlabel('Distance to end of straight', fontsize=12)
                self.grip_fig_axes['targets'].set_title('Achievable speeds')
                self.grip_fig_axes['targets'].xaxis.set_major_locator(MaxNLocator(integer=True))
                self.grip_fig_obj['target_speeds'] = []

            # Bottom right - rain forecase
            if self.grip_fig_axes.get('track', None) is None:
                self.grip_fig_axes['track'] = self.grip_fig.add_subplot(2, 2, 4)
                self.grip_fig_axes['track'].set_ylabel('Track grip (rain)', fontsize=12)
                self.grip_fig_axes['track'].set_xlabel('Move number', fontsize=12)
                self.grip_fig_axes['track'].set_title('Forecasted track grip (rain)')
                self.grip_fig_obj['track_data'] = self.grip_fig_axes['track'].plot(np.nan, np.nan, 'b+', zorder=3)[0]
                self.grip_fig_obj['track_forecast'] = []

            self.grip_fig.tight_layout()
            self.grip_fig_axes['track'].set_visible(self.weather_on)

            def fade(lines):
                if len(lines) > 20:
                    line = lines.pop(0)
                    line.remove()
                fade = np.linspace(0, 0.15, 20)
                for i, line in enumerate(lines[::-1]):
                    c = 0.8 + fade[i]
                    line.set_color((c, c, c))

            # Get data to plot for a straight of length L starting from stationary with the current tyres/weather but
            # ignoring DRS & safety car
            L = 15          # length of strraight to plot
            _, _, straight_speeds, straight_grips = self.simulate_straight(0, L, False, False)

            # ------------ Straight grip used in target speed, top left plot ------------------
            fade(self.grip_fig_obj.get('grips'))
            self.grip_fig_obj['grips'] += self.grip_fig_axes['grips'].plot(np.arange(L + 1)[::-1], straight_grips, 'b')
            self.grip_fig_axes['grips'].set_xlim([L, 0])
            self.grip_fig_axes['grips'].set_ylim([0, 1.4])

            # ----------- Target speeds, top right plot ------------------
            fade(self.grip_fig_obj.get('target_speeds'))
            self.grip_fig_obj['target_speeds'] += self.grip_fig_axes['targets'].plot(np.arange(1, L + 1)[::-1],
                                                                                     straight_speeds, 'b')
            self.grip_fig_axes['targets'].set_xlim([L, 0])
            self.grip_fig_axes['targets'].set_ylim([0, 325])

            # ------ Tyres forecast - bottom left plot -----
            # Observed tyre grip data
            latest_data = self.tyre_data[self.current_tyre_choice][:, -1]
            ages = np.arange(len(latest_data))
            b_no_nan = ~np.isnan(latest_data)
            self.grip_fig_obj['tyre_data'].set_xdata(ages[b_no_nan])
            self.grip_fig_obj['tyre_data'].set_ydata(latest_data[b_no_nan])

            # Tyre model forecast
            fade(self.grip_fig_obj.get('tyre_forecast'))
            ages_sweep = np.arange(np.sum(b_no_nan) + 100)
            tyre_sweep = self.forecast_tyre_grip(ages_sweep)
            self.grip_fig_obj['tyre_forecast'] += self.grip_fig_axes['tyres'].plot(ages_sweep, tyre_sweep, 'r')

            self.grip_fig_axes['tyres'].set_xlim([0, len(ages_sweep)])
            self.grip_fig_axes['tyres'].set_ylim([0, np.max(tyre_sweep)*1.1])

            # ------- Weather forecast - bottom right plot ----------
            if self.weather_on:
                track_data_this_race = self.track_grips[-1]
                weather_data_this_race = self.weather_data[-1]
                self.grip_fig_obj['track_data'].set_xdata(np.arange(len(track_data_this_race)))
                self.grip_fig_obj['track_data'].set_ydata(track_data_this_race)
                forecast_track_grip = self.forecast_track_grip(self.current_weather_state, 50)

                fade(self.grip_fig_obj.get('track_forecast'))
                i0 = len(track_data_this_race)
                if i0 > 0 and weather_data_this_race[-1] == self.weather_state_to_list(self.current_weather_state):
                    i0 -= 1         # forecast starts with last recorded point otherwise from next future point
                x = np.arange(i0, i0+len(forecast_track_grip))
                self.grip_fig_obj['track_forecast'] += self.grip_fig_axes['track'].plot(x, forecast_track_grip, 'r')
                self.grip_fig_axes['track'].set_xlim([0, x[-1]])
                self.grip_fig_axes['track'].set_ylim([0, 1.4])

            self.grip_fig.canvas.draw()

    @staticmethod
    def weather_state_to_list(weather_state: WeatherState):
        return [weather_state.air_temperature, weather_state.track_temperature,
                weather_state.humidity, weather_state.rain_intensity]

    @staticmethod
    def interpolate_nans(track_grips, weather_data):
        # Deal with nans - trim beginning and end but we have to interpolate missing values in the middle as we
        # need a continuous time series
        I = np.arange(len(track_grips))
        nans = np.isnan(track_grips)
        if np.sum(~nans) > 1:
            track_grips[nans] = PchipInterpolator(I[~nans], track_grips[~nans], extrapolate=False)(I[nans])
        # track_grips[nans] = np.interp(I[nans], I[~nans], track_grips[~nans], left=np.nan, right=np.nan)
        return track_grips, weather_data

    def fit_track_grip(self):
        t0 = time_fn()
        if not self.weather_on:
            return

        n = [np.sum(~np.isnan(grip)) for grip in self.track_grips]
        if len(n) == 0 or np.max(n) == 0 or n[-1] == 0:
            self.num_previous_steps = 0
            self.track_grip_model = None
            return
        num_previous_steps_predict = n[-1] - 1      # can only predict based on data from this race
        num_previous_steps_train = int(np.max(n)/10)          # leave plenty of data for training
        num_previous_steps = int(np.min([num_previous_steps_predict, num_previous_steps_train, 10]))

        y_inputs, y_targets, x_inputs, x_targets = [], [], [], []
        for data, grip in zip(self.weather_data, self.track_grips):     # loop over data from different races
            if len(grip) > num_previous_steps + 1:
                data = np.array(data)       # (N, D)
                grip = np.array(grip)       # {N, )

                # Deal with nans - trim beginning and end but we have to nterpolate missing values in the middle as we
                # need a continuous time series
                grip, data = self.interpolate_nans(grip, data)
                b_no_nan = ~np.isnan(grip)
                grip = grip[b_no_nan]
                data = data[b_no_nan, :]

                y_in, y_tar = self.format_ar_arrays_for_y(data, grip, num_previous_steps)
                x_in, x_tar = self.format_ar_arrays_for_x(data, grip, num_previous_steps)

                y_inputs.append(y_in)
                y_targets.append(y_tar)
                x_inputs.append(x_in)
                x_targets.append(x_tar)

        if len(y_inputs) > 0 and len(x_inputs):
            y_inputs_all = np.vstack(y_inputs)
            y_targets_all = np.vstack(y_targets)
            x_inputs_all = np.vstack(x_inputs)
            x_targets_all = np.vstack(x_targets)

            if self.track_grip_model_y is None:
                self.track_grip_model_y = LinearRegression()
            if self.track_grip_model_x is None:
                self.track_grip_model_x = LinearRegression()

            t1 = time_fn()
            self.track_grip_model_y.fit(y_inputs_all, y_targets_all)
            self.track_grip_model_x.fit(x_inputs_all, x_targets_all)
            self.num_previous_steps = num_previous_steps
            self.timings['fit_track_grip > linear fit'] += time_fn() - t1

        self.timings['fit_track_grip'] += time_fn() - t0

    def forecast_track_grip(self, current_weather_state: WeatherState, num_future_steps=0):
        # Predict current grip + num_future_steps into the future
        # Returns an array of length 1 + num_future_steps
        if self.track_grip_model_y is None:
            self.fit_track_grip()
        if self.track_grip_model_y is None or current_weather_state is None or not self.weather_on:
            return np.ones(num_future_steps + 1)

        historic_y, historic_x = self.interpolate_nans(np.array(self.track_grips[-1]), np.array(self.weather_data[-1]))
        b_nan = np.isnan(historic_y)
        if np.all(b_nan):
            return np.ones(num_future_steps + 1)
        elif b_nan[-1]:
            # Go back in time until we find a non-nan value
            i_last_no_nan = np.where(~b_nan)[0][-1]
            num_extra_steps = historic_y.size - i_last_no_nan - 1
            current_x = historic_x[i_last_no_nan+1, :]
            historic_x = historic_x[:i_last_no_nan+1, :]
            historic_y = historic_y[:i_last_no_nan+1]
            b_nan = b_nan[:i_last_no_nan+1]

        else:
            current_x = np.array(self.weather_state_to_list(current_weather_state))
            num_extra_steps = 0
        historic_x = historic_x[~b_nan, :]     # cut off any nans at the start
        historic_y = historic_y[~b_nan]        # cut off any nans at the start

        grips = self.autoregressive_forecast(model_y=self.track_grip_model_y, model_x=self.track_grip_model_x,
                                             historic_x=historic_x, historic_y=historic_y, current_x=current_x,
                                             num_forecast_steps=num_future_steps + 1 + num_extra_steps, bound_y=True)
        return grips[num_extra_steps:]

    @staticmethod
    def autoregressive_forecast(model_y, model_x, historic_x, historic_y, current_x, num_forecast_steps, bound_y=True,
                                bound_x=True):
        # Predict the y at the current time point and then for num_forecast_steps into the future by alternating y and
        # x predictions:
        #       predict y_t given x_t and {y_t-i, x_t-i} i = 1:num_previous_steps
        #       predict x_t+1 given [y_t, x_t] and {y_t-i, x_t-i} i = 1:num_previous_steps
        #
        # If bound_y is True then bounds y to be within (0, 1). If bound_x is True then bounds x to be within (0, 100)

        # Figure out the number of previous steps
        #   > number of model features = D*(num_previous_steps + 1) + num_previous_steps
        #   > num_previous_steps = (number of model features - D) / (D + 1)
        N, D = historic_x.shape
        num_previous_steps = int((model_y.n_features_in_ - D) / (D + 1))
        if historic_x.shape[0] < num_previous_steps:
            warnings.warn(f'Trying to make autoregressive forecast with too few historic points: '
                          f'num_previous_steps = {num_previous_steps} but historic_x has {historic_x.shape[0]} points. '
                          f'Trying a model refit...')
            self.fit_track_grip()       # something has gone wrong. Refit model and see if that fixes it
            num_previous_steps = int((model_y.n_features_in_ - D) / (D + 1))

            historic_x = np.vstack([np.zeros((num_previous_steps - historic_x.shape[0], historic_x.shape[1]))])
            if historic_x.shape[0] < num_previous_steps:
                raise ValueError("'Refitting track grip model hasn't fixed it, something has gone wrong...")
            else:
                print('Refitting track grip model has fixed it but you should probably work out where the missed train '
                      'should be called')

        if current_x.ndim == 1:
            current_x = current_x[None, :]
        if historic_y.ndim == 1:
            historic_y = historic_y[:, None]

        # Add on current_x
        if 0 == num_previous_steps:
            X = current_x
            y = np.array([np.nan])
        else:
            X = np.vstack([historic_x[-num_previous_steps:, :], current_x])
            y = np.concatenate([historic_y[-num_previous_steps:], [[np.nan]]])             # current_y is unknown

        y_input, _ = ProDriver.format_ar_arrays_for_y(X, y, num_previous_steps)           # (1, [x_t, {y_t-i, x_t-i}])

        ys = np.zeros(num_forecast_steps)
        for i in range(num_forecast_steps):
            # Predict current y
            ys[i] = model_y.predict(y_input)
            if bound_y:
                ys[i] = max(min(ys[i], 1.), 0.)

            # Predict next x
            x_input = np.atleast_2d(np.insert(y_input, 0, ys[i]))   # input for x_t+1 has additonal y_t inserted at the start
            x_next = model_x.predict(x_input)
            if bound_x:
                x_next = np.minimum(np.maximum(x_next, 0), 100)

            # Update input for next y
            y_input = np.hstack([x_next, x_input[:, :-(D + 1)]])
        return ys

    @staticmethod
    def format_ar_arrays_for_y(X, y, num_previous_steps=0):
        # Format two arrays used to predict the current value of y given the current value of x and num_previous_steps
        # of x and y.
        # y_inputs will include x at the current time step, x at the num_previous_steps time points and y at the
        # num_previous_steps time points. Hence number of cols = D*(num_previous_steps + 1) + num_previous_steps
        #
        #   num_previous_steps = 2
        #       X = [ x_00, x_01, x_02 ]          y = [ y_0 ]
        #           [ x_10, x_11, x_12 ]              [ y_1 ]
        #           [ x_20, x_21, x_22 ]              [ y_2 ]
        #           [ x_30, x_31, x_32 ]              [ y_3 ]
        #           [ x_40, x_41, x_42 ]              [ y_4 ]
        #
        #                     |- current_x -|   |-------   previous_y, previous_x   -------|
        #       y_inputs =  [ x_20, x_21, x_22, y_1, x_10, x_11, x_12, y_0, x_00, x_01, x_02 ]
        #                   [ x_30, x_31, x_32, y_2, x_20, x_21, x_22, y_1, x_10, x_11, x_12 ]
        #                   [ x_40, x_41, x_42, y_3, x_30, x_31, x_32, y_2, x_20, x_21, x_22 ]
        #
        #       y_targets = [ y_2 ]
        #                   [ y_3 ]
        #                   [ y_4 ]

        N, D = X.shape
        y = y.ravel()
        y_inputs = np.zeros((N - num_previous_steps, D + (D + 1) * num_previous_steps))
        for i in range(num_previous_steps + 1):
            y_inputs[:, i * (D+1):i * (D+1) + D] = X[num_previous_steps - i:N-i, :]
        for i in range(num_previous_steps):
            y_inputs[:, D + i * (D + 1)] = y[num_previous_steps - i - 1:N - i - 1]

        y_targets = y[num_previous_steps:]


        return y_inputs, y_targets[:, None]

    @staticmethod
    def format_ar_arrays_for_x(X, y, num_previous_steps=0):
        # Format two arrays used to predict the next value of x given the current value of x and y and
        # num_previous_steps of x and y. Hence number of cols = D*(num_previous_steps + 1) + num_previous_steps
        #
        #   num_previous_steps = 2
        #       X = [ x_00, x_01, x_02 ]          y = [ y_0 ]
        #           [ x_10, x_11, x_12 ]              [ y_1 ]
        #           [ x_20, x_21, x_22 ]              [ y_2 ]
        #           [ x_30, x_31, x_32 ]              [ y_3 ]
        #           [ x_40, x_41, x_42 ]              [ y_4 ]
        #
        #                     |-- cur_y, cur_x --|   |-------   previous_y, previous_x   -------|
        #       x_inputs =  [ y_2, x_20, x_21, x_22, y_1, x_10, x_11, x_12, y_0, x_00, x_01, x_02 ]
        #                   [ y_3, x_30, x_31, x_32, y_2, x_20, x_21, x_22, y_1, x_10, x_11, x_12 ]
        #
        #       x_targets = [x_30, x_31, x_32]
        #                   [x_40, x_41, x_42]
        #
        # Note, x_inputs has one more dimension and one fewer data point than y_inputs


        N, D = X.shape
        if y.ndim == 1:
            y = y[:, None]
        yx = np.hstack([y, X])
        x_inputs = np.hstack([yx[num_previous_steps-i:N-i-1, :] for i in range(num_previous_steps+1)])
        x_targets = X[num_previous_steps + 1:, :]
        return x_inputs, x_targets
