import scipy.interpolate

from drivers.driver import *
from drivers.youngdriver import YoungDriver


class RookieDriver(YoungDriver):
    def __init__(self, name, random_action_probability=0.5, random_action_decay=0.99,
                 min_random_action_probability=0.0, *args, **kwargs):

        super().__init__(name, random_action_probability=random_action_probability,
                         random_action_decay=random_action_decay,
                         min_random_action_probability=min_random_action_probability, *args, **kwargs)

        self.sl_data = {action: [] for action in Action.get_sl_actions()}
        self.drs_data = {action: [] for action in [Action.LightThrottle, Action.FullThrottle, Action.Continue]}
        self.end_of_straight_speed = 350        # initialising to something large
        self.lowest_crash_speed = 350
        self.target_speeds = 350 * np.ones(50)
        self.drs_was_active = False

    def prepare_for_race(self):
        pass

    def make_a_move(self, car_state: CarState, track_state: TrackState, **kwargs) -> Action:
        if track_state.distance_ahead == 0 and not (track_state.distance_left == 0 and track_state.distance_right == 0
                                                    and car_state.speed > 0):
            return self._choose_turn_direction(track_state)

        # Get the target speed
        target_speed = self._get_target_speed(track_state.distance_ahead, track_state.safety_car_active)

        # Choose action that gets us closest to target, or choose randomly
        if driver_rng().rand() > self.random_action_probability:
            action = self._choose_move_from_models(car_state.speed, target_speed, car_state.drs_active)
        else:
            action = self._choose_randomly(Action.get_sl_actions())

        # If DRS is available then need to decide whether to open DRS or not.
        if track_state.drs_available and not car_state.drs_active:
            # Simulate the straight with and without DRS and check which we think will be faster
            time_no_drs, targets_broken_no_drs, _ = self.simulate_straight(car_state.speed,
                                                                        track_state.distance_ahead,
                                                                        drs_active=False,
                                                                        safety_car_active=track_state.safety_car_active)
            time_drs, targets_broken_drs, _ = self.simulate_straight(car_state.speed, track_state.distance_ahead - 1,
                                                                     drs_active=True,
                                                                     safety_car_active=track_state.safety_car_active)
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

        if distance_ahead == 0:
            target_speed = 0                                            # dead end - need to stop!!
        else:
            target_speed = target_speeds[distance_ahead - 1]       # target for next step

        if safety_car_active:
            target_speed = min(target_speed, self.safety_car_speed)

        return target_speed

    def get_data(self, action, drs_active=False):
        if drs_active and action in self.drs_data:
            return self.drs_data[action]
        else:
            return self.sl_data[action]

    def estimate_next_speed(self, action: Action, speed, drs_active: bool, **kwargs):
        data = np.array(self.get_data(action, drs_active))
        if data.shape[0] < 2:
            return speed
        interp = scipy.interpolate.interp1d(data[:, 0], data[:, 1], fill_value='extrapolate', assume_sorted=False)
        return interp(speed)

    def estimate_previous_speed(self, test_input_speeds: np.ndarray, test_output_speeds: np.ndarray, speed):
        errors = (test_output_speeds - speed)**2
        speeds_min_error = test_input_speeds[errors == np.min(errors)]
        return np.max(speeds_min_error)

    def _choose_move_from_models(self, speed: float, target_speed: float, drs_active: bool, **kwargs):
        # Test each action to see which will get us closest to our target speed
        actions = Action.get_sl_actions()
        if 0 == speed:  # yes this is technically cheating but you can get stuck here with low grip so bending the rules
            actions = [Action.LightThrottle, Action.FullThrottle]
        next_speeds = np.array([self.estimate_next_speed(action, speed, drs_active, **kwargs) for action in actions])
        errors = next_speeds - target_speed    # difference between predicted next speed and target, +ve => above target

        # The target speed is the maximum safe speed so we want to be under the target if possible. This means we don't
        # necessarily want the action with the smallest error
        if np.any(errors <= 0):            # under or equal to the target speed
            errors[errors > 0] = np.inf    # at least one action gets us under the speed so ignore others even if close

        # Now we can choose the action with the smallest error score. At the start there will be multiple actions with
        # with the same score, so we will choose randomly from these
        min_error = np.min(errors ** 2)
        available_actions = [action for action, error in zip(actions, errors)
                             if np.abs(error ** 2 - min_error) < 1e-3]

        return self._choose_randomly(available_actions)

    def simulate_straight(self, speed, distance_ahead, drs_active, safety_car_active):
        speeds = np.zeros(distance_ahead)
        break_target_speed = False
        for d in range(distance_ahead):
            target_speed = self._get_target_speed(distance_ahead - d, safety_car_active)
            action = self._choose_move_from_models(speed, target_speed, drs_active)
            speeds[d] = self.estimate_next_speed(action, speed, drs_active)
            speed = speeds[d]
            break_target_speed |= speed > target_speed
        time = np.sum(1 / (speeds + 1))
        return time, break_target_speed, speeds

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
                                   action: Action, new_car_state: CarState, new_track_state: TrackState,
                                   result: ActionResult):

        if previous_track_state.safety_car_active:
            self._update_safety_car(previous_car_state, result)

        if previous_track_state.distance_ahead == 0:

            if result.crashed or result.spun:
                if self.print_info:
                    print(f'\tCrashed! We targeted {self.end_of_straight_speed: .0f} speed and were going '
                          f'{previous_car_state.speed: .0f}')
                self.end_of_straight_speed = min(self.end_of_straight_speed, previous_car_state.speed - 10)
                if previous_track_state.distance_left > 0 or previous_track_state.distance_right > 0:
                    self.lowest_crash_speed = min(previous_car_state.speed, self.lowest_crash_speed)
            else:
                self.end_of_straight_speed = min(max(self.end_of_straight_speed, previous_car_state.speed + 1),
                                                 self.lowest_crash_speed)

            # Update our target speeds
            self.update_target_speeds()
            self.drs_was_active = False

        elif action in self.sl_data:       # record the change in speed resulting from the action we took
            # Record the point if it is not on top of another point (interpolation doesn't like points too close
            # together in x, plus it is also a bit unnecessary) and we are below 200 points (just to keep code
            # performance up)
            current_data = self.get_data(action, previous_car_state.drs_active)
            if 0 == len(current_data):
                closest_distance = 1000
            else:
                closest_distance = np.min(np.abs(np.array(current_data)[:, 0] - previous_car_state.speed))
            if closest_distance > 1 and len(current_data) < 200:
                new_data = [previous_car_state.speed, new_car_state.speed]
                current_data.append(new_data)

    def update_target_speeds(self):
        previous_targets = np.copy(self.target_speeds)
        speed = self.end_of_straight_speed

        test_input_speeds = np.linspace(0, 350, 351)
        test_output_speeds = {action: self.estimate_next_speed(action, test_input_speeds, False)
                              for action in self.sl_data}

        for i in range(len(self.target_speeds)):
            self.target_speeds[i] = speed
            speed = np.nanmax([self.estimate_previous_speed(test_input_speeds, test_output_speeds[action], speed)
                               for action in self.sl_data])
        if self.print_info and not np.array_equal(previous_targets, self.target_speeds):
            print(f'New target speeds: mid-straight->{np.array2string(self.target_speeds[5::-1], precision=0)}<-end')