from drivers.driver import *
from drivers.learnerdriver import LearnerDriver


class YoungDriver(LearnerDriver):
    def __init__(self, name, random_action_probability=0.2, random_action_decay=0.99,
                 min_random_action_probability=0.005, speed_rounding=10, discount_factor=0.9, learning_rate=1,
                 max_distance=100, batch_learn=True):

        super().__init__(name, random_action_probability=random_action_probability,
                         random_action_decay=random_action_decay,
                         min_random_action_probability=min_random_action_probability,
                         speed_rounding=speed_rounding, discount_factor=discount_factor, learning_rate=learning_rate,
                         max_distance=max_distance)

        self.safety_car_speed = 150         # initialise it to be at a medium speed
        self.correct_turns = {}
        self.reward_cache = []
        self.state_counts = {}
        self.batch_learn = batch_learn

    def make_a_move(self, car_state: CarState, track_state: TrackState):
        state = self.get_state(car_state, track_state)
        self.state_counts[state] = self.state_counts.get(state, 0) + 1

        if track_state.safety_car_active:
            if car_state.speed > self.safety_car_speed:     # safety car active and we are above speed
                if self.print_info:
                    print(f'\tCar speed of {car_state.speed} above current safety car estimate of '
                          f'{self.safety_car_speed} so applying the brakes.')
                return Action.LightBrake                 # brake as over speed

        # Safety car not active, or we are under the speed already - use Q table to determine action
        return self._choose_move_from_q_table(car_state, track_state)

    def _choose_move_from_q_table(self, car_state: CarState, track_state: TrackState):
        # Pass it up to base class for now. Easy to override in separate function
        return super().make_a_move(car_state, track_state)

    def _choose_turn_direction(self, track_state: TrackState):
        # Check if we need to make a decision about which way to turn
        if track_state.distance_left > 0 and track_state.distance_right > 0:  # both options available, need to decide
            if len(self.correct_turns) > 0:
                # Find the closest turn we have seen previously and turn in the same direction
                distances = np.array([track_state.position.distance_to(turn_position)
                                      for turn_position in self.correct_turns])
                i_closest = np.argmin(distances)
                return list(self.correct_turns.values())[i_closest]

            else:  # First race, no data yet so choose randomly
                return driver_rng().choice([Action.TurnLeft, Action.TurnRight])

        elif track_state.distance_left > 0:  # only left turn
            return Action.TurnLeft

        else:
            return Action.TurnRight  # only right or dead-end

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
                                   action: Action, new_car_state: CarState, new_track_state: TrackState,
                                   result: ActionResult):

        if previous_track_state.safety_car_active:
            self._update_safety_car(new_car_state, result)

        self.reward_cache.append([previous_car_state, previous_track_state,
                                  action, new_car_state, new_track_state, result])
        if (previous_track_state.distance_ahead == 0) or not self.batch_learn:
            for i in range(len(self.reward_cache)):
                super().update_with_action_results(*self.reward_cache.pop(-1))

    def _update_safety_car(self, new_car_state: CarState, result: ActionResult):
        if result.safety_car_speed_exceeded:  # we ended up going too fast so safe speed must be below current speed
            if new_car_state.speed - 10 < self.safety_car_speed:
                self.safety_car_speed = new_car_state.speed - 10
                if self.print_info:
                    print(f'\tDecreasing estimate of safety car speed to {self.safety_car_speed: .1f}')
            elif self.print_info:
                print(f'Safety car speed estimate of {self.safety_car_speed: .1f} already below car speed of '
                      f'{new_car_state.speed: .1f}')

        else:  # our current speed is safe, so safety car speed must be higher
            if new_car_state.speed + 1 > self.safety_car_speed:
                self.safety_car_speed = new_car_state.speed + 1
                if self.print_info:
                    print(f'\tIncreasing estimate of safety car speed to {self.safety_car_speed: .1f}')

    def update_after_race(self, correct_turns: Dict[Position, Action]):
        # Called after the race by RaceControl
        self.correct_turns.update(correct_turns)            # dictionary mapping Position -> TurnLeft or TurnRight

    def get_safety_car_speed_estimate(self):
        return self.safety_car_speed


class YoungDriver1(YoungDriver):
    def __init__(self, name, exploration_bonus=100, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.exploration_bonus = exploration_bonus
        self.state_action_counts = defaultdict(dict)

    def _choose_move_from_q_table(self, car_state: CarState, track_state: TrackState):
        # Extract the elements of car state and track state that we care about into a state vector
        state = self.get_state(car_state, track_state)

        # Get the list of possible moves.
        available_actions = self._get_available_actions_for_state(track_state, car_state)

        # Make sure this state and action exist in Q table, adding them with default values if they don't
        self._ensure_state_action_in_q_table(state, available_actions)

        untested_actions = [action for action in available_actions if action not in self.state_action_counts[state]]
        if len(untested_actions) > 0:
            available_actions = untested_actions

        else:
            # We have tried all available actions at least once in this state
            num_times_been_in_state = sum(self.state_action_counts[state].values())
            bonuses = {action: self.exploration_bonus * np.sqrt(
                        np.log(num_times_been_in_state) / self.state_action_counts[state][action])
                       for action in available_actions}
            value_dict = {action: self.q_table[state][action] + bonuses[action] for action in available_actions}
            max_value = max(value_dict.values())
            available_actions = [action for action in available_actions if value_dict[action] == max_value]

        # Choose action randomly from remaining list of action
        action = driver_rng().choice(available_actions)
        self.state_action_counts[state][action] = self.state_action_counts[state].get(action, 0) + 1

        # If we have chosen to turn, work out which way
        if self.turn_action == action:
            return self._choose_turn_direction(track_state)

        else:
            return action


class YoungDriverNearestState(YoungDriver):
    def _choose_move_from_q_table(self, car_state: CarState, track_state: TrackState):
        # Extract the elements of car state and track state that we care about into a state vector
        state = self.get_state(car_state, track_state)

        # Get the list of possible moves.
        available_actions = self._get_available_actions_for_state(track_state, car_state)

        value_dict = None
        if state not in self.q_table:
            speeds_with_same_distance = [seen_state[0] for seen_state in self.q_table.keys()
                                         if seen_state[1] == state[1] and seen_state[0] > 0]
            if len(speeds_with_same_distance) > 0:
                i = np.argmin((state[0] - np.array(speeds_with_same_distance)) ** 2)
                value_dict = self.q_table[(speeds_with_same_distance[i], state[1])]

        # Make sure this state and action exist in Q table, adding them with default values if they don't
        self._ensure_state_action_in_q_table(state, available_actions)

        # Test if we are taking a random action or if we are using the Q table
        if driver_rng().rand() > self.random_action_probability:
            # Not taking a random action so find the action with the highest value for this state in the Q table.
            # If there is more than one action with the highest value we will choose randomly from them
            if value_dict is None:
                value_dict = self.q_table[state]
            max_value = max([value_dict[action] for action in available_actions])
            available_actions = [action for action in available_actions if value_dict[action] == max_value]

        # Choose action randomly from remaining list of action
        action = driver_rng().choice(available_actions)

        # Update randomness
        self.random_action_probability = max(self.random_action_probability * self.random_action_decay,
                                             self.min_random_action_probability)

        # If we have chosen to turn, work out which way
        if self.turn_action == action:
            return self._choose_turn_direction(track_state)

        else:
            return action
