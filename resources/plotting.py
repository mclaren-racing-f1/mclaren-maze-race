import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle
import matplotlib.colors as colours
import numpy as np
import os
import json
from typing import List
from IPython.display import clear_output, display, clear_output

from resources.actions import Action
from resources.states import *
from resources.car import *


class RacePlotterOneCar:
    def __init__(self, car, track, drs, safety_car=True, pitstops=True, weather=True, save=False):
        self.car = car
        self.track = track
        self.safety_car = safety_car
        self.pitstops = pitstops
        self.weather = weather
        self.save_flag = save

        self.num_crashes = 0
        self.num_spins = 0
        self.safety_car_start = None
        self.num_pit_stops = 0

        # Set up figure and axes
        self.figure = plt.figure(figsize=(9, 5))
        clear_output()
        self.hdisplay = display(self.figure, display_id=True)
        self.ax_track = self.figure.add_axes([0.05, 0.05, 0.45, 0.9])

        self.ax_time = self.figure.add_axes([0.55, 0.85, 0.4, 0.1], frame_on=False)
        self.ax_speed = self.figure.add_axes([0.55, 0.58, 0.4, 0.25])
        self.ax_actions = self.figure.add_axes([0.55, 0.25, 0.4, 0.3], frame_on=False)
        self.ax_race = self.figure.add_axes([0.55, 0.14, 0.4, 0.07], frame_on=False)
        self.ax_results = self.figure.add_axes([0.55, 0.05, 0.4, 0.07], frame_on=False)

        # Plot track
        track.plot_track(self.ax_track, show_drs=drs)
        self.mask = np.zeros((track.num_rows, track.num_cols, 4))
        self.mask[:, :, :3] = 0.7                                       # grey for unexplored areas
        self.mask[:, :, 3] = 1                                          # fully masked to start with
        self.mask_image = self.ax_track.imshow(self.mask, zorder=3)

        # Plot position
        position, heading = track.get_start_position_and_heading()
        self.car_marker = self.ax_track.plot(position.column, position.row, self.get_marker_from_heading(heading))[0]

        # Plot move & time
        self.move_text = self.ax_time.text(0, 1, 'Move #: 0', fontsize=20, va='top')
        self.time_text = self.ax_time.text(1, 1, 'Time: 0.0', fontsize=20, ha='right', va='top')
        self.ax_time.set_xlim([0, 1])
        self.ax_time.set_ylim([0.5, 1])
        self.ax_time.xaxis.set_visible(False)
        self.ax_time.yaxis.set_visible(False)

        # Plot speed trace
        self.speeds = [0]
        self.speed_trace = self.ax_speed.plot(0, 0, '.-')[0]
        self.ax_speed.grid(True)
        trans = transforms.blended_transform_factory(self.ax_speed.transData, self.ax_speed.transAxes)
        self.safety_car_rect = Rectangle([np.nan, 0], 0, 1, transform=trans, color='y', alpha=0.1)
        self.ax_speed.add_patch(self.safety_car_rect)
        self.ax_speed.set_ylim([0, 350])

        # Add action text
        fs = 12
        self.actions_text = {
            Action.Continue: self.ax_actions.text(0, 1, 'Continue', va='top', fontsize=fs),
            Action.LightThrottle: self.ax_actions.text(1, 1, 'Light Throttle', va='top', fontsize=fs),
            Action.FullThrottle: self.ax_actions.text(2, 1, 'Full Throttle', va='top', fontsize=fs),
            Action.LightBrake: self.ax_actions.text(1, 0.5, 'Light Brake', va='center', fontsize=fs),
            Action.HeavyBrake: self.ax_actions.text(2, 0.5, 'Heavy Brake', va='center', fontsize=fs),
            Action.TurnLeft: self.ax_actions.text(1, 0, 'Turn Left', va='bottom', fontsize=fs),
            Action.TurnRight: self.ax_actions.text(2, 0, 'Turn Right', va='bottom', fontsize=fs),
        }
        if drs:
            self.actions_text[Action.OpenDRS] = self.ax_actions.text(0, 0.5, 'DRS Active', va='center', fontsize=fs)
        if pitstops:
            self.actions_text[Action.ChangeTyres] = self.ax_actions.text(0, 0, 'Pit Stops: #0', va='bottom',
                                                                         fontsize=fs)
        self.ax_actions.set_ylim([0, 1.5])
        self.ax_actions.set_xlim([0, 2.5])
        self.ax_actions.xaxis.set_visible(False)
        self.ax_actions.yaxis.set_visible(False)

        # Race state text
        self.race_state_text = {}
        if safety_car:
            self.race_state_text['safety_car'] = self.ax_race.text(0, 0.5, 'Safety Car', va='center', ha='left',
                                                                   fontsize=fs)
        if weather:
            self.race_state_text['rain'] = self.ax_race.text(1, 0.5, f'Rain: 0', va='center', ha='right', fontsize=fs)
        self.ax_race.xaxis.set_visible(False)
        self.ax_race.yaxis.set_visible(False)

        # Result text
        self.result_text = {
            'spun': self.ax_results.text(0, 0, 'Spin: #0', fontsize=16),
            'crashed': self.ax_results.text(1, 0, 'Crash: #0', fontsize=16, ha='center'),
            'finished': self.ax_results.text(2, 0, 'Finished', fontsize=16, ha='right')
        }
        self.ax_results.xaxis.set_visible(False)
        self.ax_results.yaxis.set_visible(False)
        self.ax_results.set_xlim([0, 2])
        self.ax_results.set_ylim([0, 0.5])

        # Draw
        self.hdisplay.update(self.figure)

        # Save
        if self.save_flag:
            self.save(0)

    def update(self, position, heading, action: Action, result: ActionResult, move_number: int, race_time: float,
               safety_car_active: bool, weather_state: WeatherState):
        # Update fog-of-war
        track_state = self.track.get_state_for_position(position, heading)
        track_left, track_right, track_up, track_down = heading.rotate_from_car_to_track(track_state)
        self.mask[position.row, position.column:position.column + track_right + 2, 3] = 0
        self.mask[position.row, position.column:position.column - track_left - 2:-1, 3] = 0
        self.mask[position.row:position.row + track_down + 2, position.column, 3] = 0
        self.mask[position.row:position.row - track_up - 2:-1, position.column, 3] = 0
        self.mask_image.set_data(self.mask)

        # Update position
        self.car_marker.set_data((position.column, position.row))
        self.car_marker.set_marker(self.get_marker_from_heading(heading))

        # Update move and time
        self.move_text.set_text(f'Move #: {move_number}')
        self.time_text.set_text(f'Time: {race_time: .1f}')

        # Update speed trace
        horizon = 20
        self.speeds.append(self.car.speed)
        x = np.arange(max(0, len(self.speeds) - horizon), len(self.speeds))
        y = self.speeds[-horizon:]
        self.speed_trace.set_data(x, y)
        self.ax_speed.set_xlim(np.min(x), np.max(x))
        # self.ax_speed.set_ylim(np.min(y) - 1, np.max(y) + 1)

        # Safety car
        if safety_car_active and self.safety_car:
            if self.safety_car_start is None:
                self.safety_car_start = move_number
                self.safety_car_rect.set_x(move_number)
            else:
                self.safety_car_rect.set_width(move_number - self.safety_car_start)
        elif self.safety_car_start is not None:
            self.safety_car_start = None

        # Highlight action
        for act, object in self.actions_text.items():
            object.set_backgroundcolor('orange' if action == act else 'none')
        if Action.OpenDRS in self.actions_text:
            self.actions_text[Action.OpenDRS].set_backgroundcolor('orange' if self.car.drs_active else 'none')
        if Action.ChangeTyres == action:
            self.num_pit_stops += 1
        if self.pitstops and Action.ChangeTyres in self.actions_text:
            self.actions_text[Action.ChangeTyres].set_text(f'Pit Stops: #{self.num_pit_stops}')

        # Highlight race state
        if self.safety_car and 'safety_car' in self.race_state_text:
            self.race_state_text['safety_car'].set_backgroundcolor('orange' if safety_car_active else 'none')
        if self.weather and 'rain' in self.race_state_text:
            self.race_state_text['rain'].set_backgroundcolor('lightblue' if weather_state.rain_intensity > 0 else
                                                             'none')
            self.race_state_text['rain'].set_text(f'Rain: {weather_state.rain_intensity: .1f}')

        # Highlight result
        self.num_crashes += result.crashed
        self.num_spins += result.spun
        self.result_text['spun'].set_text(f'Spin: #{self.num_spins}')
        self.result_text['crashed'].set_text(f'Crash: #{self.num_crashes}')
        self.result_text['spun'].set_backgroundcolor('orange' if result.spun else 'none')
        self.result_text['crashed'].set_backgroundcolor('orange' if result.crashed else 'none')
        self.result_text['finished'].set_backgroundcolor('orange' if result.finished else 'none')

        # Crashed figure colour
        self.figure.set_facecolor('r' if result.crashed else 'w')

        # Draw
        self.hdisplay.update(self.figure)

        # Save if required
        if self.save_flag:
            self.save(move_number)

    def save(self, move_number):
        plt.savefig(rf'media\race_{move_number:04d}.png')

    @staticmethod
    def get_marker_from_heading(heading):
        if 1 == heading.row:
            return 'v'
        elif -1 == heading.row:
            return'^'
        elif 1 == heading.column:
            return '>'
        else:
            return '<'


def save_season_test_results(filename: str, test_name: str, race_times_array: np.ndarray, finished_array: np.ndarray):
    if not filename.endswith('.json'):
        filename += '.json'

    # Read out any existing results
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    # Add new results
    results[test_name] = {'race_times': race_times_array.tolist(), 'finished': finished_array.tolist()}

    # Write to file
    with open(filename, 'w+') as f:
        json.dump(results, f)


def plot_season_test_results(filename: str):
    if not filename.endswith('.json'):
        filename += '.json'
    if not os.path.isfile(filename):
        raise ValueError(f'Cannot file a file named {filename}')

    # Load all results
    with open(filename, 'r') as f:
        results = json.load(f)

    race_times = [np.array(test_result['race_times']) for test_result in results.values()]


def plot_race_time_boxplots(list_of_race_time_arrays, test_labels=None):

    # Plot box plots of race time results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    num_tests = len(list_of_race_time_arrays)
    if 1 == num_tests:
        offsets = [0]
        width = 0.5
    else:
        offsets = np.linspace(-0.25, 0.25, num_tests)
        width = 0.9*(offsets[1] - offsets[0])

    num_tracks = 0
    legend_lines = []
    for i, race_times in enumerate(list_of_race_time_arrays):
        ax.boxplot(race_times, positions=np.arange(race_times.shape[1]) + offsets[i], widths=width, patch_artist=True,
                   boxprops={'color': f'C{i}', 'facecolor': 'none'})
        num_tracks = max(num_tracks, race_times.shape[1])
        legend_lines.append(plt.Line2D([], [], color=f'C{i}'))

    ax.set_xlabel('Track Number', fontsize=14)
    ax.set_ylabel(f'Race Time Over Multiple Season Repeats', fontsize=14)
    ax.set_xticks(range(1, num_tracks + 1))
    if test_labels is None:
        test_labels = [f'Test {i + 1}' for i in range(num_tests)]
    ax.legend(legend_lines, test_labels)


def plot_q_table(driver):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_axes([0.07, 0.08, 0.93, 0.82])
    max_dist = 0
    colours = {Action.Continue: 'b', Action.LightBrake: 'orange', Action.HeavyBrake: 'r',
               Action.LightThrottle: (0, 1, 0.5), Action.FullThrottle: 'g', driver.turn_action: 'k'}
    for state, value_dict in driver.q_table.items():
        max_value = max(value_dict.values())
        action = next(act for act, val in value_dict.items() if val == max_value)
        ax.text(state[0], state[1], f'{max_value:.0f}', ha='center', va='center', c=colours[action], fontsize=8)
        max_dist = max(max_dist, state[1])

    ax.set_xlim([-5, 305])
    ax.set_ylim([-0.5, max_dist + 0.5])
    ax.yaxis.set_ticks(range(max_dist + 1))
    ax.yaxis.set_ticks(np.arange(0.5, max_dist + 0.5), minor=True)
    ax.grid(True, which='minor', axis='y')
    ax.set_xlabel('Speed', fontsize=16)
    ax.set_ylabel('Distance ahead', fontsize=16)

    key_ax = fig.add_axes([0.08, 0.91, 0.9, 0.05], frame_on=False)
    x = 0
    for action, colour in colours.items():
        key_ax.text(x, 0, action if isinstance(action, str) else action.name, color=colour, fontsize=12)
        x += 1
    key_ax.set_xlim([0, x-0.2])
    key_ax.xaxis.set_ticks([])
    key_ax.yaxis.set_ticks([])


def plot_straight_line_sim(actions: List[Action], speeds: np.ndarray):
    fig = plt.figure(figsize=(9, 6))
    speed_ax = fig.add_axes([0.08, 0.35, 0.9, 0.65])
    action_ax = fig.add_axes([0.08, 0, 0.9, 0.25], frame_on=False)

    speed_ax.plot(speeds)
    speed_ax.set_ylabel('Speed', fontsize=18)
    speed_ax.set_xlabel('Distance ahead', fontsize=18)
    speed_ax.set_xlim([0, len(speeds)])
    speed_ax.xaxis.set_ticks(np.arange(len(speeds)))
    speed_ax.xaxis.set_ticklabels(np.arange(len(speeds))[::-1])
    speed_ax.grid(True)

    for i, act in enumerate(actions):
        action_ax.text(i + 0.5, 0, act if isinstance(act, str) else act.name, fontsize=16, rotation=-90, ha='center',
                       va='top')
    action_ax.set_xlim([0, len(actions) + 1])
    action_ax.set_ylim([0, 0.6])
    action_ax.xaxis.set_ticks([])
    action_ax.yaxis.set_ticks([])
    action_ax.invert_yaxis()


def plot_car_dynamics(car):
    speed_in = np.linspace(0, 325, 1000)
    speed_time_ft = np.zeros(15)
    speed_time_lt = np.zeros(15)
    speed_time_hb = 299*np.ones(15)
    speed_time_lb = 299*np.ones(15)

    dynamics = car.dynamics_model

    speed_out_full_throttle = np.array([dynamics.full_throttle(s) for s in speed_in])
    speed_out_light_throttle = np.array([dynamics.light_throttle(s) for s in speed_in])
    speed_out_heavy_braking = np.array([dynamics.heavy_brake(s) for s in speed_in])
    speed_out_light_braking = np.array([dynamics.light_brake(s) for s in speed_in])

    for j in range(1, speed_time_ft.shape[0]):
        speed_time_ft[j] = dynamics.full_throttle(speed_time_ft[j - 1])
        speed_time_lt[j] = dynamics.light_throttle(speed_time_lt[j - 1])
        speed_time_hb[j] = dynamics.heavy_brake(speed_time_hb[j - 1])
        speed_time_lb[j] = dynamics.light_brake(speed_time_lb[j - 1])

    fig = plt.figure(figsize=(9, 5))
    ax_thr = fig.add_subplot(2, 2, 1)
    ft_lines = ax_thr.plot(speed_in, speed_out_full_throttle, 'r')
    lt_lines = ax_thr.plot(speed_in, speed_out_light_throttle, 'b')
    ax_thr.set_xlabel('Speed In')
    ax_thr.set_ylabel('Speed Out')
    ax_thr.set_title('Throttle')
    ax_thr.legend([ft_lines[0], lt_lines[0]], ['Full throttle', 'Light throttle'])

    # Braking transfer function
    ax_brake = fig.add_subplot(2, 2, 2)
    lb_lines = ax_brake.plot(speed_in, speed_out_light_braking, 'b')
    hb_lines = ax_brake.plot(speed_in, speed_out_heavy_braking, 'r')
    ax_brake.set_xlabel('Speed in')
    ax_brake.set_ylabel('Speed after braking')
    ax_brake.set_title('Braking')
    ax_brake.legend([hb_lines[0], lb_lines[0]], ['Heavy braking', 'Light braking'])

    # Throttle over time
    ax_time = fig.add_subplot(2, 2, 3)
    ax_time.plot(speed_time_ft, 'r')
    ax_time.plot(speed_time_lt, 'b')
    ax_time.set_xlabel('Time')
    ax_time.set_ylabel('Speed')
    ax_time.set_title('Throttle over Time')
    ax_time.legend(['Full throttle', 'Light throttle'])

    # Braking over time
    ax_time = fig.add_subplot(2, 2, 4)
    ax_time.plot(speed_time_hb, 'r')
    ax_time.plot(speed_time_lb, 'b')
    ax_time.set_xlabel('Time')
    ax_time.set_ylabel('Speed')
    ax_time.set_title('Braking over Time')
    ax_time.legend(['Heavy brake', 'Light brake'])

    fig.tight_layout()
    plt.show()


def plot_car_dynamics_full(car):
    speed_in = np.linspace(0, 325, 1000)
    speed_time_ft = np.zeros(15)
    speed_time_lt = np.zeros(15)
    speed_time_hb = 299*np.ones(15)
    speed_time_lb = 299*np.ones(15)

    dynamics = car.dynamics_model

    speed_out_full_throttle = np.array([dynamics.full_throttle(s) for s in speed_in])
    speed_out_light_throttle = np.array([dynamics.light_throttle(s) for s in speed_in])
    speed_out_heavy_braking = np.array([dynamics.heavy_brake(s) for s in speed_in])
    speed_out_light_braking = np.array([dynamics.light_brake(s) for s in speed_in])

    for j in range(1, speed_time_ft.shape[0]):
        speed_time_ft[j] = dynamics.full_throttle(speed_time_ft[j - 1])
        speed_time_lt[j] = dynamics.light_throttle(speed_time_lt[j - 1])
        speed_time_hb[j] = dynamics.heavy_brake(speed_time_hb[j - 1])
        speed_time_lb[j] = dynamics.light_brake(speed_time_lb[j - 1])

    fig = plt.figure(figsize=(9, 5))
    ax_thr = fig.add_subplot(2, 3, 1)
    ft_lines = ax_thr.plot(speed_in, speed_out_full_throttle, 'r')
    lt_lines = ax_thr.plot(speed_in, speed_out_light_throttle, 'b')
    ax_thr.set_xlabel('Speed In')
    ax_thr.set_ylabel('Speed Out')
    ax_thr.set_title('Throttle')
    ax_thr.legend([ft_lines[0], lt_lines[0]], ['Full throttle', 'Light throttle'])

    ax_thr = fig.add_subplot(2, 3, 4)
    ft_lines = ax_thr.plot(speed_in, speed_out_full_throttle - speed_in, 'r')
    lt_lines = ax_thr.plot(speed_in, speed_out_light_throttle - speed_in, 'b')
    ax_thr.set_xlabel('Speed In')
    ax_thr.set_ylabel('Delta Speed')
    ax_thr.set_title('Throttle')
    ax_thr.legend([ft_lines[0], lt_lines[0]], ['Full throttle', 'Light throttle'])

    # Braking transfer function
    ax_brake = fig.add_subplot(2, 3, 2)
    lb_lines = ax_brake.plot(speed_in, speed_out_light_braking, 'b')
    hb_lines = ax_brake.plot(speed_in, speed_out_heavy_braking, 'r')
    ax_brake.set_xlabel('Speed in')
    ax_brake.set_ylabel('Speed after braking')
    ax_brake.set_title('Braking')
    ax_brake.legend([hb_lines[0], lb_lines[0]], ['Heavy braking', 'Light braking'])

    # Braking deltas
    ax_brake = fig.add_subplot(2, 3, 5)
    lb_lines = ax_brake.plot(speed_in, speed_out_light_braking - speed_in, 'b')
    hb_lines = ax_brake.plot(speed_in, speed_out_heavy_braking - speed_in, 'r')
    ax_brake.set_xlabel('Speed in')
    ax_brake.set_ylabel('Delta Speed')
    ax_brake.set_title('Braking')
    ax_brake.legend([hb_lines[0], lb_lines[0]], ['Heavy braking', 'Light braking'])

    # Throttle over time
    ax_time = fig.add_subplot(2, 3, 3)
    ax_time.plot(speed_time_ft, 'r')
    ax_time.plot(speed_time_lt, 'b')
    ax_time.set_xlabel('Time')
    ax_time.set_ylabel('Speed')
    ax_time.set_title('Throttle over Time')
    ax_time.legend(['Full throttle', 'Light throttle'])

    # Braking over time
    ax_time = fig.add_subplot(2, 3, 6)
    ax_time.plot(speed_time_hb, 'r')
    ax_time.plot(speed_time_lb, 'b')
    ax_time.set_xlabel('Time')
    ax_time.set_ylabel('Speed')
    ax_time.set_title('Braking over Time')
    ax_time.legend(['Heavy brake', 'Light brake'])

    fig.tight_layout()
    plt.show()


def plot_multiple_championship_results(championship_results: dict):
    num_drivers = len(championship_results)
    scores = np.zeros((num_drivers, num_drivers))

    # Green to red colourmap
    pos_colours = np.hstack([np.linspace(0, 1, num_drivers)[:, None], np.linspace(1, 0, num_drivers)[:, None],
                             np.zeros((num_drivers, 1)), np.ones((num_drivers, 1))])
    colour_array = np.tile(np.reshape(pos_colours, (num_drivers, 1, 4)), [1, num_drivers, 1])  # (num_drivers, num_drivers, 4)

    for i, positions in enumerate(championship_results.values()):
        num_repeats = len(positions)
        for j in range(num_drivers):        # loop over championship ranks
            scores[j, i] = np.sum(positions == (j + 1)) / num_repeats
            colour_array[j, i, -1] = scores[j, i]                   # set transparency according to normalised count

    fig = plt.figure(figsize=(9, 5))
    ax = fig.gca()
    ax.imshow(colour_array)
    ax.set_xticks(np.arange(num_drivers))
    ax.set_xticklabels(list(championship_results.keys()))
    ax.set_ylabel('Championship Position', fontsize=12)
    ax.set_yticks(np.arange(num_drivers))
    ax.set_yticklabels(1 + np.arange(num_drivers))
    ax.set_title('Number of times a driver finished in each position', fontsize=12)

    # Add text labels
    for row in range(num_drivers):
        for col in range(num_drivers):
            ax.text(col, row, f'{scores[row, col]*100: .0f}%', va='center', ha='center')


def plot_grip_sweep():
    car = Car.get_car_for_level(Level.Pro)
    grips = [1.2, 1.0, 0.8, 0.6, 0.4]
    straight_length = 16
    fig = plt.figure(figsize=(12, 5))
    ax_speed = fig.add_axes([0.05, 0.05, 0.6, 0.8])
    ax_time = fig.add_axes([0.7, 0.05, 0.2, 0.8])
    times = np.zeros(len(grips))
    for ig, grip in enumerate(grips):
        speeds = 300 * np.ones(straight_length)
        speeds[-1] = car.dynamics_model.max_cornering_speed(grip_multiplier=grip)

        # Braking first
        for i in range(2, straight_length):
            input_speed = speeds[-i+1] + 10
            next_speed = 0
            while next_speed <= speeds[-i+1]:
                input_speed += 1
                next_speed = np.minimum(car.dynamics_model.light_brake(input_speed, grip_multiplier=grip),
                                        car.dynamics_model.heavy_brake(input_speed, grip_multiplier=grip))
            speeds[-i] = input_speed - 1
            if speeds[-i] > 300:
                break

        # Now acceleration
        speeds[0] = 0
        for i in range(1, straight_length):
            new_speed = np.maximum(car.dynamics_model.light_throttle(speeds[i-1], grip_multiplier=grip),
                                   car.dynamics_model.full_throttle(speeds[i - 1], grip_multiplier=grip))
            if new_speed >= speeds[i]:
                break
            else:
                speeds[i] = new_speed

        # Compute time
        times[ig] = np.sum(1 / (1 + speeds))

        # Plot
        x = np.arange(straight_length)[::-1]
        ax_speed.plot(x, speeds, label=f'Grip = {grip}')

    ax_speed.set_xlabel('Distance to end of straight', fontsize=12)
    ax_speed.set_ylabel('Maximum speed', fontsize=12)
    ax_speed.set_xticks(x)
    ax_speed.invert_xaxis()
    ax_speed.legend()

    p_time = times/times[1]*100
    ax_time.plot(grips, p_time, '+-')
    ax_time.set_xlabel('Grip multiplier', fontsize=12)
    ax_time.set_ylabel('Relative straight race time', fontsize=12)
    ax_time.set_yticks(p_time)
    ax_time.set_yticklabels([f'{p: .1f}%' for p in p_time])
    ax_time.invert_xaxis()
    ax_time.yaxis.tick_right()

def plot_tyre_degradation(n_samples=1):
    age = np.arange(0, 500)
    fig = plt.figure(figsize=(9, 5))
    ax = fig.gca()
    colours = {TyreChoice.Soft: 'orange', TyreChoice.Medium: 'purple', TyreChoice.Hard: 'b'}
    lines = {}

    for _ in range(n_samples):
        tyre_model = TyreModel()
        for choice in TyreChoice.get_choices():
            tyre_model.new_tyres_please(choice)         # loads up a new degradation curve
            lines[choice] = ax.plot(age, tyre_model.get_grip(age), c=colours[choice])[0]

    ax.set_xlabel('Tyre Age', fontsize=12)
    ax.set_ylabel('Grip Multiplier', fontsize=12)
    ax.legend(lines.values(), [choice.name for choice in lines], fontsize=12)


def make_sigmoid_figure():
    tyre_model = TyreModel()
    tyre_model.new_tyres_please(TyreChoice.Medium, set_number=0)
    ages = np.arange(300)
    grips = tyre_model.get_grip(ages)

    def sigmoid(a, b, c, d, age):
        return a + b / (1 + np.exp(-c + age / d))

    T = 100
    fig = plt.figure(figsize=(9, 4))
    plt.plot(ages, sigmoid(0.31, 0.7, 10, 20, ages))
    plt.plot(ages, sigmoid(-0.5, 1.5, 10, 25, ages))
    plt.plot(ages, sigmoid(0.9, 0.1, 10, 25, ages))
    plt.plot(ages, sigmoid(1.2, -0.2, 12, 18, ages))
    plt.plot(ages, sigmoid(0, 1, 110, 1, ages))
    plt.plot(ages, grips, c='k', label='True curve')
    plt.plot(ages[:T], grips[:T], 'r+', label='Observed data')
    plt.xlabel('Tyre age')
    plt.ylabel('Grip')
    plt.legend();
    fig.tight_layout()
