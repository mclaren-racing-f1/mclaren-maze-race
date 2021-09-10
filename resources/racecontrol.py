from copy import deepcopy
import scipy.stats as ss
from time import time as time_fn

from resources.track import Track, TrackStore
from resources.car import Car, TyreModel
from drivers.driver import *
from resources.plotting import *
from resources.safetycar import SafetyCar
from resources.weatherstation import WeatherStation
from resources.actions import *
from resources.rng import set_seed


def race(driver=None, car=None, track=None, track_index=None, level=Level.Learner, max_number_of_steps=1000, plot=True,
         use_safety_car=True, use_drs=True, fixed_tyre_choice=None, fixed_aero_setup=None, exit_on_crash=False,
         use_weather=True, save_plots=False):
    t0 = time_fn()

    # Setup
    if driver is None:
        driver = Driver.get_driver_class_for_level(level)('Dando')
    if car is None:
        car = Car.get_car_for_level(level)
    if track is None:
        track = TrackStore.load_track(level=level, index=track_index)

    drs = use_drs and (level not in [Level.Learner, Level.Young])
    weather = use_weather and level == Level.Pro
    plotter = RacePlotterOneCar(car=car, track=track, drs=drs, safety_car=use_safety_car,
                                pitstops=level==Level.Pro, weather=weather, save=save_plots) \
                        if plot else None
    num_steps = 0
    race_time = 0
    finished = False
    num_crashes = 0
    safety_car = SafetyCar(print_details=plot)
    safety_car_penalty = 0
    weather_station = WeatherStation()

    current_position, start_heading = track.get_start_position_and_heading()
    track_info = track.get_track_info()

    # Prepare
    prepare_driver_car(driver, car, level=level, track_info=track_info, start_heading=start_heading, plot=plot,
                       fixed_tyre_choice=fixed_tyre_choice, fixed_aero_setup=fixed_aero_setup)
    weather_station.prepare_for_race()

    # Race
    while not track.is_finished(current_position) and num_steps < max_number_of_steps:

        driver, car, action, result, new_position, race_time_i, crashed, finished, safety_car_penalty = \
            take_race_turn(driver, track, car, level, current_position=current_position, track_info=track_info,
                           safety_car=safety_car, safety_car_penalty=safety_car_penalty,
                           weather_station=weather_station, use_safety_car=use_safety_car, use_drs=use_drs,
                           use_weather=use_weather, plot=plot)

        # New becomes current
        current_position = new_position
        race_time += race_time_i
        num_steps += 1
        num_crashes += crashed

        # Update plot
        if plot:
            plotter.update(current_position, car.heading, action=action, result=result, move_number=num_steps,
                           race_time=race_time, safety_car_active=safety_car.active,
                           weather_state=weather_station.get_state())

        if exit_on_crash and crashed:
            break

    # Race is complete, update driver with post race data
    driver.update_after_race(track.correct_turns)

    return driver, race_time, finished


def prepare_driver_car(driver, car, level, track_info, start_heading, plot=True, fixed_tyre_choice=None,
                       fixed_aero_setup=None):

    if level in [Level.Pro]:
        if fixed_tyre_choice is None:
            tyre_choice = driver.choose_tyres(track_info)
        else:
            tyre_choice = fixed_tyre_choice
        if fixed_aero_setup is None:
            aero_setup = driver.choose_aero(track_info)
        else:
            aero_setup = fixed_aero_setup
    else:
        tyre_choice = TyreChoice.Learner
        aero_setup = AeroSetup.Balanced

    driver.prepare_for_race()
    car.prepare_for_race(start_heading, tyre_choice=tyre_choice, aero_setup=aero_setup)
    driver.print_info = plot  # control whether to print out information to the screen

    return track_info


def take_race_turn(driver: Driver, track: Track, car: Car, level: Level, current_position: Position, track_info: TrackInfo,
                   safety_car: SafetyCar, safety_car_penalty: int = 0, weather_station: WeatherStation = None,
                   use_safety_car=True, use_drs=True, use_weather=True, plot=True):
    t_start = time_fn()

    # Track and car state
    track_state = track.get_state_for_position(current_position, car.heading)
    car_state = car.get_state()
    if not use_drs:
        track_state.drs_available = False

    # Safety car
    if level != Level.Learner and safety_car is not None and use_safety_car:
        track_state.safety_car_active = safety_car.update()

    # Weather
    if level == Level.Pro and weather_station is not None and use_weather:
        weather_station.update()
        weather_state = weather_station.get_state()
        track_grip = weather_station.get_track_grip()
    else:
        weather_state = WeatherState()
        track_grip = 1.0

    # Gather correct states
    states = {'car_state': car_state, 'track_state': track_state}
    if level == Level.Pro:
        states['weather_state'] = weather_state

    # Driver chooses an action
    # t0 = time_fn()
    action = driver.make_a_move(**states)
    # dt = time_fn() - t0
    # print(f'\tMake a move took {dt: .5f} seconds')

    # Verify action
    if action == Action.OpenDRS and not track_state.drs_available:
        print(f'***Driver {driver.name} attempted to open DRS at {track_state.position} when it was not available***')
        action = Action.Continue
    tyre_choice = None
    if action == Action.ChangeTyres:
        if car.speed == 0:
            tyre_choice = driver.choose_tyres(track_info)
        else:
            print(f'***Driver {driver.name} attempted to change tyres when moving (speed = {car.speed}).***')
            action = Action.Continue

    # Apply action
    spun = car.apply_action(action, tyre_choice=tyre_choice, track_grip=track_grip)
    new_position, crashed, finished = track.get_new_position(current_position, car.speed, car.heading)

    # Check safety car
    safety_car_speed_exceeded = (level != Level.Learner) and safety_car is not None and safety_car.active and \
                                safety_car.has_car_exceeded_speed(car.speed)
    safety_car_penalty += int(safety_car_speed_exceeded)  # add 1 if exceeded
    if safety_car_speed_exceeded and plot:
        print(f'\tCar speed of {car.speed: .1f} exceeds safety car, penalty is now {safety_car_penalty}')

    # Get result and pass feedback to driver
    turned_ok = action in [Action.TurnLeft, Action.TurnRight] and not spun
    result = ActionResult(turned_ok=turned_ok, spun=spun, crashed=crashed, finished=finished,
                          safety_car_speed_exceeded=safety_car_speed_exceeded,
                          safety_car_penalty_level=safety_car_penalty)

    new_car_state = car.get_state()
    new_track_state = track.get_state_for_position(new_position, car.heading)
    if level == Level.Pro:
        extra = {'previous_weather_state': weather_state}
    else:
        extra = {}

    # t0 = time_fn()
    driver.update_with_action_results(previous_car_state=car_state, previous_track_state=track_state,
                                      action=action, new_car_state=new_car_state,
                                      new_track_state=new_track_state, result=result, **extra)
    # dt = time_fn() - t0
    # print(f'\tUpdating with action results {dt: .5f} seconds')

    # Update race time
    race_time = 1 / (1 + car.speed)
    if crashed:
        car.crashed()
        race_time += 10
    if safety_car_speed_exceeded:
        race_time += safety_car_penalty

    # print(f'take_race_turn took {time_fn() - t_start: .5f} seconds')

    return driver, car, action, result, new_position, race_time, crashed, finished, safety_car_penalty


class Season:
    def __init__(self, level):
        self.level = level
        self.tracks = TrackStore.load_all_tracks(level)
        self.car = Car.get_car_for_level(level)

    @property
    def number_of_tracks(self):
        return len(self.tracks)

    def get_track(self, track_index: int):
        return self.tracks[track_index]

    def race(self, driver, track_indices=None, plot=False, use_safety_car=True, use_drs=True, use_weather=True):
        if track_indices is None:
            track_indices = range(self.number_of_tracks)
        number_of_tracks = len(track_indices)

        race_times, finished = np.zeros(number_of_tracks), np.zeros(number_of_tracks)
        num_crashes = np.zeros(self.number_of_tracks)
        for i, track_idx in enumerate(track_indices):
            driver, race_times[i], finished[i] = race(driver=driver, car=self.car, track=self.tracks[track_idx],
                                                      level=self.level, plot=plot, use_safety_car=use_safety_car,
                                                      use_drs=use_drs, use_weather=use_weather)
            # print(f'Completed track {i} with a race time of {race_times[i]: .0f}. '
            #       f'{"Finished" if finished[i] else "Did not finish"} the race.')

        return driver, race_times, finished

    def race_multiple_times(self, driver: Driver, num_repeats: int, reset_driver=True):
        race_times = np.zeros((num_repeats, self.number_of_tracks))
        finished = np.zeros((num_repeats, self.number_of_tracks))

        for i in range(num_repeats):
            season_driver = deepcopy(driver) if reset_driver else driver     # don't want to learn
            season_driver, race_times[i, :], finished[i, :] = self.race(season_driver, plot=False)

        return driver, race_times, finished

    def plot_all_tracks(self):
        fig = plt.figure()
        for i in range(24):
            ax = fig.add_subplot(4, 6, i + 1)
            self.tracks[i].plot_track(ax=ax)
        fig.tight_layout()


class Championship:
    def __init__(self, drivers: List[Driver], level: Level):
        if len(np.unique([driver.name for driver in drivers])) != len(drivers):
            raise ValueError('Driver names aren''t unique')
        self.driver_to_car_dict = {driver: Car.get_car_for_level(level) for driver in drivers}
        self.drivers = drivers
        self.driver_params = [deepcopy(driver.__dict__) for driver in drivers]
        self.season = Season(level)
        self.level = level

    def run_championship(self, track_indices=None, num_repeats=1):
        if track_indices is None:
            track_indices = range(self.season.number_of_tracks)

        finishing_positions = {driver.name: np.zeros((num_repeats, len(track_indices))) for driver in self.drivers}
        all_race_times = {driver.name: np.zeros((num_repeats, len(track_indices))) for driver in self.drivers}

        # Run each race
        for repeat in range(num_repeats):
            if repeat > 0:          # reset
                for driver, param_dict in zip(self.drivers, self.driver_params):
                    for key, param in param_dict.items():
                        setattr(driver, key, deepcopy(param))

            for race_num, track_idx in enumerate(track_indices):
                track = self.season.get_track(track_idx)
                race_finishing_positions, race_times = self.run_race(track)

                for driver, position in race_finishing_positions.items():
                    finishing_positions[driver.name][repeat, race_num] = position
                for driver, time in race_times.items():
                    all_race_times[driver.name][repeat, race_num] = time

        # Determine the championship winner
        championship_points = np.hstack([np.sum(positions, axis=1)[:, None]
                                         for positions in finishing_positions.values()])
        championship_ranks = {driver.name: ss.rankdata(championship_points, axis=1, method='min')[:, i]
                              for i, driver in enumerate(self.drivers)}

        return championship_ranks, finishing_positions, all_race_times

    def run_race(self, track, max_turns=1000):

        drivers_finished = []
        driver_positions = {}
        race_times = {}
        num_steps = 0
        safety_car = SafetyCar(print_details=False)
        safety_car_penalties = {}
        weather_station = WeatherStation()

        track_info = track.get_track_info()
        start_position, start_heading = track.get_start_position_and_heading()

        # Prepare
        for driver, car in self.driver_to_car_dict.items():
            prepare_driver_car(driver, car, level=self.level, track_info=track_info, start_heading=start_heading,
                               plot=False)
            driver.print_info = False
            race_times[driver] = 0
            driver_positions[driver] = start_position
            safety_car_penalties[driver] = 0

        weather_station.prepare_for_race()

        for i_turn in range(max_turns):

            for driver in self.drivers:
                if driver not in drivers_finished:
                    car = self.driver_to_car_dict[driver]
                    current_position = driver_positions[driver]

                    driver, car, action, result, new_position, race_time_i, crashed, finished, safety_car_penalty = \
                        take_race_turn(driver, track, car, self.level, current_position, track_info=track_info,
                                       safety_car=safety_car, safety_car_penalty=safety_car_penalties[driver],
                                       weather_station=weather_station, use_safety_car=True, plot=False)

                    # New becomes current
                    driver_positions[driver] = new_position
                    race_times[driver] += race_time_i
                    safety_car_penalties[driver] = safety_car_penalty

                    if finished:
                        drivers_finished.append(driver)

            num_steps += 1

            # print(f'End of turn {i_turn}\n')

            # If all drivers have finshed the race, end
            if all(driver.name in drivers_finished for driver in self.drivers):
                break

        # Work out the finishing order
        ranks = ss.rankdata(list(race_times.values()), method='min')
        driver_finishing_positions = {driver: rank for driver, rank in zip(race_times.keys(), ranks)}

        return driver_finishing_positions, race_times


if __name__ == '__main__':
    pass
