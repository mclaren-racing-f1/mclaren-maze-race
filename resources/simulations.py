import numpy as np
from resources.racecontrol import *


def straight_line_sim(driver, level, straight_length, plot=True):
    car = Car.get_car_for_level(level)
    actions = []
    speeds = np.zeros(straight_length + 1)
    position = Position(0, 0)
    heading = Heading(0, 1)
    car.prepare_for_race(heading)
    driver.prepare_for_race()

    for i in range(straight_length):
        track_state = TrackState(distance_ahead=straight_length - i, distance_right=0, distance_left=0,
                                 distance_behind=i, position=position)
        car_state = car.get_state()
        action = driver.make_a_move(car_state=car_state, track_state=track_state)
        car.apply_action(action)

        speeds[i + 1] = car.speed
        actions.append(action)

    if plot:
        plot_straight_line_sim(actions, speeds)

    return actions, speeds


def safety_car_sim(driver, level: Level, num_moves=50, plot=True):
    print('Running safety car sim...')
    # Train on straight track first
    track_map = np.zeros((1, num_moves))
    track = Track(track_map, correct_turns={}, straight_lengths=[num_moves])
    for _ in range(100):
        driver, *_ = race(driver=driver, track=track, level=level, plot=False, use_safety_car=False)

    # Setup safety car
    safety_car = SafetyCar(print_details=plot)
    safety_car_penalty = np.zeros(num_moves + 1)
    safety_car.deploy(num_moves)

    # Setup car
    car = Car.get_car_for_level(level)
    car.prepare_for_race(Heading(0, 1))

    # Initialise result arrays
    car_speed = np.zeros(num_moves + 1)
    safety_car_estimate = np.zeros(num_moves + 1)
    safety_car_estimate[0] = driver.get_safety_car_speed_estimate()

    driver.print_info = plot
    for i in range(num_moves):
        # Get states
        car_state = car.get_state()
        track_state = TrackState(distance_ahead=num_moves-i, distance_right=0, distance_left=0,
                                 distance_behind=i, position=Position(0, i), safety_car_active=True)

        # Driver chooses an action
        action = driver.make_a_move(car_state=car_state, track_state=track_state)

        # Apply action
        car.apply_action(action)

        # Check safety car
        safety_car_speed_exceeded = safety_car.has_car_exceeded_speed(car.speed)
        safety_car_penalty[i + 1] = safety_car_penalty[i] + int(safety_car_speed_exceeded)  # add 1 if exceeded

        # Get result and pass feedback to driver
        result = ActionResult(turned_ok=False, spun=False, crashed=False, finished=False,
                              safety_car_speed_exceeded=safety_car_speed_exceeded,
                              safety_car_penalty_level=safety_car_penalty[i + 1])

        new_car_state = car.get_state()
        driver.update_with_action_results(previous_car_state=car_state, previous_track_state=track_state,
                                          action=action, new_car_state=new_car_state,
                                          new_track_state=track_state, result=result)

        car_speed[i + 1] = car.speed
        safety_car_estimate[i + 1] = driver.get_safety_car_speed_estimate()

    if plot:
        fig = plt.figure(figsize=(9, 5))
        plt.plot(car_speed, c='b')
        plt.plot(safety_car_estimate, c='orange')
        fig.gca().axhline(safety_car.current_speed, c='g')
        plt.plot(safety_car_penalty, c='r')
        plt.xlabel('Move Number', fontsize=16)
        plt.ylabel('Speed', fontsize=16)
        plt.legend(['Car Speed', 'Estimate of Safety Car Speed', 'True Safety Car Speed', 'Penalty'], fontsize=12)
        plt.grid(True)

    print('Complete')

    return car_speed, safety_car_estimate, safety_car.current_speed, safety_car_penalty
