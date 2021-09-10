from itertools import cycle
import numpy as np
from resources.rng import rng


class SafetyCar:
    def __init__(self, print_details=False):
        # These values will change in the challenge ... and won't be available here :P
        # The concept behind their structure won't change, although the exact relationships will do
        self.safety_car_speeds = cycle([100, 135, 173.5, 131.15, 169.27, 211.19, 165.07, 123.57])
        self.lengths = [9, 12, 15]

        self.activation_probability = 0.01
        self.active = False
        self.print_details = print_details

        self.current_speed = 0
        self.turns_left = 0

    def update(self):
        if self.active:         # active check if we need to end
            if 0 == self.turns_left:
                self.active = False
                if self.print_details:
                    print('Safety car no longer active')
            else:
                self.turns_left -= 1

        else:                   # not active, check if we need to deploy
            if rng().rand() < self.activation_probability:      # deploy the safety car
                self.deploy()

        return self.active      # inform race control whether we are active

    def deploy(self, length=None):
        self.active = True  # mark as active
        self.turns_left = rng().choice(self.lengths) if length is None else length              # pick a length
        self.current_speed = next(self.safety_car_speeds)  # and a speed

        if self.print_details:
            print(f'Safety car deployed for {self.turns_left} turns at {self.current_speed} speed')

    def has_car_exceeded_speed(self, speed: float):
        return self.active and (speed > self.current_speed)

