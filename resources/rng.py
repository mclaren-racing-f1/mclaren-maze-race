import numpy as np

# Two separate random streams, one for the driver and one for the race control. This means that random elements of the
# race, such as the safety car being deployed, are unaffected by changes in the driver code. This makes comparisons,
# much more stable
RNG = np.random.RandomState(0)
DRIVER_RNG = np.random.RandomState(0)


def rng():
    return RNG


def driver_rng():
    return DRIVER_RNG


def set_seed(seed):
    RNG.seed(seed)
    DRIVER_RNG.seed(seed)
