import numpy as np
from state import State


class InitStateFactory:

    def __init__(self, shape):
        self._shape = shape

    def random_init_state(self):
        """
        @shape - tuple
        """
        return State(np.random.randint(2, size=self._shape))

    def blank_slate(self):
        return State(np.zeros(self._shape))
    def black_slate(self):
        return State(np.ones(self._shape))