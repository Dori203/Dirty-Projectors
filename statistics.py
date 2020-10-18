import numpy as np


class Statistics:
    def __init__(self, initial_state):
        self._avg = np.empty(0)
        self._std = np.empty(0)
        self._score = np.array(initial_state.get_score())
        self._num_blacks = np.array(np.sum(initial_state.get_matrix()))
        self._volume_ratio = self._num_blacks / initial_state.get_dim()
        self._extra_data = {}

    def add_neighbors_data(self, data_array):
        """
        this function keeps the record of the given neighbors average score and standard deviation
        :param data_array: neighbors scores
        """
        self._avg = np.append(self._avg, np.average(data_array))
        self._std = np.append(self._std, np.std(data_array))

    def add_current_state_data(self, current_state):
        self._score = np.append(self._score, current_state.get_score())
        self._num_blacks = np.append(self._num_blacks, np.sum(current_state.get_matrix()))
        self._volume_ratio = self._num_blacks / pow(current_state.get_dim(), 3)

    def add_extra_data(self, key, value):
        self._extra_data[key] = value

    def get_from_extra_data(self, key):
        if key in self._extra_data:
            return self._extra_data[key]
        else:
            return None

    def init_stat(self, avg, std, score, num_blacks, volume_ratio):
        self._avg = avg
        self._std = std
        self._score = score
        self._num_blacks = num_blacks
        self._volume_ratio = volume_ratio

    def get_avg(self):
        return self._avg

    def get_std(self):
        return self._std

    def get_scores(self):
        return self._score

    def get_num_blacks(self):
        return self._num_blacks

    def get_volume_ratio(self):
        return self._volume_ratio

    def get_extra_data_dict(self):
        return self._extra_data
