
class State:
    def __init__(self, matrix, layer_idx=0):
        self._matrix = matrix
        self._dim = matrix.shape[0]
        self.layer = layer_idx
        self.score = 0.0

    def get_matrix(self):
        return self._matrix

    def get_depth(self):
        return self.layer

    def get_dim(self):
        return self._dim

    def save_score(self, score):
        self.score = score

    def get_score(self):
        return self.score
