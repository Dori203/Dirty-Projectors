from heuristic_solver import HeuristicSolver




# the greedy solver progresses by choosing the best change out of few changes possible
class GreedySolver(HeuristicSolver):

    # pictures - a list of same dimension matrix of boolean representing the pictures
    # fitness - the fitness function to use
    # starting state index - choose between full board, empty board or randomly fulled board
    def solve(self):
        for i in range(self._num_iterations):
            # neighbors - a list of neighbors states
            neighbors = self._get_successor(self._current_state)
            neighbors_score = [self._get_fitness(neighbor.get_matrix()) for neighbor in
                               neighbors]
            self._current_state = neighbors[neighbors_score.index(min(neighbors_score))]

    def get_answer_loss(self):
        return self._get_fitness(self._current_state.get_matrix())

    def get_answer(self):
        return self._current_state
