from heuristic_solver import HeuristicSolver
from tqdm import tqdm



# the greedy solver progresses by choosing the best change out of few changes possible
class GreedySolver(HeuristicSolver):

    # pictures - a list of same dimension matrix of boolean representing the pictures
    # fitness - the fitness function to use
    # starting state index - choose between full board, empty board or randomly fulled board

    def solve(self):
        progress_bar_iterator = tqdm(
            iterable=range(self._num_iterations),
            bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}',
            desc='greedy solver'
        )
        for i in progress_bar_iterator:
            # neighbors - a list of neighbors states
            neighbors = self._get_successor(self._current_state)
            neighbors_score = [self._get_fitness(neighbor.get_matrix()) for neighbor in
                               neighbors]
            self._current_state = neighbors[neighbors_score.index(min(neighbors_score))]
            progress_bar_iterator.set_postfix_str('loss=%.2f' % min(neighbors_score))

    def get_answer_loss(self):
        return self._get_fitness(self._current_state.get_matrix())

    def get_answer(self):
        return self._current_state
