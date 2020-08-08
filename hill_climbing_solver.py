from heuristic_solver import HeuristicSolver

NUM_OF_ITER = 100000


class HillClimbingSolver(HeuristicSolver):

    def solve(self):
        best_states = []
        for i in range(self._num_iterations):
            neighbors = self._get_successor(self._current_state)
            neighbors_score = [self._get_fitness(neighbor.get_matrix()) for neighbor in
                               neighbors]
            best = neighbors[neighbors_score.index(min(neighbors_score))]
            if self._get_fitness(best.get_matrix()) >= self._current_score:
                # if self._score > EPSILON:
                "local minimum, if the current state isn't good enough, restart"
                best_states.append((self._current_state, self._current_score))
                self._current_state = self._initial_state_generator()
                self._current_score = self._get_fitness(self._current_state.get_matrix())

            else:
                self._current_state = best
                self._current_score = self._get_fitness(self._current_state.get_matrix())
            scores = [score[1] for score in best_states]
            self._current_state, self._current_score = best_states[scores.index(min(scores))]

    def get_answer_loss(self):
        return self._current_score

    def get_answer(self):
        return self._current_state

