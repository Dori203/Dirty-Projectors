from anneal import Annealer
from heuristic_solver import HeuristicSolver
import copy
from statistics import Statistics


class SimulatedAnnealingSolver(HeuristicSolver):
    def __init__(self, fitness_func, fitness_name, successor_func, generate_initial_func,
                 num_iteration, t_max=None, t_min=None):
        super().__init__(fitness_func, fitness_name, successor_func, generate_initial_func,
                         num_iteration)
        self._t_max = t_max
        self._t_min = t_min
        self._simulated_annealing = None
        self._statistics = None

    def solve(self):
        self._simulated_annealing = SimulatedAnnealing(self._current_state, self._get_successor, self._get_fitness)
        self._simulated_annealing.steps = self._num_iterations
        self._statistics = Statistics(self._current_state)

        if self._t_max is not None:
            self._simulated_annealing.Tmax = self._t_max
        if self._t_min is not None:
            self._simulated_annealing.Tmin = self._t_min

        state, score, self._statistics = self._simulated_annealing.anneal(self._statistics)
        self._current_state = state
        self._current_state.save_score(score)

    def get_answer(self):
        return self._current_state

    def get_answer_loss(self):
        return self._current_state.get_score()

    def get_losses(self):
        return self._simulated_annealing.get_losses_array()


class SimulatedAnnealing(Annealer):
    def __init__(self, initial_state, successor_func, fitness_func):
        initial_state.save_score(fitness_func(initial_state.get_matrix()))
        super().__init__(initial_state)
        self._successor_func = successor_func
        self._fitness_func = fitness_func

    def move(self):
        neighbors = self._successor_func(self.state)
        scores = [self._fitness_func(ne.get_matrix()) for ne in neighbors]
        minimum = scores.index(min(scores))
        neighbors[minimum].save_score(scores[minimum])
        self.state = copy.deepcopy(neighbors[minimum])
        return None, scores

    def energy(self):
        return self.state.get_score()
