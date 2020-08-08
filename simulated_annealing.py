from simanneal import Annealer
from heuristic_solver import HeuristicSolver
import copy


class SimulatedAnnealingSolver(HeuristicSolver):
    def __init__(self, pictures, fitness_func, successor_func, generate_initial_func, num_iteration):
        HeuristicSolver.__init__(self, pictures, fitness_func, successor_func, generate_initial_func, num_iteration)
        self.simulated_annealing = SimulatedAnnealing(generate_initial_func(), self._get_successor, self._get_fitness)
        self._final_state = None
        self._energy = None

    def change_parameters(self, t_max, t_min, steps):
        if t_max is not None:
            self.simulated_annealing.Tmax = t_max
        if t_min is not None:
            self.simulated_annealing.Tmin = t_min
        if steps is not None:
            self.simulated_annealing.steps = steps

    def solve(self):
        self._final_state, self._energy = self.simulated_annealing.anneal()

    def get_answer(self):
        return self._final_state

    def get_answer_loss(self):
        return self._energy


class SimulatedAnnealing(Annealer):
    def __init__(self, initial_state, successor_func, fitness_func):
        Annealer.__init__(self, initial_state)
        self._successor_func = successor_func
        self._fitness_func = fitness_func

    def move(self):
        neighbors = self._successor_func(self.state)
        scores = [self._fitness_func(ne.get_matrix()) for ne in neighbors]
        minimum = scores.index(min(scores))
        neighbors[minimum].score = scores[minimum]
        self.state = copy.deepcopy(neighbors[minimum])

    def energy(self):
        return self._fitness_func(self.state.get_matrix())
