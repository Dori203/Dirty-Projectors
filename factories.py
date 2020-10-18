import greedy_solver
import hill_climbing_solver
import local_beam_search
import simulated_annealing
import successor_functions
import inital_state_factory
import csp_solvers
import image_processing


def csp_solver_factory(path, dim, solver='mrv'):
    processor = image_processing.ImageProcessor(path, image_processing.GRAY, dim)
    if len(processor.get_images()) > 3:
        print('csp does not support more than 3 pictures!')
        exit(1)
    if solver == 'mrv':
        return csp_solvers.MRV(processor)
    else:
        return csp_solvers.CSPSolver(processor)


def heuristic_solver_factory(solver):
    if solver == 'greedy':
        return greedy_solver.GreedySolver
    elif solver == 'hill':
        return hill_climbing_solver.HillClimbingSolver
    elif solver == 'local':
        return local_beam_search.LocalBeamSearch
    elif solver == 'annealing':
        return simulated_annealing.SimulatedAnnealingSolver
    else:
        raise Exception('error')


def fitness_func_factory(fitness_func, processor):
    if fitness_func == 'l1':
        return processor.loss_1
    elif fitness_func == 'l2':
        return processor.loss_2
    elif fitness_func == 'punish_empty':
        return processor.loss_penalize_empty_pixels
    elif fitness_func == 'punish_black':
        return processor.loss_penalize_black_pixels
    else:
        raise Exception('error')


def successor_func_factory(successor_func):
    if successor_func == 'naive':
        return successor_functions.naive_successors
    elif successor_func == 'onion':
        return successor_functions.onion
    elif successor_func == 'flip':
        return successor_functions.flip_all
    elif successor_func == 'neighbouring':
        return successor_functions.neighbouring_successors
    elif successor_func == 'column':
        return successor_functions.column_row_child
    else:
        raise Exception('error')


def init_func_factory(init_func, dim):
    if init_func == 'blank':
        return inital_state_factory.InitStateFactory((dim, dim, dim)).blank_slate
    elif init_func == 'black':
        return inital_state_factory.InitStateFactory((dim, dim, dim)).black_slate
    elif init_func == 'random':
        return inital_state_factory.InitStateFactory((dim, dim, dim)).random_init
    else:
        raise Exception('error')
