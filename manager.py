from image_procceesing import *
from csp import CSPSolver
from greedy_solver import GreedySolver
from simulated_annealing import *
from hill_climbing_solver import HillClimbingSolver
from successor_functions import *
from inital_state_factory import *
from GUI import *
import argparse


path = "images/pyramid 2/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs-dir', type=str, required=True)
    path = parser.parse_args()
    image_processing = ImageProcessor(path, GRAY, [40,40])
    init_state_producer = InitStateFactory((40, 40, 40))
    init_func = init_state_producer.blank_slate
    # init_func = init_state_producer.random_init_state
    num_iters = 1000
    # csp_solver = CSPSolver(image_processing.get_images())

    greedy_solver = GreedySolver(image_processing.get_images(),
                                 image_processing.loss_pyramid, naive_successors,
                                 init_func, num_iters)
    #hill_climber = HillClimbingSolver(image_processing.get_images(),
    #                                  image_processing.loss_pyramid, naive_successors,
    #                                  init_func, num_iters)
    greedy_solver.solve()
    #hill_climber.solve()
    #image_processing.export_result_csv(greedy_solver.get_answer().get_matrix(), 'results')

    print_3d_matrix(greedy_solver.get_answer().get_matrix())

    #print_3d_matrix(hill_climber.get_answer().get_matrix())

    # simulated_annealing_solver = SimulatedAnnealingSolver(image_processing.get_images(), image_processing.loss_pyramid,
    #                                                      naive_successors, init_func, num_iters)
    # simulated_annealing_solver.change_parameters(None, None, 800)
    # simulated_annealing_solver.solve()
    # print_3d_matrix(simulated_annealing_solver.get_answer().get_matrix())

main()
