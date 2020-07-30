import os

import numpy as np
import torch

from base_options import RES_FOLDER
from distillation_experiments_utils import DISTILL_EPOCHS,\
     DISTILL_STEPS, TOTAL_RESTARTS, distillation, evaluation,\
     load_distillation_results, visualise_distillation_results


RESULTS_DIR = os.path.join(RES_FOLDER, 'experiment_general_distillation')


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs(RES_FOLDER, exist_ok=True)
    os.mkdir(RESULTS_DIR)
    distillation(RESULTS_DIR)
    results = load_distillation_results(RESULTS_DIR, TOTAL_RESTARTS)
    test_at_steps = range(DISTILL_EPOCHS*DISTILL_STEPS+1)
    evaluation(RESULTS_DIR, results, TOTAL_RESTARTS, test_at_steps)
    visualise_distillation_results(exp_dir=RESULTS_DIR, total_results=results,
                                   total_restarts=TOTAL_RESTARTS,
                                   strategy_nm='original',
                                   test_at_steps=test_at_steps,
                                   convergence_xlabel='steps')


if __name__ == '__main__':
    main()
