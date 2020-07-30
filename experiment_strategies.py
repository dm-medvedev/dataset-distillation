import os

import numpy as np
import torch

from base_options import RES_FOLDER
from distillation_experiments_utils import DISTILL_EPOCHS,\
     DISTILL_STEPS, TOTAL_RESTARTS, evaluation, load_distillation_results,\
     visualise_distillation_results
from experiment_general_distillation import RESULTS_DIR


N_TIMES_EPOCH_REPEAT = 10


def get_new_steps_strategy1(steps):
    print('\nget_new_steps_strategy1\n')
    st_lr, gamma_up, gamma_down = 0.3, 1.1, 0.95
    new_lrs = [st_lr]
    for i in range(N_TIMES_EPOCH_REPEAT):
        new_lrs += [new_lrs[-1]*gamma_up]*DISTILL_STEPS
    new_lrs.pop()
    new_epochs_num = N_TIMES_EPOCH_REPEAT*DISTILL_EPOCHS
    for i in range(new_epochs_num-N_TIMES_EPOCH_REPEAT):
        new_lrs += [new_lrs[-1]*gamma_down]*DISTILL_STEPS
    new_steps = []
    for i in range(new_epochs_num*DISTILL_STEPS):
        x, y, lr = steps[i % DISTILL_STEPS]
        lr = torch.tensor(new_lrs[i]).to(lr.device)
        new_steps.append((x, y, lr))
    return new_steps


def get_new_steps_strategy2(steps):
    print('\nget_new_steps_strategy2\n')
    gamma, koef = 0.9, 1.
    new_epochs_num = N_TIMES_EPOCH_REPEAT*DISTILL_EPOCHS
    new_steps = []
    for i in range(new_epochs_num):
        koef = koef*gamma if (i % DISTILL_EPOCHS == 0) & (i > 0) else koef
        for j in range(DISTILL_STEPS):
            x, y, lr = steps[(i % DISTILL_EPOCHS)*DISTILL_STEPS+j]
            new_steps.append((x, y, lr*koef))
    return new_steps


def get_new_steps_strategy3(steps):
    print('\nget_new_steps_strategy3\n')
    gamma = 0.98
    new_epochs_num = N_TIMES_EPOCH_REPEAT*DISTILL_EPOCHS
    new_steps = steps
    last_epoch = steps[-DISTILL_STEPS:]
    for i in range(new_epochs_num - DISTILL_EPOCHS):
        for j in range(DISTILL_STEPS):
            x, y, lr = last_epoch[j]
            new_lr = lr*gamma
            last_epoch[j] = x, y, new_lr
            new_steps.append((x, y, new_lr))
    return new_steps


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    get_new_steps_l = [get_new_steps_strategy1,
                       get_new_steps_strategy2,
                       get_new_steps_strategy3]
    exp_dirnm_l = ['experiment_1Strategy',
                   'experiment_2Strategy',
                   'experiment_3Strategy']
    strategy_nm_l = ['1 Strategy', '2 Strategy', '3 Strategy']
    iterator = zip(get_new_steps_l, exp_dirnm_l, strategy_nm_l)
    for get_new_steps, exp_dirnm, strategy_nm in iterator:
        results = load_distillation_results(RESULTS_DIR, TOTAL_RESTARTS)
        exp_dir = os.path.join(RES_FOLDER, exp_dirnm)
        os.mkdir(exp_dir)
        test_at_steps = [DISTILL_STEPS*i for i in
                         range(DISTILL_EPOCHS*N_TIMES_EPOCH_REPEAT+1)]
        evaluation(exp_dir, results, TOTAL_RESTARTS,
                   test_at_steps, get_new_steps)
        visualise_distillation_results(exp_dir=exp_dir, total_results=results,
                                       total_restarts=TOTAL_RESTARTS,
                                       strategy_nm=strategy_nm,
                                       test_at_steps=test_at_steps,
                                       convergence_xlabel='epochs')


if __name__ == '__main__':
    main()
