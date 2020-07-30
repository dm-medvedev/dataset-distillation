import os
import pickle as pk
from collections import defaultdict, namedtuple
from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from base_options import BASE_OPTIONS, RES_FOLDER, State
from basics import evaluate_steps
from networks import get_networks
import train_distilled_image
from visualisation import GLOB_TITLES, plot_convergence


ResEl = namedtuple('ResEl', ['state', 'steps', 'losses', 'time'])
TOTAL_RESTARTS = 2


def distill_with_param(exp_dir, param_vars, param_initer):
    for rst_i in tqdm(range(TOTAL_RESTARTS), desc='distill_with_param'):
        res = {}
        for arch_nm in ['LinearNet', 'NonLinearNet', 'MoreNonLinearNet']:
            res[arch_nm] = {}
            for param in param_vars:
                state = deepcopy(BASE_OPTIONS)
                state['arch'] = arch_nm
                state = param_initer(state, param)
                state['test_models'] = get_networks(state, N=1)
                st = time()
                steps, losses = train_distilled_image.distill(state,
                                                              state.models)
                res[arch_nm][param] = ResEl(state, steps, losses, time() - st)
        with open(os.path.join(exp_dir, f'res_{rst_i}.pk'), 'wb') as f:
            pk.dump(res, f)


def evaluation(exp_dir, total_restarts):
    total_results = []
    for rst_i in range(total_restarts):
        with open(os.path.join(exp_dir, f'res_{rst_i}.pk'), 'rb') as f:
            total_results.append(pk.load(f))
    torch.manual_seed(42)
    np.random.seed(42)

    def _eval(state, steps):
        return evaluate_steps(state, steps,
                              f'distilled with {state.distill_steps} ' +
                              f'steps and {state.distill_epochs} epochs',
                              test_all=True, test_at_steps=None)

    for rst_i, res1 in tqdm(enumerate(total_results),
                            total=len(total_results)):
        eval_res = {}
        for arch, res2 in res1.items():
            eval_res[arch] = {}
            for param, res3 in res2.items():
                state, steps = res3.state, res3.steps
                eval_res[arch][param] = _eval(state, steps)
        with open(os.path.join(exp_dir, f'eval_{rst_i}.pk'), 'wb') as f:
            # test_step_indice, accuracy, loss, model_param
            pk.dump(eval_res, f)


def load_evaluation(exp_dir, total_restarts):
    total_eval = []
    for rst_i in range(total_restarts):
        with open(os.path.join(exp_dir, f'eval_{rst_i}.pk'), 'rb') as f:
            total_eval.append(pk.load(f))
    return total_eval


def distill_epochs_initer(state, distill_epochs):
    state.update({'distill_epochs': distill_epochs, 'distill_steps': 1})
    state = State(**state)
    state['models'] = get_networks(state, N=1)
    return state


def distill_steps_initer(state, distill_steps):
    state.update({'distill_epochs': 1, 'distill_steps': distill_steps})
    state = State(**state)
    state['models'] = get_networks(state, N=1)
    return state


def inn_models_initer(state, inn_models):
    state.update({'distill_epochs': 1, 'distill_steps': 10})
    state = State(**state)
    state['models'] = get_networks(state, N=inn_models)
    return state


def visualise(exp_dir, xlabel):
    total_eval = load_evaluation(exp_dir, TOTAL_RESTARTS)
    xs, ys = defaultdict(list), defaultdict(list)
    for rst_i, res1 in enumerate(total_eval):
        for arch_nm, res2 in res1.items():
            xs[arch_nm] += list(res2.keys())
            ys[arch_nm] += [res2[x][1][-1].item() for x in res2.keys()]
    xs, ys, labels = zip(*[(xs[arch], ys[arch],
                            GLOB_TITLES[arch]) for arch in xs.keys()])
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    plot_convergence(xs, ys, labels, xlabel,
                     ylabel='accuracy', title=None,
                     ax=ax, exp_dir=exp_dir, marker='o')


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    def _get_dir(dirnm):
        os.makedirs(RES_FOLDER, exist_ok=True)
        exp_dir = os.path.join(RES_FOLDER, dirnm)
        os.mkdir(exp_dir)
        return exp_dir

    exp_dirnm_l = [f'experiment_hyperparameters_{i}' for i in range(1, 4)]
    param_vars_l = [[1, 15, 40, 100], [1, 15, 40, 100], [1, 2, 10, 20]]
    initer_l = [distill_epochs_initer, distill_steps_initer, inn_models_initer]
    param_nm_l = ['distill epochs', 'distill steps', 'nets number']
    iterator = zip(exp_dirnm_l, param_vars_l, initer_l, param_nm_l)
    for exp_dirnm, param_vars, initer, param_nm in iterator:
        exp_dir = _get_dir(exp_dirnm)
        distill_with_param(exp_dir, param_vars, initer)
        evaluation(exp_dir, TOTAL_RESTARTS)
        visualise(exp_dir, param_nm)


if __name__ == '__main__':
    main()
