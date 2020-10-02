import os
import pickle as pk
from collections import defaultdict, namedtuple
from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from base_options import BASE_OPTIONS, State, TEST_DATASET
from basics import evaluate_steps
from networks import get_networks
import train_distilled_image
from visualisation import GLOB_TITLES, plot_convergence,\
     plot_decision_boundary, plot_dist, plot_synthetic_objects


ResEl = namedtuple('ResEl', ['state', 'steps', 'losses', 'time'])
TOTAL_RESTARTS = 10
DISTILL_EPOCHS = 5
DISTILL_STEPS = 40
EPOCHS = 50


def distillation(exp_dir):
    for rst_i in tqdm(range(TOTAL_RESTARTS), desc='distillation'):
        res = {}
        for arch_nm in ['LinearNet', 'NonLinearNet', 'MoreNonLinearNet']:
            state = deepcopy(BASE_OPTIONS)
            state.update({'arch': arch_nm,
                          'distill_epochs': DISTILL_EPOCHS,
                          'distill_steps': DISTILL_STEPS,
                          'epochs': EPOCHS,
                          })
            state = State(**state)
            state['models'] = get_networks(state, N=3)
            state['test_models'] = get_networks(state, N=1)
            st = time()
            steps, losses = train_distilled_image.distill(state, state.models)
            res[arch_nm] = ResEl(state, steps, losses, time() - st)
        with open(os.path.join(exp_dir, f'res_{rst_i}.pk'), 'wb') as f:
            pk.dump(res, f)


def load_distillation_results(exp_dir, total_restarts):
    total_results = []
    for rst_i in range(total_restarts):
        with open(os.path.join(exp_dir, f'res_{rst_i}.pk'), 'rb') as f:
            total_results.append(pk.load(f))
    return total_results


def evaluation(exp_dir, total_results, total_restarts,
               test_at_steps=None, get_new_steps=None):
    torch.manual_seed(42)
    np.random.seed(42)

    def _eval(state, steps, test_at_steps):
        return evaluate_steps(state, steps,
                              f'distilled with {state.distill_steps} ' +
                              f'steps and {state.distill_epochs} epochs',
                              test_all=True, test_at_steps=test_at_steps)

    for rst_i, res1 in tqdm(enumerate(total_results),
                            total=len(total_results)):
        eval_res = {}
        for arch, res2 in res1.items():
            steps = res2.steps if get_new_steps is None\
                    else get_new_steps(res2.steps)
            eval_res[arch] = _eval(res2.state, steps, test_at_steps)
        with open(os.path.join(exp_dir, f'eval_{rst_i}.pk'), 'wb') as f:
            # test_step_indice, accuracy, loss, model_param
            pk.dump(eval_res, f)


def load_evaluation(exp_dir, total_restarts):
    total_eval = []
    for rst_i in range(total_restarts):
        with open(os.path.join(exp_dir, f'eval_{rst_i}.pk'), 'rb') as f:
            total_eval.append(pk.load(f))
    return total_eval


def visualise_distillation_results(exp_dir, total_results,
                                   total_restarts, strategy_nm,
                                   test_at_steps, convergence_xlabel):
    total_eval = load_evaluation(exp_dir, total_restarts)
    # convergence
    xs_acc, ys_acc = defaultdict(list), defaultdict(list)
    xs_losses, ys_losses = defaultdict(list), defaultdict(list)
    res_accuracy = defaultdict(list)
    for rst_i, res1 in enumerate(total_eval):
        for arch_nm, res2 in res1.items():
            accs = res2[1].view(-1).cpu().numpy()
            losses = res2[2].view(-1).cpu().numpy()
            res_accuracy[arch_nm].append(accs[-1])
            xs_acc[arch_nm] += test_at_steps
            ys_acc[arch_nm] += accs.tolist()
            xs_losses[arch_nm] += test_at_steps
            ys_losses[arch_nm] += losses.tolist()
    for xs, ys, ylabel in zip([xs_acc, xs_losses], [ys_acc, ys_losses],
                              ['accuracy', 'losses']):
        xs, ys, labels = zip(*[(xs[arch], ys[arch], GLOB_TITLES[arch])
                             for arch in xs.keys()])
        f, ax = plt.subplots(1, 1, figsize=(5, 4))
        plot_convergence(xs, ys, labels, convergence_xlabel,\
                         ylabel=ylabel, ax=ax,
                         exp_dir=exp_dir, title=strategy_nm)
    for arch in res_accuracy.keys():
        # accuracy distribution
        plot_dist(res_accuracy[arch], xlabel='accuracy',
                  arch=arch, exp_dir=exp_dir)
    # get indxs
    idxs_d = {}
    for arch, acc_l in res_accuracy.items():
        idxs_d[arch] = np.argsort(acc_l)
    # distribution
    for arch, data in res_accuracy.items():
        # breakpoint()
        model = total_results[0][arch].state['test_models'][0].to('cpu')
        index = idxs_d[arch][len(idxs_d[arch]) // 2]
        _, accs, losses, params = total_eval[index][arch]
        params = params[0].detach().cpu()
        network = lambda x: model.forward_with_param(x, params)
        # decision boundary
        plot_decision_boundary(TEST_DATASET.X, TEST_DATASET.y, network,
                               title=f'acc: {accs[-1].item():.3f}', arch=arch,
                               exp_dir=exp_dir)
        # synthetic objects
        for step_n in np.linspace(0, 4, 4, dtype=int):
            dx, dy, _ = total_results[index][arch].steps[step_n]
            dx, dy = dx.detach().cpu().numpy(), dy.detach().cpu().numpy()
            plot_synthetic_objects(TEST_DATASET.X, TEST_DATASET.y,
                                   dx, dy, arch, step_n, exp_dir)
