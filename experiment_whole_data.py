import glob
import logging
import os
import pickle as pk
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from base_options import RES_FOLDER, State, TEST_DATASET, TRAIN_DATASET
from networks import get_networks
from visualisation import GLOB_TITLES, plot_convergence,\
     plot_decision_boundary, plot_dist


ResEl = namedtuple(
    'ResEl', ['tst_accuracies', 'tr_accuracies', 'losses', 'network'])


# the whole data
def train_on_the_whole_data(exp_dir):
    total_restarts = 2
    for rst_i in tqdm(range(total_restarts), desc='training'):
        res = {}
        for arch_nm in ['LinearNet', 'NonLinearNet', 'MoreNonLinearNet']:
            n_epochs = 200
            tr_loader = DataLoader(TRAIN_DATASET, batch_size=64,
                                   shuffle=True, pin_memory=True)
            state = State(**{'local_n_nets': None,
                             'num_classes': 2,
                             'device': 'cpu',
                             'test_nets_type': 'unknown_init',
                             'arch': arch_nm,
                             'init': 'xavier',
                             'init_param': 1,
                             'L2_coef': 1e-1
                             })
            network = get_networks(state, 1)[0]
            optimizer = optim.Adam(network.parameters(),
                                   lr=1e-3, betas=(0.5, 0.999))
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=40, gamma=0.5)
            tst_accuracies = []
            tr_accuracies = []
            losses = []
            for epoch in range(n_epochs):
                network.train()
                for data, target in tr_loader:
                    data = data.to(state.device, non_blocking=True)
                    target = target.to(state.device, non_blocking=True)
                    optimizer.zero_grad()
                    output = network(data)
                    loss = F.binary_cross_entropy_with_logits(
                        output, target.float().view_as(output))
                    losses.append(loss.detach().numpy().item())
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                network.eval()
                with torch.no_grad():
                    # tst
                    tst_data = torch.tensor(TEST_DATASET.X).float()
                    ny = torch.sigmoid(
                        network(tst_data)).detach().numpy().round()
                    tst_accuracies.append(accuracy_score(TEST_DATASET.y, ny))
                    # tr
                    tr_data = torch.tensor(TRAIN_DATASET.X).float()
                    ny = torch.sigmoid(
                        network(tr_data)).detach().numpy().round()
                    tr_accuracies.append(accuracy_score(TRAIN_DATASET.y, ny))
            res[arch_nm] = ResEl(
                tst_accuracies, tr_accuracies, losses, network)
        with open(os.path.join(exp_dir, f'res_{rst_i}.pk'), 'wb') as f:
            pk.dump(res, f)


def visualise(exp_dir):
    total_results = []
    for pth in glob.glob(os.path.join(exp_dir, '*.pk')):
        with open(pth, 'rb') as f:
            total_results.append(pk.load(f))
    losses = defaultdict(lambda: [])
    tst_accs, tr_accs = defaultdict(lambda: []), defaultdict(lambda: [])
    for res in total_results:
        for arch, res_el in res.items():
            tst_accs[arch].append(res_el.tst_accuracies)
            tr_accs[arch].append(res_el.tr_accuracies)
            losses[arch].append(res_el.losses)
    pbar = tqdm(total=len(tst_accs.keys())*3, desc='plotting')        
    # plotting
    for arch in tst_accs.keys():
        data = [el[-1] for el in tst_accs[arch]]
        # accuracy distribution
        plot_dist(data, xlabel='accuracy', arch=arch, exp_dir=exp_dir)
        # decision boundary
        indxs = np.argsort(data)
        res_ = total_results[indxs[len(indxs) // 2]][arch]
        logging.info(f'visualisation: {arch}; plot_decision_boundary')
        plot_decision_boundary(TEST_DATASET.X, TEST_DATASET.y, res_.network,
                               title=f'acc: {res_.tst_accuracies[-1]:.3f}',
                               arch=arch, exp_dir=exp_dir)
        pbar.update(1)
        f, ax = plt.subplots(1, 1, figsize=(5, 4))
        get_x = lambda x: sum((list(range(len(el))) for el in x[arch]), [])
        get_y = lambda y: sum((el for el in y[arch]), [])
        # accuracy convergence
        logging.info(f'visualisation: {arch}; plot accuracy convergence')
        plot_convergence(xs=[get_x(tst_accs), get_x(tr_accs)],
                         ys=[get_y(tst_accs), get_y(tr_accs)],
                         labels=['test', 'train'], xlabel='epochs',
                         ylabel='accuracy', title=GLOB_TITLES[arch],
                         ax=ax, exp_dir=exp_dir)
        pbar.update(1)
        # loss convergence
        logging.info(f'visualisation: {arch}; plot Loss convergence')
        f, ax = plt.subplots(1, 1, figsize=(5, 4))
        plot_convergence(xs=[get_x(losses)], ys=[get_y(losses)],
                         labels=[None], xlabel='steps', ylabel='Loss',
                         title=GLOB_TITLES[arch], ax=ax, exp_dir=exp_dir)
        pbar.update(1)
    pbar.close()


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    exp_dirnm = 'experiment_whole_data'
    os.makedirs(RES_FOLDER, exist_ok=True)
    exp_dir = os.path.join(RES_FOLDER, exp_dirnm)
    os.mkdir(exp_dir)
    train_on_the_whole_data(exp_dir)
    visualise(exp_dir)


if __name__ == '__main__':
    main()
