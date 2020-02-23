import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from data import get_dataset, get_field, get_trajectory
from model import VIN

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=1, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--max_steps', default=10000, type=int, help='number of training steps')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--t_span', default=[0, 3], type=list, help='time spam')
    parser.add_argument('--time_scale', default=15, type=int, help='steps in unit time')
    parser.add_argument('--sample_size', default=50, type=int, help='number of samples')
    parser.add_argument('--save_dir', default=FILE_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--monitor_interval', default=100, type=int, help='monitor interval')
    parser.add_argument('--evaluate_interval', default=1000, type=int, help='evaluation interval')
    parser.add_argument('--use_cuda', action='store_true')
    return parser.parse_args()


def train(args):
    # set random seed for reproducing experiments
    device = 'cuda:0' if args.use_cuda else 'cpu:0'
    np.random.seed(args.seed)
    pyro.set_rng_seed(args.seed)

    data = get_dataset(seed=args.seed, samples=args.sample_size)
    data_q = torch.tensor(data['x'][:, :, 0], dtype=torch.float32).to(device)
    test_data_q = torch.tensor(data['test_x'][:, :, 0], dtype=torch.float32).to(device)

    vin = VIN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.sample_size // 2,
        num_steps=args.t_span[-1] * args.time_scale,
        use_cuda=args.use_cuda)
    optimizer = Adam({'lr': args.learning_rate})

    svi = SVI(vin.model, vin.guide, optimizer, loss=Trace_ELBO())

    elbo_list = []
    for i in range(args.max_steps):
        # do ELBO gradient and accumulate loss
        elbo_list.append(-svi.step(data_q))
        if (i+1) % args.monitor_interval == 0:
            print('Step: {}, Log likelihood: {:.3f}'.format(i+1, elbo_list[-1]))
        if (i+1) % args.evaluate_interval == 0:
            test_logll = -svi.loss(vin.evaluate_model, svi.guide, test_data_q)
            print('test_logll: {:.3f}'.format(test_logll))

    return vin, elbo_list


if __name__ == "__main__":
    args = get_args()

    model, trace_elbo = train(args)

    # save model and elbo
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    model_path = 'vin-{}.pth'.format(args.seed)
    torch.save(model.state_dict(), model_path)
