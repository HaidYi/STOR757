# STOR 757
# UNC Chapel Hill

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist


class StateTrans(nn.Module):
    def __init__(self, input_dim, hidden_dim, mass=1.0):
        super(StateTrans, self).__init__()

        self.mass = mass
        self.trans = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q, dq, step_size):
        dfdq = torch.autograd.grad(self.trans(q).sum(), q, create_graph=True)[0]

        new_dq = dq - 1./self.mass * step_size * dfdq
        new_q = q + step_size * new_dq
        return new_q, new_dq


class VIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, num_steps, use_cuda=True):
        super(VIN, self).__init__()

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.h = 1. / num_steps
        self.trans = StateTrans(input_dim, hidden_dim)

        self.q1 = nn.Parameter(torch.randn([batch_size, self.input_dim]))
        self.q2 = nn.Parameter(torch.randn([batch_size, self.input_dim]))
        self.sigma = nn.Parameter(torch.ones(1) * 0.1)

        if use_cuda:
            self.cuda()

    def model(self, data):
        # set up parameters
        pyro.module('StateTrans', self.trans)

        q = self.q2
        dq = (self.q2 - self.q1) / self.h
        self.T = data.shape[1]
        with pyro.plate('mini_batch', len(data)):
            pyro.sample('y_0', dist.Normal(loc=self.q1, scale=self.sigma).to_event(1), obs=data[:, 0, :])
            pyro.sample('y_1', dist.Normal(loc=self.q2, scale=self.sigma).to_event(1), obs=data[:, 1, :])
            for t in range(2, self.T):
                q, dq = self.trans(q, dq, self.h)
                pyro.sample('y_{}'.format(t), dist.Normal(loc=q, scale=self.sigma).to_event(1), obs=data[:, t, :])

    def evaluate_model(self, data):
        pyro.module('StateTrans', self.trans)
        q = data[:, 1, :]
        dq = (data[:, 1, :] - data[:, 0, :]) / self.h
        T = data.shape[1]

        with pyro.plate('mini_batch', len(data)):
            for t in range(2, T):
                q, dq = self.trans(q, dq, self.h)
                pyro.sample('y_{}'.format(t), dist.Normal(loc=q, scale=self.sigma).to_event(1), obs=data[:, t, :])

    def guide(self, data):
        pass
