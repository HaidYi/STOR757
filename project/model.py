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
        q = torch.autograd.Variable(q, requires_grad=True)
        dfdq = torch.autograd.grad(self.trans(q).sum(), q, create_graph=True)[0]

        new_dq = dq - 1./self.mass * step_size * dfdq
        new_q = q + step_size * new_dq
        return new_q, new_dq


class VIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, step_size):
        super(VIN, self).__init__()

        self.input_dim = input_dim
        self.h = step_size
        self.trans = StateTrans(input_dim, hidden_dim)

    def model(self, data):
        # set up parameters
        pyro.module('StateTrans', self.trans)
        q1 = pyro.param('q1', torch.randn([len(data), self.input_dim]))
        q2 = pyro.param('q2', torch.randn([len(data), self.input_dim]))
        sigma = pyro.param('sigma', torch.ones(1) * 0.1)

        self.q = q2
        self.dq = (q2 - q1) / self.h
        self.T = data.shape[1]
        with pyro.plate('mini_batch', len(data)):
            pyro.sample('y_0', dist.Normal(loc=q1, scale=sigma).to_event(1), obs=data[:, 0, :])
            pyro.sample('y_1', dist.Normal(loc=q2, scale=sigma).to_event(1), obs=data[:, 1, :])
            for t in range(2, self.T):
                self.q, self.dq = self.trans(self.q, self.dq, self.h)
                pyro.sample('y_{}'.format(t), dist.Normal(loc=self.q, scale=sigma).to_event(1), obs=data[:, t, :])

    def guide(self, data):
        pass
