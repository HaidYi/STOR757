# STOR 757
# UNC Chapel Hill

import torch
import torch.nn as nn
import torch.distributions as distributions

import pyro
import pyro.distributions as dist


class StateTrans(nn.Module):
    def __init__(self, input_dim, hidden_dim, mass=1.0):
        super(StateTrans, self).__init__()

        self.mass = mass
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.trans = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
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
        self.sigma = torch.ones(1) * 0.1

        if use_cuda:
            self.cuda()

    def model(self, data):
        # set up parameters
        pyro.module('VIN', self)

        q_prev = self.q2
        dq_prev = (self.q2 - self.q1) / self.h
        self.T = data.shape[1]
        with pyro.plate('mini_batch', len(data)):
            pyro.sample('y_0', dist.Normal(loc=self.q1, scale=self.sigma).to_event(1), obs=data[:, 0].unsqueeze(-1))
            pyro.sample('y_1', dist.Normal(loc=self.q2, scale=self.sigma).to_event(1), obs=data[:, 1].unsqueeze(-1))
            for t in range(2, self.T):
                q, dq = self.trans(q_prev, dq_prev, self.h)
                pyro.sample('y_{}'.format(t), dist.Normal(loc=q, scale=self.sigma).to_event(1), obs=data[:, t].unsqueeze(-1))

                q_prev = q
                dq_prev = dq

    def evaluate_model(self, data):
        # set up parameters
        pyro.module('VIN', self)

        q = torch.autograd.Variable(data[:, 1].unsqueeze(-1), requires_grad=True)
        dq = (q - data[:, 0].unsqueeze(-1)) / self.h
        T = data.shape[1]

        with pyro.plate('mini_batch', len(data)):
            for t in range(2, T):
                q, dq = self.trans(q, dq, self.h)
                pyro.sample('y_{}'.format(t), dist.Normal(loc=q, scale=self.sigma).to_event(1), obs=data[:, t])

    def guide(self, data):
        pass


class VIN_torch(nn.Module):
    def __init__(self, trans_model, batch_size, num_steps, use_cuda=False):
        super(VIN_torch, self).__init__()

        self.trans = trans_model
        self.batch_size = batch_size
        self.h = 1. / num_steps

        self.q1 = nn.Parameter(torch.randn([batch_size, self.trans.input_dim]))
        self.q2 = nn.Parameter(torch.randn([batch_size, self.trans.input_dim]))
        self.sigma = torch.ones(1) * 0.1

        if use_cuda:
            self.cuda()

    def forward(self, data):
        q_prev = self.q2
        dq_prev = (self.q2 - self.q1) / self.h
        q_list = [self.q1, self.q2]
        T_max = data.shape[1]

        q1_dist = distributions.Normal(loc=self.q1, scale=self.sigma)
        q2_dist = distributions.Normal(loc=self.q2, scale=self.sigma)

        log_prob = 0.
        log_prob += q1_dist.log_prob(data[:, 0].unsqueeze(-1)).sum() + q2_dist.log_prob(data[:, 1].unsqueeze(-1)).sum()

        for t in range(2, T_max):
            q, dq = self.trans(q_prev, dq_prev, self.h)
            dist_normal = distributions.Normal(loc=q, scale=self.sigma)
            log_prob += dist_normal.log_prob(data[:, t].unsqueeze(-1)).sum()

            q_prev = q
            dq_prev = dq
            q_list.append(q)

        return torch.cat(q_list, dim=-1), log_prob
