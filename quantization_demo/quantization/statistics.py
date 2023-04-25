import json

import torch as T


class MovingAverage(T.nn.Module):
    # TODO: Step should be increased only when it is on train mode
    @T.no_grad()
    def __init__(self, x=None, inc_step=True):

        super().__init__()

        self.register_buffer('step_v', T.tensor(1))
        self.register_buffer('inc_step', T.tensor(inc_step))
        self.register_buffer('ema_min', T.tensor(0.0))
        self.register_buffer('ema_max', T.tensor(0.0))

        if x is not None:
            self(x)

    @T.no_grad()
    def forward(self, x, do_step=True):
        if len(x.size()) > 1:
            vmax, _ = T.max(x, dim=1)
            vmin, _ = T.min(x, dim=1)
        else:
            vmax = T.max(x)
            vmin = T.min(x)

        ema_w = 2.0 / (self.step_v + 1)

        self.ema_min = ema_w * (vmin.mean()) + (1.0 - ema_w) * self.ema_min
        self.ema_max = ema_w * (vmax.mean()) + (1.0 - ema_w) * self.ema_max

        self.step(do_step)

        return self.ema_min, self.ema_max

    def step(self, do_step: bool):
        if self.inc_step and do_step:
            self.step_v += 1

    def get_vmin(self):
        return self.ema_min

    def get_vmax(self):
        return self.ema_max


class MinMax(T.nn.Module):
    @T.no_grad()
    def __init__(self, x=None):
        super().__init__()

        self.register_buffer('vmin', T.Tensor([0.0]))
        self.register_buffer('vmax', T.Tensor([0.0]))

        if x is not None:
            self(x)

    @T.no_grad()
    def forward(self, x, do_step=True):
        if len(x.size()) > 1:
            self.vmin = T.min(x, dim=1)[0].mean().item()
            self.vmax = T.max(x, dim=1)[0].mean().item()
        else:
            self.vmin = T.min(x).mean()
            self.vmax = T.max(x).mean()

        return self.vmin, self.vmax

    def get_vmin(self):
        return self.vmin

    def get_vmax(self):
        return self.vmax


class StdApproach(T.nn.Module):
    @T.no_grad()
    def __init__(self, alpha, beta=2.0, x=None, inc_step=True):
        super().__init__()

        self.register_buffer('beta', T.tensor(beta))
        self.register_buffer('alpha', T.tensor(alpha))
        self.register_buffer('inc_step', T.tensor(inc_step))
        self.register_buffer('step_v', T.tensor(1.0))

        self.register_buffer('ema_w', T.tensor(0.0))
        self.register_buffer('ema_avg', T.tensor(0.0))
        self.register_buffer('ema_std', T.tensor(0.0))

        self.register_buffer('vmin', T.tensor(-1))  # TODO: Should be checked
        self.register_buffer('vmax', T.tensor(1))  # TODO: Should be checked

        if x is not None:
            self.vmin, self.vmax = self(x)

    @T.no_grad()
    def __call__(self, x, do_step=True):
        self.ema_w = self.beta / (self.step_v + 1)

        if len(x.size()) == 1:
            avg_val = T.mean(x).mean()
            std_val = T.std(x, unbiased=False).mean()
        else:

            avg_val = T.mean(x, dim=1).mean()
            std_val = T.std(x, dim=1, unbiased=False).mean()

        self.ema_avg = self.ema_w * avg_val + (1.0 - self.ema_w) * self.ema_avg
        self.ema_std = self.ema_w * std_val + (1.0 - self.ema_w) * self.ema_std

        self.vmin = self.ema_avg - self.alpha * self.ema_std
        self.vmax = self.ema_avg + self.alpha * self.ema_std

        self.step(do_step)

        return self.vmin, self.vmax

    def step(self, do_step: bool):
        if self.inc_step and do_step:
            self.step_v += 1

    def get_vmin(self):
        return self.vmin

    def get_vmax(self):
        return self.vmax


class MinMaxStd(T.nn.Module):
    def __init__(self, alpha, x=None):
        super().__init__()

        self.register_buffer('alpha', T.tensor(alpha))
        self.register_buffer('vmin', T.tensor(0.0))
        self.register_buffer('vmax', T.tensor(0.0))

        if x is not None:
            self.vmin, self.vmax = self(x)

    @T.no_grad()
    def __call__(self, x, do_step=True):
        if len(x.size()) > 1:
            average_v = T.mean(x, dim=1).mean()
            std_v = T.std(x, dim=1, unbiased=False).mean()
        else:
            average_v = T.mean(x).mean()
            std_v = T.std(x, unbiased=False).mean()

        self.vmin = average_v - self.alpha * std_v
        self.vmax = average_v + self.alpha * std_v

        return self.vmin, self.vmax

    def get_vmin(self):
        return self.vmin

    def get_vmax(self):
        return self.vmax
