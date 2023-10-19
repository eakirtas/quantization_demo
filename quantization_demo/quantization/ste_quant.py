import torch as T

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class QTensor(T.nn.Module):

    @T.no_grad()
    def __init__(self, param: T.Tensor, stats, num_bits: int):
        super().__init__()

        self.stats = stats

        self.register_buffer('num_bits', T.tensor(num_bits)).to(DEVICE)

        self.register_buffer('q_min', T.empty((1, )).to(DEVICE))
        self.register_buffer('q_max', T.empty((1, )).to(DEVICE))

        self.get_q(self.num_bits)

        self.register_buffer('scale', T.tensor(0))
        self.register_buffer('zero_point', T.tensor(0))

        # if param is not None:
        #     vmin, vmax = stats(param, False)
        #     self.scale = T.tensor(self.get_scale(vmin, vmax))

        #     self.zeropoint = T.tensor(
        #         self.get_zeropoint(self.scale, vmin, self.q_min, self.q_max))

    @T.no_grad()
    def __call__(self, x):
        vmin, vmax = self.stats(x)

        self.get_scale(vmin, vmax)
        self.get_q(self.num_bits)

        self.get_zeropoint(scale=self.scale,
                           vmin=vmin,
                           q_min=self.q_min,
                           q_max=self.q_max)
        return x

    @T.no_grad()
    def set_bits(self, num_bits):
        self.num_bits = T.tensor(num_bits)
        self.q_min, self.q_max = self.get_q(num_bits)

    @T.no_grad()
    def get_q(self, num_bits):

        self.q_min = T.tensor(0.).to(DEVICE)
        self.q_max = 2.**num_bits - 1

        return self.q_min, self.q_max

    @T.no_grad()
    def get_scale(self, vmin, vmax):
        self.scale = (vmax - vmin) / (self.q_max - self.q_min)
        return self.scale

    @T.no_grad()
    def get_zeropoint(self, scale, vmin, q_min, q_max):

        if scale == 0:
            initial_zero_point = q_min
        else:
            initial_zero_point = q_min - vmin / scale

        zero_point = T.tensor(0)

        if scale == 0:
            zero_point = 0
        elif initial_zero_point < q_min:
            zero_point = q_min
        elif initial_zero_point > q_max:
            zero_point = q_max
        else:
            zero_point = initial_zero_point

        self.zeropoint = T.tensor(int(zero_point))

        return self.zeropoint


class DequantizeTensor(T.autograd.Function):

    @staticmethod
    def forward(ctx, x, x_q):
        return x_q.scale * (x.float() - x_q.zeropoint)

    @staticmethod
    def backward(ctx, x_grad):
        return x_grad


class QuantizeTensor(T.autograd.Function):

    @staticmethod
    def forward(ctx, x, x_q):
        q_x = (x_q.zeropoint + x / x_q.scale)
        q_x.clamp_(x_q.q_min, x_q.q_max).round_()
        return q_x.byte()

    @staticmethod
    def backward(ctx, x_grad):
        return x_grad


class Calculate(T.autograd.Function):

    @staticmethod
    def forward(ctx, x, x_q: QTensor):
        vmin, vmax = x_q.stats(x)
        x_q.get_scale(vmin, vmax)
        x_q.get_q(x_q.num_bits)

        x_q.get_zeropoint(scale=x_q.scale,
                          vmin=vmin,
                          q_min=x_q.q_min,
                          q_max=x_q.q_max)
        return x

    @staticmethod
    def backward(ctx, x_grad):
        return x_grad


class FakeQuant(T.autograd.Function):

    @staticmethod
    def forward(ctx, x, x_q: QTensor):

        x = Calculate.apply(x, x_q)
        x = QuantizeTensor.apply(x, x_q)

        x = DequantizeTensor.apply(x, x_q)

        return x

    @staticmethod
    def backward(ctx, x_grad):
        x_grad = Calculate.backward(ctx, x_grad)
        x_grad = QuantizeTensor.backward(ctx, x_grad)
        x_grad = DequantizeTensor.backward(ctx, x_grad)

        return x_grad, None
