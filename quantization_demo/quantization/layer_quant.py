import torch as T
from pnn_library.quantize.quantize_v2.ste_quant import FakeQuant, QTensor


class Dequantize():
    def __call__(self, x):
        if isinstance(x, QTensor):
            return x.o_tensor

        return x


class LayerQuantWrap(T.nn.Module):
    def __init__(self, layer: T.nn.Module, is_quant, num_bits, stat_lambda):
        super().__init__()

        self.registered_pre_hook = None
        self.registered_after_hook = None
        self.original_weight = None
        self.original_bias = None

        self.register_buffer('is_quant', T.tensor(False))
        self.register_buffer('num_bits', T.tensor(num_bits))

        self.layer = layer

        if stat_lambda is not None:
            self.stats = T.nn.ModuleDict({
                'input': stat_lambda(),
                'weight': stat_lambda(),
                'bias': stat_lambda(),
                'output': stat_lambda(),
            })

        self.set_quant_mode(is_quant)

    def set_stats(self, stats):
        self.stats = stats

    def set_quant_mode(self, is_quant):
        if self.is_quant != is_quant:
            self.is_quant = T.tensor(is_quant)
            if is_quant:
                self.q_weight, self.q_bias = None, None
                self.registered_pre_hook = self.layer.register_forward_pre_hook(
                    self.pre_hook)

                self.registered_after_hook = self.layer.register_forward_hook(
                    self.after_hook)
                self.q_weight = QTensor(self.layer.weight,
                                        self.stats['weight'], self.num_bits)

                if self.layer.bias is not None:
                    self.q_bias = QTensor(self.layer.bias, self.stats['bias'],
                                          self.num_bits)

            if not self.is_quant and self.registered_after_hook is not None:
                self.registered_after_hook.remove()
                self.registered_after_hook = None
                self.q_bias, self.q_weight = None, None

            if not self.is_quant and self.registered_pre_hook is not None:
                self.registered_pre_hook.remove()
                self.registered_pre_hook = None
                self.q_bias, self.q_weight = None, None

    def set_bits(self, num_bits):
        self.num_bits = num_bits

        if self.q_bias is not None:
            self.q_bias.set_bits(num_bits)

        if self.q_weight is not None:
            self.q_weight.set_bits(num_bits)

    def pre_hook(self, m, x):

        x_q = QTensor(x[0], self.stats['input'], self.num_bits)
        x = FakeQuant.apply(x[0], x_q)

        self.quant_layer()

        return x

    def after_hook(self, m, inputs, z):
        self.dequant_layer()

        z_q = QTensor(z[0], self.stats['output'], self.num_bits)
        z = FakeQuant.apply(z, z_q)

        return z

    def forward(self, x):
        return self.layer(x)

    def quant_layer(self):
        self.original_weight = self.layer.weight.data
        self.layer.weight.data = FakeQuant.apply(self.layer.weight.data,
                                                 self.q_weight)

        if self.layer.bias is not None:
            self.original_bias = self.layer.bias.data
            self.layer.bias.data = FakeQuant.apply(self.layer.bias.data,
                                                   self.q_bias)

    def dequant_layer(self):
        self.layer.weight.data = self.original_weight

        if self.layer.bias is not None:
            self.layer.bias.data = self.original_bias

    # def __repr__(self):
    #     r = {}
    #     for key, item in self.stats.items():
    #         r[key] = item.__dict__
    #     return r
