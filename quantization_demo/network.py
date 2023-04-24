import torch as T
from torch_fquant.v2 import LayerQuantWrap


class MnistNet(T.nn.Module):

    def __init__(self, input, output, activation, q_dict: dict):
        super().__init__()

        self.layer_1 = T.nn.Linear(input, output)
        self.layer_1_q = LayerQuantWrap(
            self.layer_1,
            q_dict['train_quant'],
            q_dict['bits'],
            q_dict['stats_lmbd'],
        )
        self.a_func_1 = activation()

        self.layer_2 = T.nn.Linear(10, 20)
        self.layer_2_q = LayerQuantWrap(
            self.layer_2,
            q_dict['train_quant'],
            q_dict['bits'],
            q_dict['stats_lmbd'],
        )

        self.a_func_2 = activation()

        self.layer_3 = T.nn.Linear(20, 20)
        self.layer_3_q = LayerQuantWrap(
            self.layer_3,
            q_dict['train_quant'],
            q_dict['bits'],
            q_dict['stats_lmbd'],
        )
        self.a_func_3 = activation()

        self.layer_4 = T.nn.Linear(20, output)
        self.layer_4_q = LayerQuantWrap(
            self.layer_4,
            q_dict['train_quant'],
            q_dict['bits'],
            q_dict['stats_lmbd'],
        )

        self.q_layers = [
            self.layer_1_q, self.layer_2_q, self.layer_3_q, self.layer_4_q
        ]

        self.q_dict = q_dict

    def train(self, mode: bool = True):
        if mode:
            self.set_q_properties(self.q_dict['train_quant'])
        else:
            self.set_q_properties(self.q_dict['eval_quant'])

        return super().train(mode)

    def eval(self):
        self.set_q_properties(self.q_dict['eval_quant'])
        return super().eval()

    def set_q_properties(self, is_quant):
        for q_layer in self.q_layers:
            q_layer.set_quant_mode(is_quant)

    def forward(self, x):
        x = T.flatten(x, start_dim=1)

        x = self.layer_1_q(x)
        x = self.a_func_1(x)

        x = self.layer_2_q(x)
        x = self.a_func_2(x)

        x = self.layer_3_q(x)
        x = self.a_func_3(x)

        x = self.layer_4_q(x)

        return x
