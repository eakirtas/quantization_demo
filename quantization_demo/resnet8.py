import torch as T
import torch.nn as nn
from torch_fquant.v2 import LayerQuantWrap


class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,
                 activation, q_dict):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   bias=False)

        self.q_conv_res1 = LayerQuantWrap(
            self.conv_res1,
            q_dict['train_quant'],
            q_dict['init_bits'],
            q_dict['stats_lmbd'],
        )

        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels,
                                           momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   bias=False)
        self.q_conv_res2 = LayerQuantWrap(
            self.conv_res2,
            q_dict['train_quant'],
            q_dict['init_bits'],
            q_dict['stats_lmbd'],
        )

        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels,
                                           momentum=0.9)

        self.q_dwn_conv_3 = None
        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.dwn_conv_3 = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        stride=stride,
                                        bias=False),
            self.q_dwn_conv_3 = LayerQuantWrap(
                self.conv_res2,
                q_dict['train_quant'],
                q_dict['init_bits'],
                q_dict['stats_lmbd'],
            )
            self.dwn_norm_3 = nn.BatchNorm2d(num_features=out_channels,
                                             momentum=0.9)

        else:
            self.downsample = None

        self.activation = activation()

        self.q_layers = [self.q_conv_res1, self.q_conv_res2, self.q_dwn_conv_3]
        self.q_dict = q_dict

    def forward(self, x):
        residual = x

        x = self.q_conv_res1(x)
        x = self.conv_res1_bn(x)
        out = self.activation(x)

        out = self.q_conv_res2(out)
        out = self.conv_res2_bn(out)

        if self.downsample is not None and self.q_dwn_conv_3 is not None:
            residual = self.q_dwn_conv_3(residual)
            residual = self.dwn_norm_3(residual)

        out = self.activation(out)
        out = out + residual

        return out


class ResNet8(nn.Module):
    """
    A Residual network.
    """

    def __init__(self, q_dict: dict, activation=T.nn.ReLU):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.q_conv_1 = LayerQuantWrap(
            self.conv_1,
            q_dict['train_quant'],
            q_dict['init_bits'],
            q_dict['stats_lmbd'],
        )
        self.norm_1 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.act_1 = activation()

        self.conv_2 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.q_conv_2 = LayerQuantWrap(
            self.conv_2,
            q_dict['train_quant'],
            q_dict['init_bits'],
            q_dict['stats_lmbd'],
        )

        self.norm_2 = nn.BatchNorm2d(num_features=128, momentum=0.9)
        self.act_2 = activation()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_3 = ResidualBlock(in_channels=128,
                                   out_channels=128,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   activation=activation,
                                   q_dict=q_dict)

        self.conv_4 = nn.Conv2d(in_channels=128,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.q_conv_4 = LayerQuantWrap(
            self.conv_4,
            q_dict['train_quant'],
            q_dict['init_bits'],
            q_dict['stats_lmbd'],
        )

        self.norm_4 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.act_4 = activation()
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv_5 = nn.Conv2d(in_channels=256,
        #                         out_channels=256,
        #                         kernel_size=3,
        #                         stride=1,
        #                         padding=1,
        #                         bias=False)

        # self.q_conv_5 = LayerQuantWrap(
        #     self.conv_5,
        #     q_dict['train_quant'],
        #     q_dict['init_bits'],
        #     q_dict['stats_lmbd'],
        # )

        # self.norm_5 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        # self.act_5 = activation()
        # self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_6 = ResidualBlock(in_channels=256,
                                   out_channels=256,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   activation=activation,
                                   q_dict=q_dict)
        self.pool_6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_7 = nn.Linear(in_features=4096, out_features=10, bias=True)

        self.q_fc_7 = LayerQuantWrap(
            self.fc_7,
            q_dict['train_quant'],
            q_dict['init_bits'],
            q_dict['stats_lmbd'],
        )

        self.res_3.conv_res1

        self.q_layers = [
            self.q_conv_1, self.q_conv_2, self.res_3.q_conv_res1,
            self.res_3.q_conv_res2, self.q_conv_4, self.res_6.q_conv_res1,
            self.res_6.q_conv_res2, self.q_fc_7
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
        x = self.q_conv_1(x)
        x = self.norm_1(x)
        x = self.act_1(x)

        x = self.q_conv_2(x)
        x = self.norm_2(x)
        x = self.act_2(x)
        x = self.pool_2(x)

        x = self.res_3(x)

        x = self.q_conv_4(x)
        x = self.norm_4(x)
        x = self.act_4(x)
        x = self.pool_4(x)

        # x = self.q_conv_5(x)
        # x = self.norm_5(x)
        # x = self.act_5(x)
        # x = self.pool_5(x)

        x = self.res_6(x)
        x = self.pool_6(x)

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.q_fc_7(x)

        return x
