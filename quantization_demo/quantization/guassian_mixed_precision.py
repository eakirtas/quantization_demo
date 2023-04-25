import numpy as np
import torch as T


class MixedGaussianQScheduler():
    def __init__(
        self,
        model,
        gamma,
        delta,
        gamma_thrs,
        min_bits,
        bit_step,
    ):
        self.gamma = gamma
        self.delta = delta
        self.gamma_thrs = gamma_thrs
        self.min_bits = min_bits
        self.bit_step = bit_step

        self.init_quant_dict(model)

    def init_quant_dict(self, model):

        q_dict = model.q_dict
        layer_num = len(model.q_layers)

        borders = np.linspace(-self.gamma_thrs, self.gamma_thrs, layer_num + 1)

        q_dict['generators'] = []
        q_dict['possible_range'] = []
        q_dict['active_range'] = []
        q_dict['stepper'] = []
        q_dict['counter'] = []
        q_dict['bits'] = []
        q_dict['point'] = []
        q_dict['reset'] = []

        for i in range(layer_num):

            generator = T.distributions.normal.Normal(loc=0, scale=self.gamma)
            pos_range = borders[i:i + 2].copy()

            if pos_range[0] < 0:
                stepper = -(pos_range[0] - pos_range[1]) / self.delta
                active_range = [pos_range[0], pos_range[0]]

            else:
                stepper = (pos_range[0] - pos_range[1]) / self.delta
                active_range = [pos_range[1], pos_range[1]]

            q_dict['generators'].append(generator)
            q_dict['possible_range'].append(pos_range)
            q_dict['stepper'].append(stepper)
            q_dict['counter'].append(0)
            q_dict['active_range'].append(active_range)
            q_dict['bits'].append(q_dict['init_bits'])
            q_dict['point'].append(0)
            q_dict['reset'].append(False)

        return q_dict

    def step(self, model):
        q_dict = model.q_dict
        q_layers = model.q_layers

        for i, q_layer in enumerate(q_layers):
            q_dict['reset'][i] = False
            if q_layer.num_bits > self.min_bits:

                q_dict['point'][i] = q_dict['generators'][i].sample().item()

                if q_dict['point'][i] > q_dict['active_range'][i][0] and q_dict[
                        'point'][i] < q_dict['active_range'][i][1]:

                    q_layer.set_bits(int(q_layer.num_bits - self.bit_step))

                    if q_dict['stepper'][i] > 0:
                        q_dict['active_range'][i][1] = q_dict[
                            'possible_range'][i][0]
                    else:
                        q_dict['active_range'][i][0] = q_dict[
                            'possible_range'][i][1]
                    q_dict['bits'][i] = q_layer.num_bits
                    q_dict['reset'][i] = True

                    print('Reset at epoch {}| Layer: {} - {}bits'.format(
                        q_dict['counter'][i], i, q_layer.num_bits))

                elif q_dict['stepper'][i] > 0:

                    q_dict['active_range'][i][1] = np.clip(
                        q_dict['active_range'][i][1] + q_dict['stepper'][i],
                        a_min=None,
                        a_max=q_dict['possible_range'][i][1])

                else:
                    q_dict['active_range'][i][0] = np.clip(
                        q_dict['active_range'][i][0] + q_dict['stepper'][i],
                        a_min=q_dict['possible_range'][i][0],
                        a_max=None)

            q_dict['counter'][i] += 1

        return q_dict
