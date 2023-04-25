import torch as T

from quantization_demo.network import MnistNet
from quantization_demo.quantization.statistics import (MovingAverage,
                                                       StdApproach)
from quantization_demo.runners.multiclass_runner import MulticlassRunner
from utils import get_mnist

CONFIG = {'lr': 0.0001, 'batch_size': 256, 'epochs': 10}

QUANTIZATION_METHODS = {
    'regular': {
        'train_quant': False,
        'eval_quant': False,
        'stats_lmbd': None,
    },
    'post_train': {
        'train_quant': False,
        'eval_quant': True,
        'stats_lmbd': MovingAverage,
    },
    'quantization_aware': {
        'train_quant': True,
        'eval_quant': True,
        'stats_lmbd': MovingAverage,
    },
    'std_quant_aware': {
        'train_quant': True,
        'eval_quant': True,
        'stats_lmbd': lambda: StdApproach(alpha=3.5, beta=2),
    },
    'std_post_train': {
        'train_quant': False,
        'eval_quant': True,
        'stats_lmbd': lambda: StdApproach(alpha=3.5, beta=2),
    }
}


def run(q_dict, config):
    train_dl, test_dl = get_mnist(batch_size=config['batch_size'])

    model = MnistNet(
        784,
        10,
        T.nn.ReLU,
        q_dict,
    )

    runner = MulticlassRunner(T.nn.CrossEntropyLoss())

    optimizer = T.optim.RMSprop(model.parameters(), lr=config['lr'])

    train_loss, train_accuracy = runner.fit(model,
                                            optimizer,
                                            train_dl,
                                            num_epochs=config['epochs'],
                                            verbose=1)

    print("Train Loss={:.4f}  Accuracy={:.4f}".format(train_loss,
                                                      train_accuracy))

    eval_loss, eval_accuracy = runner.eval(model, test_dl, verbose=0)

    print("Evaluation: Loss={:.4f} Accuracy={:.4f}".format(
        eval_loss, eval_accuracy))


BITS = 4
for q_method, q_dict in QUANTIZATION_METHODS.items():
    print('Running for method {}...'.format(q_method))
    q_dict['bits'] = BITS

    run(q_dict, CONFIG)
