import torch as T
from torch_fquant.v2.mixed import MixedGaussianQScheduler
from torch_fquant.v2.observers import MovingAverage, Normalized

from quantization_demo.network import MnistNet
from quantization_demo.resnet8 import ResNet8
from quantization_demo.runners.multiclass_runner import QuantRunner
from utils import get_cifar10_dl, get_mnist

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")

CONFIG = {'lr': 0.0001, 'batch_size': 256, 'epochs': 30}

SIMPLE_QUANTIZATION_METHODS = {
    # 'regular': {
    #     'train_quant': False,
    #     'eval_quant': False,
    #     'stats_lmbd': None,
    #     'q_scheduler': None
    # },
    # 'post_train': {
    #     'train_quant': False,
    #     'eval_quant': True,
    #     'stats_lmbd': MovingAverage,
    #     'q_scheduler': None
    # },
    # 'quantization_aware': {
    #     'train_quant': True,
    #     'eval_quant': True,
    #     'stats_lmbd': MovingAverage,
    #     'q_scheduler': None
    # },
    'std_quant_aware': {
        'train_quant': True,
        'eval_quant': True,
        'stats_lmbd': lambda: Normalized(alpha=3.5, beta=2),
        'q_scheduler': None
    },
    'std_post_train': {
        'train_quant': False,
        'eval_quant': True,
        'stats_lmbd': lambda: Normalized(alpha=3.5, beta=2),
        'q_scheduler': None
    }
}

MIXED_QUANTIZATION_METHODS = {
    'proposed': {
        'train_quant':
        True,
        'eval_quant':
        True,
        'stats_lmbd':
        MovingAverage,
        'gamma':
        1,
        'gamma_thrs':
        3,
        'bit_step':
        2,
        'min_bits':
        2,
        'q_scheduler':
        lambda model, config: MixedGaussianQScheduler(
            model,
            config['gamma'],
            config['epochs'] / 6,
            config['gamma_thrs'],
            config['min_bits'],
            config['bit_step'],
        ),
    }
}


def run(config):

    train_dl, test_dl = get_mnist(batch_size=config['batch_size'])

    model = MnistNet(
        784,
        10,
        T.nn.ReLU,
        q_dict,
    ).to(DEVICE)

    runner = QuantRunner(T.nn.CrossEntropyLoss())

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


def run_resnet(config):

    print(config)

    train_dl, test_dl = get_cifar10_dl(config['batch_size'])

    model = ResNet8(config).to(DEVICE)

    q_scheduler = None
    if config['q_scheduler'] is not None:
        q_scheduler = config['q_scheduler'](model, config)

    runner = QuantRunner(
        T.nn.CrossEntropyLoss(),
        q_scheduler=q_scheduler,
    )

    optimizer = T.optim.RMSprop(model.parameters(), lr=config['lr'])

    train_loss, train_accuracy = runner.fit(model,
                                            optimizer,
                                            train_dl,
                                            num_epochs=config['epochs'],
                                            verbose=2)

    print("Train Loss={:.4f}  Accuracy={:.4f}".format(train_loss,
                                                      train_accuracy))

    eval_loss, eval_accuracy = runner.eval(model, test_dl, verbose=0)

    print("Evaluation: Loss={:.4f} Accuracy={:.4f}".format(
        eval_loss, eval_accuracy))


BITS = 8
for q_method, q_dict in SIMPLE_QUANTIZATION_METHODS.items():
    print('Running for method {}...'.format(q_method))
    q_dict['init_bits'] = BITS
    q_dict['bits'] = BITS

    config = CONFIG.copy()
    config.update(q_dict)

    run_resnet(config)
