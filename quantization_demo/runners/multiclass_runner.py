from typing import Tuple

import numpy as np
import torch as T

from .runner import Runner

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class MulticlassRunner(Runner):

    def __init__(self,
                 criterion=T.nn.NLLLoss,
                 device=DEVICE,
                 gradient_clipping=None,
                 scheduler=None):
        super().__init__(criterion=criterion,
                         device=device,
                         gradient_clipping=gradient_clipping,
                         scheduler=scheduler)

    def predict(self, output: T.Tensor):
        return T.argmax(output, dim=1)

    def prep_target(self, target: T.Tensor):
        if len(target.size()) > 1:
            target = target.squeeze(dim=1)
        return target

    def compute_cost(self, output: T.Tensor, target: T.Tensor):
        return self.criterion(output, target)


class ImbalanceMulticlassRunner(MulticlassRunner):

    def __init__(self,
                 criterion=T.nn.NLLLoss,
                 device=DEVICE,
                 gradient_clipping=None,
                 scheduler=None):

        super().__init__(criterion=criterion,
                         device=device,
                         gradient_clipping=gradient_clipping,
                         scheduler=scheduler)

    def eval(
        self,
        model: T.nn.Module,
        eval_dataloader: T.utils.data.DataLoader,
        verbose: int = 2,
        type=T.cuda.FloatTensor
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:

        model.to(self.device)
        model.eval()

        running_loss, running_corrects, total = 0.0, 0.0, 0

        acc_pred = np.zeros(len(eval_dataloader.dataset), dtype=np.int)
        acc_target = np.zeros(len(eval_dataloader.dataset), dtype=np.int)

        for inputs, target in eval_dataloader:

            inputs = inputs.to(self.device).type(type)
            target = self.prep_target(target.to(self.device))

            with T.no_grad():
                output = model(inputs)
                pred = self.predict(output)
                loss = self.compute_cost(output, target)

            running_loss += loss.detach()
            total += target.size(0)
            running_corrects += (pred.detach()
                                 == target.detach()).sum().item() * 1.0

            acc_pred[total - target.size(0):total] = pred.detach().long().cpu(
            ).numpy().squeeze()
            acc_target[total - target.size(0):total] = target.detach().cpu(
            ).numpy().squeeze()

            del output, pred, loss

        final_loss, final_accuracy = running_loss / float(
            total), running_corrects / float(total)

        if (verbose > 1):
            print('Evaluation: Loss: {:.4f} Acc: {:.4f}'.format(
                final_loss, final_accuracy))

        return final_loss, final_accuracy, acc_target, acc_pred


class QuantRunner(MulticlassRunner):

    def __init__(self,
                 criterion=T.nn.NLLLoss,
                 device=DEVICE,
                 gradient_clipping=None,
                 scheduler=None,
                 q_scheduler=None):
        super().__init__(
            criterion=criterion,
            device=device,
            gradient_clipping=gradient_clipping,
            scheduler=scheduler,
        )

        self.q_scheduler = q_scheduler

    def run_epoch(self,
                  model: T.nn.Module,
                  optimizer: T.optim.Optimizer,
                  train_dataloader: T.utils.data.DataLoader,
                  type=T.cuda.FloatTensor):
        loss, accuracy = super().run_epoch(model, optimizer, train_dataloader,
                                           type)

        if self.q_scheduler is not None:
            self.q_scheduler.step(model)

        return loss, accuracy
