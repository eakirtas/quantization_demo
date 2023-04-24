import numpy as np
import torch as T
from train_utils.runners.runner import Runner

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class BinaryclassRunner(Runner):
    def __init__(self, criterion=T.nn.BCELoss(), device=DEVICE):
        super().__init__(criterion=criterion, device=device)

    def predict(self, output):
        pred = T.gt(output, 0.5)
        return pred.flatten()

    def compute_cost(self, output, labels):
        return self.criterion(output.flatten(), labels.float())


class BinaryImbalanceRunner(BinaryclassRunner):
    def __init__(self,
                 criterion=T.nn.BCELoss(),
                 device=DEVICE,
                 use_onehot=False):
        super().__init__(criterion=criterion, device=device)
        self.use_onehot = use_onehot

    def predict(self, output):
        if self.use_onehot:
            return T.argmax(output, dim=1)
        else:
            pred = T.gt(output, 0.5)
            return pred.flatten()

    def compute_cost(self, output, labels):
        if self.use_onehot:
            return self.criterion(output, labels.float())
        else:
            return self.criterion(output.flatten(), labels.float())

    def run_epoch(self, model: T.nn.Module, optimizer: T.optim.Optimizer,
                  train_dataloader: T.utils.data.DataLoader, type):

        running_loss, running_accuracy, total = 0.0, 0.0, 0.0

        for inputs, target in train_dataloader:

            model.zero_grad()
            optimizer.zero_grad()

            inputs = inputs.to(self.device).type(type)
            target = self.prep_target(target.to(self.device))

            with T.set_grad_enabled(True):
                z_out, y_out = model(inputs)
                pred = self.predict(y_out)
                loss = self.compute_cost(z_out, target)

                loss.backward()

                if self.gradient_clipping is not None:
                    self.gradient_clipping(model.parameters())

                if self.scheduler is not None:
                    self.scheduler.step()

                optimizer.step()

            running_loss += loss.detach()
            total += target.size(0)

            if self.use_onehot:
                target = T.argmax(target, axis=1)
                running_accuracy += (pred.detach()
                                     == target.detach()).sum().item() * 1.0
            else:
                running_accuracy += (pred.detach()
                                     == target.detach()).sum().item() * 1.0

            del z_out, y_out, pred, loss

        return running_loss / total, running_accuracy / total

    def eval(self, model, eval_dataloader, verbose=2, type=T.cuda.FloatTensor):
        model.to(self.device)
        model.eval()

        running_loss, running_corrects, total = 0.0, 0.0, 0

        acc_pred = np.empty(len(eval_dataloader.dataset), dtype=int)
        acc_target = np.empty(len(eval_dataloader.dataset), dtype=int)

        for inputs, target in eval_dataloader:

            inputs = inputs.to(self.device).type(type)
            target = self.prep_target(target.to(self.device))

            with T.no_grad():
                z_out, y_out = model(inputs)
                pred = self.predict(y_out)
                loss = self.compute_cost(z_out, target)

            running_loss += loss.detach()
            total += target.size(0)

            if self.use_onehot:
                target = T.argmax(target, axis=1)
                running_corrects += (pred.detach()
                                     == target.detach()).sum().item() * 1.0
            else:
                running_corrects += (pred.detach()
                                     == target.detach()).sum().item() * 1.0

            acc_pred[total - target.size(0):total] = pred.detach().long().cpu(
            ).numpy().squeeze()

            acc_target[total - target.size(0):total] = target.detach().cpu(
            ).numpy().squeeze()

            del z_out, y_out, pred, loss

        final_loss, final_accuracy = running_loss / total, running_corrects / total

        if (verbose > 1):
            print('Evaluation: Loss: {:.4f} Acc: {:.4f}'.format(
                final_loss, final_accuracy))

        return acc_target, acc_pred
