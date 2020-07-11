import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseEvaluater, BaseEvaluaterEnsemble
from utils import *


class EvaluaterDEEnsemble(BaseEvaluaterEnsemble):
    """
    Evaluater class
    """
    def __init__(self, models, criterion, metric_ftns, config, test_data_loader):
        super().__init__(models, criterion, metric_ftns, config)
        self.config = config
        self.test_data_loader = test_data_loader
        self.test_metrics = MetricTracker(*['loss_' + str(i)  for i in range(self.n_ensembles)], *[m.__name__ + '_' + str(i) for m in self.metric_ftns for i in range(self.n_ensembles)],
                                          *[m.__name__ + '_all' for m in self.metric_ftns])

    def evaluate(self):
        """
        Evaluate after training procedure finished

        :return: A log that contains information about validation
        """
        Outputs = torch.zeros(self.test_data_loader.n_samples, self.models[0].output_dim, self.n_ensembles)
        global targets

        for i, model in enumerate(self.models):
            model.eval()

            outputs = torch.zeros(self.test_data_loader.n_samples, model.output_dim).to(self.device)
            targets = torch.zeros(self.test_data_loader.n_samples, model.output_dim)

            with torch.no_grad(): # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
                start = 0
                for batch_idx, (data, target) in enumerate(self.test_data_loader):
                    end = len(data) + start
                    data, target = data.to(self.device), target.to(self.device)

                    output = model(data)
                    outputs[start:end, :] = output
                    targets[start:end, :] = target
                    start = end

                    loss = self.criterion(output, target)
                    self.test_metrics.update('loss_' + str(i), loss.item())

                    for met in self.metric_ftns:
                        self.test_metrics.update(met.__name__ + '_' + str(i), met(output, target, type="DE"))

            self._visualization(outputs, targets, i)
            Outputs[:, :, i] = outputs

        self._info_ensemble(Outputs, targets)
        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        self._visualization_ensemble(Outputs, targets)

    def _visualization(self, outputs, targets, i):
        save_path = str(self.result_dir)
        train = False
        scatter = self.config["evaluater"]["visualization"]["scatter"]
        file_name = 'Net_' + str(i) + '_Uncertainty'
        mu = outputs[:, :1]
        sig = outputs[:, 1:]
        sig_pos = torch.log(1 + torch.exp(sig)) + 1e-06
        std = torch.sqrt(sig_pos)
        plot_uncertainty(mu, std, targets[:, :1], save_path, train, scatter, file_name)

    def _visualization_ensemble(self, Outputs, targets):
        save_path = str(self.result_dir)
        train = False
        scatter = self.config["evaluater"]["visualization"]["scatter"]
        file_name = 'Net_Ensemble_Uncertainty'

        mu =  Outputs[:, :1, :]
        sig = Outputs[:, 1:, :]
        sig_pos = torch.log(1 + torch.exp(sig)) + 1e-06
        means = torch.mean(mu, dim=2)
        vars = torch.mean(sig_pos + torch.pow(mu,2), dim=2) - torch.pow(means,2)
        stds = torch.sqrt(vars)
        plot_uncertainty(means, stds, targets[:, :1], save_path, train, scatter, file_name)

    def _info_ensemble(self, Outputs, targets):
        mu = Outputs[:, :1, :]
        sig = Outputs[:, 1:, :]
        sig_pos = torch.log(1 + torch.exp(sig)) + 1e-06
        means = torch.mean(mu, dim=2)
        vars = torch.mean(sig_pos + torch.pow(mu, 2), dim=2) - torch.pow(means, 2)

        outputs = torch.zeros(Outputs.shape[0], Outputs.shape[1])
        outputs[:, :1] = means
        outputs[:, 1:] = vars

        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__ + '_all', met(outputs, targets, type="DE_ensemble"))

