import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseEvaluater
from utils import *


class Evaluater(BaseEvaluater):
    """
    Evaluater class
    """
    def __init__(self, model, criterion, metric_ftns, config, test_data_loader):
        super().__init__(model, criterion, metric_ftns, config)
        self.config = config
        self.test_data_loader = test_data_loader
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        if config['trainer']['type'] == 'Deep_Ensemble':
            self.type = 'DE'
        else:
            self.type = 'Normal'

    def evaluate(self):
        """
        Evaluate after training procedure finished

        :return: A log that contains information about validation
        """
        self.model.eval()

        Outputs = torch.zeros(self.test_data_loader.n_samples, self.model.output_dim).to(self.device)
        targets = torch.zeros(self.test_data_loader.n_samples, self.model.output_dim)

        with torch.no_grad(): # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
            start = 0
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                end = len(data) + start
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                Outputs[start:end, :] = output
                targets[start:end, :] = target
                start = end

                loss = self._compute_loss(output, target)
                self.test_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self._compute_metric(self.test_metrics, met,output, target, self.type)

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        self._visualization(Outputs, targets)

    def _compute_loss(self, output, target):
        if self.model.regression_type == 'homo':
            loss = self.criterion(output, target, self.model.log_noise.exp(), self.model.output_dim)
        elif self.model.regression_type == 'hetero':
            loss = self.criterion(output, target, self.model.output_dim/2)
        else:
            loss = self.criterion(output, target)
        return loss

    def _compute_metric(self, metrics, met, output, target, type=None):
        if self.model.regression_type == 'homo':
            metrics.update(met.__name__, met([output, self.model.log_noise.exp()], target,type))
        else:
            metrics.update(met.__name__, met(output, target, type))

    def _visualization(self, Outputs, targets):
        save_path = str(self.result_dir)
        train = False
        scatter = self.config["evaluater"]["visualization"]["scatter"]
        if self.type == 'DE':
            mu = Outputs[:, :1]
            sig = Outputs[:, 1:]
            sig_pos = torch.log(1 + torch.exp(sig)) + 1e-06
            plot_uncertainty(mu, sig_pos, targets[:, :1], save_path, train, scatter)
        else:
            plot_prediction(Outputs, targets, save_path, train, scatter)


