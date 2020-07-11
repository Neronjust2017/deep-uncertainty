import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseEvaluater
from utils import *
from model.metric import *

class EvaluaterVd(BaseEvaluater):
    """
    Evaluater class
    """
    def __init__(self, model, criterion, metric_ftns, config, test_data_loader):
        super().__init__(model, criterion, metric_ftns, config)
        self.config = config
        self.test_data_loader = test_data_loader
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def evaluate(self, samples=1000):
        """
        Evaluate after training procedure finished

        :return: A log that contains information about validation
        """
        Outputs = torch.zeros(self.test_data_loader.n_samples, self.model.output_dim, samples).to(self.device)
        targets = torch.zeros(self.test_data_loader.n_samples, self.model.output_dim)

        self.model.eval()

        with torch.no_grad(): # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
            start = 0
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                end = len(data) + start
                data, target = data.to(self.device), target.to(self.device)

                loss = 0
                outputs = torch.zeros(data.shape[0], self.model.output_dim, samples).to(self.device)

                if samples == 1:
                    out, _ = self.model(data)
                    loss = self._compute_loss(out, target)
                    outputs[:, :, 0] = out

                elif samples > 1:
                    mlpdw_cum = 0

                    for i in range(samples):
                        out, _ = self.model(data, sample=True)
                        mlpdw_i = self._compute_loss(out, target)
                        mlpdw_cum = mlpdw_cum + mlpdw_i
                        outputs[:, :, i] = out

                    mlpdw = mlpdw_cum / samples
                    loss = mlpdw

                Outputs[start:end, :, :] = outputs
                targets[start:end, :] = target
                start = end

                self.test_metrics.update('loss', loss.item(), n=len(target))
                for met in self.metric_ftns:
                    self._compute_metric(self.test_metrics, met, outputs, target)

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

    def _compute_metric(self, metrics, met, output, target, type="VD"):
        if self.model.regression_type == 'homo':
            metrics.update(met.__name__, met([output, self.model.log_noise.exp()], target,type))
        else:
            metrics.update(met.__name__, met(output, target, type))

    def _visualization(self, Outputs, targets):
        save_path = str(self.result_dir)
        train = False
        scatter = self.config["evaluater"]["visualization"]["scatter"]
        if self.model.regression_type == 'homo':
            means = torch.mean(Outputs, dim=2)
            stds = torch.std(Outputs, dim=2)
            noise = self.model.log_noise.exp().detach()
            total_unc_1 = (noise ** 2 + stds ** 2) ** 0.5
            total_unc_2 = (noise ** 2 + (2 * stds) ** 2) ** 0.5
            total_unc_3 = (noise ** 2 + (3 * stds) ** 2) ** 0.5

            plot_uncertainty_noise(means, noise, [total_unc_1, total_unc_2, total_unc_3], targets, save_path, train, scatter)

        elif self.model.regression_type == 'hetero':
            means = torch.mean(Outputs[:,:1,:], dim=2)
            stds = torch.std(Outputs[:,:1,:], dim=2)
            noise = torch.mean(Outputs[:, 1:, :].exp() **2, dim=2) ** 0.5
            total_unc_1 = (noise ** 2 + stds ** 2) ** 0.5
            total_unc_2 = (noise ** 2 + (2 * stds) ** 2) ** 0.5
            total_unc_3 = (noise ** 2 + (3 * stds) ** 2) ** 0.5
            plot_uncertainty_noise(means, noise, [total_unc_1, total_unc_2, total_unc_3], targets[:, :1], save_path, train, scatter)

        else:
            means = torch.mean(Outputs, dim=2)
            stds = torch.std(Outputs, dim=2)
            plot_uncertainty(means, stds, targets, save_path, train, scatter)