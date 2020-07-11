import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseEvaluater
from utils import *

class EvaluaterDE(BaseEvaluater):
    """
    Evaluater class
    """
    def __init__(self, model, criterion, metric_ftns, config, test_data_loader):
        super().__init__(model, criterion, metric_ftns, config)
        self.config = config
        self.test_data_loader = test_data_loader
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

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

                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target, type="DE"))

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        self._visualization(Outputs, targets)

    def _visualization(self, Outputs, targets):
        save_path = str(self.result_dir)
        train = False
        scatter = self.config["evaluater"]["visualization"]["scatter"]
        mu = Outputs[:, :1]
        sig = Outputs[:, 1:]
        sig_pos = torch.log(1 + torch.exp(sig)) + 1e-06
        std = torch.sqrt(sig_pos)
        plot_uncertainty(mu, std, targets[:, :1], save_path, train, scatter)



