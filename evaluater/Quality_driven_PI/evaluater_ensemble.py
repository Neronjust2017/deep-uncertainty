import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseEvaluater, BaseEvaluaterEnsemble
from utils import *
from trainer.Quality_driven_PI.trainer import Loss
from model.metric import *

class EvaluaterQdEnsemble(BaseEvaluaterEnsemble):
    """
    Evaluater class
    """
    def __init__(self, models, criterion, metric_ftns, config, test_data_loader):
        super().__init__(models, criterion, metric_ftns, config)
        self.config = config
        self.test_data_loader = test_data_loader
        self.test_metrics = MetricTracker(*['loss_' + str(i)  for i in range(self.n_ensembles)], *[m.__name__ + '_' + str(i) for m in self.metric_ftns for i in range(self.n_ensembles)],
                                          *[m.__name__ + '_all' for m in self.metric_ftns])
        cfg_loss = config['evaluater']['loss']
        self.type_in = cfg_loss['type_in']
        self.alpha = cfg_loss['alpha']
        self.loss_type = cfg_loss['loss_type']
        self.censor_R = cfg_loss['censor_R']
        self.soften = cfg_loss['soften']
        self.lambda_in = cfg_loss['lambda_in']
        self.sigma_in = cfg_loss['sigma_in']

        self._create_loss()

    def _create_loss(self):
        self.loss = Loss(device=self.device,type_in=self.type_in, alpha=self.alpha,
                         loss_type=self.loss_type, censor_R=self.censor_R,
                         soften=self.soften, lambda_in=self.lambda_in,
                         sigma_in=self.sigma_in)

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
    
                    loss, PICP, MPIW = self.loss(output, target)
                    self.test_metrics.update('loss_' + str(i), loss.item())
                    for met in self.metric_ftns:
                        self.test_metrics.update(met.__name__ + '_' + str(i), met(output, target, type="QD"))

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

        y_pred_U = outputs[:, :1]
        y_pred_L = outputs[:, 1:]
        y_pred_mid = (y_pred_L + y_pred_U) / 2
        plot_err_bars(y_pred_U, y_pred_L, targets[:, :1], save_path)
        plot_uncertainty_QD(y_pred_U, y_pred_L, y_pred_mid, targets[:, :1], save_path, scatter, train, file_name)

    def _visualization_ensemble(self, Outputs, targets):
        save_path = str(self.result_dir)
        train = False
        scatter = self.config["evaluater"]["visualization"]["scatter"]
        file_name = 'Net_Ensemble_Uncertainty'

        alpha = self.config['evaluater']['loss']['alpha']
        n_std_devs = 0
        if alpha == 0.05:
            n_std_devs = 1.96
        elif alpha == 0.10:
            n_std_devs = 1.645
        elif alpha == 0.01:
            n_std_devs = 2.575

        # model uncertainty method - perc norm (paper uses norm)
        y_pred_U = torch.mean(Outputs[:, :1, :], dim=2) + 1.96 * torch.std(Outputs[:, :1, :], dim=2) / np.sqrt(Outputs.shape[2])
        y_pred_L = torch.mean(Outputs[:, 1:, :], dim=2) - 1.96 * torch.std(Outputs[:, 1:, :], dim=2) / np.sqrt(Outputs.shape[2])
        y_pred_mid = (y_pred_L + y_pred_U) / 2
        plot_uncertainty_QD(y_pred_U, y_pred_L, y_pred_mid, targets[:, :1], save_path, scatter, train, file_name)

    def _info_ensemble(self, Outputs, targets):

        outputs = torch.zeros(Outputs.shape[0], Outputs.shape[1])
        y_pred_U = torch.mean(Outputs[:, :1, :], dim=2) + 1.96 * torch.std(Outputs[:, :1, :], dim=2) / np.sqrt(
            Outputs.shape[2])
        y_pred_L = torch.mean(Outputs[:, 1:, :], dim=2) - 1.96 * torch.std(Outputs[:, 1:, :], dim=2) / np.sqrt(
            Outputs.shape[2])

        outputs[:, :1] = y_pred_U
        outputs[:, 1:] = y_pred_L

        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__ + '_all', met(outputs, targets[:, :1], type="QD"))