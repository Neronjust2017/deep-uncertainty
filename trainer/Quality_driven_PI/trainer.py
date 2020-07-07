import numpy as np
import math
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker

class Loss(nn.Module):
    def __init__(self, device, type_in="pred_intervals", alpha=0.1,
                 loss_type='qd_soft', censor_R=False,
		         soften=100., lambda_in=10., sigma_in=0.5):
        super().__init__()
        self.alpha = alpha
        self.lambda_in = lambda_in
        self.soften = soften
        self.loss_type = loss_type
        self.type_in = type_in
        self.censor_R = censor_R
        self.sigma_in = sigma_in
        self.device = device

    def forward(self, y_pred, y_true):

        # compute loss

        if self.type_in == "pred_intervals":

            metric = []
            metric_name = []

            # get components
            y_U = y_pred[:, 0]
            y_L = y_pred[:, 1]
            y_T = y_true[:, 0]

            # set inputs and constants
            N_ = y_T.shape[0]
            alpha_ = self.alpha
            lambda_ = self.lambda_in

            # N_ = torch.tensor(y_T.shape[0])
            # alpha_ = torch.tensor(self.alpha)
            # lambda_ = torch.tensor(self.lambda_in)

            # in case want to do point predictions
            y_pred_mean = torch.mean(y_pred, dim=1)
            MPIW = torch.mean(y_U - y_L)

            # soft uses sigmoid
            gamma_U = torch.sigmoid((y_U - y_T) * self.soften)
            gamma_L = torch.sigmoid((y_T - y_L) * self.soften)
            gamma_ = torch.mul(gamma_U, gamma_L)
            ones_ = torch.ones_like(gamma_)

            # hard uses sign step fn
            zeros = torch.zeros_like(y_U)
            gamma_U_hard = torch.max(zeros, torch.sign(y_U - y_T))
            gamma_L_hard = torch.max(zeros, torch.sign(y_T - y_L))
            gamma_hard = torch.mul(gamma_U_hard, gamma_L_hard)

            # lube - lower upper bound estimation
            qd_lhs_hard = torch.div(torch.mean(torch.abs(y_U - y_L) * gamma_hard), torch.mean(gamma_hard) + 0.001)
            qd_lhs_soft = torch.div(torch.mean(torch.abs(y_U - y_L) * gamma_),
                                    torch.mean(gamma_) + 0.001)  # add small noise in case 0
            PICP_soft = torch.mean(gamma_)
            PICP_hard = torch.mean(gamma_hard)

            zero = torch.tensor(0.).to(self.device)
            qd_rhs_soft = lambda_ * math.sqrt(N_) * torch.pow(torch.max(zero, (1. - alpha_) - PICP_soft), 2)
            qd_rhs_hard = lambda_ * math.sqrt(N_) * torch.pow(torch.max(zero, (1. - alpha_) - PICP_hard), 2)

            # old method
            qd_loss_soft = qd_lhs_hard + qd_rhs_soft  # full LUBE w sigmoid for PICP
            qd_loss_hard = qd_lhs_hard + qd_rhs_hard  # full LUBE w step fn for PICP

            umae_loss = 0  # ignore this

            # gaussian log likelihood
            # already defined output nodes
            # y_U = mean, y_L = variance
            y_mean = y_U

            # from deep ensemble paper

            y_var_limited = torch.min(y_L, torch.tensor(10.).to(self.device))  # seem to need to limit otherwise causes nans occasionally
            y_var = torch.max(torch.log(1. + torch.exp(y_var_limited)), torch.tensor(10e-6).to(self.device))

            # to track nans
            self.y_mean = y_mean
            self.y_var = y_var

            gauss_loss = torch.log(y_var) / 2. + torch.div(torch.pow(y_T - y_mean, 2), 2. * y_var)  # this is -ve already
            gauss_loss = torch.mean(gauss_loss)
            # use mean so has some kind of comparability across datasets
            # but actually need to rescale and add constant if want to get actual results

            # set main loss type
            if self.loss_type == 'qd_soft':
                loss = qd_loss_soft
            elif self.loss_type == 'qd_hard':
                loss = qd_loss_hard
            # elif self.loss_type == 'umae_R_cens':
            #     loss = umae_loss_cens_R
            elif self.loss_type == 'gauss_like':
                loss = gauss_loss
            elif self.loss_type == 'picp':  # for loss visualisation
                loss = PICP_hard
            elif self.loss_type == 'mse':
                loss = torch.mean(torch.pow(y_U - y_T, 2))

            # add metrics
            u_capt = torch.mean(gamma_U_hard)  # apparently is quicker if define these
            l_capt = torch.mean(gamma_L_hard)  # here rather than in train loop

            # all_capt = torch.mean(gamma_hard)
            PICP = torch.mean(gamma_hard)

            # metric.append(u_capt)
            # metric_name.append('U capt.')
            # metric.append(l_capt)
            # metric_name.append('L capt.')
            metric.append(PICP)
            metric_name.append('PICP')
            metric.append(MPIW)
            metric_name.append('MPIW')
            # metric.append(tf.reduce_mean(tf.pow(y_T - y_pred_mean,2)))
            # metric_name.append("MSE mid")

        return loss, PICP, MPIW

class TrainerQd(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        if self.do_validation:
            keys_val = ['val_' + k for k in self.keys]
            for key in self.keys + keys_val:
                self.log[key] = []

        cfg_loss = config['trainer']['loss']
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

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss, PICP, MPIW = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            #######
            # y_U = output[:, :1]
            # y_L = output[:, 1:]
            # y_mean = torch.mean(output, dim=1)
            #######
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, type="QD"))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss, PICP, MPIW = self.loss(output, target)

                #######
                # y_U = output[:, :1]
                # y_L = output[:, 1:]
                # y_mean = torch.mean(output, dim=1)
                #######

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, type="QD"))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
