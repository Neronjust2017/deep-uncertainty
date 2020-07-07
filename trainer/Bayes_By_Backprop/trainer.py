import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker

class TrainerBayes(BaseTrainer):
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
        self.n_batches = data_loader.n_samples / data_loader.batch_size
        self.n_batches_valid = valid_data_loader.n_samples / valid_data_loader.batch_size

        self.train_metrics = MetricTracker('loss', 'kl_cost', 'pred_cost', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', 'kl_cost', 'pred_cost', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.keys.extend(['kl_cost', 'pred_cost'])
        if self.do_validation:
            keys_val = ['val_' + k for k in self.keys]
            for key in self.keys + keys_val:
                self.log[key] = []


    def _train_epoch(self, epoch, samples=10):
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

            outputs = torch.zeros(data.shape[0], self.model.output_dim, samples).to(self.device)
            if samples == 1:
                out, tlqw, tlpw = self.model(data)
                mlpdw = self._compute_loss(out, target)
                Edkl = (tlqw - tlpw) / self.n_batches
                outputs[:, :, 0] = out

            elif samples > 1:
                mlpdw_cum = 0
                Edkl_cum = 0

                for i in range(samples):
                    out, tlqw, tlpw = self.model(data, sample=True)
                    mlpdw_i = self._compute_loss(out, target)
                    Edkl_i = (tlqw - tlpw) / self.n_batches
                    mlpdw_cum = mlpdw_cum + mlpdw_i
                    Edkl_cum = Edkl_cum + Edkl_i

                    outputs[:, :, i] = out

                mlpdw = mlpdw_cum / samples
                Edkl = Edkl_cum / samples

            mean = torch.mean(outputs, dim=2)
            loss = Edkl + mlpdw
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('kl_cost', Edkl.item())
            self.train_metrics.update('pred_cost', mlpdw.item())
            for met in self.metric_ftns:
                self._compute_metric(self.train_metrics, met, outputs, target)

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

    def _valid_epoch(self, epoch, samples=100):
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

                loss = 0
                outputs = torch.zeros(data.shape[0], self.model.output_dim, samples).to(self.device)

                if samples == 1:
                    out, tlqw, tlpw = self.model(data)
                    mlpdw_i = self._compute_loss(out, target)
                    Edkl_i = (tlqw - tlpw) / self.n_batches_valid
                    mlpdw_cum = mlpdw_cum + mlpdw_i
                    Edkl_cum = Edkl_cum + Edkl_i
                    outputs[:, :, 0] = out

                elif samples > 1:
                    mlpdw_cum = 0
                    Edkl_cum = 0

                    for i in range(samples):

                        out, tlqw, tlpw = self.model(data, sample=True)
                        mlpdw_i = self._compute_loss(out, target)
                        Edkl_i = (tlqw - tlpw) / self.n_batches_valid
                        mlpdw_cum = mlpdw_cum + mlpdw_i
                        Edkl_cum = Edkl_cum + Edkl_i

                        outputs[:, :, i] = out

                    mlpdw = mlpdw_cum / samples
                    Edkl = Edkl_cum / samples
                    loss = Edkl + mlpdw

                mean = torch.mean(outputs, dim=2)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('kl_cost', Edkl.item())
                self.valid_metrics.update('pred_cost', mlpdw.item())

                for met in self.metric_ftns:
                    self._compute_metric(self.valid_metrics, met, outputs, target)
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

    def _compute_loss(self, output, target):
        if self.model.regression_type == 'homo':
            loss = self.criterion(output, target, self.model.log_noise.exp(), self.model.output_dim)
        elif self.model.regression_type == 'hetero':
            loss = self.criterion(output, target, self.model.output_dim/2)
        else:
            loss = self.criterion(output, target)
        return loss

    def _compute_metric(self, metrics, met, output, target, type="BBB"):
        if self.model.regression_type == 'homo':
            metrics.update(met.__name__, met([output, self.model.log_noise.exp()], target,type))
        else:
            metrics.update(met.__name__, met(output, target, type))
