import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import TrainerDeEnsemble, TrainerQdEnsemble
from evaluater import EvaluaterDEEnsemble, EvaluaterQdEnsemble

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# n_ensembles
N = 5

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.valid_data_loader
    test_data_loader = data_loader.test_data_loader

    # build model architecture, then print to console
    models = []
    for i in range(N):
        model = config.init_obj('arch', module_arch)
        models.append(model)
        logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizers = []
    for model in models:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        optimizers.append(optimizer)

    lr_schedulers = []
    for optimizer in optimizers:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        lr_schedulers.append(lr_scheduler)

    if config.config['trainer']['type'] == 'Quality_driven_PI':
        trainer = TrainerQdEnsemble(models, criterion, metrics, optimizers,
                              config=config,
                              data_loader=data_loader,
                              valid_data_loader=valid_data_loader,
                              lr_schedulers=lr_schedulers)
    elif config.config['trainer']['type'] == 'Deep_Ensemble':
        trainer = TrainerDeEnsemble(models, criterion, metrics, optimizers,
                                    config=config,
                                    data_loader=data_loader,
                                    valid_data_loader=valid_data_loader,
                                    lr_schedulers=lr_schedulers)
    else:
        print("type error")
        exit(1)

    trainer.train()

    if config.config['trainer']['type'] == 'Quality_driven_PI':
        evaluater = EvaluaterQdEnsemble(models, criterion, metrics,
                    config=config,
                    test_data_loader=test_data_loader)
    elif config.config['trainer']['type'] == 'Deep_Ensemble':
        evaluater = EvaluaterDEEnsemble(models, criterion, metrics,
                                        config=config,
                                        test_data_loader=test_data_loader)
    else:
        print("type error")
        exit(1)

    evaluater.evaluate()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)