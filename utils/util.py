import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self._keys = keys
        self._log = self.create_log()
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        # self._data.total[key] += value * n
        self._data.total[key] += value
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def add(self, key, value):
        self._data.total[key] = value
        self._data.counts[key] = 1
        self._data.average[key] = value

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

    def create_log(self):
        log_dict = {}
        for key in self._keys:
            log_dict[key] = []
        return log_dict

    def update_log(self, key, value):
        self._log[key].append(value)

    def log(self):
        for key in self._keys:
            key_list = self._log[key]
            self._log[key] = np.array(key_list)
        return self._log

def plot_metric(metric_train, metric_val, metric, save_path):

    textsize = 15
    marker = 5
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.plot(metric_train, 'r')
    ax1.plot(metric_val, 'b')
    ax1.set_ylabel(metric)
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    if metric_val is not None:
        lgd = plt.legend(['train', 'val'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    else:
        lgd = plt.legend(['train'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title(metric)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(save_path + '/' + metric + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_prediction(outputs, target, save_path, scatter=True, train=False, file_name='Uncertainty'):
    outputs = outputs.cpu()
    target = target.cpu()

    textsize = 15
    marker = 5

    c = ['#1f77b4', '#ff7f0e']
    ind = np.arange(0, len(target))
    plt.figure()
    fig, ax1 = plt.subplots()
    if scatter:
        plt.scatter(ind, target, color='black', alpha=0.5)
    else:
        ax1.plot(ind, target, 'b')
    ax1.plot(ind, outputs, 'r')

    ax1.set_ylabel('prediction')
    if train:
        plt.xlabel('all points')
    else:
        plt.xlabel('test points')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['prediction'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('Prediction')

    plt.savefig(save_path + '/' + file_name + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_uncertainty_noise(means, noise, total_unc, target, save_path, train=False, scatter=True, file_name='Uncertainty'):
    means = means.cpu()
    noise = noise.cpu()
    target = target.cpu()

    textsize = 15
    marker = 5

    means = means.reshape((means.shape[0],))
    noise = noise.reshape((noise.shape[0],))
    total_unc_1 = total_unc[0].reshape((total_unc[0].shape[0],)).cpu()
    total_unc_2 = total_unc[1].reshape((total_unc[1].shape[0],)).cpu()
    total_unc_3 = total_unc[2].reshape((total_unc[2].shape[0],)).cpu()

    c = ['#1f77b4', '#ff7f0e']
    ind = np.arange(0, len(target))
    plt.figure()
    fig, ax1 = plt.subplots()
    if scatter:
        plt.scatter(ind, target, color='black', alpha=0.5)
    else:
        ax1.plot(ind, target, 'b')
    ax1.plot(ind, means, 'r')
    # plt.fill_between(ind, means - total_unc_3, means + total_unc_3,
    #                  alpha=0.25, label='99.7% Confidence')
    # plt.fill_between(ind, means - total_unc_2, means + total_unc_2,
    #                  alpha=0.25, label='95% Confidence')
    # plt.fill_between(ind, means - total_unc_1, means + total_unc_1,
    #                  alpha=0.25, label='68% Confidence')
    # plt.fill_between(ind, means - noise, means + noise,
    #                  alpha=0.25, label='Noise')
    plt.fill_between(ind, means - total_unc_2, means + total_unc_2,
                     alpha=0.25, label='95% Confidence')
    plt.fill_between(ind, means - noise, means + noise,
                     alpha=0.25, label='Noise')
    ax1.set_ylabel('prediction')
    if train:
        plt.xlabel('all points')
    else:
        plt.xlabel('test points')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['prediction mean'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('Uncertainty')

    plt.savefig(save_path + '/' + file_name + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_uncertainty(means, stds, target, save_path, train=False, scatter=True, file_name='Uncertainty'):
    means = means.cpu()
    stds = stds.cpu()
    target = target.cpu()

    means = means.cpu()
    stds = stds.cpu()
    target = target.cpu()

    textsize = 15
    marker = 5

    means = means.reshape((means.shape[0],))
    stds = stds.reshape((stds.shape[0],))

    c = ['#1f77b4', '#ff7f0e']
    ind = np.arange(0, len(target))
    plt.figure()
    fig, ax1 = plt.subplots()
    if scatter:
        plt.scatter(ind, target, color='black', alpha=0.5)
    else:
        ax1.plot(ind, target, 'b')
    ax1.plot(ind, means, 'r')
    # plt.fill_between(ind, means - 3 * stds, means + 3 * stds,
    #                  alpha=0.25, label='99.7% Confidence')
    # plt.fill_between(ind, means - 2 * stds, means + 2 * stds,
    #                  alpha=0.25, label='95% Confidence')
    # plt.fill_between(ind, means - stds, means + stds,
    #                  alpha=0.25, label='68% Confidence')
    plt.fill_between(ind, means - 2 * stds, means + 2 * stds,
                     alpha=0.25, label='95% Confidence')
    ax1.set_ylabel('prediction')
    if train:
        plt.xlabel('all points')
    else:
        plt.xlabel('test points')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['prediction mean'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('Uncertainty')

    plt.savefig(save_path + '/' + file_name + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_err_bars(y_pred_U, y_pred_L, target, save_path, file_name='Errorbar'):
    y_pred_U = y_pred_U.cpu()
    y_pred_L = y_pred_L.cpu()
    target = target.cpu()

    """
	plot the error bar plot
	"""
    fig, ax = plt.subplots(1)
    # plot as is
    ind = np.arange(0, len(target))
    ax.errorbar(ind, target,
        yerr=[target-y_pred_L,
        y_pred_U-target],
        ecolor='r', alpha=0.6, elinewidth=0.6, capsize=2.0, capthick=1., fmt='none', label='PIs')
    ax.scatter(ind, target, c='b',s=2.0)
    ax.set_xlabel('X')

    ax.set_title('Errorbar', fontsize=10)

    ax.legend(loc='upper left')

    fig.savefig(save_path + '/' + file_name + '.png', bbox_inches='tight')

def plot_uncertainty_QD(y_pred_U, y_pred_L, y_pred_Mid, target, save_path, scatter=True, train=False, file_name='Uncertainty'):
    y_pred_U = y_pred_U.cpu()
    y_pred_L = y_pred_L.cpu()
    y_pred_Mid = y_pred_Mid.cpu()
    target = target.cpu()

    textsize = 15
    marker = 5

    y_pred_U = y_pred_U.reshape((y_pred_U.shape[0],))
    y_pred_L = y_pred_L.reshape((y_pred_L.shape[0],))
    y_pred_Mid = y_pred_Mid.reshape((y_pred_Mid.shape[0],))

    c = ['#1f77b4', '#ff7f0e']
    ind = np.arange(0, len(target))
    plt.figure()
    fig, ax1 = plt.subplots()
    if scatter:
        plt.scatter(ind, target, color='black', alpha=0.5)
    else:
        ax1.plot(ind, target, 'b')
    ax1.plot(ind, y_pred_Mid, 'r')
    plt.fill_between(ind, y_pred_L, y_pred_U,
                     alpha=0.25)

    ax1.set_ylabel('prediction')
    plt.xlabel('test points')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['prediction mean'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('Uncertainty')

    plt.savefig(save_path + '/' + file_name + '.png', bbox_inches='tight')
