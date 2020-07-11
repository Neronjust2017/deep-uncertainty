from .layer import *
import torch.nn.functional as F
import torch.nn as nn
import copy
from base import BaseModel

def sample_weights(W_mu, b_mu, W_p, b_p):
    """Quick method for sampling weights and exporting weights"""
    eps_W = W_mu.data.new(W_mu.size()).normal_()
    # sample parameters
    std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)
    W = W_mu + 1 * std_w * eps_W

    if b_mu is not None:
        std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)
        eps_b = b_mu.data.new(b_mu.size()).normal_()
        b = b_mu + 1 * std_b * eps_b
    else:
        b = None

    return W, b

class Bayes_MLP(BaseModel):
    def __init__(self, input_dim, output_dim, n_hid, prior, activation='ReLU', regression_type=None):
        super().__init__()
        # prior_instance = isotropic_gauss_prior(mu=0, sigma=0.1)
        # prior_instance = spike_slab_2GMM(mu1=0, mu2=0, sigma1=0.135, sigma2=0.001, pi=0.5)
        # prior_instance = isotropic_gauss_prior(mu=0, sigma=0.1)
        if prior['type'] == 'Gaussian_prior':
            self.prior_instance = isotropic_gauss_prior(mu=prior['paramters']['mu'], sigma=prior['paramters']['sigma'])
        elif prior['type']  == 'Laplace_prior':
            self.prior_instance = laplace_prior(mu=prior['paramters']['mu'], b=prior['paramters']['sigma'])
        elif prior['type']  == 'GMM_prior':
            self.prior_instance = spike_slab_2GMM(mu1=prior['paramters']['mu1'], mu2=prior['paramters']['mu2'],
                                                  sigma1=prior['paramters']['sigma1'], sigma2=prior['paramters']['sigma2'],
                                                  pi=prior['paramters']['pi'])
        else:
            print('Invalid prior type')
            exit(1)

        self.input_dim = input_dim
        self.output_dim = 2 * output_dim if regression_type == 'hetero' else  output_dim
        self.n_hid = n_hid
        self.regression_type = regression_type

        fc = []
        fc.append(BayesLinear_Normalq(self.input_dim, self.n_hid[0], self.prior_instance))
        for i in range(1, len(self.n_hid)):
            fc.append(BayesLinear_Normalq(self.n_hid[i - 1], self.n_hid[i], self.prior_instance))
        fc.append(BayesLinear_Normalq(self.n_hid[-1], self.output_dim, self.prior_instance))

        self.fc = nn.ModuleList(fc)

        if regression_type == 'homo':
            init_log_noise = 0
            # self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))
            self.log_noise = nn.Parameter(torch.FloatTensor([init_log_noise]))

        # Non linearity
        if activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        elif activation == 'Sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x, sample=False):
        tlqw = 0
        tlpw = 0

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)

        for i in range(len(self.fc)-1):
            # -----------------
            x, lqw, lpw = self.fc[i](x, sample)
            tlqw = tlqw + lqw
            tlpw = tlpw + lpw
            # -----------------
            x = self.act(x)

        # -----------------
        y, lqw, lpw = self.fc[-1](x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw

        return y, tlqw, tlpw

    def sample_predict(self, x, Nsamples):
        """Used for estimating the data's likelihood by approximately marginalising the weights with MC"""
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tlqw_vec = np.zeros(Nsamples)
        tlpw_vec = np.zeros(Nsamples)

        for i in range(Nsamples):
            y, tlqw, tlpw = self.forward(x, sample=True)
            predictions[i] = y
            tlqw_vec[i] = tlqw
            tlpw_vec[i] = tlpw

        return predictions, tlqw_vec, tlpw_vec

class Bayes_MLP_LR(BaseModel):
    def __init__(self, input_dim, output_dim, n_hid, prior_sig, activation='ReLU', regression_type=None):
        super().__init__()
        # prior_instance = isotropic_gauss_prior(mu=0, sigma=0.1)
        # prior_instance = spike_slab_2GMM(mu1=0, mu2=0, sigma1=0.135, sigma2=0.001, pi=0.5)
        # prior_instance = isotropic_gauss_prior(mu=0, sigma=0.1)
        self.prior_sig = prior_sig
        self.input_dim = input_dim
        self.output_dim = 2 * output_dim if regression_type == 'hetero' else  output_dim
        self.n_hid = n_hid
        self.regression_type = regression_type

        fc = []
        fc.append(BayesLinear_local_reparam(self.input_dim, self.n_hid[0], prior_sig))
        for i in range(1, len(self.n_hid)):
            fc.append(BayesLinear_local_reparam(self.n_hid[i - 1], self.n_hid[i], prior_sig))
        fc.append(BayesLinear_local_reparam(self.n_hid[-1], self.output_dim, prior_sig))

        self.fc = nn.ModuleList(fc)

        if regression_type == 'homo':
            init_log_noise = 0
            # self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))
            self.log_noise = nn.Parameter(torch.FloatTensor([init_log_noise]))

        # Non linearity
        if activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        elif activation == 'Sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x, sample=False):
        tlqw = 0
        tlpw = 0

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)

        for i in range(len(self.fc)-1):
            # -----------------
            x, lqw, lpw = self.fc[i](x, sample)
            tlqw = tlqw + lqw
            tlpw = tlpw + lpw
            # -----------------
            x = self.act(x)

        # -----------------
        y, lqw, lpw = self.fc[-1](x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw

        return y, tlqw, tlpw

    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tlqw_vec = np.zeros(Nsamples)
        tlpw_vec = np.zeros(Nsamples)

        for i in range(Nsamples):
            y, tlqw, tlpw = self.forward(x, sample=True)
            predictions[i] = y
            tlqw_vec[i] = tlqw
            tlpw_vec[i] = tlpw

        return predictions, tlqw_vec, tlpw_vec

class Bayes_CNN(BaseModel):
    def __init__(self, input_dim, output_dim, filters, kernels, prior, activation='ReLU', regression_type=None):
        super().__init__()
        
        if prior['type'] == 'Gaussian_prior':
            self.prior_instance = isotropic_gauss_prior(mu=prior['paramters']['mu'], sigma=prior['paramters']['sigma'])
        elif prior['type'] == 'Laplace_prior':
            self.prior_instance = laplace_prior(mu=prior['paramters']['mu'], b=prior['paramters']['sigma'])
        elif prior['type'] == 'GMM_prior':
            self.prior_instance = spike_slab_2GMM(mu1=prior['paramters']['mu1'], mu2=prior['paramters']['mu2'],
                                                  sigma1=prior['paramters']['sigma1'],
                                                  sigma2=prior['paramters']['sigma2'],
                                                  pi=prior['paramters']['pi'])
        else:
            print('Invalid prior type')
            exit(1)
            
        self.input_dim = input_dim
        self.output_dim = 2 * output_dim if regression_type == 'hetero' else output_dim
        self.filters = filters
        self.kernels = kernels
        self.regression_type = regression_type

        num_features = 50*30
        conv = []
        conv.append(BayesConv1D_Normalq(self.input_dim, self.filters[0], self.kernels[0], self.prior_instance))
        for i in range(1, len(self.filters)):
            conv.append(BayesConv1D_Normalq(self.filters[i - 1], self.filters[i], self.kernels[i], self.prior_instance))
        self.conv = nn.ModuleList(conv)
        self.fc = BayesLinear_Normalq(num_features, self.output_dim, self.prior_instance)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)

        if regression_type == 'homo':
            init_log_noise = 0
            self.log_noise = nn.Parameter(torch.FloatTensor([init_log_noise]))
            # self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))

        # Non linearity
        if activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        elif activation == 'Sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x, sample=False):
        # -----------------
        tlqw = 0
        tlpw = 0

        for i in range(len(self.conv)):
            # -----------------
            x, lqw, lpw = self.conv[i](x, sample)
            tlqw = tlqw + lqw
            tlpw = tlpw + lpw
            # -----------------
            x = self.act(x)
            x = self.pool(x)

        x = x.reshape(x.size(0), -1)

        y, lqw, lpw = self.fc(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw

        return y, tlqw, tlpw
