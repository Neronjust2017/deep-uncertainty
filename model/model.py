import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

def MC_dropout(act_vec, p=0.5, mask=True):
    # return F.dropout(act_vec, p=p, training=mask, inplace=True)
    return F.dropout(act_vec, p=p, training=mask, inplace=False)

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MLP(BaseModel):
    def __init__(self, input_dim, output_dim, n_hid, activation='ReLU', regression_type=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * output_dim if regression_type == 'hetero' else  output_dim
        self.n_hid = n_hid
        self.regression_type = regression_type

        fc = []
        fc.append(nn.Linear(self.input_dim, self.n_hid[0]))
        for i in range(1, len(self.n_hid)):
            fc.append(nn.Linear(self.n_hid[i-1], self.n_hid[i]))
        fc.append(nn.Linear(self.n_hid[-1], self.output_dim))

        self.fc = nn.ModuleList(fc)

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

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        for i in range(len(self.fc) - 1):
            x = self.act(self.fc[i](x))
        y = self.fc[-1](x)
        return y

class MLP_dropout(BaseModel):
    def __init__(self, input_dim, output_dim, n_hid, pdrop, activation='ReLU', regression_type=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * output_dim if regression_type == 'hetero' else output_dim
        self.n_hid = n_hid
        self.regression_type = regression_type
        self.pdrop = pdrop

        fc = []
        fc.append(nn.Linear(self.input_dim, self.n_hid[0]))
        for i in range(1, len(self.n_hid)):
            fc.append(nn.Linear(self.n_hid[i-1], self.n_hid[i]))
        fc.append(nn.Linear(self.n_hid[-1], self.output_dim))

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

    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        # -----------------
        for i in range(len(self.fc) - 1):
            x = self.act(self.fc[i](x))
            x = MC_dropout(x, p=self.pdrop, mask=mask)
        y = self.fc[-1](x)

        return y

class CNN(BaseModel):
    def __init__(self, input_dim, output_dim, filters, kernels, activation='ReLU', regression_type=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * output_dim if regression_type == 'hetero' else output_dim
        self.filters = filters
        self.kernels = kernels
        self.regression_type = regression_type

        num_features = 50*30
        conv = []
        conv.append(nn.Conv1d(self.input_dim, self.filters[0], self.kernels[0]))
        for i in range(1, len(self.filters)):
            conv.append(nn.Conv1d(self.filters[i - 1], self.filters[i], self.kernels[i]))
        self.conv = nn.ModuleList(conv)
        self.fc = nn.Linear(num_features, self.output_dim)

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

    def forward(self, x):
        # -----------------
        for i in range(len(self.conv)):
            x = self.pool(self.act(self.conv[i](x)))
        x = x.reshape(x.size(0), -1)
        y = self.fc(x)
        return y

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN_dropout(BaseModel):
    def __init__(self, input_dim, output_dim, filters, kernels, pdrop, activation='ReLU', regression_type=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * output_dim if regression_type == 'hetero' else output_dim
        self.filters = filters
        self.kernels = kernels
        self.pdrop = pdrop
        self.regression_type = regression_type

        num_features = 50*30
        conv = []
        conv.append(nn.Conv1d(self.input_dim, self.filters[0], self.kernels[0]))
        for i in range(1, len(self.filters)):
            conv.append(nn.Conv1d(self.filters[i - 1], self.filters[i], self.kernels[i]))
        self.conv = nn.ModuleList(conv)
        self.fc = nn.Linear(num_features, self.output_dim)

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

    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations
        # -----------------
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        for i in range(len(self.conv)):
            x = self.pool(self.act(self.conv[i](x)))
            x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = x.reshape(x.size(0), -1)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        y = self.fc(x)
        return y

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = CNN(input_dim=1, output_dim=1, filters=[50,50], kernels=[3,3], activation='ReLU')
x = torch.randn(48, 1, 36)
model(x)