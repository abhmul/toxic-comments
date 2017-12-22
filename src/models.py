import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pyjet.models import SLModel
import pyjet.backend as J

from modules import kmax_pooling, pad_torch_embedded_sequences, unpad_torch_embedded_sequences, pad_numpy_to_length


class CNNEmb(SLModel):
    __name__ = "cnn-emb"

    def __init__(self, num_features, kernel_size=3, pool_kernel_size=2, pool_stride=1, n1_filters=512, n2_filters=256,
                 k=5, conv_dropout=0.2, fc_dropout=0.5, batchnorm=True):
        super(CNNEmb, self).__init__()

        self.num_features = num_features
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.n1_filters = n1_filters
        self.n2_filters = n2_filters
        self.k = k
        self.conv_dropout = conv_dropout
        self.fc_dropout = fc_dropout
        self.batchnorm = batchnorm
        self.padding = (kernel_size - 1) // 2

        # Block 1
        self.conv1 = nn.Conv1d(num_features, n1_filters, kernel_size, padding=self.padding)
        self.m1 = nn.MaxPool1d(pool_kernel_size, stride=pool_stride)
        self.bn1 = nn.BatchNorm1d(n1_filters) if batchnorm else None
        self.dropout1 = nn.Dropout(conv_dropout)
        # Block 2
        self.conv2 = nn.Conv1d(n1_filters, n2_filters, kernel_size, padding=self.padding)
        self.m2 = nn.MaxPool1d(pool_kernel_size, stride=pool_stride)
        self.bn2 = nn.BatchNorm1d(n2_filters) if batchnorm else None
        self.dropout2 = nn.Dropout(conv_dropout)
        # Evaluation Layer
        self.fc1 = nn.Linear(k * n2_filters, 6)
        self.dropout3 = nn.Dropout(fc_dropout)

    def cast_input_to_torch(self, x, volatile=False):
        # If a sample is too short extend it
        x = [pad_numpy_to_length(sample, length=(self.k + 2)) for sample in x]
        # Transpose to get features x length
        return [Variable(J.Tensor(sample).transpose(0, 1), volatile=volatile) for sample in x]

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(torch.from_numpy(y).cuda().float(), volatile=volatile)

    def forward(self, x):
        x = [self.m1(F.relu(self.conv1(sample.unsqueeze(0)))).squeeze(0) for sample in x]  # B x 512 x Li-1
        # Do the batchnorm
        if self.batchnorm:
            x, lens = pad_torch_embedded_sequences(x, length_last=True)
            x = self.bn1(x)
            x = unpad_torch_embedded_sequences(x, lens, length_last=True)

        x = [self.dropout1(sample.unsqueeze(0)).squeeze(0) for sample in x]

        x = [self.m2(F.relu(self.conv2(sample.unsqueeze(0)))).squeeze(0) for sample in x]  # B x 256 x Li-2
        # Do the batchnorm
        if self.batchnorm:
            x, lens = pad_torch_embedded_sequences(x, length_last=True)
            x = self.bn2(x)
            x = unpad_torch_embedded_sequences(x, lens, length_last=True)

        x = [self.dropout2(sample.unsqueeze(0)).squeeze(0) for sample in x]

        x = torch.stack([kmax_pooling(sample.unsqueeze(0), 2, self.k).squeeze(0) for sample in x])  # B x 256 x k
        x = J.flatten(x)  # B x 256k
        x = self.dropout3(x)
        self.loss_in = self.fc1(x)  # B x 6
        return F.sigmoid(self.loss_in)
