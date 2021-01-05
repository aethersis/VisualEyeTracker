import numpy as np
from torch import nn
import torch.nn.functional as F

from models.detection_model import DetectionModel


class VggModel(DetectionModel):
    default_params = {
        "model_type": "resnet",
        "model_version": 1.0,
        "input_shape": (1, 240, 320),
        "initial_filters": 16,
        "num_fc1": 500,
        "dropout_rate": 0.00,
        "num_classes": 2,
    }

    def __init__(self, params=None):
        if params is None:
            params = VggModel.default_params

        if "version" not in params:
            params["version"] = self._model_version
        if "model_type" not in params:
            params["model_type"] = "resnet"

        super(VggModel, self).__init__(params)
        self.load_structure(params)

    def load_structure(self, hyperparameters):
        C_in, H_in, W_in = hyperparameters["input_shape"]
        init_f = hyperparameters["initial_filters"]
        num_fc1 = hyperparameters["num_fc1"]
        num_classes = hyperparameters["num_classes"]
        self.dropout_rate = hyperparameters["dropout_rate"]

        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(init_f)
        h, w = self._find_conv_2d_output_shape(H_in, W_in, self.conv1)

        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(2 * init_f)
        h, w = self._find_conv_2d_output_shape(h, w, self.conv2)

        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        self.batch_norm3 = nn.BatchNorm2d(4 * init_f)
        h, w = self._find_conv_2d_output_shape(h, w, self.conv3)

        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        self.batch_norm4 = nn.BatchNorm2d(8 * init_f)
        h, w = self._find_conv_2d_output_shape(h, w, self.conv4)

        # compute the flatten size
        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)
        return self

    @staticmethod
    def _find_conv_2d_output_shape(H_in, W_in, conv, pool=2):
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        dilation = conv.dilation
        # Ref: https://pytorch.org/docs/stable/nn.html
        H_out = np.floor((H_in + 2 * padding[0] -
                          dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        W_out = np.floor((W_in + 2 * padding[1] -
                          dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        if pool:
            H_out /= pool
        W_out /= pool
        return int(H_out), int(W_out)

    def forward(self, x):
        x = self.batch_norm1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.batch_norm2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.batch_norm3(F.relu(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.batch_norm4(F.relu(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return x

