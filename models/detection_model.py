import json
from abc import abstractmethod
from pathlib import Path
from typing import Dict

import torch
from torch import nn


class DetectionModel(nn.Module):
    """
    Base class describing any single object detection model
    """
    def __init__(self, params: {}):
        self._params = params
        assert params is not None
        super(DetectionModel, self).__init__()

    def save(self, folder: Path, name):
        with open(folder / (name + ".json"), "w") as f:
            json.dump(self._params, f, indent=2)
        torch.save(self.state_dict(), folder / (name + ".pt"))

    def load(self, folder: Path, name):
        with open(folder / (name + ".json"), "r") as f:
            params = json.load(f)

        model = self.load_structure(params)
        model.load_state_dict(torch.load(folder / (name + ".pt")))
        return model

    @abstractmethod
    def load_structure(self, config_dictionary: Dict):
        """
        Loads model structure from a json description.
        The structure json file describes non-trainable hyperparameters such as number of layers, filters etc.
        :param config_dictionary: dictionary with all the parameters necessary to load the model
        :return: None
        """
        return NotImplemented

