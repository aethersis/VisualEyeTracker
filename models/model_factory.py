import json
from pathlib import Path

from models.detection_model import DetectionModel
from models.vgg_model import VggModel
from models.resnet_model import ResNetModel


def load_model_from_directory(model_directory: Path) -> DetectionModel:
    """
    Loads a model of given type with trained weights given a directory where the state dictionary and json config
    describing the model structure are stored.
    The type of the model (e.g. VGG, ResNet) is automatically recognized from the config.
    :param model_directory: path to the directory containing the model. The model config and state dict files must be
    named the same as this directory.
    :return: Model of the type stored in the config with preloaded weights. Can be used for inference or for continuing
    an interrupted training.
    """

    if not model_directory.is_dir() or not model_directory.exists():
        raise FileNotFoundError("Model directory must exist")

    model_name = str(model_directory.stem)

    model_config_path = model_directory / (model_name + ".json")
    if not model_config_path.is_file() or not model_config_path.exists():
        raise FileNotFoundError("Model config file must be named the same as its parent directory and must exist")

    with open(model_config_path, "r") as fp:
        model_config = json.load(fp)

    if model_config is None:
        raise IOError("Failed to load model config")
    if "model_type" not in model_config:
        raise IOError("Failed to load model_type from the config")

    model_type = model_config["model_type"]

    if model_type == "resnet":
        model = ResNetModel()
    elif model_type == "vgg":
        model = VggModel()
    else:
        raise ValueError(f"Invalid model type {model_type}")

    model.load(model_directory, model_name)
    return model
