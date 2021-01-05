import random
from moviepy.editor import *
from pathlib import Path
from typing import Optional, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class Augmentations:
    default_augmentations_config = {
        "target_size": (240, 320),
        "p_hflip": 0.0,
        "p_vflip": 0.0,
        "p_shift": 0.0,
        "max_translate": (0.0, 0.0),
        "p_brightness": 0.0,
        "brightness_factor": 0.4,
        "p_contrast": 0.0,
        "contrast_factor": 0.4,
        "p_gamma": 0.0,
        "gamma": 0.2,
    }

    @staticmethod
    def scale_label(image, label, target_size):
        w_orig, h_orig = image.shape[1], image.shape[2]
        w_target, h_target = target_size
        cx, cy = label
        image_new = TF.resize(image, target_size)
        label_new = cx / w_orig * w_target, cy / h_orig * h_target
        return image_new, label_new

    @staticmethod
    def random_hflip(image, label):
        w, h = image.shape[2], image.shape[1]
        x, y = label
        image = TF.hflip(image)
        label = w - x, y
        return image, label

    @staticmethod
    def random_vflip(image, label):
        w, h = image.shape[2], image.shape[1]
        x, y = label
        image = TF.vflip(image)
        label = x, h - y
        return image, label

    @staticmethod
    def random_shift(image, label, max_translate=(0.2, 0.2)):
        w, h = image.shape[2], image.shape[1]

        max_t_w, max_t_h = max_translate
        cx, cy = label
        trans_coef = np.random.rand() * 2 - 1
        w_t = int(trans_coef * max_t_w * w)
        h_t = int(trans_coef * max_t_h * h)
        image = TF.affine(image, translate=[w_t, h_t], shear=[0.0, 0.0], angle=0, scale=1)
        label = cx + w_t, cy + h_t
        return image, label

    @staticmethod
    def transform(image, label, augmentations_config=None):
        if augmentations_config is None:
            augmentations_config = Augmentations.default_augmentations_config
        image, label = Augmentations.scale_label(image, label, augmentations_config["target_size"])

        if random.random() < augmentations_config["p_hflip"]:
            image, label = Augmentations.random_hflip(image, label)
        if random.random() < augmentations_config["p_vflip"]:
            image, label = Augmentations.random_vflip(image, label)
        if random.random() < augmentations_config["p_shift"]:
            max_translate_x = augmentations_config["max_translate"][0]
            max_translate_y = augmentations_config["max_translate"][1]
            translate_x = random.uniform(-max_translate_x, max_translate_x)
            translate_y = random.uniform(-max_translate_y, max_translate_y)
            image, label = Augmentations.random_shift(image, label, [translate_x, translate_y])

        if random.random() < augmentations_config["p_brightness"]:
            brightness_factor = 1 + (np.random.rand() * 2 - 1) * augmentations_config["brightness_factor"]
            image = TF.adjust_brightness(image, brightness_factor)
        if random.random() < augmentations_config["p_contrast"]:
            contrast_factor = 1 + (np.random.rand() * 2 - 1) * augmentations_config["contrast_factor"]
            image = TF.adjust_contrast(image, contrast_factor)
        if random.random() < augmentations_config["p_gamma"]:
            gamma = 1 + (np.random.rand() * 2 - 1) * augmentations_config["gamma"]
            image = TF.adjust_gamma(image, gamma)

        image = TF.rgb_to_grayscale(image)
        label = np.array(label)

        # Label is converted from pixel space coordinates to a normalized system so
        # 0,0 is at the center and 1 or -1 towards the corners.
        label /= [augmentations_config["target_size"][0], augmentations_config["target_size"][1]]
        label -= 0.5
        label *= 2
        return image, label


class PupilsDatasetLoader(Dataset):
    def __init__(self, dataset_folder_path: Path, augmentations_config: Optional[Dict] = None, device=torch.device("cuda")):
        """
        Loads the dataset prepared for this project (see readme for more details on the data format)
        :param dataset_folder_path: folder with the dataset (subfolders are supported too!)
        :param augmentations_config: a dictionary with augmentations (see default training config for details)
        :param device: torch device. If 'cuda' is used, augmentations will be ran on the GPU which can speed up the training
        """
        # To achieve repeatability of training runs we need to convert the files to a sorted list. Otherwise the
        # iteration order is random and can't be fixed with a seed because loading order depends on the OS.
        if augmentations_config is None:
            self._augmentations_config = Augmentations.default_augmentations_config
        else:
            self._augmentations_config = augmentations_config

        self._device = device
        self._image_paths = sorted(list(dataset_folder_path.rglob("*.jpg")))

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, np.ndarray]:
        image_path = random.choice(self._image_paths)
        labels_path = Path(str(image_path.parent / image_path.stem) + ".txt")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = image.transpose((2, 0, 1))  # width, height, channels to channels, height, width used by PyTorch
        image = torch.tensor(image).to(self._device)

        with open(labels_path, "r") as fp:
            label = fp.readline().strip().split(',')
        label = np.array((float(label[0]), float(label[1])))

        frame, label = Augmentations.transform(image, (float(label[0]), float(label[1])), self._augmentations_config)
        return frame.float(), label


class VideoDataset(Dataset):
    """
    Loads only frames (without labels) from a video file. The frames are loaded as NCHW grayscale tensors
    """
    def __init__(self, video_file_path: Path):
        # This library handles video decoding reasonably fast and has better support for many formats than OpenCV
        self._video = VideoFileClip(str(video_file_path))

    @property
    def fps(self) -> float:
        return self._video.fps

    def __len__(self):
        return int(self._video.duration * self._video.fps - 1)

    def __getitem__(self, idx) -> torch.Tensor:
        frame = self._video.get_frame(idx/self.fps)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[np.newaxis, ...].astype(np.float32)
        frame = torch.tensor(frame)
        return frame