import json
import linecache
import os
import random
import shutil
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

import numpy as np
import cv2

"""
This is a collection of scripts that were used to convert data from two datasets into a common format used for
training of the models included in this project.

The common format is 320x240 jpg files with eyes and a text file with the same name that stores two floating point 
numbers describing the x and y coordinate of the pupil in the pixel space (e.g. 122.30,189.22)

Links to the datasets:
  gaze-in-wild: http://www.cis.rit.edu/~rsk3900/gaze-in-wild/
  UnityEyes: https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/
"""


def _compute_iris_center_from_labels(json_path: Path, image_height: int):
    with open(json_path, "r") as fp:
        half_height = image_height/2
        json_data = json.load(fp)
        iris_coordinates = []
        iris_data = json_data["iris_2d"]
        for row in iris_data:
            points = row[1:-1].split(',')
            # y axis is inverted so we need to invert it back
            iris_coordinates.append((float(points[0]), (float(points[1]) - half_height) * -1 + half_height))
        iris_coordinates = np.array(iris_coordinates)
        center = np.sum(iris_coordinates, axis=0) / iris_coordinates.shape[0]

        return center


def convert_unity_eyes(input_path: Path, output_path: Path, output_resolution: Tuple[int, int], initial_index=0):
    input_files = sorted(list(input_path.glob("*.jpg")))
    if not output_path.exists():
        os.makedirs(output_path)

    idx = 0
    for input_file in tqdm(input_files):
        image = cv2.imread(str(input_file), cv2.IMREAD_GRAYSCALE)
        input_resolution = np.array((image.shape[1], image.shape[0]))
        image = cv2.resize(image, output_resolution)

        json_path = Path(str(input_file.parent / input_file.stem) + ".json")
        iris_center = _compute_iris_center_from_labels(json_path, image_height=input_resolution[1])
        scale_factor = np.array(output_resolution) / input_resolution
        iris_center *= scale_factor

        output_file = str(output_path / "{:06d}".format(idx + initial_index))
        cv2.imwrite(output_file + ".jpg", image)
        with open(output_file + ".txt", "w") as fp:
            fp.write(str(iris_center[0]) + "," + str(iris_center[1]))

        idx += 1


def convert_gaze_in_wild(input_path: Path, output_path: Path, output_resolution: Tuple[int, int], frame_count: int = 20000, initial_index: int = 0):
    if not output_path.exists():
        os.makedirs(output_path)

    input_files = sorted(list(input_path.rglob("*.avi")))
    video_loaders = [cv2.VideoCapture(str(f)) for f in input_files]
    idx = 0
    for i in tqdm(range(frame_count)):
        loader_id = random.randint(0, len(video_loaders) - 1)
        labels_path = Path(str(input_files[loader_id].parent / input_files[loader_id].stem) + ".txt")
        amount_of_frames = int(video_loaders[loader_id].get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame_id = random.randint(0, amount_of_frames - 1)
        pupil_center = linecache.getline(str(labels_path), random_frame_id + 1).strip().split(' ')
        pupil_center = np.array((float(pupil_center[0]), float(pupil_center[1])))

        video_loaders[loader_id].set(cv2.CAP_PROP_POS_FRAMES, random_frame_id)
        ok, frame = video_loaders[loader_id].read()
        if not ok:
            raise IOError(f"Something went wrong reading frame {random_frame_id} from file {input_files[loader_id]}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input_resolution = np.array((frame.shape[1], frame.shape[0]))
        scale_factor = np.array(output_resolution) / input_resolution
        pupil_center *= scale_factor
        frame = cv2.resize(frame, output_resolution)

        output_file = str(output_path / "{:06d}".format(idx + initial_index))
        idx += 1
        cv2.imwrite(output_file + ".jpg", frame)
        with open(output_file + ".txt", "w") as fp:
            fp.write(str(pupil_center[0]) + "," + str(pupil_center[1]))
