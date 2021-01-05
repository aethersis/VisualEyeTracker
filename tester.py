import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import VideoDataset
from models.model_factory import load_model_from_directory


def preview_images(loader: DataLoader):
    """
    Keeps displaying images generated by the data loader specified until 'q' is pressed
    :param loader: an arbitrary pytorch data loader
    :return: None
    """
    for xb, yb in loader:
        xb = xb.numpy()
        yb = yb.numpy()
        for i in range(0, xb.shape[0]):
            yb[i] += 1
            yb[i] /= 2
            yb[i] *= [240, 320]
            cv2.circle(xb[i][0], tuple(yb[i].astype(np.int)), 10, 255)
            cv2.imshow("image", xb[i][0] / 255)
            key = cv2.waitKey(0)
            if key == 'q':
                return


def run_model_on_video(model_path: Path, video_path: Path):
    model = load_model_from_directory(model_path)

    test_dataset = VideoDataset(video_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter("out.mp4", fourcc, test_dataset.fps, (320, 240))

    with torch.no_grad():
        for xb in test_loader:
            coordinates = model(xb)
            xb = xb.numpy().astype(np.uint8)
            coordinates = coordinates.numpy()
            for i in range(0, xb.shape[0]):
                coordinates[i] += 1
                coordinates[i] /= 2
                coordinates[i] *= [240, 320]
                cv2.circle(xb[i][0], tuple(coordinates[i].astype(np.int)), 10, 255)
                cv2.imshow("image", xb[i][0].astype(np.uint8))
                cv2.waitKey(33)
                writer.write(cv2.cvtColor(xb[i][0], cv2.COLOR_GRAY2BGR))
        writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Loads model from file and runs it on a video. The video must be 320x240 px MP4."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model file (model weights and structure name must be the same as the name of the parent "
             "directory name)"
    )
    parser.add_argument("video_path", type=str, help="Path to the mp4 video")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    video_path = Path(args.video_path)
    run_model_on_video(model_path, video_path)