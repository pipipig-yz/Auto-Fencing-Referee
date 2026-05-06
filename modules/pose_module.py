"""YOLOv8-Pose inference wrapper.

Responsibilities:
- Load yolov8x-pose.pt once and keep it on GPU.
- Run inference on a single BGR frame.
- Return per-detection keypoints (17×3) and bounding boxes.
- Flag low-confidence keypoints.
"""

import numpy as np
import torch
from ultralytics import YOLO

import config
from utils.angle_utils import compute_angles


class PoseDetection:
    """One person detected in a single frame."""

    def __init__(self, box_xyxy: np.ndarray, keypoints: np.ndarray):
        """
        Args:
            box_xyxy:  (4,) float array — x1, y1, x2, y2
            keypoints: (17, 3) float array — x, y, confidence per keypoint
        """
        self.box_xyxy  = box_xyxy
        self.keypoints = keypoints  # (17, 3)

        # Centre X of bounding box — used to assign left / right athlete
        self.center_x = float((box_xyxy[0] + box_xyxy[2]) / 2.0)

        # Boolean mask: True where keypoint confidence is unreliable
        self.low_conf_mask = keypoints[:, 2] < config.KEYPOINT_CONF_THRESHOLD

    def compute_angles(self, side: str) -> dict:
        """Return joint angle dict for this detection given screen side."""
        return compute_angles(self.keypoints, side)


class PoseModule:
    """Manages YOLOv8-Pose model loading and per-frame inference."""

    def __init__(self):
        self._model = YOLO(config.MODEL_PATH)
        # Warm up on a dummy frame so the first real frame isn't slow
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)

    def infer(self, frame_bgr: np.ndarray) -> list[PoseDetection]:
        """Run pose estimation on one frame.

        Returns a list of PoseDetection objects, one per detected person.
        Sorted left-to-right by bounding box centre X.
        """
        results = self._model(frame_bgr, verbose=False)
        detections = []

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            kps  = result.keypoints.data   # (N, 17, 3) tensor
            boxes = result.boxes.xyxy       # (N, 4) tensor

            for i in range(len(boxes)):
                box_np = boxes[i].cpu().numpy().astype(float)
                kp_np  = kps[i].cpu().numpy().astype(float)
                detections.append(PoseDetection(box_np, kp_np))

        detections.sort(key=lambda d: d.center_x)
        return detections
