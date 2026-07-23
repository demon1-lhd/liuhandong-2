# ============================================================
# recognizers/selfie_segmentation_recognizer.py
# ============================================================
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (RunningMode, ImageSegmenter, ImageSegmenterOptions)
from recognizers.base import BaseRecognizer

class SelfieSegmentationRecognizer(BaseRecognizer):
    def __init__(self):
        super().__init__()
        self._bg = None

    def _model_file(self) -> str:
        return "selfie_segmenter.tflite"

    def _create_task(self):
        opts = ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.IMAGE, output_category_mask=True)
        return ImageSegmenter.create_from_options(opts)

    def process(self, frame, mp_image):
        result = self._task.segment(mp_image)
        mask = result.category_mask.numpy_view()
        if mask is None or mask.size == 0:
            return frame
        cond = np.stack((mask > 127,) * 3, axis=-1)
        if self._bg is None:
            self._bg = np.zeros(frame.shape, dtype=np.uint8)
            self._bg[:] = (192, 192, 192)
        return np.where(cond, frame, self._bg)