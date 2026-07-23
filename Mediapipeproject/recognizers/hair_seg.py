# ============================================================
# recognizers/hair_segmentation_recognizer.py
# ============================================================
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (RunningMode, ImageSegmenter, ImageSegmenterOptions)
from recognizers.base import BaseRecognizer

class HairSegmentationRecognizer(BaseRecognizer):
    def _model_file(self) -> str:
        return "hair_segmenter.tflite"

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
        hair = (mask == 1)
        if hair.any():
            overlay = frame.copy()
            overlay[hair] = (0, 0, 255)
            frame = cv.addWeighted(frame, 0.6, overlay, 0.4, 0)
        return frame