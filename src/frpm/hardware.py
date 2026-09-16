"""Best-effort hardware/acceleration detection.

The pipeline is designed to run fully on CPU: the prebuilt wheels used here
(``opencv-contrib-python``, ``mediapipe``) are CPU-only on PyPI. This module
reports what, if anything, looks accelerated so it can be logged - it never
changes correctness if no acceleration is found.
"""
from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass

logger = logging.getLogger("frpm.hardware")


@dataclass
class HardwareInfo:
    platform_str: str
    cpu_count: int
    opencv_cuda_devices: int
    onnxruntime_gpu: bool

    def summary(self) -> str:
        if self.opencv_cuda_devices > 0 or self.onnxruntime_gpu:
            bits = []
            if self.opencv_cuda_devices > 0:
                bits.append(f"OpenCV CUDA devices={self.opencv_cuda_devices}")
            if self.onnxruntime_gpu:
                bits.append("onnxruntime CUDAExecutionProvider")
            return f"GPU acceleration detected ({', '.join(bits)})."
        return f"No GPU acceleration detected - running on CPU ({self.cpu_count} cores). This is fully supported."


def detect_hardware() -> HardwareInfo:
    opencv_cuda_devices = 0
    try:
        import cv2

        opencv_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        opencv_cuda_devices = 0

    onnxruntime_gpu = False
    try:
        import onnxruntime as ort

        onnxruntime_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        onnxruntime_gpu = False

    info = HardwareInfo(
        platform_str=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        opencv_cuda_devices=opencv_cuda_devices,
        onnxruntime_gpu=onnxruntime_gpu,
    )
    logger.info(info.summary())
    return info
