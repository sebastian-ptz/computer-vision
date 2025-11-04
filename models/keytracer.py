import cv2
import numpy as np
from typing import Tuple

class Keytracer:
    @staticmethod
    def trace_edges(gray: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        # Canny-Algorithm -> binary corner image
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges

    @staticmethod
    def overlay_edges(color_img: np.ndarray, edges: np.ndarray, 
                               color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        overlay = color_img.copy()
        overlay[edges != 0] = color
        return overlay
