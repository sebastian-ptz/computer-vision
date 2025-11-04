
import cv2
import numpy as np
from typing import Tuple

class Keypoints:
    @staticmethod
    def load_color_gray(path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    @staticmethod
    def detect_features(gray: np.ndarray, max_corners: int = 500,
                       quality: float = 0.01, min_dist: int = 8) -> np.ndarray:
        """
        Shi‑Tomasi Eckdetektor. Gibt Nx2 float-Array (x,y) zurück.
        https://docs.opencv.org/4.12.0/dd/d1a/group__imgproc__feature.html
        """
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners,
                                          qualityLevel=quality, 
                                          minDistance=min_dist,
                                          useHarrisDetector=False)
        if corners is None:
            return np.empty((0,2), dtype=np.float32)
        return corners.reshape(-1,2).astype(np.float32)

    @staticmethod
    def draw_freatures(color_img: np.ndarray, points: np.ndarray, radius: int = 3,
                      color: Tuple[int,int,int] = (0,255,255)) -> np.ndarray:
        out = color_img.copy()
        if points is None:
            return out
        for point in points:
            x, y = int(round(point[0])), int(round(point[1]))
            # Draw cross at each keypoint location
            cv2.line(out, (x - radius, y), (x + radius, y), color, 1, lineType=cv2.LINE_AA)
            cv2.line(out, (x, y - radius), (x, y + radius), color, 1, lineType=cv2.LINE_AA)
        return out

    @staticmethod
    def resize_keep_aspect(img: np.ndarray, max_width: int, max_heigth: int) -> np.ndarray:
        heigth, width = img.shape[:2]
        scale = min(max_width / width, max_heigth / heigth)
        if scale >= 1.0:
            return img.copy()
        scaled_width, scaled_heigth = int(width*scale), int(heigth*scale)
        return cv2.resize(img, (scaled_width, scaled_heigth), interpolation=cv2.INTER_AREA)