import cv2
import numpy as np
from typing import Tuple

class Features:
    @staticmethod
    def load_color_gray(path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray
   
    """
        Shi‑Tomasi corner detector. Returns an Nx2 float array of (x, y) coordinates.
        https://docs.opencv.org/4.12.0/dd/d1a/group__imgproc__feature.html

        Algorithm Steps:
        1. Matrix for each pixel in gray image that describes the structure of the gradients.
        2. For each image region, computed a value that indicates how "corner-like" it is (Shi-Tomasi).
        3. Collect all pixels with a value above a threshold (quality Level).
        4. The best corners are selected, maintaining a minimum distance between them.
        5. The coordinates of these corners are returned as the result.
    """
    @staticmethod
    def detect_features(gray: np.ndarray, max_corners: int = 500,
                        quality: float = 0.01, min_dist: int = 8) -> np.ndarray:

        corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners,
                                          qualityLevel=quality, 
                                          minDistance=min_dist,
                                          useHarrisDetector=False)
        if corners is None:
            return np.empty((0,2), dtype=np.float32)
        return corners.reshape(-1,2).astype(np.float32)

    @staticmethod
    def draw_freatures_crosses(color_img: np.ndarray, points: np.ndarray, radius: int = 3,
                                color: Tuple[int,int,int] = (0,255,255)) -> np.ndarray:
        
        out = color_img.copy()
        if points is None:
            return out
        
        for point in points:
            x, y = int(round(point[0])), int(round(point[1]))
            # Draw cross at each freature location
            cv2.line(out, (x - radius, y), (x + radius, y), color, 1)
            cv2.line(out, (x, y - radius), (x, y + radius), color, 1)
        return out

    @staticmethod
    def resize_keep_aspect(img: np.ndarray, max_width: int, max_heigth: int) -> np.ndarray:
        heigth, width = img.shape[:2]
        # Scale by aspect ratio
        scale = min(max_width / width, max_heigth / heigth)
        if scale >= 1.0:
            return img.copy()
        scaled_width, scaled_heigth = int(width*scale), int(heigth*scale)
        return cv2.resize(img, (scaled_width, scaled_heigth), interpolation=cv2.INTER_AREA)