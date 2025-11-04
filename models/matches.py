import cv2
import numpy as np
from typing import List, Tuple

class Matches:

    @staticmethod
    def find_machtes(
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        keypoints_left: np.ndarray,
        keypoints_right: np.ndarray,
        window_size: int = 11,
        search_range: int = 50
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    
        """
        For each keypoint in left, find the best matching keypoint in right:
        SSD in a horizontal window -> returns list of ((x1, y1), (x2, y2)) pairs.
        """

        half_window = window_size // 2
        height, width = left_gray.shape
        matches = []
        for pt1 in keypoints_left:
            x1, y1 = int(round(pt1[0])), int(round(pt1[1]))

            if (x1 - half_window < 0 or x1 + half_window >= width 
                or y1 - half_window < 0 or y1 + half_window >= height):
                continue

            patch_left = left_gray[y1 - half_window:y1 + half_window + 1, 
                                   x1 - half_window:x1 + half_window + 1]
            
            best_score = float('inf')
            best_pt2 = None
            for pt2 in keypoints_right:
                x2, y2 = int(round(pt2[0])), int(round(pt2[1]))

                if y2 != y1:
                    continue
                if x2 - half_window < 0 or x2 + half_window >= width:
                    continue
                patch_right = right_gray[y2 - half_window:y2 + half_window + 1,
                                         x2 - half_window:x2 + half_window + 1]
                
                ssd = np.sum((patch_left.astype(np.float32) - patch_right.astype(np.float32)) ** 2)
                if ssd < best_score:
                    best_score = ssd
                    best_pt2 = (x2, y2)

            if best_pt2 is not None:
                matches.append(((x1, y1), best_pt2))

        return matches

    @staticmethod
    def remove_extrema(
        matches: list[tuple[tuple[int, int], tuple[int, int]]],
    ) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[tuple[int, int], tuple[int, int]]]]:
        
        # ToDo: Filtering Extrema
        
        # Dummy: Remove last 10% as fake extrema
        n = len(matches)
        n_extrema = max(1, n // 10)
        
        filtered = matches[:-n_extrema]
        extrema = matches[-n_extrema:]
        return filtered, extrema
    
    @staticmethod
    def add_extrema(
        matches: list[tuple[tuple[int, int], tuple[int, int]]],
        extrema: list[tuple[tuple[int, int], tuple[int, int]]],
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        # Add extrema back to matches
        return matches + extrema

    @staticmethod
    def draw_matches(
        img_left: np.ndarray,
        img_right: np.ndarray,
        matches: list[tuple[tuple[int, int], tuple[int, int]]],
        line_color: tuple[int, int, int] = (0, 200, 200)
    ) -> np.ndarray: 
        # Visualize lines of tuples for matching features.
        height = max(img_left.shape[0], img_right.shape[0])
        left_width, right_width = img_left.shape[1], img_right.shape[1]
        vis = np.zeros((height, left_width + right_width, 3), dtype=np.uint8)
        vis[:img_left.shape[0], :left_width] = img_left
        vis[:img_right.shape[0], left_width:left_width + right_width] = img_right

        for (x1, y1), (x2, y2) in matches:
            pt1 = (x1, y1)
            pt2 = (x2 + left_width, y2)
            cv2.line(vis, pt1, pt2, line_color, 1, lineType=cv2.LINE_AA)
        return vis
