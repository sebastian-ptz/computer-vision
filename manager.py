from models.keytracer import Keytracer
from models.features import Features
from models.matches import Matches
import cv2

class AppManager:
    def __init__(self, gui):
        self.gui = gui
        self.matches = []
        self.extrema = []
        self.extrema_removed = False
        self.feature_points_left = None
        self.feature_points_right = None
        self._last_max_corners = None

    def _recompute_features(self):
        max_corners = self.gui._get_max_corners()
        left_gray = cv2.cvtColor(self.gui.left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.gui.right_img, cv2.COLOR_BGR2GRAY)
        self.feature_points_left = Features.detect_features(left_gray, max_corners=max_corners)
        self.feature_points_right = Features.detect_features(right_gray, max_corners=max_corners)
        self._last_max_corners = max_corners

    def handle_control_change(self):
        if self.gui.left_img is None or self.gui.right_img is None:
            return

        # Feature-Points ggf. neu berechnen, wenn maxCorners geändert wurde
        max_corners = self.gui._get_max_corners()
        max_corners_changed = self._last_max_corners != max_corners
        if max_corners_changed or self.feature_points_left is None or self.feature_points_right is None:
            self._recompute_features()
            self.matches = None

        # Check if max_matches changed and handle all reset logic here
        max_matches = self.gui._get_max_matches() if hasattr(self.gui, '_get_max_matches') else None
        if not hasattr(self, '_last_max_matches'):
            self._last_max_matches = max_matches
        max_matches_changed = (max_matches != self._last_max_matches)
        if max_matches_changed:
            self._last_max_matches = max_matches
            # Reset extrema and checkbox
            if hasattr(self.gui, 'checkbox_remove_extrema'):
                self.gui.checkbox_remove_extrema.setChecked(False)
            self.extrema_removed = False
            self.extrema = []
            self.matches = None

        # ...existing code for display_mode and drawing...

        if self.gui.radio_trace.isChecked():
            display_mode = "edges"
        elif self.gui.radio_features.isChecked():
            display_mode = "features"
        elif self.gui.radio_matches.isChecked():
            display_mode = "matches"
        else:
            display_mode = "original"

        self.gui.display_mode = display_mode

        match display_mode:
            case "edges":
                left_gray = cv2.cvtColor(self.gui.left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(self.gui.right_img, cv2.COLOR_BGR2GRAY)
                edges_left = Keytracer.trace_edges(left_gray)
                edges_right = Keytracer.trace_edges(right_gray)
                overlay_left = Keytracer.overlay_edges(self.gui.left_img, edges_left)
                overlay_right = Keytracer.overlay_edges(self.gui.right_img, edges_right)
                self.gui._set_pixmap(self.gui.left_widget, overlay_left)
                self.gui._set_pixmap(self.gui.right_widget, overlay_right)

            case "features":
                drawnpoint_left = Features.draw_freatures_crosses(self.gui.left_img, self.feature_points_left)
                drawnpoints_right = Features.draw_freatures_crosses(self.gui.right_img, self.feature_points_right)
                self.matches = None
                self.extrema = []
                self.extrema_removed = False
                self.gui._set_pixmap(self.gui.left_widget, drawnpoint_left)
                self.gui._set_pixmap(self.gui.right_widget, drawnpoints_right)

            case "matches":
                left_gray = cv2.cvtColor(self.gui.left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(self.gui.right_img, cv2.COLOR_BGR2GRAY)
                left_with_kp = Features.draw_freatures_crosses(self.gui.left_img, self.feature_points_left)
                right_with_kp = Features.draw_freatures_crosses(self.gui.right_img, self.feature_points_right)

                # Recalculate matches if needed
                if self.matches is None or len(self.matches) == 0 or max_corners_changed or max_matches_changed:
                    self.matches = Matches.find_machtes(left_gray, right_gray, self.feature_points_left, self.feature_points_right)
                    self.extrema = []
                    self.extrema_removed = False

                # Remove or add extrema based on checkbox
                if hasattr(self.gui, 'checkbox_remove_extrema') and self.gui.checkbox_remove_extrema.isChecked():
                    if not self.extrema_removed:
                        filtered_matches, extrema = Matches.remove_extrema(self.matches)
                        self.matches = filtered_matches
                        self.extrema = extrema
                        self.extrema_removed = True
                else:
                    if self.extrema_removed:
                        self.matches = Matches.add_extrema(self.matches, self.extrema)
                        self.extrema = []
                        self.extrema_removed = False

                # Begrenze die Anzahl der gezeichneten Matches, aber behalte alle Matches intern
                matches_to_draw = self.matches
                if max_matches is not None and len(self.matches) > max_matches:
                    matches_to_draw = self.matches[:max_matches]

                # Get alpha value from GUI
                alpha = 0.5
                if hasattr(self.gui, '_get_alpha'):
                    alpha = self.gui._get_alpha()

                vis = Matches.draw_matches(left_with_kp, right_with_kp, matches_to_draw, line_color=(0, 200, 200), alpha=alpha)

                height, width, _ = vis.shape
                left_width = self.gui.left_img.shape[1]
                left_vis = vis[:, :left_width]
                right_vis = vis[:, left_width:]

                self.gui._set_pixmap(self.gui.left_widget, left_vis)
                self.gui._set_pixmap(self.gui.right_widget, right_vis)

            case "original":
                self.gui._set_pixmap(self.gui.left_widget, self.gui.left_img)
                self.gui._set_pixmap(self.gui.right_widget, self.gui.right_img)

            case _:
                self.gui._set_pixmap(self.gui.left_widget, self.gui.left_img)
                self.gui._set_pixmap(self.gui.right_widget, self.gui.right_img)

