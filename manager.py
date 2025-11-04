from models.keypoints import Keypoints
from models.keytracer import Keytracer
from models.matches import Matches
import cv2

class AppManager:
    def __init__(self, gui):
        self.gui = gui  # Reference to the GUI (view)
        self.matches = None
        self.extrema = []
        self.extrema_removed = False

    def handle_control_change(self):

        if self.gui.radio_trace.isChecked():
            display_mode = "edges"
        elif self.gui.radio_features.isChecked():
            display_mode = "features"
        elif self.gui.radio_matches.isChecked():
            display_mode = "matches"
        else:
            display_mode = "original"

        self.gui.display_mode = display_mode

        # Main display logic (merged from refresh_display)
        if self.gui.left_img is None or self.gui.right_img is None:
            return

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
                max_corners = self.gui._get_max_corners()
                left_gray = cv2.cvtColor(self.gui.left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(self.gui.right_img, cv2.COLOR_BGR2GRAY)
                keypoints_left = Keypoints.detect_features(left_gray, max_corners=max_corners)
                keypoints_right = Keypoints.detect_features(right_gray, max_corners=max_corners)
                drawnpoint_left = Keypoints.draw_freatures(self.gui.left_img, keypoints_left)
                drawnpoints_right = Keypoints.draw_freatures(self.gui.right_img, keypoints_right)
                self.matches = None  # Drop matches if features are recalculated
                self.extrema = []
                self.extrema_removed = False
                self.gui._set_pixmap(self.gui.left_widget, drawnpoint_left)
                self.gui._set_pixmap(self.gui.right_widget, drawnpoints_right)

            case "matches":
                max_corners = self.gui._get_max_corners()
                left_gray = cv2.cvtColor(self.gui.left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(self.gui.right_img, cv2.COLOR_BGR2GRAY)
                keypoints_left = Keypoints.detect_features(left_gray, max_corners=max_corners)
                keypoints_right = Keypoints.detect_features(right_gray, max_corners=max_corners)
                left_with_kp = Keypoints.draw_freatures(self.gui.left_img, keypoints_left)
                right_with_kp = Keypoints.draw_freatures(self.gui.right_img, keypoints_right)

                # Recalculation on empty or corner changed
                if self.matches is None:
                    self.matches = Matches.find_machtes(left_gray, right_gray, keypoints_left, keypoints_right)
                    self.extrema = []
                    self.extrema_removed = False

                # Remove or add extrema based on checkbox
                if hasattr(self.gui, 'checkbox_remove_extrema') and self.gui.checkbox_remove_extrema.isChecked():
                    if not self.extrema_removed:
                        # Entferne Extrema, speichere sie in self.extrema
                        filtered_matches, extrema = Matches.remove_extrema(self.matches)
                        self.matches = filtered_matches
                        self.extrema = extrema
                        self.extrema_removed = True

                        # Checkbox nach einmaliger Anwendung wieder deaktivieren
                        self.gui.checkbox_remove_extrema.setChecked(False)
                else:
                    if self.extrema_removed:
                        # Re-add extrema using Matches.add_extrema
                        self.matches = Matches.add_extrema(self.matches, self.extrema)
                        self.extrema = []
                        self.extrema_removed = False

                vis = Matches.draw_matches(left_with_kp, right_with_kp, self.matches)
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
        
    def __init__(self, gui):
        self.gui = gui  # Reference to the GUI (view)

