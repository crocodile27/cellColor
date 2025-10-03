import os
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer
from cellpose import utils


class CellposeMixin:
    def _generate_outlines_and_update(self):
        # Derive the original mask path from the color image path if available
        if hasattr(self, 'cellpose_mask_color_image') and hasattr(self, 'cellpose_masks'):
            # Guess filename from npy file (you could store this explicitly if unsure)
            mask_shape = self.cellpose_masks.shape
            color_image_shape = getattr(
                self.cellpose_mask_color_image, 'shape', None)
            if color_image_shape and color_image_shape[:2] == mask_shape:
                base_path = None
                for ext in ['_color.npy', '_masks.npy', '.npy']:
                    try:
                        color_path = next(
                            p for p in sys.argv if p.endswith(ext)
                        )
                        base_path = color_path.replace(ext, '')
                        break
                    except StopIteration:
                        continue
            else:
                base_path = None
        else:
            base_path = None

        outline_path = getattr(self, 'cellpose_mask_base_path', None)

        if outline_path:
            outline_path += "_outlines.npy"
            if os.path.exists(outline_path):
                self.cellpose_outlines = np.load(
                    outline_path, allow_pickle=True).tolist()
            else:
                # This is likely where it's hanging
                self.cellpose_outlines = utils.outlines_list(
                    self.cellpose_masks)

                # new
                # Fixed: was mask_shape
                h_img, w_img = self.cellpose_masks.shape[:2]
                h_m, w_m = self.cellpose_mask_color_image.shape[:2]
                scale_x = w_m / w_img
                scale_y = h_m / h_img
                scaled = []
                for i, outline in enumerate(self.cellpose_outlines):
                    pts = np.array(outline)
                    # multiply then cast back to int
                    pts_scaled = [(int(x * scale_x), int(y * scale_y))
                                  for x, y in pts]
                    if len(pts_scaled) > 1:
                        scaled.append(pts_scaled)
                self.cellpose_outlines = scaled

                np.save(outline_path, np.array(
                    self.cellpose_outlines, dtype=object))

        else:
            self.cellpose_outlines = utils.outlines_list(self.cellpose_masks)

        self.toggle_cellpose_button.setEnabled(True)
        self.toggle_cellpose_outline_button.setEnabled(True)

        if self.cell_centers is not None:            # Enable the cluster button
            # Automatically create cluster masks if both cell_centers and cellpose_masks are available
            self.make_cluster_data()
        self.status_bar.showMessage("Cellpose masks loaded successfully")
        self.update_display()

    def load_cellpose_masks(self):
        """Load Cellpose masks - used internally by auto_load_files"""
        if self.original_image is None:
            self.status_bar.showMessage(
                "Please make sure to upload an image to scale to.")
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Cellpose Masks", "", "NumPy Files (*.npy)")
        if not file_name:
            return

        self._load_cellpose_masks_from_path(file_name)

    def toggle_cellpose_masks(self):
        self.show_cellpose_masks = self.toggle_cellpose_button.isChecked()
        self.toggle_cellpose_button.setText(
            "Hide Cellpose Masks" if self.show_cellpose_masks else "Show Cellpose Masks")
        self.update_display()

    def toggle_cellpose_outlines(self):
        self.show_cellpose_outlines = self.toggle_cellpose_outline_button.isChecked()
        self.toggle_cellpose_outline_button.setText(
            "Hide Cellpose Outlines" if self.show_cellpose_outlines else "Show Cellpose Outlines")
        self.update_display()

    def _draw_cellpose_mask_fill(self, image):

        if not hasattr(self, 'cellpose_mask_color_image'):
            return
        if hasattr(self, 'current_zoom') and self.current_zoom is not None:
            zoom = self.current_zoom
            crop = self.cellpose_mask_color_image[
                zoom['y_start']:zoom['y_end'],
                zoom['x_start']:zoom['x_end']
            ]
        else:
            crop = self.cellpose_mask_color_image

        if crop.size == 0:
            return
        # highlight
        resized = cv2.resize(
            crop, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Force both images to be np.uint8
        if resized.dtype != np.uint8:
            resized = resized.astype(np.uint8)
        if image.dtype != np.uint8:
            image[:] = image.astype(np.uint8)
        cv2.addWeighted(image, 0.5, resized, 0.5, 0, dst=image)

    def _draw_cellpose_mask_outlines(self, image):
        if self.cellpose_outlines is None:
            return

        if hasattr(self, 'current_zoom') and self.current_zoom is not None:
            zoom = self.current_zoom
            x0, y0 = zoom['x_start'], zoom['y_start']
            x1, y1 = zoom['x_end'], zoom['y_end']
            rect_width, rect_height = x1 - x0, y1 - y0

            scale_x = image.shape[1] / rect_width
            scale_y = image.shape[0] / rect_height
            for outline in self.cellpose_outlines:
                outline = np.array(outline)
                in_x = (outline[:, 0] >= x0) & (outline[:, 0] < x1)
                in_y = (outline[:, 1] >= y0) & (outline[:, 1] < y1)
                valid = in_x & in_y
                if not np.any(valid):
                    continue
                outline = outline[valid]
                outline_zoom = outline - np.array([x0, y0])
                # apply the *same* scaling that cv2.resize used
                outline_scaled = (
                    outline_zoom * np.array([scale_x, scale_y])).astype(int)

                if len(outline_scaled) > 1:
                    cv2.polylines(image, [outline_scaled], isClosed=True, color=(
                        0, 0, 255), thickness=1)
        else:
            mask_h, mask_w = self.cellpose_mask_color_image.shape[:2]
            scale_x = image.shape[1] / mask_w
            scale_y = image.shape[0] / mask_h

            for outline in self.cellpose_outlines:
                outline = np.array(outline)
                # shift origin to (0,0)—not strictly necessary if mask coords already start at zero
                outline_zoom = outline
                outline_scaled = (
                    outline_zoom * np.array([scale_x, scale_y])).astype(int)
                if len(outline_scaled) > 1:
                    cv2.polylines(image, [outline_scaled], isClosed=True, color=(
                        0, 0, 255), thickness=1)
