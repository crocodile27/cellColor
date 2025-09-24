import cv2
from PyQt5.QtCore import QRectF, QTimer


class ZoomMixin:
    """Mixin for zooming and resetting zoom in the image viewer."""

    def zoom_to_selection(self, rect):
        if self.resized_image is None or self.original_image is None:
            return

        pixmap = self.image_label.pixmap()
        if not pixmap:
            return
        pixmap_rect = self.get_pixmap_rect()
        if not pixmap_rect.isValid():
            return
        print(f'[DEBUG] rubberband rect x, y{rect.x(), rect.y()}')
        print(
            f'[DEBUG] pixmap_rect.width, height {pixmap_rect.width()}, {pixmap_rect.height()}')
        normalized_rect = QRectF(
            (rect.x() - pixmap_rect.x()) / pixmap_rect.width(),
            (rect.y() - pixmap_rect.y()) / pixmap_rect.height(),
            rect.width() / pixmap_rect.width(),
            rect.height() / pixmap_rect.height()
        )

        normalized_rect = QRectF(
            max(0, normalized_rect.x()),
            max(0, normalized_rect.y()),
            min(1 - normalized_rect.x(), normalized_rect.width()),
            min(1 - normalized_rect.y(), normalized_rect.height())
        )

        orig_height, orig_width = self.original_image.shape[:2]
        print(f'[DEBUG] Original Image {orig_height}, {orig_width}')

        if getattr(self, "current_zoom", None) is not None:
            self.zoom_history.append(self.current_zoom.copy())
            cur = self.current_zoom
            orig_x1 = int(cur['x_start'] + normalized_rect.x()
                          * (cur['x_end'] - cur['x_start']))
            orig_y1 = int(cur['y_start'] + normalized_rect.y()
                          * (cur['y_end'] - cur['y_start']))
            orig_x2 = int(cur['x_start'] + (normalized_rect.x() +
                          normalized_rect.width()) * (cur['x_end'] - cur['x_start']))
            orig_y2 = int(cur['y_start'] + (normalized_rect.y() +
                          normalized_rect.height()) * (cur['y_end'] - cur['y_start']))
        else:
            orig_x1 = int(normalized_rect.x() * orig_width)
            orig_y1 = int(normalized_rect.y() * orig_height)
            orig_x2 = int(
                (normalized_rect.x() + normalized_rect.width()) * orig_width)
            orig_y2 = int(
                (normalized_rect.y() + normalized_rect.height()) * orig_height)

        # Clamp coordinates
        orig_x1 = max(0, min(orig_x1, orig_width - 1))
        orig_x2 = max(0, min(orig_x2, orig_width))
        orig_y1 = max(0, min(orig_y1, orig_height - 1))
        orig_y2 = max(0, min(orig_y2, orig_height))

        if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
            print("[Warning] Invalid zoom box: zero width/height")
            return

        selected_region = self.original_image[orig_y1:orig_y2, orig_x1:orig_x2]
        view_height, view_width = self.image_label.height(), self.image_label.width()
        scale_factor = min(
            view_height / selected_region.shape[0], view_width / selected_region.shape[1])

        # ✅ Update zoom state
        self.current_zoom = {
            'x_start': orig_x1,
            'y_start': orig_y1,
            'x_end': orig_x2,
            'y_end': orig_y2,
            'scale_factor': scale_factor,
        }
        print(
            f"[DEBUG] Zoomed Region: x={orig_x1}:{orig_x2}, y={orig_y1}:{orig_y2}")
        print(f"[DEBUG] Scale factor: {scale_factor}")

        self.resized_image = cv2.resize(
            selected_region, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR
        )

        self.reset_zoom_button.setEnabled(True)

        # Clear cached overlay coords
        for attr in ['visible_gene_x_coords', 'visible_gene_y_coords', 'visible_gene_colors',
                     'cell_center_x_coords', 'cell_center_y_coords']:
            if hasattr(self, attr):
                delattr(self, attr)

        self.update_display()

        self.status_bar.showMessage(
            f"Zoomed to region. Zoom level: {len(self.zoom_history) + 1}")

    def get_pixmap_rect(self):
        """Calculate the actual rectangle of the pixmap within the label."""
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return QRectF()

        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        x = (label_width - pixmap_width) / \
            2 if pixmap_width < label_width else 0
        y = (label_height - pixmap_height) / \
            2 if pixmap_height < label_height else 0

        return QRectF(x, y, pixmap_width, pixmap_height)

    def reset_zoom(self):
        if self.zoom_history:
            previous_zoom = self.zoom_history.pop()

            orig_x1, orig_y1 = previous_zoom['x_start'], previous_zoom['y_start']
            orig_x2, orig_y2 = previous_zoom['x_end'], previous_zoom['y_end']
            scale_factor = previous_zoom['scale_factor']

            selected_region = self.original_image[orig_y1:orig_y2,
                                                  orig_x1:orig_x2]

            self.resized_image = cv2.resize(
                selected_region, (0, 0),
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_LINEAR
            )

            self.current_zoom = previous_zoom.copy()
        else:
            self.do_full_reset()
            return

        self.update_display()
