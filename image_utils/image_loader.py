import os
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QProgressDialog
from PyQt5.QtCore import Qt, QCoreApplication, QTimer
from PyQt5.QtWidgets import QMessageBox


class ImageMixin:
    """Mixin for Image loading"""

    def load_image(self):
        self.status_bar.showMessage("Checking Image...")
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.tif *.tiff)"
        )
        self.status_bar.showMessage("Opening File...")

        if file_name:
            root, ext = os.path.splitext(file_name)
            downsized_path = f"{root}_downsizedby4{ext}"
            # --- Priority 1: load downsized if already exists ---
            if os.path.exists(downsized_path):
                print('found another downsized file')
                self.status_bar.showMessage("Loading downsized image...")
                self.original_image = cv2.imread(downsized_path)
            elif "_downsizedby4" in file_name:
                print('this is a downsized file')
                self.status_bar.showMessage("Loading downsized image...")
                self.original_image = cv2.imread(file_name)
            else:
                # --- Otherwise load full image and downsize---
                self.status_bar.showMessage("Loading full image...")
                print('loaded full image')

                downsized_path = self.downsize_image(file_name)
                self.show_downsize_message(downsized_path)

            self.reset_zoom_button.setEnabled(False)
            self.do_full_reset()

            if self.gene_data is not None:
                self.filter_genes()

            self.status_bar.showMessage("Image loaded successfully")

    def downsize_image(self, file_path):
        # Load the original image
        self.original_image = cv2.imread(file_path)
        orig_height, orig_width = self.original_image.shape[:2]

        # Scale factor
        scale_factor = 0.25
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)

        # Resize
        downsized_image = cv2.resize(
            self.original_image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Save downsized file in same folder
        root, ext = os.path.splitext(file_path)
        downsized_path = f"{root}_downsizedby4{ext}"
        cv2.imwrite(downsized_path, downsized_image)
        self.original_image = downsized_image

        return downsized_path

    def display_image(self):
        if self.resized_image is not None:
            resized_image_rgb = cv2.cvtColor(
                self.resized_image, cv2.COLOR_BGR2RGB)

            height, width, channel = resized_image_rgb.shape
            bytes_per_line = 3 * width

            q_img = QImage(resized_image_rgb.data, width, height,
                           bytes_per_line, QImage.Format_RGB888)

            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            self.image_label.setMinimumSize(1, 1)

            self.status_bar.showMessage(
                f"Image displayed successfully ({width}x{height})")
        else:
            self.status_bar.showMessage("Resized image is None")

    def resize_event(self, event):
        """Handle window resize events to adjust the image size"""
        super(type(self), self).resizeEvent(event)
        if hasattr(self, 'original_image') and self.original_image is not None:
            QTimer.singleShot(50, self.resize_image_to_fit)

    def resize_image_to_fit(self):
        """Resize the current image to fit the display after window resize"""
        if hasattr(self, 'current_zoom') and self.current_zoom is not None:
            return
        self.do_full_reset()

    def do_full_reset(self):
        """Reset to original unzoomed state"""
        if self.original_image is not None:
            self.view_height = self.image_label.height()
            self.view_width = self.image_label.width()
            self.orig_height, self.orig_width = self.original_image.shape[:2]
            scale_factor = min(self.view_height / self.orig_height,
                               self.view_width / self.orig_width)
            new_width = int(self.orig_width * scale_factor)
            new_height = int(self.orig_height * scale_factor)

            # # Make sure new dimensions don't exceed view
            # if new_height > self.view_height or new_width > self.view_width:
            #     scale_factor = min(self.view_height / self.orig_height, self.view_width / self.orig_width) * 0.9  # 10% margin
            #     new_width = int(self.orig_width * scale_factor)
            #     new_height = int(self.orig_height * scale_factor)

            # Resize the image
            self.resized_image = cv2.resize(
                self.original_image,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR
            )
            print("resized here")
            print(f'height, width {self.resized_image.shape[:2]}')

            # Clear zoom state and history
            self.zoom_history = []
            self.current_zoom = None

            # Update UI
            # Disable since we're at base zoom
            self.reset_zoom_button.setEnabled(False)

            # Store the scale factor for use in overlay_genes
            self.full_view_scale_factor = scale_factor

            # Clear any cached coordinates since we have a new zoom level
            if hasattr(self, 'visible_gene_x_coords'):
                delattr(self, 'visible_gene_x_coords')
            if hasattr(self, 'visible_gene_y_coords'):
                delattr(self, 'visible_gene_y_coords')
            if hasattr(self, 'visible_gene_colors'):
                delattr(self, 'visible_gene_colors')
            if hasattr(self, 'cell_center_x_coords'):
                delattr(self, 'cell_center_x_coords')
            if hasattr(self, 'cell_center_y_coords'):
                delattr(self, 'cell_center_y_coords')
            # Call filter_genes to redraw genes if data exists
            self.update_display()
            if self.gene_data is not None and self.selected_genes:
                self.filter_genes()

            self.status_bar.showMessage("View reset to original")

    def show_downsize_message(self, downsized_path):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Image Downsized")
        msg.setText(
            "Your original image is slow for loading.\n\n"
            "A downsized image (1/4 x 1/4) has been created "
            "for faster loading in the future.\n\n"
            f"Saved in the same folder as:\n{downsized_path}"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
