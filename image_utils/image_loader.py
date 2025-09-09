import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QProgressDialog
from PyQt5.QtCore import Qt, QCoreApplication, QTimer

class ImageMixin:
    """Mixin for Image loading"""
    def load_image(self):
        self.status_bar.showMessage(f"Checking Image...")
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.tif *.tiff)")
        self.status_bar.showMessage(f"Opening File...")
        if file_name:
            self.status_bar.showMessage("Processing...")
            QTimer.singleShot(100, lambda: self.process_image(file_name))


    def process_image(self, file_name):
        progress = QProgressDialog("Loading image...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QCoreApplication.processEvents()

        try:
            # Step 1: File name selected
            progress.setValue(10)
            QCoreApplication.processEvents()

            # Step 2: Reading image
            if file_name.lower().endswith(('.tif', '.tiff')):
                print(f"Reading image, {file_name}")
                self.image = cv2.imread(file_name)
            else:
                self.image = cv2.imread(file_name)
            progress.setValue(40)
            QCoreApplication.processEvents()

            # Step 3: Copy original image
            if self.image is not None:
                self.original_image = self.image.copy()
                progress.setValue(60)
                QCoreApplication.processEvents()

                # Step 4: Reset zoom and resize
                self.reset_zoom_button.setEnabled(False)
                self.do_full_reset()
                progress.setValue(80)
                QCoreApplication.processEvents()

                # Step 5: Overlay genes if available
                if self.gene_data is not None:
                    self.overlay_genes()
                progress.setValue(90)
                QCoreApplication.processEvents()

                self.status_bar.showMessage("Image loaded and resized successfully")
            else:
                self.status_bar.showMessage("Failed to load image")
                progress.setValue(100)
                QCoreApplication.processEvents()
                return

            progress.setValue(100)
            QCoreApplication.processEvents()

        except Exception as e:
            self.status_bar.showMessage(f"Error loading image: {str(e)}")
            print(f"Error loading image: {str(e)}")
            progress.setValue(100)
            QCoreApplication.processEvents()
        finally:
            progress.close()


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

            self.status_bar.showMessage(f"Image displayed successfully ({width}x{height})")
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
                # Get the current dimensions of the image display area
                view_height = self.image_label.height()
                view_width = self.image_label.width()
                
                # If dimensions are too small, use minimum reasonable values
                if view_height < 100 or view_width < 100:
                    view_height = max(view_height, 600)
                    view_width = max(view_width, 800)
                
                # Get original image dimensions
                orig_height, orig_width = self.original_image.shape[:2]
                
                # Calculate scale factor to fit the image in the view while maintaining aspect ratio
                scale_factor = min(view_height / orig_height, view_width / orig_width)
                
                # Calculate new dimensions
                new_width = int(orig_width * scale_factor)
                new_height = int(orig_height * scale_factor)
                
                # Make sure new dimensions don't exceed view
                if new_height > view_height or new_width > view_width:
                    scale_factor = min(view_height / orig_height, view_width / orig_width) * 0.9  # 10% margin
                    new_width = int(orig_width * scale_factor)
                    new_height = int(orig_height * scale_factor)
                
                # Resize the image
                self.resized_image = cv2.resize(
                    self.original_image, 
                    (new_width, new_height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Clear zoom state and history
                self.zoom_history = []
                self.current_zoom = None
                
                # Update UI
                self.reset_zoom_button.setEnabled(False)  # Disable since we're at base zoom
                
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
                # Call overlay_genes to redraw genes if data exists
                self.update_display()
                if self.gene_data is not None and self.selected_genes:
                    self.overlay_genes()
                    
                self.status_bar.showMessage("View reset to original")