# isort: skip
# # Use GDAL for large TIFF files
# import gdal     
import random
import tkinter as tk
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QTimer, QRectF, QPointF
from qtpy.QtGui import QImage, QPixmap, QColor, QPainter, QPen
from qtpy.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget, QFileDialog,
                            QMenuBar, QAction, QStatusBar, QProgressBar, QToolBar,
                            QComboBox, QHBoxLayout, QPushButton, QScrollArea,
                            QFrame, QColorDialog, QSlider)
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
root = tk.Tk()
screen_height = root.winfo_screenheight() - 75
screen_width = root.winfo_screenwidth()
print("Width, height of screen: ", screen_width, screen_height)

class ZoomableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setMouseTracking(True)
        self.rubberband_active = False
        self.origin = QPointF()
        self.rubberband_rect = QRectF()
        self.setAlignment(Qt.AlignCenter)
        
    def mousePressEvent(self, event):
        if not hasattr(self.parent, 'resized_image') or self.parent.resized_image is None:
            return
            
        if event.button() == Qt.LeftButton:
            self.rubberband_active = True
            self.origin = event.pos()
            self.rubberband_rect = QRectF(self.origin, self.origin)
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.rubberband_active:
            self.rubberband_rect = QRectF(self.origin, event.pos()).normalized()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rubberband_active:
            self.rubberband_active = False
            # Only process zoom if the rectangle has a reasonable size
            if self.rubberband_rect.width() > 10 and self.rubberband_rect.height() > 10:
                self.parent.zoom_to_selection(self.rubberband_rect)
            self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.rubberband_active:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.rubberband_rect)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gene Visualization Tool")
        self.setGeometry(0, 0, screen_width, screen_height)
        self.screenWidth = screen_width
        self.screenHeight = screen_height
        
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Image Area
        self.image_area = QWidget()
        self.image_layout = QVBoxLayout(self.image_area)

        # Custom Zoomable Image Label
        self.image_label = ZoomableImageLabel(self)
        self.image_layout.addWidget(self.image_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.image_layout.addWidget(self.progress_bar)
        self.progress_bar.hide()

        # Toolbar Area
        self.toolbar_area = QWidget()
        self.toolbar_layout = QVBoxLayout(self.toolbar_area)

        # Zoom Controls
        self.zoom_controls_frame = QFrame()
        self.zoom_controls_layout = QVBoxLayout(self.zoom_controls_frame)
        
        self.zoom_label = QLabel("Zoom Instructions:")
        self.zoom_instructions = QLabel("Click and drag to select an area to zoom into")
        self.zoom_controls_layout.addWidget(self.zoom_label)
        self.zoom_controls_layout.addWidget(self.zoom_instructions)
        
        # Reset Zoom Button
        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        self.reset_zoom_button.setEnabled(False)
        self.zoom_controls_layout.addWidget(self.reset_zoom_button)
        
        self.toolbar_layout.addWidget(self.zoom_controls_frame)

        # Gene Selection Dropdown
        self.gene_dropdown = QComboBox()
        self.gene_dropdown.setPlaceholderText("Select a Gene")
        self.gene_dropdown.currentTextChanged.connect(self.on_gene_selected)
        self.toolbar_layout.addWidget(self.gene_dropdown)
        self.gene_dropdown.setSizeAdjustPolicy(2)

        # Selected Genes Scroll Area
        self.selected_genes_scroll = QScrollArea()
        self.selected_genes_widget = QWidget()
        self.selected_genes_layout = QVBoxLayout(self.selected_genes_widget)
        self.selected_genes_scroll.setWidget(self.selected_genes_widget)
        self.selected_genes_scroll.setWidgetResizable(True)
        self.toolbar_layout.addWidget(self.selected_genes_scroll)

        # Main Layout Organization
        self.main_layout.addWidget(self.image_area, stretch=4)
        self.main_layout.addWidget(self.toolbar_area, stretch=1)

        # Menu Bar
        self.menu_bar = self.menuBar()
        self.file_menu = self.menu_bar.addMenu("File")

        # Load Image Action
        self.load_image_action = QAction("Load Image", self)
        self.load_image_action.triggered.connect(self.load_image)
        self.file_menu.addAction(self.load_image_action)

        # Other menu items...
        self.load_detected_transcripts_action = QAction("Load Detected Transcripts", self)
        self.load_detected_transcripts_action.triggered.connect(self.load_detected_transcripts)
        self.file_menu.addAction(self.load_detected_transcripts_action)

        self.load_transformation_matrix_action = QAction("Load Transformation Matrix", self)
        self.load_transformation_matrix_action.triggered.connect(self.load_transformation_matrix)
        self.file_menu.addAction(self.load_transformation_matrix_action)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Data Storage
        self.image = None
        self.original_image = None
        self.gene_data = None
        self.transformation_matrix = None
        self.resized_image = None
        self.selected_genes = {}
        self.zoom_history = []  # Stack to track zoom levels
        
    def zoom_to_selection(self, rect):
        if self.resized_image is None or self.original_image is None:
            return
            
        # Get the pixmap geometry
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return
            
        pixmap_rect = self.get_pixmap_rect()
        if not pixmap_rect.isValid():
            return
            
        # Adjust rect coordinates relative to pixmap position
        normalized_rect = QRectF(
            (rect.x() - pixmap_rect.x()) / pixmap_rect.width(), 
            (rect.y() - pixmap_rect.y()) / pixmap_rect.height(),
            rect.width() / pixmap_rect.width(),
            rect.height() / pixmap_rect.height()
        )
        
        # Ensure rect is within valid bounds [0-1]
        normalized_rect = QRectF(
            max(0, normalized_rect.x()),
            max(0, normalized_rect.y()),
            min(1 - normalized_rect.x(), normalized_rect.width()),
            min(1 - normalized_rect.y(), normalized_rect.height())
        )
        
        # Calculate region in original image coordinates
        orig_height, orig_width = self.original_image.shape[:2]
        
        # If we're already zoomed in, calculate based on current zoom
        if hasattr(self, 'current_zoom') and self.current_zoom is not None:
            # Save current zoom state to history for back navigation
            self.zoom_history.append(self.current_zoom.copy())
            
            # Calculate new zoom coordinates relative to current zoom
            current_x_start = self.current_zoom['x_start']
            current_y_start = self.current_zoom['y_start']
            current_width = self.current_zoom['x_end'] - self.current_zoom['x_start']
            current_height = self.current_zoom['y_end'] - self.current_zoom['y_start']
            
            orig_x1 = int(current_x_start + normalized_rect.x() * current_width)
            orig_y1 = int(current_y_start + normalized_rect.y() * current_height)
            orig_x2 = int(current_x_start + (normalized_rect.x() + normalized_rect.width()) * current_width)
            orig_y2 = int(current_y_start + (normalized_rect.y() + normalized_rect.height()) * current_height)
        else:
            # First zoom level from original image
            orig_x1 = int(normalized_rect.x() * orig_width)
            orig_y1 = int(normalized_rect.y() * orig_height)
            orig_x2 = int((normalized_rect.x() + normalized_rect.width()) * orig_width)
            orig_y2 = int((normalized_rect.y() + normalized_rect.height()) * orig_height)
        
        # Extract the region from the original high-resolution image
        selected_region = self.original_image[orig_y1:orig_y2, orig_x1:orig_x2]
        
        # Calculate scale factor to fit the region in the view
        view_height = self.image_label.height()
        view_width = self.image_label.width()
        scale_factor = view_height / selected_region.shape[0]
        scale_factor2 = view_width / selected_region.shape[1]
        scale_factor = min(scale_factor, scale_factor2)
        
        # Resize the selected region
        self.resized_image = cv2.resize(
            selected_region, (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR
        )
        
        # Store zoom information for gene overlay calculations
        self.current_zoom = {
            'x_start': orig_x1,
            'y_start': orig_y1,
            'x_end': orig_x2,
            'y_end': orig_y2,
            'scale_factor': scale_factor
        }
        
        # Update UI state
        self.reset_zoom_button.setEnabled(True)
        self.display_image()
        
        # Overlay genes if data is available
        if self.gene_data is not None:
            self.overlay_genes()
        
        self.status_bar.showMessage(f"Zoomed to region. Zoom level: {len(self.zoom_history) + 1}")
    
    def get_pixmap_rect(self):
        """Calculate the actual rectangle of the pixmap within the label"""
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return QRectF()
            
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        
        # Calculate position based on alignment
        x = (label_width - pixmap_width) / 2 if pixmap_width < label_width else 0
        y = (label_height - pixmap_height) / 2 if pixmap_height < label_height else 0
        
        return QRectF(x, y, pixmap_width, pixmap_height)

    def reset_zoom(self):
        # Check if we have zoom history
        if self.zoom_history:
            # Pop the last zoom level and apply it
            previous_zoom = self.zoom_history.pop()
            
            # If we're back at the first level, completely reset
            if not self.zoom_history:
                self.do_full_reset()
                return
                
            # Otherwise, go back to previous zoom level
            orig_x1 = previous_zoom['x_start']
            orig_y1 = previous_zoom['y_start']
            orig_x2 = previous_zoom['x_end']
            orig_y2 = previous_zoom['y_end']
            scale_factor = previous_zoom['scale_factor']
            
            # Extract region from original image
            selected_region = self.original_image[orig_y1:orig_y2, orig_x1:orig_x2]
            
            # Resize selected region
            self.resized_image = cv2.resize(
                selected_region, (0, 0),
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_LINEAR
            )
            
            # Update current zoom
            self.current_zoom = previous_zoom.copy()  # Use copy to avoid reference issues
            
        else:
            # No history, reset to original view
            self.do_full_reset()
        
        # Update display
        self.display_image()
        
        # Overlay genes if data is available
        if self.gene_data is not None:
            self.overlay_genes()
            
        # Update status
        zoom_level = len(self.zoom_history) + (1 if hasattr(self, 'current_zoom') and self.current_zoom else 0)
        self.status_bar.showMessage(f"Zoom level: {zoom_level}")

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
            if hasattr(self, 'current_zoom'):
                self.current_zoom = None
            self.zoom_history = []
            
            # Update UI
            self.reset_zoom_button.setEnabled(False)  # Disable since we're at base zoom
            self.display_image()
            
            # Store the scale factor for use in overlay_genes
            self.full_view_scale_factor = scale_factor
            
            # Call overlay_genes to redraw genes if data exists
            if self.gene_data is not None and hasattr(self, 'selected_genes') and self.selected_genes:
                self.overlay_genes()
                
            self.status_bar.showMessage("View reset to original")

            
    def load_transformation_matrix(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.progress_bar.show()
            self.status_bar.showMessage("Loading Transformation Matrix...")
            QTimer.singleShot(0, lambda: self.process_csv(file_name))

    def load_detected_transcripts(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.progress_bar.show()
            self.status_bar.showMessage(
                "Loading Detected Transcripts...(this may take a while)")
            QTimer.singleShot(0, lambda: self.process_csv(file_name))

    def process_csv(self, file_name):
        try:
            if "transform" in file_name.lower():
                # Load transformation matrix
                self.transformation_matrix = pd.read_csv(
                    file_name, header=None)
                self.transformation_matrix = self.transformation_matrix[0].str.split(
                    expand=True).astype(float).values
                self.status_bar.showMessage(
                    "Transformation matrix loaded successfully")
            else:
                # Load gene transcript data
                self.gene_data = pd.read_csv(file_name)
                self.status_bar.showMessage("Gene data loaded successfully")

                # Populate gene dropdown
                unique_genes = self.gene_data['gene'].unique()
                self.gene_dropdown.clear()
                self.gene_dropdown.addItems(unique_genes)

                if self.image is not None:
                    self.overlay_genes()
        except Exception as e:
            self.status_bar.showMessage(
                f"Error loading CSV file {file_name}: {str(e)}")
        finally:
            self.progress_bar.hide()

    def on_gene_selected(self, gene):
        if gene in self.selected_genes:
            self.status_bar.showMessage("Gene already selected, choose a different gene.")
            return
        elif not gene:
            self.status_bar.showMessage("Gene does not exist, choose a different gene.")
            return

        # Generate a unique color
        color = self.generate_unique_color()

        # Create a gene selection widget
        gene_widget = QFrame()
        gene_widget_layout = QHBoxLayout(gene_widget)

        # Color indicator
        color_label = QLabel()
        color_label.setFixedSize(20, 20)
        color_label.setStyleSheet(f"background-color: rgb({color[2]}, {color[1]}, {color[0]}); border-radius: 10px;")

        # Gene name label
        gene_name_label = QLabel(gene)

        # Remove button
        remove_button = QPushButton("cancel")
        remove_button.setFixedSize(75, 25)
        remove_button.clicked.connect(
            lambda _, g=gene: self.remove_gene_selection(g))

        gene_widget_layout.addWidget(color_label)
        gene_widget_layout.addWidget(gene_name_label)
        gene_widget_layout.addStretch()
        gene_widget_layout.addWidget(remove_button)

        # Store gene and color
        self.selected_genes[gene] = color

        # Add to selected genes layout
        self.selected_genes_layout.addWidget(gene_widget)

        # Overlay genes
        self.overlay_genes()

    def remove_gene_selection(self, gene):
        # Remove from selected genes
        if gene in self.selected_genes:
            del self.selected_genes[gene]

        # Remove widget from layout
        for i in range(self.selected_genes_layout.count()):
            widget = self.selected_genes_layout.itemAt(i).widget()
            if widget:
                # Find all labels in the widget
                labels = widget.findChildren(QLabel)
                # Check if any label contains our gene name
                for label in labels:
                    if label.text() == gene:  # Exact match instead of partial match
                        # Remove the widget from layout
                        self.selected_genes_layout.removeWidget(widget)
                        # Hide the widget first
                        widget.hide()
                        self.overlay_genes()
                        # Schedule for deletion
                        widget.deleteLater()#deferred
                        # Reoverlay genes
                        
                        return  # Exit after finding and removing
                        break

        self.overlay_genes()

    def generate_unique_color(self):
        # Generate a random color that isn't too light
        while True:
            color = (random.randint(50, 200),
                     random.randint(50, 200),
                     random.randint(50, 200))
            # Ensure color is not already used
            if color not in self.selected_genes.values():
                return color

    def overlay_genes(self):
        if self.gene_data is None or self.image is None or self.resized_image is None:
            print("Please make sure to upload the detected transcripts")
            return

        # Create a copy of the resized image to draw on
        overlay_image = self.resized_image.copy()

        # Filter out only selected genes to make it faster
        selected_gene_mask = self.gene_data["gene"].isin(self.selected_genes)
        filtered_data = self.gene_data[selected_gene_mask]

        if filtered_data.empty:
            self.status_bar.showMessage("No selected genes to overlay.")
            # Display the original resized image without any overlay
            self.display_image()
            return

        # Extract coordinates and genes
        coords = filtered_data[["global_x", "global_y"]].to_numpy()
        genes = filtered_data["gene"].to_numpy()

        # Apply transformation matrix in bulk
        if self.transformation_matrix is not None:
            ones = np.ones((coords.shape[0], 1))
            transformed_coords = np.dot(self.transformation_matrix, np.hstack([coords, ones]).T).T
            x_coords, y_coords = transformed_coords[:, 0], transformed_coords[:, 1]
        else:
            self.status_bar.showMessage("There is no transformation matrix. Please load a transformation matrix.")
            return

        # Handle zoomed view differently from the full view
        if hasattr(self, 'current_zoom') and self.current_zoom is not None:
            # Filter for genes only in the zoomed region (in original image coordinates)
            zoom_x_start = self.current_zoom['x_start']
            zoom_y_start = self.current_zoom['y_start']
            zoom_x_end = self.current_zoom['x_end']
            zoom_y_end = self.current_zoom['y_end']
            
            # Create masks for genes inside the zoomed area
            in_zoom_region = (
                (x_coords >= zoom_x_start) & 
                (x_coords < zoom_x_end) & 
                (y_coords >= zoom_y_start) & 
                (y_coords < zoom_y_end)
            )
            
            # Filter to only include genes in the zoomed region
            if not any(in_zoom_region):
                self.status_bar.showMessage("No genes in the zoomed region")
                return
                
            x_coords = x_coords[in_zoom_region]
            y_coords = y_coords[in_zoom_region]
            genes = genes[in_zoom_region]
            
            # Adjust coordinates for zoomed view
            x_coords = (x_coords - zoom_x_start) * self.current_zoom['scale_factor']
            y_coords = (y_coords - zoom_y_start) * self.current_zoom['scale_factor']
        else:
            # Normal full-image view
            # Use the scale factor calculated in do_full_reset
            if hasattr(self, 'full_view_scale_factor'):
                scale_factor = self.full_view_scale_factor
            else:
                # Fall back to calculating scale factor if not already stored
                orig_height, orig_width = self.original_image.shape[:2]
                view_height = self.image_label.height()
                view_width = self.image_label.width()
                scale_factor = min(view_height / orig_height, view_width / orig_width)
                if scale_factor <= 0:
                    scale_factor = 0.5  # Fallback if calculation fails
                self.full_view_scale_factor = scale_factor
                
            x_coords = x_coords * scale_factor
            y_coords = y_coords * scale_factor

        # Vectorized color mapping
        colors = np.array([self.selected_genes[gene] for gene in genes])
        
        # Convert to integer coordinates
        x_coords = x_coords.astype(int)
        y_coords = y_coords.astype(int)

        # Filter out genes that would be outside the visible area
        height, width = overlay_image.shape[:2]
        valid_coords = (
            (x_coords >= 0) & 
            (x_coords < width) & 
            (y_coords >= 0) & 
            (y_coords < height)
        )
        
        x_coords = x_coords[valid_coords]
        y_coords = y_coords[valid_coords]
        colors = colors[valid_coords]

        # Draw visible genes
        for x, y, color in zip(x_coords, y_coords, colors):
            color = tuple(map(int, color))
            cv2.circle(overlay_image, (x, y), 1, color, -1)

        # Convert image for display
        overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        height, width, channel = overlay_image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(overlay_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
        
        self.status_bar.showMessage(f"Genes overlaid: {len(x_coords)} visible points")

    def load_image(self):
        self.status_bar.showMessage(f"Checking Image...")
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.tif *.tiff)")
        self.status_bar.showMessage(f"Opening File...")
        if file_name:
            self.progress_bar.show()
            self.status_bar.showMessage("Loading image...")
            QTimer.singleShot(100, lambda: self.process_image(file_name))

    def process_image(self, file_name):
        try:
            # Create an image pyramid for large images
            if file_name.lower().endswith(('.tif', '.tiff')):
                
            #     dataset = gdal.Open(file_name)
            #     # Store dataset reference and load only visible portion
            #     self.image_dataset = dataset
            #     # Get lower resolution thumbnail for initial display
            #     self.image = self.get_thumbnail_from_dataset(dataset)
            # else:
            #     # For smaller images, load normally
                self.image = cv2.imread(file_name)
            
            self.original_image = self.image.copy()

            if self.image is not None:
                # Instead of using fixed screen dimensions, use the actual image label dimensions
                self.reset_zoom_button.setEnabled(False)  # Disable zoom reset initially
                self.do_full_reset()  # This will properly resize the image to fit
                
                # If gene data is already loaded, reoverlay genes
                if self.gene_data is not None:
                    self.overlay_genes()

                self.status_bar.showMessage("Image loaded and resized successfully")
            else:
                self.status_bar.showMessage("Failed to load image")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading image: {str(e)}")
            print(f"Error loading image: {str(e)}")
        finally:
            self.progress_bar.hide()

    def display_image(self):
        if self.resized_image is not None:
            # Convert the image to RGB format
            resized_image_rgb = cv2.cvtColor(
                self.resized_image, cv2.COLOR_BGR2RGB)

            # Get image dimensions
            height, width, channel = resized_image_rgb.shape
            bytes_per_line = 3 * width

            # Create QImage from the RGB image data
            q_img = QImage(resized_image_rgb.data, width, height,
                        bytes_per_line, QImage.Format_RGB888)

            # Display the image in the QLabel
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            
            # Ensure the image label size hint is appropriate
            self.image_label.setMinimumSize(1, 1)  # Allow the label to shrink if needed
            
            self.status_bar.showMessage(f"Image displayed successfully ({width}x{height})")
        else:
            self.status_bar.showMessage("Resized image is None")

    def resizeEvent(self, event):
        """Handle window resize events to adjust the image size"""
        super().resizeEvent(event)
        
        # If we have an image loaded, resize it to fit the new dimensions
        if hasattr(self, 'original_image') and self.original_image is not None:
            # Delay the resize slightly to ensure all UI components have updated their sizes
            QTimer.singleShot(50, self.resize_image_to_fit)

    def resize_image_to_fit(self):
        """Resize the current image to fit the display after window resize"""
        if hasattr(self, 'current_zoom') and self.current_zoom is not None:
            # We're in a zoomed state, don't resize
            return
            
        # We're in the full view, resize to fit
        self.do_full_reset()
        
if __name__ == '__main__':
    from qtpy.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
    