# isort: skip
import random
import tkinter as tk
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage, QPixmap, QColor
from qtpy.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget, QFileDialog,
                            QMenuBar, QAction, QStatusBar, QProgressBar, QToolBar,
                            QComboBox, QHBoxLayout, QPushButton, QScrollArea,
                            QFrame, QColorDialog, QSlider)
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
root = tk.Tk()
height = root.winfo_screenheight() - root.winfo_screenmmheight()
width = root.winfo_screenwidth()
print("Width, height of screen: ", width, height)

class GridLabel(QLabel):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.parent = parent
            self.grid_size = 1
            self.selected_cell = None
            self.setMouseTracking(True)

        def mousePressEvent(self, event):
            if not hasattr(self.parent, 'resized_image') or self.parent.resized_image is None:
                return

            pos = event.pos()
            pixmap = self.pixmap()
            if pixmap:
                width = pixmap.width()
                height = pixmap.height()
                cell_width = width / self.grid_size
                cell_height = height / self.grid_size
                
                # Calculate label dimensions and position
                label_rect = self.rect()
                label_width = label_rect.width()
                label_height = label_rect.height()
                
                # Calculate pixmap position within label (due to alignment)
                pixmap_x = (label_width - width) / 2 if width < label_width else 0
                pixmap_y = (label_height - height) / 2 if height < label_height else 0
                
                # Adjust coordinates relative to pixmap position
                adjusted_x = pos.x() - pixmap_x
                adjusted_y = pos.y() - pixmap_y
                
                # Ensure adjusted coordinates are within pixmap boundaries
                adjusted_x = max(0, min(width-1, adjusted_x))
                adjusted_y = max(0, min(height-1, adjusted_y))
                
                # Calculate which cell was clicked
                row = int(adjusted_y / cell_height)
                col = int(adjusted_x / cell_width)

                self.selected_cell = (row, col)
                self.parent.update_grid_overlay()
            
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gene Visualization Tool")
        self.setGeometry(0, 0, width, height)
        self.screenWidth = width
        self.screenHeight = height
        
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Image Area
        self.image_area = QWidget()
        self.image_layout = QVBoxLayout(self.image_area)

        # Custom Image Label with Grid Support
        self.image_label = GridLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.image_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.image_layout.addWidget(self.progress_bar)
        self.progress_bar.hide()

        # Toolbar Area
        self.toolbar_area = QWidget()
        self.toolbar_layout = QVBoxLayout(self.toolbar_area)

        # Grid Size Slider
        self.grid_slider_label = QLabel("Grid Size: 1x1")
        self.toolbar_layout.addWidget(self.grid_slider_label)
        
        self.grid_slider = QSlider(Qt.Horizontal)
        self.grid_slider.setMinimum(1)
        self.grid_slider.setMaximum(10)
        self.grid_slider.setValue(1)
        self.grid_slider.setTickPosition(QSlider.TicksBelow)
        self.grid_slider.setTickInterval(1)
        self.grid_slider.valueChanged.connect(self.update_grid_size)
        self.toolbar_layout.addWidget(self.grid_slider)

        # Confirm Zoom Button
        self.zoom_button = QPushButton("Confirm Zoom")
        self.zoom_button.clicked.connect(self.zoom_to_selection)
        self.zoom_button.setEnabled(False)
        self.toolbar_layout.addWidget(self.zoom_button)

        # Reset Zoom Button
        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        self.reset_zoom_button.setEnabled(False)
        self.toolbar_layout.addWidget(self.reset_zoom_button)

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
        self.original_image = None  # Store the original image for reset functionality
        self.gene_data = None
        self.transformation_matrix = None
        self.resized_image = None
        self.selected_genes = {}

    
    def update_grid_size(self):
        size = self.grid_slider.value()
        self.grid_slider_label.setText(f"Grid Size: {size}x{size}")
        self.image_label.grid_size = size
        self.image_label.selected_cell = None
        self.zoom_button.setEnabled(False)
        self.update_grid_overlay()
    def update_grid_overlay(self):
        if self.resized_image is None:
            return

        overlay_image = self.resized_image.copy()
        height, width = overlay_image.shape[:2]
        grid_size = self.image_label.grid_size

        # Draw grid lines
        cell_height = height / grid_size
        cell_width = width / grid_size

        # Draw horizontal lines
        for i in range(1, grid_size):
            y = int(i * cell_height)
            cv2.line(overlay_image, (0, y), (width, y), (255, 255, 255), 1)

        # Draw vertical lines
        for i in range(1, grid_size):
            x = int(i * cell_width)
            cv2.line(overlay_image, (x, 0), (x, height), (255, 255, 255), 1)

        # Highlight selected cell
        if self.image_label.selected_cell is not None:
            row, col = self.image_label.selected_cell
            x1 = int(col * cell_width)
            y1 = int(row * cell_height)
            x2 = int((col + 1) * cell_width)
            y2 = int((row + 1) * cell_height)
            
            overlay = overlay_image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, overlay_image, 0.7, 0, overlay_image)
            self.zoom_button.setEnabled(True)

        # Convert and display the image
        overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        height, width, channel = overlay_image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(overlay_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def zoom_to_selection(self):
        if self.image_label.selected_cell is None or self.resized_image is None or self.original_image is None:
            return

        row, col = self.image_label.selected_cell
        grid_size = self.image_label.grid_size

        # Calculate the region to zoom into in the resized image coordinates
        resized_height, resized_width = self.resized_image.shape[:2]
        cell_height = resized_height / grid_size
        cell_width = resized_width / grid_size

        # Calculate the proportion of the cell in the resized image
        y_prop_start = row / grid_size
        x_prop_start = col / grid_size
        y_prop_end = (row + 1) / grid_size
        x_prop_end = (col + 1) / grid_size

        # Calculate the corresponding region in the original image
        orig_height, orig_width = self.original_image.shape[:2]
        orig_y1 = int(y_prop_start * orig_height)
        orig_x1 = int(x_prop_start * orig_width)
        orig_y2 = int(y_prop_end * orig_height)
        orig_x2 = int(x_prop_end * orig_width)

        # Extract the region from the original high-resolution image
        selected_region = self.original_image[orig_y1:orig_y2, orig_x1:orig_x2]
        
        # Determine the appropriate scale factor to fit the region to the display
        scale_factor = self.screenHeight / selected_region.shape[0]
        
        # Resize the selected region directly from the original image
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
        self.image_label.selected_cell = None
        self.zoom_button.setEnabled(False)
        self.reset_zoom_button.setEnabled(True)
        self.update_grid_overlay()

        # Overlay genes if data is available
        if self.gene_data is not None:
            self.overlay_genes()

    def reset_zoom(self):
        if self.original_image is not None:
            # Restore the original resized image with proper scaling
            scale_factor = self.screenHeight / self.original_image.shape[0]
            self.resized_image = cv2.resize(
                self.original_image, (0, 0), 
                fx=scale_factor, 
                fy=scale_factor
            )
            # Clear zoom state
            self.current_zoom = None
            
            self.reset_zoom_button.setEnabled(False)
            self.update_grid_overlay()
            
            if self.gene_data is not None:
                self.overlay_genes()
                
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
    '''
    process_csv:
    This takes in the transformation function and the gene transcript data and stores it.
    Variables Changed:
    self.transformation matrix,
    self.gene_data
    self.gene_dropdown
    
    Then calls the overlay_genes function
    '''
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
        color_label.setStyleSheet(
            f"background-color: rgb{color}; border-radius: 10px;")

        # Gene name label
        gene_name_label = QLabel(gene)

        # Remove button
        remove_button = QPushButton("cancel")
        remove_button.setFixedSize(25, 25)
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
            if widget and gene in widget.findChild(QLabel).text():
                self.selected_genes_layout.removeWidget(widget)
                widget.deleteLater()
                break

        # Reoverlay genes
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
    '''
    overlay_genes:
    This function should create a copy of the resized image and draw dots on the matrix for the genes that were selected.

    '''
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
            print("No selected genes to overlay.")
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
            print("There is no transformation matrix. Please load a transformation matrix.")
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
                print("No genes in the zoomed region")
                return
                
            x_coords = x_coords[in_zoom_region]
            y_coords = y_coords[in_zoom_region]
            genes = genes[in_zoom_region]
            
            # Adjust coordinates for zoomed view
            x_coords = (x_coords - zoom_x_start) * self.current_zoom['scale_factor']
            y_coords = (y_coords - zoom_y_start) * self.current_zoom['scale_factor']
        else:
            # Normal full-image view
            scale_factor = self.screenHeight / self.image.shape[0]
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


    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.tif *.tiff)")
        if file_name:
            self.progress_bar.show()
            self.status_bar.showMessage("Loading image...")
            QTimer.singleShot(0, lambda: self.process_image(file_name))

    def process_image(self, file_name):
        try:
            # Load the image
            self.image = cv2.imread(file_name)
            self.original_image = self.image.copy()  # Store original image

            if self.image is not None:
                # Resize the image for display
                self.resized_image = cv2.resize(
                    self.image, (0, 0), 
                    fx=self.screenHeight/self.image.shape[0], 
                    fy=self.screenHeight/self.image.shape[0]
                )
                self.update_grid_overlay()

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
            self.status_bar.showMessage("Image displayed successfully")
        else:
            self.status_bar.showMessage("Resized image is None")


if __name__ == '__main__':
    from qtpy.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
