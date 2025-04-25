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
                            QMenuBar, QAction, QStatusBar, QToolBar,
                            QComboBox, QHBoxLayout, QPushButton, QScrollArea,
                            QFrame, QColorDialog, QSlider)

root = tk.Tk()
screen_height = root.winfo_screenheight() - 50
screen_width = root.winfo_screenwidth()
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2 

from cellpose import utils, plot, io as cp_io
import tifffile
from ui.zoomable_image_label import ZoomableImageLabel
from resources.colors import colors_rgb
from logic.gene_overlay import (
    on_gene_selected,
    remove_gene_selection,
    overlay_genes,
)
from logic.cell_centers import (
    toggle_cell_centers,
    display_cell_centers,
)
from logic.image_processing import process_image, display_image, resize_image_to_fit
from logic.zoom_utils import zoom_to_selection, reset_zoom, get_pixmap_rect



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
        self.reset_zoom_button.clicked.connect(lambda: reset_zoom(self))
        self.reset_zoom_button.setEnabled(False)
        self.zoom_controls_layout.addWidget(self.reset_zoom_button)
        
        self.toolbar_layout.addWidget(self.zoom_controls_frame)

        # Gene Selection Dropdown
        self.gene_dropdown = QComboBox()
        self.gene_dropdown.setPlaceholderText("Select a Gene")
        self.gene_dropdown.currentTextChanged.connect(lambda gene: on_gene_selected(self, gene))
        self.toolbar_layout.addWidget(self.gene_dropdown)
        self.gene_dropdown.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

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
        
        self.load_anndata_action = QAction('Load Anndata Cell Centers', self)
        self.load_anndata_action.triggered.connect(self.load_anndata)
        self.file_menu.addAction(self.load_anndata_action)

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
        
        self.cell_centers_frame = QFrame()
        self.cell_centers_layout = QVBoxLayout(self.cell_centers_frame)

        self.cell_centers_label = QLabel("Cell Centers:")
        self.cell_centers_layout.addWidget(self.cell_centers_label)

        self.toggle_cell_centers_button = QPushButton("Show Cell Centers")
        self.toggle_cell_centers_button.setCheckable(True)
        self.toggle_cell_centers_button.clicked.connect(lambda: toggle_cell_centers(self))
        self.cell_centers_layout.addWidget(self.toggle_cell_centers_button)

        self.toolbar_layout.addWidget(self.cell_centers_frame)

        # Add to data storage section
        self.show_cell_centers = False
        self.cell_center_color = (255, 0, 0)  # Don't know why but their color scheme is flipped
        self.cell_center_size = 2  # Default size
        self.x_coords_valid = []
        self.y_coords_valid = []
        
        # plotting the cellpose mask
        self.cell_mask = None
        self.cell_mask_rgb = None
        self.cell_mask_outlines = None
        self.show_cell_mask = False
        self.show_cell_outlines = False
        
        # menu items for mask: 
        self.load_mask_action = QAction("Load Cell Mask", self)
        self.load_mask_action.triggered.connect(self.load_cell_mask)
        self.file_menu.addAction(self.load_mask_action)

        self.toggle_mask_frame = QFrame()
        self.toggle_mask_layout = QVBoxLayout(self.toggle_mask_frame)

        self.toggle_mask_label = QLabel("Masks:")
        self.toggle_mask_layout.addWidget(self.toggle_mask_label)

        self.toggle_mask_button = QPushButton("Overlay Masks")
        self.toggle_mask_button.setCheckable(True)
        self.toggle_mask_button.clicked.connect(self.toggle_mask_overlay)
        self.toggle_mask_layout.addWidget(self.toggle_mask_button)
        
        self.toolbar_layout.addWidget(self.toggle_mask_frame)
        
        self.toggle_outline_frame = QFrame()
        self.toggle_outline_layout = QVBoxLayout(self.toggle_outline_frame)

        self.toggle_outline_label = QLabel("Outlines:")
        self.toggle_outline_layout.addWidget(self.toggle_outline_label)

        self.toggle_outline_button = QPushButton("Outline Cell")
        self.toggle_outline_button.setCheckable(True)
        self.toggle_outline_button.clicked.connect(self.toggle_outline_overlay)
        self.toggle_outline_layout.addWidget(self.toggle_outline_button)

        self.toolbar_layout.addWidget(self.toggle_outline_frame)

        
    def toggle_cell_centers(self):
        """Toggle display of cell centers"""
        self.show_cell_centers = self.toggle_cell_centers_button.isChecked()
        
        if self.show_cell_centers:
            self.toggle_cell_centers_button.setText("Hide Cell Centers")
            cell_centers = getattr(self, 'cell_centers', None)
            if cell_centers is not None:
                if self.image is not None:
                    display_cell_centers(self)
                else:
                    self.status_bar.showMessage("Please load an image first")
            else:
                self.status_bar.showMessage("No cell centers loaded. Please load anndata file first.")
        else:
            self.toggle_cell_centers_button.setText("Show Cell Centers")
            # If we're hiding cells, redisplay the image without cells
            if self.image is not None:
                display_image(self)
                # Reapply gene overlay if we have genes selected
                if self.gene_data is not None and self.selected_genes:
                    overlay_genes(self)

    
    def _process_cell_centers(self):
        """Process cell center coordinates for the current view."""
        if not hasattr(self, 'cell_centers') or self.cell_centers is None or self.cell_centers.empty:
            return
        
        # Process and overlay cell centers
        x_coords, y_coords = self.cell_centers[['global_x', 'global_y']].to_numpy().T
        
        if self.transformation_matrix is not None:
            coords = np.dot(self.transformation_matrix, np.hstack([x_coords[:, None], y_coords[:, None], np.ones((len(x_coords), 1))]).T).T[:, :2]
            x_coords, y_coords = coords[:, 0], coords[:, 1]
        
        if getattr(self, 'current_zoom', None):
            zoom = self.current_zoom
            in_zoom = (zoom['x_start'] <= x_coords) & (x_coords < zoom['x_end']) & \
                    (zoom['y_start'] <= y_coords) & (y_coords < zoom['y_end'])
            if not any(in_zoom):
                self.cell_center_visible = False
                return
            x_coords, y_coords = (x_coords[in_zoom] - zoom['x_start']) * zoom['scale_factor'], \
                                (y_coords[in_zoom] - zoom['y_start']) * zoom['scale_factor']
        else:
            scale_factor = getattr(self, 'full_view_scale_factor', None) or min(
                self.image_label.height() / self.original_image.shape[0],
                self.image_label.width() / self.original_image.shape[1])
            x_coords, y_coords = x_coords * scale_factor, y_coords * scale_factor
        
        x_coords, y_coords = x_coords.astype(int), y_coords.astype(int)
        
        # Filter valid coordinates
        height, width = self.resized_image.shape[:2]
        valid = (0 <= x_coords) & (x_coords < width) & (0 <= y_coords) & (y_coords < height)
        
        self.cell_center_x_coords = x_coords[valid]
        self.cell_center_y_coords = y_coords[valid]
        self.cell_center_visible = valid.sum() > 0


    def _draw_cell_centers(self, image):
        """Draw cell centers on the given image and display it."""
        if not hasattr(self, 'cell_center_x_coords') or not hasattr(self, 'cell_center_y_coords'):
            self._process_cell_centers()
        
        if hasattr(self, 'cell_center_x_coords') and hasattr(self, 'cell_center_y_coords'):
            for x, y in zip(self.cell_center_x_coords, self.cell_center_y_coords):
                cv2.circle(image, (x, y), self.cell_center_size, self.cell_center_color, -1)
        
        # Convert and display the final image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
        
        num_points = len(getattr(self, 'cell_center_x_coords', []))
        self.status_bar.showMessage(f"Cell centers displayed: {num_points} visible points")
    

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
            display_image(self)
            
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
            if self.gene_data is not None and hasattr(self, 'selected_genes') and self.selected_genes:
                overlay_genes(self)
            # If no genes but showing cell centers, display them
            elif self.show_cell_centers:
                display_cell_centers(self)
                
            self.status_bar.showMessage("View reset to original")

            
    def load_transformation_matrix(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.status_bar.showMessage("Loading Transformation Matrix...")
            QTimer.singleShot(0, lambda: self.process_csv(file_name))

    def load_detected_transcripts(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.status_bar.showMessage(
                "Loading Detected Transcripts...(this may take a while)")
            QTimer.singleShot(0, lambda: self.process_csv(file_name))
    
    def load_anndata(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Anndata File", "", "Anndata Files (*.h5ad);"
            "All Files (*)")
        if file_name:
            self.status_bar.showMessage(
                "Loading Anndata...")
            QTimer.singleShot(0, lambda: self.process_anndata(file_name))
            
    def process_anndata(self, file_name):
        """Process anndata file to extract cell centers"""
        try:
            import anndata as ad
            adata = ad.read_h5ad(file_name)
            self.status_bar.showMessage("AnnData loaded successfully")
            
            # Check for spatial coordinates in different possible locations
            if 'spatial' in adata.obsm:
                cell_coords = adata.obsm['spatial']
                x_coords = cell_coords[:, 0]
                y_coords = cell_coords[:, 1]
            elif 'X_spatial' in adata.obsm:
                cell_coords = adata.obsm['X_spatial']
                x_coords = cell_coords[:, 0]
                y_coords = cell_coords[:, 1]
            elif 'center_x' in adata.obs and 'center_y' in adata.obs:
                x_coords = adata.obs['center_x'].values
                y_coords = adata.obs['center_y'].values
            elif 'x' in adata.obs and 'y' in adata.obs:
                x_coords = adata.obs['x'].values
                y_coords = adata.obs['y'].values
            else:
                # Last resort: try to find any columns that might contain coordinates
                potential_x_cols = [col for col in adata.obs.columns if 'x' in col.lower()]
                potential_y_cols = [col for col in adata.obs.columns if 'y' in col.lower()]
                
                if potential_x_cols and potential_y_cols:
                    x_coords = adata.obs[potential_x_cols[0]].values
                    y_coords = adata.obs[potential_y_cols[0]].values
                    self.status_bar.showMessage(f"Using columns '{potential_x_cols[0]}' and '{potential_y_cols[0]}' for coordinates")
                else:
                    self.status_bar.showMessage("Could not find cell center coordinates in AnnData file")
                    return
            
            # Create DataFrame to store cell centers
            self.cell_centers = pd.DataFrame({
                'global_x': x_coords,
                'global_y': y_coords
            })
            
            num_cells = len(self.cell_centers)
            self.status_bar.showMessage(f"Loaded {num_cells} cell centers from AnnData file")
            
            # Enable the cell centers button
            self.toggle_cell_centers_button.setEnabled(True)
            
            # If already toggled to show cells and we have an image, display them
            cell_centers = getattr(self, 'cell_centers', None)
            if (cell_centers is not None and not cell_centers.empty) and (self.image is not None):
                display_cell_centers(self)
                
        except ImportError:
            self.status_bar.showMessage("Please install the 'anndata' package to load AnnData files using `pip install anndata`")
        except Exception as e:
            self.status_bar.showMessage(f"Error processing AnnData file: {str(e)}")
            print(f"Error processing AnnData file: {str(e)}")
            
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
                self.gene_data = pd.read_csv(file_name)

                unique_genes = self.gene_data['gene'].unique()
                self.gene_dropdown.clear()
                self.gene_dropdown.addItems(unique_genes)

                if self.image is not None:
                    overlay_genes(self)
                    
                self.status_bar.showMessage("Gene data loaded successfully")
        except Exception as e:
            self.status_bar.showMessage(
                f"Error loading file {file_name}: {str(e)}")
    
    def load_image(self):
        self.status_bar.showMessage(f"Checking Image...")
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.tif *.tiff)")
        self.status_bar.showMessage(f"Opening File...")
        if file_name:
            self.status_bar.showMessage("Loading image...")
            QTimer.singleShot(100, lambda: process_image(self, file_name))

    def process_image(self, file_name):
        try:
            # Create an image pyramid for large images
            if file_name.lower().endswith(('.tif', '.tiff')):
                cv2.setNumThreads(0)
                cv2.setUseOptimized(True)
                cv2.utils.setOpenCVIOMaxImagePixels(pow(2, 40))
                self.image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
            
            self.original_image = self.image.copy()

            if self.image is not None:
                # Instead of using fixed screen dimensions, use the actual image label dimensions
                self.reset_zoom_button.setEnabled(False)  # Disable zoom reset initially
                self.do_full_reset()  # This will properly resize the image to fit
                
                # If gene data is already loaded, reoverlay genes
                if self.gene_data is not None:
                    overlay_genes(self)

                self.status_bar.showMessage("Image loaded and resized successfully")
            else:
                self.status_bar.showMessage("Failed to load image")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading image: {str(e)}")
            print(f"Error loading image: {str(e)}")

    def resizeEvent(self, event):
        """Handle window resize events to adjust the image size"""
        super().resizeEvent(event)
        
        # If we have an image loaded, resize it to fit the new dimensions
        if hasattr(self, 'original_image') and self.original_image is not None:
            # Delay the resize slightly to ensure all UI components have updated their sizes
            QTimer.singleShot(50, lambda: resize_image_to_fit(self))

    def load_cell_mask(self):
        mask_file, _ = QFileDialog.getOpenFileName(
            self, "Open Cellpose Segmentation (.npy)", "", "NPY Files (*.npy)"
        )
        if not mask_file:
            return

        try:
            self.status_bar.showMessage("Loading mask...")
            loaded_data = np.load(mask_file, allow_pickle=True)
            
            print(f"Loaded data type: {type(loaded_data)}")
            
                    
            #     self.cell_mask_rgb = plot.mask_overlay(img, dat['masks'], colors=colors, alpha=0.3)
            #     self.cell_mask_outlines = utils.outlines_list(dat['masks'])
                
            #     self.show_cell_mask = True
            #     self.show_cell_outlines = False
            #     self.toggle_mask_button.setChecked(True)
            #     self.toggle_outline_button.setChecked(False)
                
            #     self.status_bar.showMessage("Mask loaded successfully.")
            #     self.redraw_image()
            # else:
            #     self.status_bar.showMessage("Invalid mask file format: 'masks' key not found")
                
        except Exception as e:
            self.status_bar.showMessage(f"Error loading mask: {str(e)}")
            print(f"Error loading mask: {e}")
            import traceback
            traceback.print_exc()
            
    def redraw_image(self):
        if self.resized_image is None:
            return

        display = self.resized_image.copy()

        # Overlay genes
        if hasattr(self, 'visible_gene_x_coords'):
            for x, y, color in zip(self.visible_gene_x_coords,
                                self.visible_gene_y_coords,
                                self.visible_gene_colors):
                bgr = (int(color[2]), int(color[1]), int(color[0]))
                cv2.circle(display, (x, y), 1, bgr, -1)

        # Overlay cell centers
        if self.show_cell_centers:
            self._draw_cell_centers(display)

        # Overlay cell mask RGB
        if self.show_cell_mask and self.cell_mask_rgb is not None:
            # Resize mask_rgb to match display
            mask_resized = cv2.resize(self.cell_mask_rgb, (display.shape[1], display.shape[0]))
            display = cv2.addWeighted(display, 1.0, mask_resized, 0.3, 0)

        # Draw outlines
        if self.show_cell_outlines and self.cell_mask_outlines is not None:
            for outline in self.cell_mask_outlines:
                pts = outline.astype(np.int32)
                cv2.polylines(display, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

        # Display final image
        image_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
        
    def toggle_mask_overlay(self, checked):
        self.show_cell_mask = checked
        self.redraw_image()

    def toggle_outline_overlay(self, checked):
        self.show_cell_outlines = checked
        self.redraw_image()


        
if __name__ == '__main__':
    from qtpy.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())