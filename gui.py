# isort: skip

import random
import tkinter as tk
import numpy as np
import pandas as pd
import anndata as ad

from qtpy.QtCore import Qt, QTimer, QRectF, QPointF
from qtpy.QtGui import QImage, QPixmap, QPainter, QPen
from qtpy.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget, QFileDialog, QAction, QStatusBar, QToolBar,
                            QComboBox, QHBoxLayout, QPushButton, QScrollArea,
                            QFrame)
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
from cellpose import utils
from qtpy.QtWidgets import QApplication
import sys

#Helper functions:
from image_utils.zoom import ZoomMixin
from image_utils.image_loader import ImageMixin
from overlays.genes import GenesMixin
from overlays.cell_centers import CellCentersMixin
from overlays.cellpose_loader import CellposeMixin
from overlays.colors import ColorMixin

root = tk.Tk()
screen_height = root.winfo_screenheight() - 50
screen_width = root.winfo_screenwidth()

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

class MainWindow(QMainWindow, ZoomMixin, CellposeMixin, CellCentersMixin, ImageMixin, GenesMixin, ColorMixin):
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
        
        # Cellpose Mask Toggle Button
        self.toggle_cellpose_button = QPushButton("Show Cellpose Masks")
        self.toggle_cellpose_button.setCheckable(True)
        self.toggle_cellpose_button.clicked.connect(self.toggle_cellpose_masks)
        self.toggle_cellpose_button.setEnabled(False)  # Initially disabled until masks are loaded
        self.toolbar_layout.addWidget(self.toggle_cellpose_button)

        # Cellpose Outline Toggle Button
        self.toggle_cellpose_outline_button = QPushButton("Show Cellpose Outlines")
        self.toggle_cellpose_outline_button.setCheckable(True)
        self.toggle_cellpose_outline_button.clicked.connect(self.toggle_cellpose_outlines)
        self.toggle_cellpose_outline_button.setEnabled(False)
        self.toolbar_layout.addWidget(self.toggle_cellpose_outline_button)

        # Make Cluster Button
        self.make_cluster_button = QPushButton("Make Cluster Masks")
        self.make_cluster_button.setCheckable(True)
        self.make_cluster_button.clicked.connect(self.make_cluster_data)
        self.make_cluster_button.setEnabled(False)
        self.toolbar_layout.addWidget(self.make_cluster_button)

        # Outline visibility state
        self.show_cellpose_outlines = False

        # Data storage
        self.cellpose_masks = None
        self.cellpose_colors = None
        self.cellpose_outlines = None
        self.show_cellpose_masks = False

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
        self.gene_dropdown.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

        # Selected Genes Scroll Area
        self.selected_genes_scroll = QScrollArea()
        self.selected_genes_widget = QWidget()
        self.selected_genes_layout = QVBoxLayout(self.selected_genes_widget)
        self.selected_genes_scroll.setWidget(self.selected_genes_widget)
        self.selected_genes_scroll.setWidgetResizable(True)
        self.toolbar_layout.addWidget(self.selected_genes_scroll)
        
        # cluster Selection Dropdown
        self.cluster_dropdown = QComboBox()
        self.cluster_dropdown.setPlaceholderText("Select a Cluster")
        self.cluster_dropdown.currentTextChanged.connect(self.on_cluster_selected)
        self.toolbar_layout.addWidget(self.cluster_dropdown)
        self.cluster_dropdown.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

        # Selected clusters Scroll Area
        self.selected_clusters_scroll = QScrollArea()
        self.selected_clusters_widget = QWidget()
        self.selected_clusters_layout = QVBoxLayout(self.selected_clusters_widget)
        self.selected_clusters_scroll.setWidget(self.selected_clusters_widget)
        self.selected_clusters_scroll.setWidgetResizable(True)
        self.toolbar_layout.addWidget(self.selected_clusters_scroll)

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

        # Load Cellpose Masks Action
        self.load_cellpose_masks_action = QAction('Load Cellpose Masks', self)
        self.load_cellpose_masks_action.triggered.connect(self.load_cellpose_masks)
        self.file_menu.addAction(self.load_cellpose_masks_action)
        
        # Cell centers
        self.cell_centers_frame = QFrame()
        self.cell_centers_layout = QVBoxLayout(self.cell_centers_frame)

        self.cell_centers_label = QLabel("Cell Centers:")
        self.cell_centers_layout.addWidget(self.cell_centers_label)
        self.toggle_cell_centers_button = QPushButton("Show Cell Centers")
        self.toggle_cell_centers_button.setCheckable(True)
        self.toggle_cell_centers_button.clicked.connect(self.toggle_cell_centers)
        self.cell_centers_layout.addWidget(self.toggle_cell_centers_button)
        self.toolbar_layout.addWidget(self.cell_centers_frame)

        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Data Storage
        self.cluster_mask = None
        self.image = None
        self.original_image = None
        self.gene_data = None
        self.cluster_data = None
        self.transformation_matrix = None
        self.resized_image = None
        self.selected_genes = {}
        self.zoom_history = []  # Stack to track zoom levels
        self.cell_centers = None
        self.show_cell_centers = False
        self.cell_center_color = (255, 0, 0)  # Don't know why but their color scheme is flipped
        self.cell_center_size = 2  # Default size
        self.x_coords_valid = []
        self.y_coords_valid = []
        
        self.selected_clusters = {}
        self.cached_resized_mask_view = None  # cache per zoom
    
    def update_display(self):
        if self.resized_image is None:
            return
        base_image = self.resized_image.copy()
        # Overlay genes
        if hasattr(self, 'visible_gene_x_coords'):
            for x, y, color in zip(self.visible_gene_x_coords, self.visible_gene_y_coords, self.visible_gene_colors):
                # Ensure color is a tuple of integers
                bgr_color = tuple(int(c) for c in color[::-1])  # Reverse RGB to BGR and convert to int
                cv2.circle(base_image, (x, y), 1, bgr_color, -1)

        # Overlay cell centers
        if self.show_cell_centers:
            self._draw_cell_centers(base_image)

        # Overlay Cellpose masks
        if self.show_cellpose_masks and self.cellpose_masks is not None:
            self._draw_cellpose_mask_fill(base_image)

        if self.show_cellpose_outlines and self.cellpose_outlines is not None:
            self._draw_cellpose_mask_outlines(base_image)

        # Overlay cluster masks
        if self.selected_clusters is not None and self.cluster_mask is not None:
            self._draw_cluster_mask(base_image)
        
        # Display final image
        overlay_image_rgb = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        height, width, channel = overlay_image_rgb.shape
        q_img = QImage(overlay_image_rgb.data, width, height, 3 * width, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
        
    def on_cluster_selected(self, cluster):
        
        if cluster in self.selected_clusters:
            self.status_bar.showMessage("cluster already selected, choose a different cluster.")
            return
        elif not cluster:
            self.status_bar.showMessage("cluster does not exist, choose a different cluster.")
            return

        # generate a unique color
        cluster_color = self.generate_unique_cluster_color()

        # Create a cluster selection widget
        cluster_widget = QFrame()
        cluster_widget_layout = QHBoxLayout(cluster_widget)

        # Color indicator
        color_label = QLabel()
        color_label.setFixedSize(20, 20)
        color_label.setStyleSheet(
            f"background-color: rgb({cluster_color[0]}, {cluster_color[1]}, {cluster_color[2]}); border-radius: 10px;"
        )

        # cluster name label
        cluster_name_label = QLabel(cluster)

        # Remove button
        remove_button = QPushButton("cancel")
        remove_button.setFixedSize(75, 25)
        remove_button.clicked.connect(lambda _, g=cluster: self.remove_cluster_selection(g))

        cluster_widget_layout.addWidget(color_label)
        cluster_widget_layout.addWidget(cluster_name_label)
        cluster_widget_layout.addStretch()
        cluster_widget_layout.addWidget(remove_button)

        # Store cluster and color
        self.selected_clusters[cluster] = (cluster_color[0], cluster_color[1], cluster_color[2])

        # Add to selected clusters layout
        self.selected_clusters_layout.addWidget(cluster_widget)

        # Overlay clusters
        self.update_display()
        
    def remove_cluster_selection(self, cluster):
        if cluster in self.selected_clusters:
            del self.selected_clusters[cluster]

        for i in range(self.selected_clusters_layout.count()):
            widget = self.selected_clusters_layout.itemAt(i).widget()
            if widget:
                labels = widget.findChildren(QLabel)
                for label in labels:
                    if label.text() == cluster:
                        self.selected_clusters_layout.removeWidget(widget)
                        widget.hide()
                        widget.deleteLater()
                        self.overlay_clusters()
                        return
        self.overlay_clusters()

    def make_cluster_data(self):
        """Create cluster image masks with cluster,x,y data from anndata and cellpose mask indexs from cellpose masks"""
        # Step 1: Create a dictionary for mask index to cluster type in anndata
        dict_index_to_cluster = {}
        # loop through all anndata cell centers, retrieve the index and assign each index to the right cluster.
        print('make sure they are the same dimensions')
        print('masks:', self.cellpose_masks.shape)
        print('color_image', self.cellpose_mask_color_image.shape)
        
        print(max(self.cell_centers['global_x']), max(self.cell_centers['global_y']))
        
        for _, cell_center in self.cell_centers.iterrows():
            curr_index = self.cellpose_masks[int(cell_center['global_x']), int(cell_center['global_y'])]
            dict_index_to_cluster[curr_index] = cell_center['cluster']

        # Step 2: Create a mask where each pixel's value is the cluster id, or 0 if not in dict
        
        # Initialize the mask
        mask_shape = self.cellpose_masks.shape
        cluster_mask = np.zeros(mask_shape, dtype=np.int32)
        print("Finished initializing mask")
        # Vectorized mapping: create a lookup array for mask indices
        max_index = int(self.cellpose_masks.max())
        
        lookup = np.zeros(max_index + 1, dtype=np.int32)
        for idx, cluster in dict_index_to_cluster.items():
            if idx > 0 and idx <= max_index:
                lookup[int(idx)] = int(cluster)

        # Map mask indices to cluster ids
        cluster_mask = lookup[self.cellpose_masks.astype(np.int32)]
        print('First row of cluster mask', cluster_mask[0])
        print('sum of row of cluster mask', sum(cluster_mask[1]))
        
        self.cluster_mask = cluster_mask
        return
        
    def _draw_cluster_mask(self, base_image):
        """Overlay selected clusters on the current image."""
        if not hasattr(self, 'cluster_mask') or self.cluster_mask is None or not self.selected_clusters:
            return
        print('drawing cluster on mask')
        
        # Prepare a color image for the cluster mask
        mask = self.cluster_mask
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

        # Assign colors to selected clusters
        for cluster, color in self.selected_clusters.items():
            print(f'Cluster {cluster} was selected and given color {color}')
            cluster_id = int(cluster)
            color_mask[mask == cluster_id] = color  # color is (r, g, b)

        # Handle zoom/crop
        if hasattr(self, 'current_zoom') and self.current_zoom is not None:
            zoom = self.current_zoom
            crop = color_mask[
                zoom['y_start']:zoom['y_end'],
                zoom['x_start']:zoom['x_end']
            ]
        else:
            crop = color_mask

        if crop.size == 0:
            return

        # Resize to match base_image
        resized = cv2.resize(crop, (base_image.shape[1], base_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Blend with base image
        if resized.dtype != np.uint8:
            resized = resized.astype(np.uint8)
        if base_image.dtype != np.uint8:
            base_image[:] = base_image.astype(np.uint8)
        cv2.addWeighted(base_image, 0.5, resized, 0.5, 0, dst=base_image)
        print('finished adding clusters')
       
        

if __name__ == '__main__':
    from qtpy.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())