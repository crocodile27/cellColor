# isort: skip
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

from .overlays.colors import ColorMixin
from .overlays.cellpose_loader import CellposeMixin
from .overlays.cell_centers import CellCentersMixin
from .overlays.genes import GenesMixin
from .image_utils.image_loader import ImageMixin
from .image_utils.zoom import ZoomMixin
import sys
from qtpy.QtWidgets import QApplication
from cellpose import utils
import cv2
import random
import tkinter as tk
import numpy as np
import pandas as pd
import anndata as ad
import glob

from qtpy.QtCore import Qt, QTimer, QRectF, QPointF
from qtpy.QtGui import QImage, QPixmap, QPainter, QPen
from qtpy.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget, QFileDialog, QAction, QStatusBar, QToolBar,
                            QComboBox, QHBoxLayout, QPushButton, QScrollArea, QMessageBox,
                            QFrame)


# Helper functions:

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
            self.rubberband_rect = QRectF(
                self.origin, event.pos()).normalized()
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

        # Zoom Controls
        self.zoom_controls_frame = QFrame()
        self.zoom_controls_layout = QVBoxLayout(self.zoom_controls_frame)

        self.zoom_label = QLabel("Zoom Instructions:")
        self.zoom_instructions = QLabel(
            "Click and drag to select an area to zoom into")
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
        self.gene_dropdown.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)

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
        self.cluster_dropdown.currentTextChanged.connect(
            self.on_cluster_selected)
        self.toolbar_layout.addWidget(self.cluster_dropdown)
        self.cluster_dropdown.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)

        # Selected clusters Scroll Area
        self.selected_clusters_scroll = QScrollArea()
        self.selected_clusters_widget = QWidget()
        self.selected_clusters_layout = QVBoxLayout(
            self.selected_clusters_widget)
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

        # Add separator
        self.file_menu.addSeparator()

        # Auto Load Files Action
        self.auto_load_files_action = QAction('Auto Load Files', self)
        self.auto_load_files_action.triggered.connect(self.auto_load_files)
        self.file_menu.addAction(self.auto_load_files_action)

        # Cell centers
        self.toggle_cell_centers_button = QPushButton("Show Cell Centers")
        self.toggle_cell_centers_button.setCheckable(True)
        self.toggle_cell_centers_button.clicked.connect(
            self.toggle_cell_centers)
        self.toggle_cell_centers_button.setEnabled(False)
        self.toolbar_layout.addWidget(self.toggle_cell_centers_button)

        # Cellpose Mask Toggle Button
        self.toggle_cellpose_button = QPushButton("Show Cellpose Masks")
        self.toggle_cellpose_button.setCheckable(True)
        self.toggle_cellpose_button.clicked.connect(self.toggle_cellpose_masks)
        # Initially disabled until masks are loaded
        self.toggle_cellpose_button.setEnabled(False)
        self.toolbar_layout.addWidget(self.toggle_cellpose_button)

        # Cellpose Outline Toggle Button
        self.toggle_cellpose_outline_button = QPushButton(
            "Show Cellpose Outlines")
        self.toggle_cellpose_outline_button.setCheckable(True)
        self.toggle_cellpose_outline_button.clicked.connect(
            self.toggle_cellpose_outlines)
        self.toggle_cellpose_outline_button.setEnabled(False)
        self.toolbar_layout.addWidget(self.toggle_cellpose_outline_button)

        # Outline visibility state
        self.show_cellpose_outlines = False

        # Data storage
        self.cellpose_masks = None
        self.cellpose_colors = None
        self.cellpose_outlines = None
        self.show_cellpose_masks = False

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        # Dimensions
        self.view_height = self.image_label.height()
        self.view_width = self.image_label.width()
        self.orig_height = None
        self.orig_width = None
        # Data Storage
        self.cluster_mask = None
        self.image = None
        self.cluster_to_id = None
        self.original_image = None
        self.gene_data = None
        self.cluster_data = None
        self.transformation_matrix = None
        self.resized_image = None
        self.selected_genes = {}
        self.zoom_history = []  # Stack to track zoom levels
        self.cell_centers = None
        self.show_cell_centers = False
        self.visible_gene_x_coords = None
        self.visible_gene_y_coords = None
        self.visible_gene_colors = None
        self.cell_center_x_coords = None
        self.cell_center_y_coords = None
        self.cell_center_colors = None
        # Don't know why but their color scheme is flipped
        self.cell_center_color = (255, 0, 0)
        self.cell_center_size = 2  # Default size
        self.x_coords_valid = []
        self.y_coords_valid = []
        self.region = None
        self.run = None

        self.selected_clusters = {}
        self.cached_resized_mask_view = None  # cache per zoom

    def update_display(self):
        if self.resized_image is None:
            return
        base_image = self.resized_image.copy()
        # Overlay genes
        if self.selected_genes is not None:
            if hasattr(self, 'visible_gene_x_coords') and self.visible_gene_x_coords is not None:
                for x, y, color in zip(self.visible_gene_x_coords, self.visible_gene_y_coords, self.visible_gene_colors):
                    # Ensure color is a tuple of integers
                    # Reverse RGB to BGR and convert to int
                    bgr_color = tuple(int(c) for c in color[::-1])
                    cv2.circle(base_image, (x, y), 1, bgr_color, -1)
            else:
                self.filter_genes()
                if self.visible_gene_x_coords is not None:
                    for x, y, color in zip(self.visible_gene_x_coords, self.visible_gene_y_coords, self.visible_gene_colors):
                        # Ensure color is a tuple of integers
                        # Reverse RGB to BGR and convert to int
                        bgr_color = tuple(int(c) for c in color[::-1])
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
        q_img = QImage(overlay_image_rgb.data, width, height,
                       3 * width, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def on_cluster_selected(self, cluster):

        # Check if cluster is already selected (using cluster_id as key)
        if cluster in self.cluster_to_id and self.cluster_to_id[cluster] in self.selected_clusters:
            self.status_bar.showMessage(
                "cluster already selected, choose a different cluster.")
            return
        elif not cluster:
            self.status_bar.showMessage(
                "cluster does not exist, choose a different cluster.")
            return

        # Check if cluster exists in cluster_to_id mapping
        if not hasattr(self, 'cluster_to_id') or self.cluster_to_id is None:
            self.status_bar.showMessage(
                "No cluster mapping available. Please create cluster masks first.")
            return

        if cluster not in self.cluster_to_id:
            self.status_bar.showMessage(
                f"Cluster '{cluster}' not found in current data. Available clusters: {list(self.cluster_to_id.keys())}")
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
        remove_button.clicked.connect(
            lambda _, g=cluster: self.remove_cluster_selection(g))

        cluster_widget_layout.addWidget(color_label)
        cluster_widget_layout.addWidget(cluster_name_label)
        cluster_widget_layout.addStretch()
        cluster_widget_layout.addWidget(remove_button)

        # Store cluster and color
        cluster_id = self.cluster_to_id[cluster]
        self.selected_clusters[cluster_id] = (
            cluster_color[2], cluster_color[1], cluster_color[0])

        # Add to selected clusters layout
        self.selected_clusters_layout.addWidget(cluster_widget)

        # Overlay clusters
        self.update_display()

    def remove_cluster_selection(self, cluster):

        # Check if cluster exists in cluster_to_id mapping
        if cluster in self.cluster_to_id:
            cluster_id = self.cluster_to_id[cluster]
            if cluster_id in self.selected_clusters:
                del self.selected_clusters[cluster_id]

        # Remove the widget from the UI
        for i in range(self.selected_clusters_layout.count()):
            widget = self.selected_clusters_layout.itemAt(i).widget()
            if widget:
                labels = widget.findChildren(QLabel)
                for label in labels:
                    if label.text() == cluster:
                        self.selected_clusters_layout.removeWidget(widget)
                        widget.hide()
                        widget.deleteLater()
                        self.update_display()
                        return

        self.update_display()

    def make_cluster_data(self):
        """
        Efficient mapping from cellpose mask index -> cluster id using cell_id.
        Uses cell_id to mask_id direct mapping for one-pass mask creation.
        Stores result in self.cluster_mask (int32).
        """
        print("\n=== DEBUG: make_cluster_data started ===")
        print(f"DEBUG: cell_centers available: {self.cell_centers is not None}")
        if self.cell_centers is not None:
            print(f"DEBUG: cell_centers shape: {self.cell_centers.shape}")
            print(f"DEBUG: cell_centers columns: {list(self.cell_centers.columns)}")

        self.cluster_dropdown.clear()
        if self.cell_centers is not None and "cluster" in self.cell_centers.columns:
            unique_clusters = pd.Series(
                self.cell_centers["cluster"]).dropna().astype(str).unique()
            print(f"DEBUG: Found {len(unique_clusters)} unique clusters: {unique_clusters[:10]}")
            self.cluster_dropdown.addItems(list(map(str, unique_clusters)))
        else:
            print("DEBUG: No cluster column found in cell_centers")
            self.cluster_dropdown.addItem("No cluster annotations found")

        # Try to load cached cluster mask and mappings for faster startup
        try:
            # Base path from cellpose masks if available; otherwise fall back to original image path
            cache_base = None
            if hasattr(self, 'cellpose_mask_base_path') and self.cellpose_mask_base_path:
                cache_base = self.cellpose_mask_base_path
            elif hasattr(self, 'original_image_path') and self.original_image_path:
                cache_base = os.path.splitext(self.original_image_path)[0]

            if cache_base is not None:
                cluster_mask_path = cache_base + "_cluster_mask.npy"
                cluster_map_path = cache_base + "_cluster_map.csv"
                if os.path.exists(cluster_mask_path) and os.path.exists(cluster_map_path):
                    loaded_mask = np.load(cluster_mask_path)
                    # Only accept if shape matches current masks (after scaling)
                    if (hasattr(self, 'cellpose_masks') and self.cellpose_masks is not None and
                            loaded_mask.shape == self.cellpose_masks.shape):
                        self.cluster_mask = loaded_mask.astype(np.int32)
                        # Load mapping
                        df_map = pd.read_csv(cluster_map_path)
                        # Expect two columns: 'cluster_name','cluster_id'
                        if {'cluster_name', 'cluster_id'}.issubset(df_map.columns):
                            self.cluster_to_id = {
                                row['cluster_name']: int(row['cluster_id']) for _, row in df_map.iterrows()
                            }
                            self.cluster_id_to_name = {
                                int(row['cluster_id']): row['cluster_name'] for _, row in df_map.iterrows()
                            }

                            # Check if cached clusters match current clusters
                            if 'unique_clusters' in locals():
                                current_clusters = set(unique_clusters)
                                cached_clusters = set(
                                    self.cluster_to_id.keys())
                                if current_clusters != cached_clusters:

                                    # Clear the cached data and continue to regenerate
                                    self.cluster_mask = None
                                    self.cluster_to_id = None
                                    self.cluster_id_to_name = None

                                    # Also clear the cache files to prevent future mismatches
                                    try:
                                        if os.path.exists(cluster_mask_path):
                                            os.remove(cluster_mask_path)
                                    except Exception as e:
                                        print(
                                            f"Failed to remove cache files: {e}")
        except Exception as e:
            print(f"Failed loading cached cluster mask: {e}")

        # Check if we have cell_id information for efficient mapping
        print(f"DEBUG: Checking for cell_id-based mapping...")
        print(f"DEBUG: cellpose_masks available: {hasattr(self, 'cellpose_masks') and self.cellpose_masks is not None}")
        if hasattr(self, 'cellpose_masks') and self.cellpose_masks is not None:
            print(f"DEBUG: cellpose_masks shape: {self.cellpose_masks.shape}")
            print(f"DEBUG: cellpose_masks max value: {self.cellpose_masks.max()}")
            print(f"DEBUG: cellpose_masks unique values count: {len(np.unique(self.cellpose_masks))}")

        if (self.cell_centers is not None and
            "cell_id" in self.cell_centers.columns and
                "cluster" in self.cell_centers.columns):

            print("DEBUG: cell_id column found, using cell_id-based method")
            # Get valid data (non-null cell_id and cluster)
            valid_data = self.cell_centers.dropna(
                subset=['cell_id', 'cluster'])

            print(f"DEBUG: Valid data rows (non-null cell_id and cluster): {len(valid_data)}")
            if len(valid_data) > 0:
                print(f"DEBUG: Sample cell_ids: {valid_data['cell_id'].head().tolist()}")
                print(f"DEBUG: cell_id range: {valid_data['cell_id'].min()} to {valid_data['cell_id'].max()}")

            if len(valid_data) == 0:
                print("DEBUG: No valid cell_id data, falling through to coordinate-based method")
                # Fall through to coordinate-based method
                pass
            else:
                # Create cell_id to cluster mapping
                cell_id_to_cluster = dict(
                    zip(valid_data['cell_id'], valid_data['cluster']))

                # Create cluster name to integer ID mapping
                unique_clusters = valid_data['cluster'].unique()
                self.cluster_to_id = {cluster: idx + 1 for idx,
                                      cluster in enumerate(unique_clusters)}
                self.cluster_id_to_name = {
                    v: k for k, v in self.cluster_to_id.items()}

                # Get max mask index for lookup table size
                max_mask_index = int(self.cellpose_masks.max())
                lookup = np.zeros(max_mask_index + 1, dtype=np.int32)

                # Create cell_id to cluster_id mapping
                cell_id_to_cluster_id = {}
                for cell_id, cluster in cell_id_to_cluster.items():
                    if cluster in self.cluster_to_id:
                        cell_id_to_cluster_id[cell_id] = self.cluster_to_id[cluster]

                # Fill lookup table using cell_id as mask_id
                for cell_id, cluster_id in cell_id_to_cluster_id.items():
                    if 1 <= cell_id <= max_mask_index:  # Valid mask index range
                        lookup[cell_id] = cluster_id

                # Create cluster mask by mapping cellpose_masks through lookup table
                self.cluster_mask = np.take(
                    lookup, self.cellpose_masks.astype(np.int32))

                print(f"DEBUG: cell_id-based method succeeded")
                print(f"DEBUG: cluster_mask shape: {self.cluster_mask.shape}")
                print(f"DEBUG: cluster_mask unique values: {len(np.unique(self.cluster_mask))}")
                # Skip the fallback method if cell_id method succeeded
                return

        # Fallback to original coordinate-based method if no cell_id available or cell_id method failed
        print("\n=== DEBUG: Using coordinate-based method ===")

        if not hasattr(self, 'cellpose_masks') or self.cellpose_masks is None:
            print("ERROR: cellpose_masks not loaded! Cannot create cluster data.")
            return

        if self.cell_centers is None:
            print("ERROR: cell_centers not loaded! Cannot create cluster data.")
            return

        if 'global_x' not in self.cell_centers.columns or 'global_y' not in self.cell_centers.columns:
            print(f"ERROR: Required columns 'global_x' and 'global_y' not found in cell_centers!")
            print(f"Available columns: {list(self.cell_centers.columns)}")
            return

        xs = self.cell_centers['global_x'].to_numpy().astype(np.intp)
        ys = self.cell_centers['global_y'].to_numpy().astype(np.intp)
        clusters = self.cell_centers['cluster']

        print(f"DEBUG: Number of cell centers: {len(xs)}")
        print(f"DEBUG: X coordinate range: {xs.min()} to {xs.max()}")
        print(f"DEBUG: Y coordinate range: {ys.min()} to {ys.max()}")

        H, W = self.cellpose_masks.shape
        print(f"DEBUG: Cellpose mask dimensions: H={H}, W={W}")

        # ensure coords are in bounds (clip avoids IndexError)
        xs = np.clip(xs, 0, H - 1)
        ys = np.clip(ys, 0, W - 1)

        print(f"DEBUG: After clipping - X range: {xs.min()} to {xs.max()}")
        print(f"DEBUG: After clipping - Y range: {ys.min()} to {ys.max()}")

        # Vectorized fetch of mask indices for all centers at once
        mask_indices = self.cellpose_masks[xs, ys]
        print(f"DEBUG: Mask indices at cell centers - range: {mask_indices.min()} to {mask_indices.max()}")
        print(f"DEBUG: Number of non-zero mask indices: {np.count_nonzero(mask_indices)}")

        # Build lookup table (index -> cluster). Use max on the mask once.
        max_index = int(self.cellpose_masks.max())
        lookup = np.zeros(max_index + 1, dtype=np.int32)

        # Only set for valid indices (ignore background 0 and out-of-range)
        valid = (mask_indices > 0) & (mask_indices <= max_index)

        print(f"DEBUG: Valid mask indices: {np.count_nonzero(valid)} out of {len(valid)}")
        if np.count_nonzero(valid) > 0:
            print(f"DEBUG: Sample valid mask indices: {mask_indices[valid][:10]}")

        if np.any(valid):
            # Handle string cluster names by creating a mapping to integer IDs
            unique_clusters = clusters[valid].unique()
            print(f"DEBUG: Creating cluster mapping for {len(unique_clusters)} clusters")
            self.cluster_to_id = {cluster: idx + 1 for idx,
                                  cluster in enumerate(unique_clusters)}

            # Convert cluster names to integer IDs
            cluster_ids = clusters[valid].map(
                self.cluster_to_id).astype(np.int32)

            # Store the mapping for later use in cluster selection
            self.cluster_name_to_id = self.cluster_to_id
            self.cluster_id_to_name = {
                v: k for k, v in self.cluster_to_id.items()}

            # np.put is vectorized and avoids Python loops
            np.put(lookup, mask_indices[valid].astype(
                np.intp), cluster_ids)
            print(f"DEBUG: Successfully mapped {np.count_nonzero(valid)} cells to clusters")
        else:
            print("ERROR: No valid mask indices found in coordinate-based method!")
            print("This usually means:")
            print("  1. Cell center coordinates don't align with cellpose mask coordinates")
            print("  2. Coordinate systems are different (e.g., X/Y swapped or different scaling)")
            print("  3. The cellpose masks are all zeros at the cell center locations")

        # Map whole mask at once
        self.cluster_mask = np.take(
            lookup, self.cellpose_masks.astype(np.int32))

        # Persist the computed cluster mask and mapping for faster reload next time
        try:
            if 'cache_base' not in locals() or cache_base is None:
                if hasattr(self, 'cellpose_mask_base_path') and self.cellpose_mask_base_path:
                    cache_base = self.cellpose_mask_base_path
                elif hasattr(self, 'original_image_path') and self.original_image_path:
                    cache_base = os.path.splitext(self.original_image_path)[0]
            if cache_base is not None:
                cluster_mask_path = cache_base + "_cluster_mask.npy"
                cluster_map_path = cache_base + "_cluster_map.csv"
                # Ensure directory exists
                os.makedirs(os.path.dirname(cluster_mask_path), exist_ok=True)
                np.save(cluster_mask_path, self.cluster_mask.astype(np.int32))
                # Save mapping
                if hasattr(self, 'cluster_to_id') and self.cluster_to_id is not None:
                    df_map = pd.DataFrame([
                        {"cluster_name": k, "cluster_id": v}
                        for k, v in self.cluster_to_id.items()
                    ])
                    df_map.to_csv(cluster_map_path, index=False)
        except Exception as e:
            print(f"Failed saving cached cluster mask: {e}")

        return

    def _draw_cluster_mask(self, base_image):
        """
        Create a colored overlay for selected clusters and blend with base_image.
        Improvements:
        - Crop before coloring (if zoomed) to avoid full-image allocation.
        - Use a small colors lookup and vectorized indexing to build the RGB crop.
        """
        if not hasattr(self, 'cluster_mask') or self.cluster_mask is None or not self.selected_clusters:
            return

        mask = self.cluster_mask

        # Crop according to zoom (do this early so we only color a small area)
        if hasattr(self, 'current_zoom') and self.current_zoom is not None:
            z = self.current_zoom
            y0, y1 = z['y_start'], z['y_end']
            x0, x1 = z['x_start'], z['x_end']
            mask_crop = mask[y0:y1, x0:x1]
        else:
            mask_crop = mask

        if mask_crop.size == 0:
            return

        # If only a few clusters are selected and mask_crop is large,
        # using boolean assignment per-cluster can be faster/memory-savvier than building a huge colors_lookup.
        sel_ids = [k for k in self.selected_clusters.keys()]

        # Strategy A: vectorized lookup for compact max cluster id (fast if max cluster id small)
        max_cluster_id = int(mask_crop.max())
        if max_cluster_id <= 5000:  # heuristic: avoid huge lookup arrays if cluster ids are sparse and very large
            colors_lookup = np.zeros((max_cluster_id + 1, 3), dtype=np.uint8)
            for cid, color in self.selected_clusters.items():
                cid = int(cid)
                if 0 <= cid <= max_cluster_id:
                    colors_lookup[cid] = color  # color should be (r,g,b)
            color_crop = colors_lookup[mask_crop]
        else:
            # Strategy B: allocate minimal RGB crop and paint cluster-by-cluster (better for sparse large ids)
            color_crop = np.zeros((*mask_crop.shape, 3), dtype=np.uint8)
            for cid, color in self.selected_clusters.items():
                cid = int(cid)
                if cid == 0:
                    continue
                # boolean mask on the cropped region only
                sel = (mask_crop == cid)
                if sel.any():
                    color_crop[sel] = color

        # Resize to base_image shape and blend
        resized = cv2.resize(color_crop, (base_image.shape[1], base_image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        # Ensure correct dtype and in-place blending
        if resized.dtype != np.uint8:
            resized = resized.astype(np.uint8)
        if base_image.dtype != np.uint8:
            base_image[:] = base_image.astype(np.uint8)

        cv2.addWeighted(base_image, 0.5, resized, 0.5, 0, dst=base_image)

    def auto_load_files(self):
        """
        Automatically find and load files based on patterns:
        1. anndata file (.h5ad)
        2. transformation matrix (contains 'transform' in filename)
        3. detected transcripts (contains 'detected_transcripts' in filename)
        4. cellpose masks (.npy file, excluding outlines)
        """

        # Ask user to select a directory
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Auto Load Files From"
        )

        if not directory:
            return

        self.status_bar.showMessage("Auto loading files...")

        loaded_files = []
        errors = []

        try:

            # 1. Find and load transformation matrix
            csv_files = glob.glob(os.path.join(directory, "*.csv"))
            transform_file = None
            for csv_file in csv_files:
                if "transform" in os.path.basename(csv_file).lower():
                    transform_file = csv_file
                    break

            if transform_file:
                self.status_bar.showMessage(
                    f"Loading transformation matrix: {os.path.basename(transform_file)}")
                self.load_transformation_matrix(transform_file)
                loaded_files.append(
                    f"Transformation matrix: {os.path.basename(transform_file)}")
            else:
                errors.append(
                    "No transformation matrix file found (looking for 'transform' in filename)")
            # 2. Find and load anndata file (.h5ad)
            h5ad_files = glob.glob(os.path.join(directory, "*.h5ad"))
            if h5ad_files:
                if len(h5ad_files) == 1:
                    anndata_file = h5ad_files[0]
                    self.status_bar.showMessage(
                        f"Loading anndata: {os.path.basename(anndata_file)}")
                    self.load_anndata(anndata_file)
                    loaded_files.append(
                        f"Anndata: {os.path.basename(anndata_file)}")
                else:
                    errors.append(
                        f"Multiple .h5ad files found: {[os.path.basename(f) for f in h5ad_files]}")
            else:
                errors.append("No .h5ad files found")

            # 3. Find and load detected transcripts
            transcripts_file = None
            for csv_file in csv_files:
                if "detected_transcripts" in os.path.basename(csv_file).lower():
                    transcripts_file = csv_file
                    break

            if transcripts_file:
                self.status_bar.showMessage(
                    f"Loading detected transcripts: {os.path.basename(transcripts_file)}")
                self.load_detected_transcripts(transcripts_file)
                loaded_files.append(
                    f"Detected transcripts: {os.path.basename(transcripts_file)}")
            else:
                errors.append(
                    "No detected transcripts file found (looking for 'detected_transcripts' in filename)")

            # 4. Find and load cellpose masks (.npy, excluding outlines)
            npy_files = glob.glob(os.path.join(directory, "*.npy"))
            masks_file = None
            for npy_file in npy_files:
                filename = os.path.basename(npy_file)
                if "outlines" not in filename.lower():
                    masks_file = npy_file
                    break

            if masks_file:
                self.status_bar.showMessage(
                    f"Loading cellpose masks: {os.path.basename(masks_file)}")
                # Load the masks directly using the existing logic
                self._load_cellpose_masks_from_path(masks_file)
                loaded_files.append(
                    f"Cellpose masks: {os.path.basename(masks_file)}")
            else:
                errors.append(
                    "No cellpose masks file found (looking for .npy file without 'outlines' in name)")

            # Show results
            if loaded_files:
                success_msg = "Successfully loaded:\n" + \
                    "\n".join(loaded_files)
                if errors:
                    success_msg += "\n\nErrors:\n" + "\n".join(errors)

                QMessageBox.information(
                    self, "Auto Load Complete", success_msg)
                self.status_bar.showMessage("Auto load completed")
            else:
                QMessageBox.warning(
                    self, "Auto Load Failed", "No files were loaded.\n\n" + "\n".join(errors))
                self.status_bar.showMessage("Auto load failed")

        except Exception as e:
            QMessageBox.critical(
                self, "Auto Load Error", f"An error occurred during auto loading:\n{str(e)}")
            self.status_bar.showMessage("Auto load error")

    def _load_cellpose_masks_from_path(self, file_path):
        """
        Load cellpose masks from a specific file path (extracted from load_cellpose_masks)
        """
        try:
            print(f"\n=== DEBUG: Loading Cellpose masks ===")
            print(f"DEBUG: File path: {file_path}")

            data = np.load(file_path)

            print(f"DEBUG: Loaded data shape: {data.shape}")
            print(f"DEBUG: Loaded data dtype: {data.dtype}")
            print(f"DEBUG: Loaded data dimensions: {data.ndim}")

            if isinstance(data, np.ndarray) and data.ndim == 2 and np.issubdtype(data.dtype, np.integer):
                print("DEBUG: Processing as 2D integer mask")
                # Original 2D case
                self.cellpose_masks = data
            elif isinstance(data, np.ndarray) and data.ndim == 3:
                print("DEBUG: Processing as 3D mask stack")
                # Handle 3D case - assume it's a stack of 2D masks
                # Convert to integer if needed
                if not np.issubdtype(data.dtype, np.integer):
                    print("DEBUG: Converting to int32")
                    data = data.astype(np.int32)

                # For 3D data, we need to decide which slice to use
                # Let's use the slice with the most non-zero values (most cells)
                slice_counts = [(i, np.count_nonzero(slice))
                                for i, slice in enumerate(data)]
                best_slice_idx = max(slice_counts, key=lambda x: x[1])[0]

                print(f"DEBUG: Selected slice {best_slice_idx} with {slice_counts[best_slice_idx][1]} non-zero values")
                self.cellpose_masks = data[best_slice_idx]
            else:
                print(f"ERROR: Unsupported mask format - ndim={data.ndim}, dtype={data.dtype}")
                raise ValueError("Unsupported mask format")

            print(f"DEBUG: Cellpose masks shape: {self.cellpose_masks.shape}")
            print(f"DEBUG: Cellpose masks dtype: {self.cellpose_masks.dtype}")
            print(f"DEBUG: Cellpose masks value range: {self.cellpose_masks.min()} to {self.cellpose_masks.max()}")
            print(f"DEBUG: Number of unique mask values: {len(np.unique(self.cellpose_masks))}")

            # Continue with the rest of the processing
            if hasattr(self, 'cellpose_masks'):
                num_labels = int(self.cellpose_masks.max())
                rng = np.random.default_rng(42)
                self.cellpose_colors = rng.integers(
                    0, 255, size=(num_labels, 3), dtype=np.uint8)

                # Try to load precomputed color image
                self.cellpose_mask_base_path = file_path.replace(".npy", "")
                color_path = self.cellpose_mask_base_path + "_color.npy"
                outline_path = self.cellpose_mask_base_path + "_outlines.npy"
                if os.path.exists(color_path):
                    self.cellpose_mask_color_image = np.load(color_path)
                else:
                    # Ensure background = black
                    color_lut = np.vstack(([0, 0, 0], self.cellpose_colors))
                    indices = self.cellpose_masks.astype(np.int32)
                    self.cellpose_mask_color_image = color_lut[indices].astype(
                        np.uint8)
                    # Save for future use

                h_img, w_img = self.original_image.shape[:2]
                print(f"\nDEBUG: Scaling masks to match image...")
                print(f"DEBUG: Original image size: {h_img}x{w_img}")

                # Scale the cellpose masks
                h_cm, w_cm = self.cellpose_masks.shape[:2]
                print(f"DEBUG: Cellpose masks size before scaling: {h_cm}x{w_cm}")

                if (h_img, w_img) != (h_cm, w_cm):
                    print(f"DEBUG: Resizing cellpose masks from {h_cm}x{w_cm} to {h_img}x{w_img}")
                    self.cellpose_masks = cv2.resize(
                        self.cellpose_masks,
                        (w_img, h_img),
                        interpolation=cv2.INTER_NEAREST
                    )
                    print(f"DEBUG: Cellpose masks after scaling: {self.cellpose_masks.shape}")
                    print(f"DEBUG: Value range after scaling: {self.cellpose_masks.min()} to {self.cellpose_masks.max()}")
                else:
                    print("DEBUG: Cellpose masks already match image size, no scaling needed")

                # Scale the color image
                h_m, w_m = self.cellpose_mask_color_image.shape[:2]
                if (h_img, w_img) != (h_m, w_m):
                    print(f"DEBUG: Resizing color image from {h_m}x{w_m} to {h_img}x{w_img}")
                    self.cellpose_mask_color_image = cv2.resize(
                        self.cellpose_mask_color_image,
                        (w_img, h_img),
                        interpolation=cv2.INTER_NEAREST
                    )
                    # overwrite cache so next load is already scaled
                    np.save(color_path, self.cellpose_mask_color_image)
                    print("DEBUG: Saved scaled color image to cache")

                # Enable buttons
                self.toggle_cellpose_button.setEnabled(True)
                self.toggle_cellpose_outline_button.setEnabled(True)

                # Automatically create cluster masks if both cell_centers and cellpose_masks are available
                if self.cell_centers is not None:
                    self.make_cluster_data()

                # Queue outline generation
                self.status_bar.showMessage(
                    "Generating Cellpose outlines... this may take a moment")
                QTimer.singleShot(100, self._generate_outlines_and_update)

        except Exception as e:
            self.status_bar.showMessage(
                f"Error loading Cellpose masks: {str(e)}")
            raise


if __name__ == '__main__':
    from qtpy.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
