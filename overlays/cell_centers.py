import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import anndata as ad
import pandas as pd

class CellCentersMixin:
    def toggle_cell_centers(self):
        """Toggle display of cell centers"""
        self.show_cell_centers = self.toggle_cell_centers_button.isChecked()

        if self.show_cell_centers:
            self.toggle_cell_centers_button.setText("Hide Cell Centers")
            cell_centers = getattr(self, 'cell_centers', None)
            if cell_centers is not None:
                if self.image is not None:
                    self.display_cell_centers()
                else:
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Missing Image")
                    msg.setText("Please load an image first.")
                    msg.exec_()
            else:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("No Cell Centers")
                msg.setText("No cell centers loaded. Please load anndata file first.")
                msg.exec_()
        else:
            self.toggle_cell_centers_button.setText("Show Cell Centers")
            if self.image is not None:
                self.display_image()
                if self.gene_data is not None and self.selected_genes:
                    self.overlay_genes()


    def _process_cell_centers(self):
        """Process cell center coordinates for the current view."""
        if not hasattr(self, 'cell_centers') or self.cell_centers is None or self.cell_centers.empty:
            return

        x_coords, y_coords = self.cell_centers[['global_x', 'global_y']].to_numpy().T

        if self.transformation_matrix is not None:
            coords = np.dot(
                self.transformation_matrix,
                np.hstack([x_coords[:, None], y_coords[:, None], np.ones((len(x_coords), 1))]).T
            ).T[:, :2]
            x_coords, y_coords = coords[:, 0], coords[:, 1]

        if getattr(self, 'current_zoom', None):
            zoom = self.current_zoom
            in_zoom = (
                (zoom['x_start'] <= x_coords) & (x_coords < zoom['x_end']) &
                (zoom['y_start'] <= y_coords) & (y_coords < zoom['y_end'])
            )
            if not any(in_zoom):
                self.cell_center_visible = False
                return
            x_coords, y_coords = (
                (x_coords[in_zoom] - zoom['x_start']) * zoom['scale_factor'],
                (y_coords[in_zoom] - zoom['y_start']) * zoom['scale_factor']
            )
        else:
            scale_factor = getattr(self, 'full_view_scale_factor', None) or min(
                self.image_label.height() / self.original_image.shape[0],
                self.image_label.width() / self.original_image.shape[1]
            )
            x_coords, y_coords = x_coords * scale_factor, y_coords * scale_factor

        x_coords, y_coords = x_coords.astype(int), y_coords.astype(int)

        height, width = self.resized_image.shape[:2]
        valid = (0 <= x_coords) & (x_coords < width) & (0 <= y_coords) & (y_coords < height)

        self.cell_center_x_coords = x_coords[valid]
        self.cell_center_y_coords = y_coords[valid]
        self.cell_center_visible = valid.sum() > 0


    def _draw_cell_centers(self, image):
        """Draw cell centers on the given image and display it."""
        if not hasattr(self, 'cell_center_x_coords') or not hasattr(self, 'cell_center_y_coords'):
            self._process_cell_centers

        if hasattr(self, 'cell_center_x_coords') and hasattr(self, 'cell_center_y_coords'):
            for x, y in zip(self.cell_center_x_coords, self.cell_center_y_coords):
                cv2.circle(image, (x, y), self.cell_center_size, self.cell_center_color, -1)

        # Convert and display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

        num_points = len(getattr(self, 'cell_center_x_coords', []))
        self.status_bar.showMessage(f"Cell centers displayed: {num_points} visible points")


    def display_cell_centers(self):
        """Display cell centers from anndata on the image and overlay genes if enabled."""
        cell_centers = getattr(self, 'cell_centers', None)
        if cell_centers is None or cell_centers.empty:
            self.status_bar.showMessage("No cell centers loaded")
            return
        if self.transformation_matrix is None:
            self.status_bar.showMessage("Please load transformation matrix first")
            return
        if self.image is None or self.resized_image is None:
            self.status_bar.showMessage("Please load an image first")
            return

        base_image = self.resized_image.copy()
        self._process_cell_centers()

        if hasattr(self, 'gene_data') and self.gene_data is not None and hasattr(self, 'selected_genes') and self.selected_genes:
            if hasattr(self, 'visible_gene_x_coords'):
                for x, y, color in zip(self.visible_gene_x_coords, self.visible_gene_y_coords, self.visible_gene_colors):
                    color = (int(color[2]), int(color[1]), int(color[0]))
                    cv2.circle(base_image, (x, y), 1, color, -1)
            else:
                temp_show_cell_centers = self.show_cell_centers
                self.show_cell_centers = False
                self.overlay_genes()
                self.show_cell_centers = temp_show_cell_centers
                if hasattr(self, 'visible_gene_x_coords'):
                    for x, y, color in zip(self.visible_gene_x_coords, self.visible_gene_y_coords, self.visible_gene_colors):
                        color = (int(color[2]), int(color[1]), int(color[0]))
                        cv2.circle(base_image, (x, y), 1, color, -1)

        self.update_display()


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
                self.display_cell_centers()
                
        except ImportError:
            self.status_bar.showMessage("Please install the 'anndata' package to load AnnData files using `pip install anndata`")
        except Exception as e:
            self.status_bar.showMessage(f"Error processing AnnData file: {str(e)}")
            print(f"Error processing AnnData file: {str(e)}")