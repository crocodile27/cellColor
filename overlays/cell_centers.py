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
            print('toggle_cell_centers')
            self.toggle_cell_centers_button.setText("Hide Cell Centers")
            self._process_cell_centers()
            self.update_display()
        else:
            self.toggle_cell_centers_button.setText("Show Cell Centers")
            self.update_display()

    def _process_cell_centers(self):
        """Process cell center coordinates for the current view."""
        if not hasattr(self, 'cell_centers') or self.cell_centers is None or self.cell_centers.empty:
            return

        x_coords, y_coords = self.cell_centers[[
            'global_x', 'global_y']].to_numpy().T

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
        valid = (0 <= x_coords) & (x_coords < width) & (
            0 <= y_coords) & (y_coords < height)

        self.cell_center_x_coords = x_coords[valid]
        self.cell_center_y_coords = y_coords[valid]
        self.cell_center_visible = valid.sum() > 0

    def _draw_cell_centers(self, image):
        """Draw cell centers on the given image and display it."""
        if getattr(self, 'cell_center_x_coords', None) is None or getattr(self, 'cell_center_y_coords', None) is None:
            self._process_cell_centers()

        for x, y in zip(self.cell_center_x_coords, self.cell_center_y_coords):
            cv2.circle(image, (x, y), self.cell_center_size,
                       self.cell_center_color, -1)

        # Convert and display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(image_rgb.data, width, height,
                       bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

        num_points = len(getattr(self, 'cell_center_x_coords', []))
        print(
            f"Cell centers displayed: {num_points} visible points")

    def load_anndata(self):
        if self.transformation_matrix is None:
            QMessageBox.warning(
                self, "Missing Transformation Matrix",
                "Please load transformation matrix first")
            return
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Anndata File", "", "Anndata Files (*.h5ad);"
            "All Files (*)")
        if file_name:
            self.status_bar.showMessage(
                "Loading Anndata...")
            QTimer.singleShot(0, lambda: self.process_anndata(file_name))

    def process_anndata(self, file_name):
        """Process anndata file to extract cell centers and cluster annotations"""
        try:

            adata = ad.read_h5ad(file_name)
            self.status_bar.showMessage("AnnData loaded successfully")

            # --- Extract coordinates ---
            x_coords = y_coords = None
            # Try common obsm keys
            for key in ['spatial', 'X_spatial']:
                if key in adata.obsm:
                    cell_coords = adata.obsm[key]
                    x_coords, y_coords = cell_coords[:, 0], cell_coords[:, 1]
                    break
            # Try common obs columns
            if x_coords is None or y_coords is None:
                for x_key, y_key in [('center_x', 'center_y'), ('x', 'y')]:
                    if x_key in adata.obs and y_key in adata.obs:
                        x_coords, y_coords = adata.obs[x_key].values, adata.obs[y_key].values
                        break
            # Try any columns with 'x' and 'y' in their names
            if x_coords is None or y_coords is None:
                x_cols = [col for col in adata.obs.columns if 'x' in col.lower()]
                y_cols = [col for col in adata.obs.columns if 'y' in col.lower()]
                if x_cols and y_cols:
                    x_coords, y_coords = adata.obs[x_cols[0]
                                                   ].values, adata.obs[y_cols[0]].values
                    self.status_bar.showMessage(
                        f"Using columns '{x_cols[0]}' and '{y_cols[0]}' for coordinates"
                    )
            if x_coords is None or y_coords is None:
                self.status_bar.showMessage(
                    "Could not find cell center coordinates in AnnData file")
                return

            # --- Extract cluster/type columns ---
            potential_cluster_cols = ["leiden", "cluster", "type", "celltype"]
            cluster_cols = [
                col for col in adata.obs.columns if col.lower() in potential_cluster_cols]
            cluster_series = None
            selected_cluster_col = None
            if cluster_cols:
                selected_cluster_col = cluster_cols[0]
                cluster_series = pd.Series(
                    adata.obs[selected_cluster_col].values, name="cluster")
            else:
                # No cluster annotations found; create a placeholder series to match lengths
                cluster_series = pd.Series(
                    [None] * len(x_coords), name="cluster")

            if self.transformation_matrix is not None:
                coords = np.dot(
                    self.transformation_matrix,
                    np.hstack([x_coords[:, None], y_coords[:, None],
                              np.ones((len(x_coords), 1))]).T
                ).T[:, :2]
                x_coords, y_coords = coords[:, 0] * 0.25, coords[:, 1] * 0.25

            # --- Store everything in DataFrame ---
            data = {
                "global_x": x_coords,
                "global_y": y_coords,
            }
            # Only include cluster if its length matches coordinates
            if len(cluster_series) == len(x_coords):
                data["cluster"] = cluster_series.values

            self.cell_centers = pd.DataFrame(data)

            num_cells = len(self.cell_centers)
            if selected_cluster_col is not None:
                self.status_bar.showMessage(
                    f"Loaded {num_cells} cell centers and '{selected_cluster_col}' cluster annotations from AnnData"
                )
            else:
                self.status_bar.showMessage(
                    f"Loaded {num_cells} cell centers (no cluster annotations found)"
                )

            # --- Populate dropdown with available unique clusters ---
            if self.cellpose_masks is not None:
                self.cluster_dropdown.clear()
                if selected_cluster_col is not None and "cluster" in self.cell_centers.columns:
                    unique_clusters = pd.Series(
                        self.cell_centers["cluster"]).dropna().astype(str).unique()
                    # Ensure list of strings for the dropdown
                    self.cluster_dropdown.addItems(
                        list(map(str, unique_clusters)))
                else:
                    self.cluster_dropdown.addItem(
                        "No cluster annotations found")
                self.make_cluster_button.setEnabled(True)

            # Enable the cell centers button
            if self.original_image is not None:
                self.toggle_cell_centers_button.setEnabled(True)

        except ImportError:
            self.status_bar.showMessage(
                "Please install the 'anndata' package: `pip install anndata`")
        except Exception as e:
            self.status_bar.showMessage(
                f"Error processing AnnData file: {str(e)}")
            print(f"Error processing AnnData file: {str(e)}")
