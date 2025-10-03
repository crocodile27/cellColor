import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import anndata as ad
import pandas as pd
import os
import re


class CellCentersMixin:
    def toggle_cell_centers(self):
        """Toggle display of cell centers"""
        self.show_cell_centers = self.toggle_cell_centers_button.isChecked()

        if self.show_cell_centers:
            self.toggle_cell_centers_button.setText("Hide Cell Centers")
            self._process_cell_centers()
            self.update_display()
        else:
            self.toggle_cell_centers_button.setText("Show Cell Centers")
            self.update_display()

    def _process_cell_centers(self):
        """Process cell center coordinates for the current view."""
        if not hasattr(self, 'cell_centers') or self.cell_centers is None or self.cell_centers.empty:
            self.cell_center_x_coords = np.array([], dtype=int)
            self.cell_center_y_coords = np.array([], dtype=int)
            self.cell_center_visible = False
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
                self.cell_center_x_coords = np.array([], dtype=int)
                self.cell_center_y_coords = np.array([], dtype=int)
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
        # enable the cell centers button
        self.toggle_cell_centers_button.setEnabled(True)

    def _draw_cell_centers(self, image):
        """Draw cell centers on the given image and display it."""
        x_coords = getattr(self, 'cell_center_x_coords', None)
        y_coords = getattr(self, 'cell_center_y_coords', None)
        if x_coords is None or y_coords is None:

            self._process_cell_centers()
            x_coords = getattr(self, 'cell_center_x_coords', [])
            y_coords = getattr(self, 'cell_center_y_coords', [])
        if len(x_coords) == 0 or len(y_coords) == 0:
            # Nothing to draw; just display the current image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(image_rgb.data, width, height,
                           bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            return

        for x, y in zip(x_coords, y_coords):
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

    def load_anndata(self, file_name):
        """Load AnnData file - used internally by auto_load_files"""
        if file_name:
            self.status_bar.showMessage(
                "Loading Anndata...")
            QTimer.singleShot(0, lambda: self.process_anndata(file_name))

    def process_anndata(self, file_name):
        """Process anndata file to extract cell centers and cluster annotations"""
        try:
            folder_name = os.path.basename(os.path.dirname(file_name))
            match = re.search(r'_rn(\d+)_rg(\d+)', folder_name)
            if match:
                self.run = int(match.group(1))
                self.region = int(match.group(2))
            else:
                self.run = None
                self.region = None
            adata = ad.read_h5ad(file_name)
            self.status_bar.showMessage("AnnData loaded successfully")

            # Filter data for the specific region and run
            filtered_data = adata.obs[
                (adata.obs['region'].astype(int) == self.region) &
                (adata.obs['run'].astype(int) == self.run)
            ]

            # --- Extract coordinates ---
            x_coords = y_coords = None
            # Try common obsm keys
            for key in ['spatial', 'X_spatial']:
                if key in adata.obsm:
                    # Get indices of filtered data
                    filtered_indices = filtered_data.index
                    cell_coords = adata.obsm[key][adata.obs.index.isin(
                        filtered_indices)]
                    x_coords, y_coords = cell_coords[:, 0], cell_coords[:, 1]
                    break
            # Try common obs columns
            if x_coords is None or y_coords is None:
                for x_key, y_key in [('center_x', 'center_y'), ('x', 'y')]:
                    if x_key in adata.obs and y_key in adata.obs:
                        x_coords, y_coords = filtered_data[x_key].values, filtered_data[y_key].values
                        break
            # Try any columns with 'x' and 'y' in their names
            if x_coords is None or y_coords is None:
                x_cols = [col for col in adata.obs.columns if 'x' in col.lower()]
                y_cols = [col for col in adata.obs.columns if 'y' in col.lower()]
                if x_cols and y_cols:
                    x_coords, y_coords = filtered_data[x_cols[0]
                                                       ].values, filtered_data[y_cols[0]].values
                    self.status_bar.showMessage(
                        f"Using columns '{x_cols[0]}' and '{y_cols[0]}' for coordinates"
                    )
            if x_coords is None or y_coords is None:
                self.status_bar.showMessage(
                    "Could not find cell center coordinates in AnnData file")
                return

            # --- Extract cell_id and parse numeric ID ---
            # Get cell_id from obs_names (the index of the AnnData object)
            cell_id_series = filtered_data.index

            # Parse numeric ID from cell_id format (e.g., "1_rg0_rn3" -> 1)
            def parse_cell_id(cell_id_str):
                try:
                    # Extract the first number before the first underscore
                    return int(cell_id_str.split('_')[0])
                except (ValueError, AttributeError):
                    return None

            numeric_cell_ids = pd.Series([parse_cell_id(cell_id) for cell_id in cell_id_series],
                                         index=cell_id_series, name="cell_id")
            # --- Extract cluster/type columns ---
            potential_cluster_cols = ["final_assignment", "leiden", "cluster",
                                      "type", "celltype", "cell_type"]

            cluster_cols = [
                col for col in adata.obs.columns if col.lower() in potential_cluster_cols]
            cluster_series = None
            selected_cluster_col = None
            if cluster_cols:
                selected_cluster_col = cluster_cols[0]
                cluster_series = filtered_data[selected_cluster_col]
                cluster_series = pd.Series(
                    cluster_series.values, name="cluster")
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

            # Include cell_id if available
            if len(numeric_cell_ids) == len(x_coords):
                data["cell_id"] = numeric_cell_ids.values

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
                # Automatically create cluster masks if both cell_centers and cellpose_masks are available
                self.make_cluster_data()
            self._process_cell_centers()
            self.toggle_cell_centers_button.setEnabled(True)

        except ImportError:
            self.status_bar.showMessage(
                "Please install the 'anndata' package: `pip install anndata`")
        except Exception as e:
            self.status_bar.showMessage(
                f"Error processing AnnData file: {str(e)}")
            print(f"Error processing AnnData file: {str(e)}")
