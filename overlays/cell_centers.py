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
            print(f"\n=== DEBUG: Loading AnnData file ===")
            print(f"DEBUG: File path: {file_name}")

            folder_name = os.path.basename(os.path.dirname(file_name))
            print(f"DEBUG: Folder name: {folder_name}")

            match = re.search(r'_rn(\d+)_rg(\d+)', folder_name)
            if match:
                self.run = int(match.group(1))
                self.region = int(match.group(2))
                print(f"DEBUG: Extracted run={self.run}, region={self.region} from folder name")
            else:
                self.run = None
                self.region = None
                print("DEBUG: Could not extract run/region from folder name")

            adata = ad.read_h5ad(file_name)
            print(f"DEBUG: AnnData loaded successfully")
            print(f"DEBUG: AnnData shape: {adata.shape}")
            print(f"DEBUG: AnnData obs columns: {list(adata.obs.columns)}")
            print(f"DEBUG: AnnData obsm keys: {list(adata.obsm.keys()) if hasattr(adata, 'obsm') else 'None'}")

            self.status_bar.showMessage("AnnData loaded successfully")

            # Filter data for the specific region and run
            print(f"DEBUG: Filtering for run={self.run}, region={self.region}")
            filtered_data = adata.obs[
                (adata.obs['region'].astype(int) == self.region) &
                (adata.obs['run'].astype(int) == self.run)
            ]
            print(f"DEBUG: Filtered data shape: {filtered_data.shape}")

            # --- Extract coordinates ---
            print("DEBUG: Attempting to extract coordinates...")
            x_coords = y_coords = None
            # Try common obsm keys
            for key in ['spatial', 'X_spatial']:
                if key in adata.obsm:
                    print(f"DEBUG: Found coordinates in obsm['{key}']")
                    # Get indices of filtered data
                    filtered_indices = filtered_data.index
                    cell_coords = adata.obsm[key][adata.obs.index.isin(
                        filtered_indices)]
                    x_coords, y_coords = cell_coords[:, 0], cell_coords[:, 1]
                    print(f"DEBUG: Extracted {len(x_coords)} coordinates from obsm")
                    break
            # Try common obs columns
            if x_coords is None or y_coords is None:
                print("DEBUG: Trying common obs column names...")
                for x_key, y_key in [('center_x', 'center_y'), ('x', 'y')]:
                    if x_key in adata.obs and y_key in adata.obs:
                        x_coords, y_coords = filtered_data[x_key].values, filtered_data[y_key].values
                        print(f"DEBUG: Found coordinates in obs columns '{x_key}' and '{y_key}'")
                        break
            # Try any columns with 'x' and 'y' in their names
            if x_coords is None or y_coords is None:
                print("DEBUG: Trying to find any columns with 'x' and 'y' in names...")
                x_cols = [col for col in adata.obs.columns if 'x' in col.lower()]
                y_cols = [col for col in adata.obs.columns if 'y' in col.lower()]
                print(f"DEBUG: Found x columns: {x_cols}")
                print(f"DEBUG: Found y columns: {y_cols}")
                if x_cols and y_cols:
                    x_coords, y_coords = filtered_data[x_cols[0]
                                                       ].values, filtered_data[y_cols[0]].values
                    print(f"DEBUG: Using columns '{x_cols[0]}' and '{y_cols[0]}' for coordinates")
                    self.status_bar.showMessage(
                        f"Using columns '{x_cols[0]}' and '{y_cols[0]}' for coordinates"
                    )
            if x_coords is None or y_coords is None:
                print("ERROR: Could not find cell center coordinates in AnnData file")
                self.status_bar.showMessage(
                    "Could not find cell center coordinates in AnnData file")
                return

            print(f"DEBUG: Successfully extracted {len(x_coords)} coordinates")
            print(f"DEBUG: X range: {x_coords.min()} to {x_coords.max()}")
            print(f"DEBUG: Y range: {y_coords.min()} to {y_coords.max()}")

            # --- Extract cell_id and parse numeric ID ---
            # Get cell_id from obs_names (the index of the AnnData object)
            cell_id_series = filtered_data.index
            print(f"DEBUG: Sample cell_id strings: {list(cell_id_series[:5])}")

            # Parse numeric ID from cell_id format (e.g., "1_rg0_rn3" -> 1)
            def parse_cell_id(cell_id_str):
                try:
                    # Extract the first number before the first underscore
                    return int(cell_id_str.split('_')[0])
                except (ValueError, AttributeError):
                    return None

            numeric_cell_ids = pd.Series([parse_cell_id(cell_id) for cell_id in cell_id_series],
                                         index=cell_id_series, name="cell_id")
            print(f"DEBUG: Parsed {numeric_cell_ids.notna().sum()} valid numeric cell_ids")
            if numeric_cell_ids.notna().sum() > 0:
                print(f"DEBUG: Sample numeric cell_ids: {numeric_cell_ids[numeric_cell_ids.notna()].head().tolist()}")
                print(f"DEBUG: cell_id range: {numeric_cell_ids.min()} to {numeric_cell_ids.max()}")

            # --- Extract cluster/type columns ---
            potential_cluster_cols = ["final_assignment", "leiden", "cluster",
                                      "type", "celltype", "cell_type"]
            print(f"DEBUG: Looking for cluster columns in: {potential_cluster_cols}")

            cluster_cols = [
                col for col in adata.obs.columns if col.lower() in potential_cluster_cols]
            print(f"DEBUG: Found cluster columns: {cluster_cols}")
            cluster_series = None
            selected_cluster_col = None
            if cluster_cols:
                selected_cluster_col = cluster_cols[0]
                print(f"DEBUG: Using cluster column: '{selected_cluster_col}'")
                cluster_series = filtered_data[selected_cluster_col]
                unique_clusters = cluster_series.unique()
                print(f"DEBUG: Found {len(unique_clusters)} unique clusters")
                print(f"DEBUG: Sample clusters: {list(unique_clusters[:10])}")
                cluster_series = pd.Series(
                    cluster_series.values, name="cluster")
            else:
                print("DEBUG: No cluster annotations found in AnnData")
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
            print("\nDEBUG: Creating cell_centers DataFrame...")
            data = {
                "global_x": x_coords,
                "global_y": y_coords,
            }
            # Only include cluster if its length matches coordinates
            if len(cluster_series) == len(x_coords):
                data["cluster"] = cluster_series.values
                print(f"DEBUG: Added cluster column ({len(cluster_series)} values)")
            else:
                print(f"WARNING: Cluster series length ({len(cluster_series)}) doesn't match coordinates ({len(x_coords)})")

            # Include cell_id if available
            if len(numeric_cell_ids) == len(x_coords):
                data["cell_id"] = numeric_cell_ids.values
                print(f"DEBUG: Added cell_id column ({len(numeric_cell_ids)} values)")
            else:
                print(f"WARNING: cell_id series length ({len(numeric_cell_ids)}) doesn't match coordinates ({len(x_coords)})")

            self.cell_centers = pd.DataFrame(data)

            print(f"DEBUG: Final cell_centers DataFrame shape: {self.cell_centers.shape}")
            print(f"DEBUG: Final cell_centers columns: {list(self.cell_centers.columns)}")
            print(f"DEBUG: Sample cell_centers data:\n{self.cell_centers.head()}")

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
