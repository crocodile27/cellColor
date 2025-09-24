
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtCore import QTimer
import cv2


class GenesMixin:
    def load_detected_transcripts(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.status_bar.showMessage(
                "Loading Detected Transcripts...(this may take a while)")
            QTimer.singleShot(0, lambda: self.process_csv(file_name))

    def load_transformation_matrix(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.status_bar.showMessage("Loading Transformation Matrix...")
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
                self.gene_data = pd.read_csv(file_name)

                unique_genes = self.gene_data['gene'].unique()
                self.gene_dropdown.clear()
                self.gene_dropdown.addItems(unique_genes)

                if self.image is not None:
                    self.overlay_genes()

                self.status_bar.showMessage("Gene data loaded successfully")
        except Exception as e:
            self.status_bar.showMessage(
                f"Error loading file {file_name}: {str(e)}")

    def on_gene_selected(self, gene):
        print(f"{gene} is selected")
        if gene in self.selected_genes:
            self.status_bar.showMessage(
                "Gene already selected, choose a different gene.")
            return
        elif not gene:
            self.status_bar.showMessage(
                "Gene does not exist, choose a different gene.")
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
            f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border-radius: 10px;"
        )

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
        self.selected_genes[gene] = (color[0], color[1], color[2])

        # Add to selected genes layout
        self.selected_genes_layout.addWidget(gene_widget)

        # Overlay genes
        self.overlay_genes()

    def remove_gene_selection(self, gene):
        print(f"[DEBUG-genes.py] removing gene {gene}")
        if gene in self.selected_genes:
            del self.selected_genes[gene]

        for i in range(self.selected_genes_layout.count()):
            widget = self.selected_genes_layout.itemAt(i).widget()
            if widget:
                labels = widget.findChildren(QLabel)
                for label in labels:
                    if label.text() == gene:
                        self.selected_genes_layout.removeWidget(widget)
                        widget.hide()
                        widget.deleteLater()
                        self.overlay_genes()
                        return
        self.overlay_genes()

    def overlay_genes(self):
        """Overlay selected genes on the current image."""
        if self.gene_data is None or self.resized_image is None:
            print('no data')
            return

        overlay_image = self.resized_image.copy()

        # Only selected genes
        print(f"[DEBUG-genes.py] selected genes: {self.selected_genes}")
        selected_gene_mask = self.gene_data["gene"].isin(self.selected_genes)
        filtered_data = self.gene_data[selected_gene_mask]

        if filtered_data.empty:
            print('filtered data is empty')
            self.status_bar.showMessage("No selected genes to overlay.")
            self.visible_gene_x_coords = None
            self.visible_gene_y_coords = None
            self.visible_gene_colors = None
            self.update_display()
            return

        coords = filtered_data[["global_x", "global_y"]].to_numpy()
        genes = filtered_data["gene"].to_numpy()

        # Apply transformation
        if self.transformation_matrix is not None:
            ones = np.ones((coords.shape[0], 1))
            transformed_coords = np.dot(
                self.transformation_matrix, np.hstack([coords, ones]).T).T
            x_coords, y_coords = transformed_coords[:,
                                                    0], transformed_coords[:, 1]
            x_coords, y_coords = x_coords * 0.25, y_coords * 0.25
        else:
            self.status_bar.showMessage(
                "Please load a transformation matrix first.")
            return

        # Handle zoom/full-view scaling
        if getattr(self, "current_zoom", None):
            zoom = self.current_zoom
            in_zoom = (
                (x_coords >= zoom["x_start"])
                & (x_coords < zoom["x_end"])
                & (y_coords >= zoom["y_start"])
                & (y_coords < zoom["y_end"])
            )
            if not any(in_zoom):
                self.status_bar.showMessage("No genes in the zoomed region")
                if self.show_cell_centers:
                    self._draw_cell_centers(overlay_image)
                return
            x_coords, y_coords, genes = (
                (x_coords[in_zoom] - zoom["x_start"]) * zoom["scale_factor"],
                (y_coords[in_zoom] - zoom["y_start"]) * zoom["scale_factor"],
                genes[in_zoom],
            )
        else:
            scale_factor = getattr(self, "full_view_scale_factor", None)
            if not scale_factor:
                orig_h, orig_w = self.original_image.shape[:2]
                view_h, view_w = self.image_label.height(), self.image_label.width()
                scale_factor = min(view_h / orig_h, view_w / orig_w) or 0.5
                self.full_view_scale_factor = scale_factor
            x_coords, y_coords = x_coords * scale_factor, y_coords * scale_factor

        # Assign colors
        colors = np.array([self.selected_genes[g] for g in genes])
        x_coords, y_coords = x_coords.astype(int), y_coords.astype(int)

        # Filter valid
        h, w = overlay_image.shape[:2]
        valid = (0 <= x_coords) & (x_coords < w) & (
            0 <= y_coords) & (y_coords < h)
        self.visible_gene_x_coords = x_coords[valid]
        self.visible_gene_y_coords = y_coords[valid]
        self.visible_gene_colors = colors[valid]

        self.update_display()
        self.status_bar.showMessage(
            f"Genes overlaid: {len(self.visible_gene_x_coords)} visible points")
