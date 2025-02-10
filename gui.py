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
                            QFrame, QColorDialog)
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
root = tk.Tk()
height = root.winfo_screenheight() - root.winfo_screenmmheight()
width = root.winfo_screenwidth()
print("Width, height of screen: ", width, height)



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

        # Image Label
        self.image_label = QLabel("Upload an image to start")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.image_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.image_layout.addWidget(self.progress_bar)
        self.progress_bar.hide()

        # Toolbar Area
        self.toolbar_area = QWidget()
        self.toolbar_layout = QVBoxLayout(self.toolbar_area)

        # Gene Selection Dropdown
        self.gene_dropdown = QComboBox()
        # self.gene_dropdown.setPlaceholderText("Select a Gene") #don't think this works because declared before added?
        self.gene_dropdown.currentTextChanged.connect(self.on_gene_selected)
        self.toolbar_layout.addWidget(self.gene_dropdown)
        self.gene_dropdown.setPlaceholderText("Select a Gene")

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

        # Load Detected Transcripts Action
        self.load_detected_transcripts_action = QAction(
            "Load Detected Transcripts", self)
        self.load_detected_transcripts_action.triggered.connect(
            self.load_detected_transcripts)
        self.file_menu.addAction(self.load_detected_transcripts_action)

        # Load Transformation Matrix Action
        self.load_transformation_matrix_action = QAction(
            "Load Transformation Matrix", self)
        self.load_transformation_matrix_action.triggered.connect(
            self.load_transformation_matrix)
        self.file_menu.addAction(self.load_transformation_matrix_action)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Data Storage
        self.image = None
        self.gene_data = None
        self.transformation_matrix = None
        self.resized_image = None
        self.selected_genes = {}  # Dictionary to store selected genes and their colors

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

    def overlay_genes(self):
        if self.gene_data is not None and self.image is not None and self.resized_image is not None:
            # Create a copy of the resized image to draw on
            overlay_image = self.resized_image.copy()

            for _, row in self.gene_data.iterrows():
                x, y, gene = row['global_x'], row['global_y'], row['gene']

                # Check if this gene is selected
                if gene in self.selected_genes:
                    if self.transformation_matrix is not None:
                        # Apply transformation matrix to coordinates
                        coords = np.dot(self.transformation_matrix, [x, y, 1])
                        x, y = coords[0], coords[1]
                    else:
                        print(
                            "There is no transformation matrix. Please load a transformation matrix.")
                        return

                    # Scale coordinates to match resized image
                    x, y = int(x * self.screenHeight/(self.image.shape[0])), int(y * self.screenHeight/(self.image.shape[0]))

                    # Get color for this gene
                    color = self.selected_genes[gene]

                    # Draw circle on the image
                    cv2.circle(overlay_image, (x, y), 0.1, color, -1)

            # Convert and display the image
            overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            height, width, channel = overlay_image_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(overlay_image_rgb.data, width, height,
                           bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
        else:
            print("Please make sure to upload the detected transcripts")
        return

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

            if self.image is not None:
                # Resize the image for display
                self.resized_image = cv2.resize(
                    self.image, (0, 0), fx=self.screenHeight/ (self.image.shape[0]), fy=self.screenHeight / (self.image.shape[0]))

                self.display_image()

                # If gene data is already loaded, reoverlay genes
                if self.gene_data is not None:
                    self.overlay_genes()

                self.status_bar.showMessage(
                    "Image loaded and resized successfully")
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
