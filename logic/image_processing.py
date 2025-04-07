import cv2
from qtpy.QtGui import QImage, QPixmap
import numpy as np

def process_image(main_window, file_name):
    try:
        if file_name.lower().endswith(('.tif', '.tiff')):
            print(f"Reading image, {file_name}")
            main_window.image = cv2.imread(file_name)

        print("Making copy of original image")
        main_window.original_image = main_window.image.copy()

        if main_window.image is not None:
            main_window.reset_zoom_button.setEnabled(False)
            from .zoom_utils import do_full_reset
            do_full_reset(main_window)

            if main_window.gene_data is not None:
                from .gene_overlay import overlay_genes
                overlay_genes(main_window)

            main_window.status_bar.showMessage("Image loaded and resized successfully")
        else:
            main_window.status_bar.showMessage("Failed to load image")
    except Exception as e:
        main_window.status_bar.showMessage(f"Error loading image: {str(e)}")
        print(f"Error loading image: {str(e)}")

def display_image(main_window):
    if main_window.resized_image is not None:
        image_rgb = cv2.cvtColor(main_window.resized_image, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        q_img = QImage(image_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        main_window.image_label.setPixmap(QPixmap.fromImage(q_img))
        main_window.image_label.setMinimumSize(1, 1)
        main_window.status_bar.showMessage(f"Image displayed successfully ({w}x{h})")
    else:
        main_window.status_bar.showMessage("Resized image is None")

def resize_image_to_fit(main_window):
    if hasattr(main_window, 'current_zoom') and main_window.current_zoom is not None:
        return
    from .zoom_utils import do_full_reset
    do_full_reset(main_window)