import cv2
import numpy as np
from qtpy.QtCore import QRectF
from qtpy.QtGui import QImage, QPixmap

from logic.gene_overlay import overlay_genes
from logic.cell_centers import display_cell_centers

def get_pixmap_rect(main_window):
    pixmap = main_window.image_label.pixmap()
    if not pixmap:
        return QRectF()

    label_width = main_window.image_label.width()
    label_height = main_window.image_label.height()
    pixmap_width = pixmap.width()
    pixmap_height = pixmap.height()

    x = (label_width - pixmap_width) / 2 if pixmap_width < label_width else 0
    y = (label_height - pixmap_height) / 2 if pixmap_height < label_height else 0

    return QRectF(x, y, pixmap_width, pixmap_height)

def zoom_to_selection(main_window, rect):
    if main_window.resized_image is None or main_window.original_image is None:
        return

    pixmap = main_window.image_label.pixmap()
    if not pixmap:
        return

    pixmap_rect = get_pixmap_rect(main_window)
    if not pixmap_rect.isValid():
        return

    normalized_rect = QRectF(
        (rect.x() - pixmap_rect.x()) / pixmap_rect.width(),
        (rect.y() - pixmap_rect.y()) / pixmap_rect.height(),
        rect.width() / pixmap_rect.width(),
        rect.height() / pixmap_rect.height()
    )

    normalized_rect = QRectF(
        max(0, normalized_rect.x()),
        max(0, normalized_rect.y()),
        min(1 - normalized_rect.x(), normalized_rect.width()),
        min(1 - normalized_rect.y(), normalized_rect.height())
    )

    orig_h, orig_w = main_window.original_image.shape[:2]

    if hasattr(main_window, 'current_zoom') and main_window.current_zoom is not None:
        main_window.zoom_history.append(main_window.current_zoom.copy())
        zoom = main_window.current_zoom

        x1 = int(zoom['x_start'] + normalized_rect.x() * (zoom['x_end'] - zoom['x_start']))
        y1 = int(zoom['y_start'] + normalized_rect.y() * (zoom['y_end'] - zoom['y_start']))
        x2 = int(zoom['x_start'] + (normalized_rect.x() + normalized_rect.width()) * (zoom['x_end'] - zoom['x_start']))
        y2 = int(zoom['y_start'] + (normalized_rect.y() + normalized_rect.height()) * (zoom['y_end'] - zoom['y_start']))
    else:
        x1 = int(normalized_rect.x() * orig_w)
        y1 = int(normalized_rect.y() * orig_h)
        x2 = int((normalized_rect.x() + normalized_rect.width()) * orig_w)
        y2 = int((normalized_rect.y() + normalized_rect.height()) * orig_h)

    selected_region = main_window.original_image[y1:y2, x1:x2]

    view_h = main_window.image_label.height()
    view_w = main_window.image_label.width()
    scale = min(view_h / selected_region.shape[0], view_w / selected_region.shape[1])

    main_window.resized_image = cv2.resize(selected_region, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    main_window.current_zoom = {
        'x_start': x1,
        'y_start': y1,
        'x_end': x2,
        'y_end': y2,
        'scale_factor': scale
    }

    main_window.reset_zoom_button.setEnabled(True)

    from logic.image_processing import display_image
    display_image(main_window)

    for attr in ['visible_gene_x_coords', 'visible_gene_y_coords', 'visible_gene_colors',
                 'cell_center_x_coords', 'cell_center_y_coords']:
        if hasattr(main_window, attr):
            delattr(main_window, attr)

    if main_window.gene_data is not None and main_window.selected_genes and main_window.show_cell_centers:
        overlay_genes(main_window)
    elif main_window.gene_data is not None and main_window.selected_genes:
        overlay_genes(main_window)
    elif main_window.show_cell_centers:
        display_cell_centers(main_window)

    main_window.status_bar.showMessage(f"Zoomed to region. Zoom level: {len(main_window.zoom_history) + 1}")

def reset_zoom(main_window):
    if main_window.zoom_history:
        previous_zoom = main_window.zoom_history.pop()

        if not main_window.zoom_history:
            do_full_reset(main_window)
            return

        x1 = previous_zoom['x_start']
        y1 = previous_zoom['y_start']
        x2 = previous_zoom['x_end']
        y2 = previous_zoom['y_end']
        scale = previous_zoom['scale_factor']

        region = main_window.original_image[y1:y2, x1:x2]
        main_window.resized_image = cv2.resize(region, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        main_window.current_zoom = previous_zoom.copy()
    else:
        do_full_reset(main_window)
        return

    from logic.image_processing import display_image
    display_image(main_window)

    if main_window.gene_data is not None and main_window.selected_genes:
        overlay_genes(main_window)
    elif main_window.show_cell_centers:
        display_cell_centers(main_window)

    level = len(main_window.zoom_history) + (1 if main_window.current_zoom else 0)
    main_window.status_bar.showMessage(f"Zoom level: {level}")

def do_full_reset(main_window):
    if main_window.original_image is not None:
        vh = main_window.image_label.height()
        vw = main_window.image_label.width()
        if vh < 100 or vw < 100:
            vh = max(vh, 600)
            vw = max(vw, 800)

        oh, ow = main_window.original_image.shape[:2]
        scale = min(vh / oh, vw / ow)
        nw, nh = int(ow * scale), int(oh * scale)

        if nh > vh or nw > vw:
            scale *= 0.9
            nw, nh = int(ow * scale), int(oh * scale)

        main_window.resized_image = cv2.resize(
            main_window.original_image, (nw, nh), interpolation=cv2.INTER_LINEAR
        )

        main_window.current_zoom = None
        main_window.zoom_history = []
        main_window.reset_zoom_button.setEnabled(False)
        from logic.image_processing import display_image
        display_image(main_window)

        main_window.full_view_scale_factor = scale

        for attr in ['visible_gene_x_coords', 'visible_gene_y_coords', 'visible_gene_colors',
                     'cell_center_x_coords', 'cell_center_y_coords']:
            if hasattr(main_window, attr):
                delattr(main_window, attr)

        if main_window.gene_data is not None and main_window.selected_genes:
            overlay_genes(main_window)
        elif main_window.show_cell_centers:
            display_cell_centers(main_window)

        main_window.status_bar.showMessage("View reset to original")
