import numpy as np
import cv2
from qtpy.QtGui import QImage, QPixmap

def toggle_cell_centers(main_window):
    main_window.show_cell_centers = main_window.toggle_cell_centers_button.isChecked()
    
    if main_window.show_cell_centers:
        main_window.toggle_cell_centers_button.setText("Hide Cell Centers")
        cell_centers = getattr(main_window, 'cell_centers', None)
        if cell_centers is not None:
            if main_window.image is not None:
                display_cell_centers(main_window)
            else:
                main_window.status_bar.showMessage("Please load an image first")
        else:
            main_window.status_bar.showMessage("No cell centers loaded. Please load anndata file first.")
    else:
        main_window.toggle_cell_centers_button.setText("Show Cell Centers")
        if main_window.image is not None:
            main_window.display_image()
            if main_window.gene_data is not None and main_window.selected_genes:
                from logic.gene_overlay import overlay_genes
                overlay_genes(main_window)

def _process_cell_centers(main_window):
    if not hasattr(main_window, 'cell_centers') or main_window.cell_centers is None or main_window.cell_centers.empty:
        return
    
    x_coords, y_coords = main_window.cell_centers[['global_x', 'global_y']].to_numpy().T

    if main_window.transformation_matrix is not None:
        coords = np.dot(
            main_window.transformation_matrix,
            np.hstack([x_coords[:, None], y_coords[:, None], np.ones((len(x_coords), 1))]).T
        ).T[:, :2]
        x_coords, y_coords = coords[:, 0], coords[:, 1]

    if getattr(main_window, 'current_zoom', None):
        zoom = main_window.current_zoom
        in_zoom = (
            (zoom['x_start'] <= x_coords) & (x_coords < zoom['x_end']) &
            (zoom['y_start'] <= y_coords) & (y_coords < zoom['y_end'])
        )
        if not any(in_zoom):
            main_window.cell_center_visible = False
            return
        x_coords, y_coords = (x_coords[in_zoom] - zoom['x_start']) * zoom['scale_factor'], \
                             (y_coords[in_zoom] - zoom['y_start']) * zoom['scale_factor']
    else:
        scale_factor = getattr(main_window, 'full_view_scale_factor', None) or min(
            main_window.image_label.height() / main_window.original_image.shape[0],
            main_window.image_label.width() / main_window.original_image.shape[1]
        )
        x_coords, y_coords = x_coords * scale_factor, y_coords * scale_factor

    x_coords, y_coords = x_coords.astype(int), y_coords.astype(int)

    height, width = main_window.resized_image.shape[:2]
    valid = (0 <= x_coords) & (x_coords < width) & (0 <= y_coords) & (y_coords < height)

    main_window.cell_center_x_coords = x_coords[valid]
    main_window.cell_center_y_coords = y_coords[valid]
    main_window.cell_center_visible = valid.sum() > 0

def _draw_cell_centers(main_window, image):
    if not hasattr(main_window, 'cell_center_x_coords') or not hasattr(main_window, 'cell_center_y_coords'):
        _process_cell_centers(main_window)

    if hasattr(main_window, 'cell_center_x_coords') and hasattr(main_window, 'cell_center_y_coords'):
        for x, y in zip(main_window.cell_center_x_coords, main_window.cell_center_y_coords):
            cv2.circle(image, (x, y), main_window.cell_center_size, main_window.cell_center_color, -1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    q_img = QImage(image_rgb.data, width, height, 3 * width, QImage.Format_RGB888)
    main_window.image_label.setPixmap(QPixmap.fromImage(q_img))

    num_points = len(getattr(main_window, 'cell_center_x_coords', []))
    main_window.status_bar.showMessage(f"Cell centers displayed: {num_points} visible points")

def display_cell_centers(main_window):
    if main_window.cell_centers is None or main_window.cell_centers.empty:
        main_window.status_bar.showMessage("No cell centers loaded")
        return
    if main_window.transformation_matrix is None:
        main_window.status_bar.showMessage("Please load transformation matrix first")
        return
    if main_window.image is None or main_window.resized_image is None:
        main_window.status_bar.showMessage("Please load an image first")
        return

    base_image = main_window.resized_image.copy()

    _process_cell_centers(main_window)

    if hasattr(main_window, 'gene_data') and main_window.gene_data is not None and main_window.selected_genes:
        if hasattr(main_window, 'visible_gene_x_coords'):
            for x, y, color in zip(main_window.visible_gene_x_coords,
                                   main_window.visible_gene_y_coords,
                                   main_window.visible_gene_colors):
                color = (int(color[2]), int(color[1]), int(color[0]))  # RGB -> BGR
                cv2.circle(base_image, (x, y), 1, color, -1)
        else:
            from logic.gene_overlay import overlay_genes
            temp = main_window.show_cell_centers
            main_window.show_cell_centers = False
            overlay_genes(main_window)
            main_window.show_cell_centers = temp

    _draw_cell_centers(main_window, base_image)
