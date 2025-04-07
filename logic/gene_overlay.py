import cv2
import numpy as np
from resources.colors import colors_rgb

def generate_unique_color(selected_genes):
    available_colors = [value for key, value in colors_rgb.items()
                        if value not in selected_genes.values()]
    if not available_colors:
        return random.choice(list(colors_rgb.values()))
    return random.choice(available_colors)

def on_gene_selected(main_window, gene):
    if gene in main_window.selected_genes:
        main_window.status_bar.showMessage("Gene already selected, choose a different gene.")
        return
    elif not gene:
        main_window.status_bar.showMessage("Gene does not exist, choose a different gene.")
        return

    color = generate_unique_color(main_window.selected_genes)

    from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton
    gene_widget = QFrame()
    layout = QHBoxLayout(gene_widget)

    color_label = QLabel()
    color_label.setFixedSize(20, 20)
    color_label.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border-radius: 10px;")

    gene_name_label = QLabel(gene)

    remove_button = QPushButton("cancel")
    remove_button.setFixedSize(75, 25)
    remove_button.clicked.connect(lambda _, g=gene: main_window.remove_gene_selection(g))

    layout.addWidget(color_label)
    layout.addWidget(gene_name_label)
    layout.addStretch()
    layout.addWidget(remove_button)

    main_window.selected_genes[gene] = color
    main_window.selected_genes_layout.addWidget(gene_widget)

    overlay_genes(main_window)

def remove_gene_selection(main_window, gene):
    if gene in main_window.selected_genes:
        del main_window.selected_genes[gene]

    for i in range(main_window.selected_genes_layout.count()):
        widget = main_window.selected_genes_layout.itemAt(i).widget()
        if widget:
            labels = widget.findChildren(type(widget))
            for label in labels:
                if label.text() == gene:
                    main_window.selected_genes_layout.removeWidget(widget)
                    widget.hide()
                    widget.deleteLater()
                    overlay_genes(main_window)
                    return
    overlay_genes(main_window)

def overlay_genes(main_window):
    if main_window.gene_data is None or main_window.image is None or main_window.resized_image is None:
        print("Please make sure to upload the detected transcripts")
        return

    overlay_image = main_window.resized_image.copy()
    selected_mask = main_window.gene_data["gene"].isin(main_window.selected_genes)
    filtered = main_window.gene_data[selected_mask]

    if filtered.empty:
        main_window.status_bar.showMessage("No selected genes to overlay.")
        main_window.display_image()
        if main_window.show_cell_centers:
            main_window._draw_cell_centers(overlay_image)
        return

    coords = filtered[["global_x", "global_y"]].to_numpy()
    genes = filtered["gene"].to_numpy()

    if main_window.transformation_matrix is not None:
        ones = np.ones((coords.shape[0], 1))
        transformed = np.dot(main_window.transformation_matrix, np.hstack([coords, ones]).T).T
        x_coords, y_coords = transformed[:, 0], transformed[:, 1]
    else:
        main_window.status_bar.showMessage("There is no transformation matrix. Please load one.")
        return

    if hasattr(main_window, 'current_zoom') and main_window.current_zoom is not None:
        zoom = main_window.current_zoom
        in_zoom = (
            (x_coords >= zoom['x_start']) &
            (x_coords < zoom['x_end']) &
            (y_coords >= zoom['y_start']) &
            (y_coords < zoom['y_end'])
        )
        if not any(in_zoom):
            main_window.status_bar.showMessage("No genes in the zoomed region")
            main_window.display_image()
            if main_window.show_cell_centers:
                main_window._draw_cell_centers(overlay_image)
            return

        x_coords = (x_coords[in_zoom] - zoom['x_start']) * zoom['scale_factor']
        y_coords = (y_coords[in_zoom] - zoom['y_start']) * zoom['scale_factor']
        genes = genes[in_zoom]
    else:
        scale = getattr(main_window, 'full_view_scale_factor', None)
        if scale is None:
            h, w = main_window.original_image.shape[:2]
            vh = main_window.image_label.height()
            vw = main_window.image_label.width()
            scale = min(vh / h, vw / w)
            main_window.full_view_scale_factor = scale
        x_coords = x_coords * scale
        y_coords = y_coords * scale

    colors = np.array([main_window.selected_genes[g] for g in genes])
    x_coords = x_coords.astype(int)
    y_coords = y_coords.astype(int)

    h, w = overlay_image.shape[:2]
    valid = (0 <= x_coords) & (x_coords < w) & (0 <= y_coords) & (y_coords < h)
    x_coords = x_coords[valid]
    y_coords = y_coords[valid]
    colors = colors[valid]

    main_window.visible_gene_x_coords = x_coords
    main_window.visible_gene_y_coords = y_coords
    main_window.visible_gene_colors = colors

    for x, y, color in zip(x_coords, y_coords, colors):
        bgr = (color[2], color[1], color[0])
        cv2.circle(overlay_image, (x, y), 1, bgr, -1)

    if main_window.show_cell_centers:
        main_window._draw_cell_centers(overlay_image)
    else:
        overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        h, w, _ = overlay_image_rgb.shape
        q_img = QImage(overlay_image_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        main_window.image_label.setPixmap(QPixmap.fromImage(q_img))

    main_window.status_bar.showMessage(f"Genes overlaid: {len(x_coords)} visible points")
