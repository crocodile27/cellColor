# Gene Visualization Tool (`cellColor`)

An interactive desktop application for visualizing **spatial transcriptomics data**, **cell segmentation masks**, **cell clustering data** and **microscopy images**.
Perfect for verifying cell segmentation accuracy and exploring spatial gene expression patterns.

---

## 👥 Credits & Project Team

- **Developer:** Anthea Guo
- **Mentor:** Kushal Nimkar
- **Principal Investigator (PI):** Prof. Karthik Shekhar

---

## Motivation

The motivation behind this was to create a tool that allows the user to easily verify the accuracy of cellpose segmentation masks while also highlighting spatially unusual genes.

---

## ✨ Features

- **Image Loading & Zooming:**Load tissue/microscopy images, zoom into regions, and reset to full view.
- **Cellpose Segmentation Overlay:**Overlay Cellpose-generated segmentation masks or outlines with smooth, cached zooming.
- **Transcript Visualization:**Import transcript coordinates (`x`, `y`, `gene`), align with images using transformation matrices, and overlay selected genes.
- **Single-Cell Integration:**Load AnnData cell center positions (`.h5ad`), toggle display, and customize appearance.
- **User-Friendly Toolbar:**Intuitive controls for overlays and zoom, live status feedback, and collapsible navigation frames.
- **Data Alignment:**
  Load transformation matrices for accurate transcript-image alignment.

---

## 🚀 Installation

**Option 1: Local Development (Editable Mode)**

```bash
git clone https://github.com/crocodile27/cellColor.git
cd cellColor
conda create -n cellcolor python=3.10
conda activate cellcolor
pip install -e .
```

Run locally:

```bash
cellColor
```

**Option 2: Install via PyPI (v0.1.0)**

```bash
pip install cellColor
```

Release: September 1, 2025 ([PyPI link](https://pypi.org/project/cellColor/0.1.0/))

Launch:

```bash
cellColor
```

---

## ‼️ IMPORTANT: Required Data Formats

## **Images**

**Tissue Section Images:** `.png`, `.jpg`, `.tif`. High resolution tissues that will act as the canvas and which images you choose is up to your discretion. Click on file-> load image and choose desired image. A downsized version of the image will be produced after it is loaded for the first time. Feel free to choose either of them in future runs.

## All other data:

Place all of the following files in a single folder with the format **[prefix]\*\*\***rn[insert run number]**\*\_**rg[insert region number]\*\*. E.g. 140g_rn3_rg0 -> \_this is really important for the auto_load_file function to correctly detect with run and region you are working on and automatically load all necessary files.

Files include:

- **Cellpose Masks:** `.npy` image mask generated from cellpose.
- **Detected Transcripts:** CSV/TSV with `barcode_id`, `global_x`, `global_z`, `x`, `y`, `fov`, `gene`, `transcript_id` as columns. global_x & global_y will be used as the coordinates.
- **Transformation Matrix:** CSV/TSV for alignment of gene and tissue data.
- **AnnData:**`.h5ad` with cell center coordinates saved in `global_x` and `global_y` observations.

---

## 🧪 Example Workflow

1. **Open the app:** `cellColor`
2. **Load image:** _File → Load Image_ to display tissue section.
3. **Auto load all files:** _→ click on desired folder_
4. **Overlay Data:** choose the data you'd like to overlay including: cell centers, gene transcripts, cellpose masks/outlines, cell clusters.
5. **Zoom & reset:**
   Zoom into areas of interest; use _Reset Zoom_ to return to previous zoom level.
