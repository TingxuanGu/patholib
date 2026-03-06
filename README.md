# patholib - Pathology Image Analysis Library

[English](README.md) | [中文](README_CN.md)

![Status](https://img.shields.io/badge/status-experimental-orange) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.8%2B-blue)

**⚠️ Experimental / Beta**: This library is under active development. APIs may change. Test coverage is incomplete. Use with caution in production environments.

Automated quantitative analysis of IHC (immunohistochemistry) and H&E stained tissue sections.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/TingxuanGu/patholib.git
cd patholib

# Install with pip
pip install -e .

# For GPU support (recommended for H&E analysis)
pip install cellpose torch --index-url https://download.pytorch.org/whl/cu121

# For whole-slide image support
pip install openslide-python
```

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install openslide-tools
```

**macOS:**
```bash
brew install openslide
```

## Usage

### IHC Analysis
```bash
# Ki-67 (nuclear stain)
python analyze_ihc.py --input ki67_slide.tif --stain-type nuclear --marker Ki67 \
    --output-dir ./results --save-overlay --save-csv

# HER2 (membrane stain)
python analyze_ihc.py --input her2_slide.tif --stain-type membrane --marker HER2 \
    --output-dir ./results --save-overlay

# ER/PR (nuclear, with Allred score)
python analyze_ihc.py --input er_slide.tif --stain-type nuclear --marker ER \
    --output-dir ./results --save-overlay --save-csv
```

### H&E Analysis
```bash
# Inflammation scoring
python analyze_he.py --input tissue.tif --mode inflammation --use-gpu \
    --output-dir ./results --save-overlay --save-heatmap --save-csv

# Tumor/necrosis area ratio
python analyze_he.py --input tissue.tif --mode area-ratio \
    --output-dir ./results --save-overlay

# Both analyses
python analyze_he.py --input tissue.tif --mode both --use-gpu \
    --output-dir ./results --save-overlay --save-heatmap --save-csv
```

### Key Flags
- `--fail-fast`: Abort if specialized analyzer is unavailable (instead of silent fallback)
- `--use-gpu`: Enable GPU acceleration for Cellpose (H&E analysis)
- `--normalize-stain`: Apply Reinhard stain normalization before analysis

## Output
- `*_report.json`: Full analysis report with all metrics
- `*_cells.csv`: Per-cell data (centroid, area, grade, intensity)
- `*_overlay.png`: Annotated image with cell detections
- `*_heatmap.png`: Density heatmap (inflammation mode)

## Package Structure
```
patholib_deploy/
├── analyze_ihc.py          # IHC CLI
├── analyze_he.py           # H&E CLI
├── setup.sh                # One-click deploy script
├── setup.py                # pip install -e .
├── requirements.txt        # Dependencies
├── patholib/
│   ├── io/                 # Image loading + WSI tiling
│   ├── stain/              # Color deconvolution + normalization
│   ├── detection/          # Tissue/cell detection (CV + Cellpose)
│   ├── analysis/           # IHC + H&E analysis modules
│   ├── scoring/            # H-score, Allred, percentage
│   └── viz/                # Overlay, heatmap, report generation
└── references/             # Documentation
```

## Features

- **IHC Analysis**: Nuclear, membrane, and cytoplasmic staining quantification
- **H&E Analysis**: Inflammation scoring and tumor/necrosis area ratio
- **Scoring Methods**: H-score, Allred score, positive percentage
- **Detection Methods**: Classical CV (watershed) and deep learning (Cellpose)
- **Visualization**: Annotated overlays, density heatmaps, comprehensive reports

## Examples

See the `examples/` directory for batch processing scripts and parameter tuning examples.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```
@software{patholib2026,
  author = {Gu, Tingxuan},
  title = {patholib: Pathology Image Analysis Library},
  year = {2026},
  url = {https://github.com/TingxuanGu/patholib}
}
```

## Contributing

This is an experimental project. Contributions, bug reports, and feature requests are welcome via GitHub issues.
