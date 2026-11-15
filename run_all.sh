#!/usr/bin/env bash
# Full TGCT paper pipeline. Run from this folder:   conda activate scvi-env ; bash run_all.sh
set -e
cd "$(dirname "$0")"
python 01_fetal_T.py
python 02_tcga_P.py
python 03_validation_V.py
python 04_figures.py
