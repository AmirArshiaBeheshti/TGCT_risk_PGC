# TGCT inherited-risk programme: fetal origin and tumour retention — analysis code

Run order (each step reads the previous step's results):

| Step | Script | Data | Output |
|---|---|---|---|
| 01 | `01_fetal_T.py` | GSE86146 (fetal testis, Smart-seq2) | `results/T` |
| 02 | `02_tcga_P.py` | TCGA-TGCT (UCSC Xena) | `results/P` |
| 03 | `03_validation_V.py` | GSE3218 (MSKCC, U133A+B) | `results/V` |
| 04 | `04_figures.py` | results only | `figures/` (PDF, PNG 600 dpi, TIFF 600 dpi), Table 1, Supplementary_Tables.xlsx |

All folders are created under `~/TGCT_paper/`. Downloads go to `~/TGCT_paper/raw/`
(files already downloaded by earlier scripts in `~/TGCT_GSE86146` or `~/TGCT_P` are reused).

```
conda activate scvi-env
bash run_all.sh            # or run the four scripts one by one
```
Random seeds are fixed; bootstrap/permutation counts: 10,000 (env vars TGCT_NBOOT / TGCT_NPERM).
`tgct_common.py` holds paths, the fixed gene sets (Pluta 2021; Ugalde-Morales 2025) and shared helpers.
