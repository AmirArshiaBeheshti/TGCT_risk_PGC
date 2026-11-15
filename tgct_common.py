"""Shared paths, logging, downloads and gene sets for the TGCT paper pipeline.

Folder layout (created automatically):
    ~/TGCT_paper/raw/        downloaded public data (kept between runs)
    ~/TGCT_paper/results/T   fetal single-cell analysis (GSE86146)
    ~/TGCT_paper/results/P   TCGA-TGCT analysis
    ~/TGCT_paper/results/V   independent validation (GSE3218)
    ~/TGCT_paper/figures     publication figures (PDF + PNG)
"""
import os, re, io, gzip, time, shutil, urllib.request
import numpy as np, pandas as pd

HOME = os.path.expanduser("~")
BASE = os.environ.get("TGCT_BASE", os.path.join(HOME, "TGCT_paper"))
RAW = os.path.join(BASE, "raw")
RES = {k: os.path.join(BASE, "results", k) for k in ["T", "P", "V"]}
FIG = os.path.join(BASE, "figures")
for d in [RAW, FIG, *RES.values()]:
    os.makedirs(d, exist_ok=True)

# raw files downloaded by earlier versions of the scripts are reused instead of downloaded again
OLD_RAW_DIRS = [os.path.join(HOME, "TGCT_GSE86146"), os.path.join(HOME, "TGCT_P"),
                os.path.join(HOME, "TGCT_P", "geo_cache"), os.path.join(HOME, "Downloads")]


class Logger:
    def __init__(self, path):
        self.fh = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.fh.write(s + "\n"); self.fh.flush()

    def close(self):
        self.fh.close()


def fetch(urls, fname, log=print, min_bytes=1000, timeout=180):
    """Return a local path for `fname` in RAW: reuse RAW or an old folder, else download from the first working URL."""
    dest = os.path.join(RAW, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > min_bytes:
        return dest
    for d in OLD_RAW_DIRS:
        old = os.path.join(d, fname)
        if os.path.exists(old) and os.path.getsize(old) > min_bytes:
            log(f"  reusing {old}")
            shutil.copy(old, dest)
            return dest
    for url in ([urls] if isinstance(urls, str) else urls):
        for k in range(2):
            try:
                log("  downloading", url)
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout) as r, open(dest + ".part", "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                if os.path.getsize(dest + ".part") > min_bytes:
                    os.replace(dest + ".part", dest)
                    return dest
            except Exception as e:
                log(f"   failed: {e}")
                time.sleep(2)
    return None


def geo_dir(acc, kind):
    num = re.sub(r"\D", "", acc)
    stub = acc[:len(acc) - 3] + "nnn" if len(num) > 3 else acc[:3] + "nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/{kind}/{stub}/{acc}/"


# ---------------- gene sets (defined once, shared by all steps) ----------------
# Pluta et al. 2021 Nat Commun, Table 1: target genes of the 22 novel loci
GWAS_NOVEL = ["PPP2R5A", "PACC1", "BCL2L11", "TERT", "TNXB", "BAK1", "DEPTOR", "DMRT1", "SSNA1", "ANAPC2",
              "TPRN", "ARL14EP", "MPPED2", "RAD52", "SP1", "AMHR2", "CYTH1", "USP36", "ENOSF1", "ZNF217",
              "SUPT20HL1", "PDK3", "AR", "CENPI", "DRP2", "TKTL1"]
# established loci genes named in Pluta 2021 / Ugalde-Morales 2025
GWAS_ESTABLISHED = ["KITLG", "SPRY4", "TEX14", "PRDM14", "DAZL", "TFCP2L1", "PITX1", "PKNOX2", "GATA4", "HNF1B",
                    "CLPTM1L", "AIFM3", "ATF7IP", "UCK2", "PMF1", "ZFPM1", "MCM3AP", "MPV17L", "GAB2", "RFWD3"]
# Ugalde-Morales et al. 2025 AJHG, Table 1: 46 leading TWAS genes
TWAS46 = ["KLHL21", "CCDC17", "PMF1", "UCK2", "MSH6", "DYSF", "BCL2L11", "KLF7", "EXOG", "PA2G4P4", "GPR160",
          "ZNF518B", "BOD1L1", "MIER3", "HSPA1L", "BAK1", "GINM1", "AC110781.3", "ZC3HAV1", "DEPTOR", "SMU1", "TPRN",
          "GAB2", "TMEM52B", "ATF7IP", "SP1", "CRYL1", "ARID3B", "RP11-20I23.5", "ZNF213-AS1", "MPV17L", "DCTPP1",
          "HEATR3", "TERF2", "ZFPM1", "USP36", "C18orf32", "CTB-31O20.4", "ZNF257", "ZNF675", "CAPNS1",
          "CTD-3162L10.1", "SMOX", "MCM3AP", "XXbac-B135H6.18", "RP4-539M6.14"]
GENESETS = {"GWAS": sorted(set(GWAS_NOVEL + GWAS_ESTABLISHED)), "TWAS": TWAS46,
            "Risk_union": sorted(set(GWAS_NOVEL + GWAS_ESTABLISHED + TWAS46))}

# reference programmes used for tumour scores (identical in TCGA and GSE3218)
REF = {
    "Pluripotency": ["POU5F1", "NANOG", "LIN28A", "DPPA3", "TFAP2C", "PRDM1", "KIT", "SOX17", "NANOS3"],
    "Proliferation": ["MKI67", "TOP2A", "CCNB1", "CDK1", "BUB1", "PCNA", "MCM2", "CENPF", "AURKA", "BIRC5"],
    "Sertoli_remnant": ["SOX9", "AMH", "INHA", "CLDN11", "FATE1", "DHH"],
    "Leydig_remnant": ["INSL3", "CYP17A1", "HSD3B2", "LHCGR", "STAR"],
    "Fibroblast": ["DCN", "LUM", "COL1A1", "COL1A2", "COL3A1", "PDGFRB", "THY1"],
    "Immune": ["PTPRC", "CD3E", "CD8A", "CD4", "CD19", "MS4A1", "CD68", "CD14"],
}


def load_T_modules():
    """Risk-gene modules and germ-specific background genes from step 01 (fetal single-cell)."""
    f1 = os.path.join(RES["T"], "T5_risk_gene_modules.csv")
    f2 = os.path.join(RES["T"], "T5_germ_specific_background.csv")
    if not (os.path.exists(f1) and os.path.exists(f2)):
        raise SystemExit("Run 01_fetal_T.py first (results/T/T5_*.csv not found).")
    risk = pd.read_csv(f1)
    bg = pd.read_csv(f2, index_col=0)
    germ_risk = risk.loc[risk.module.str.startswith("germ"), "gene"].tolist()
    niche_risk = risk.loc[risk.module == "niche", "gene"].tolist()
    B = bg[~bg.is_risk]
    return dict(risk=risk, bg=bg, GERM_RISK=germ_risk, NICHE_RISK=niche_risk,
                BG_PGC=B.index[B.module == "germ_PGC"].tolist(), BG_PROSG=B.index[B.module == "germ_ProSG"].tolist())


def zscore_rows(M):
    return M.sub(M.mean(1), axis=0).div(M.std(1) + 1e-9, axis=0)


def matched_null_mean(values, genes, level, n_perm, rng, n_bins=10):
    """Null distribution of mean(values[genes]) over random gene sets matched on `level` deciles."""
    pool_genes = level.index.intersection(values.index)
    dec = pd.qcut(level[pool_genes].rank(method="first"), n_bins, labels=False)
    pools = {d: values[dec.index[dec == d]].to_numpy() for d in range(n_bins)}
    comp = pd.Series([dec[g] for g in genes if g in dec.index]).value_counts()
    null = np.zeros(n_perm)
    for d, n in comp.items():
        pool = pools[d]
        null += pool[rng.integers(0, len(pool), size=(n_perm, n))].sum(1)
    return null / comp.sum()
