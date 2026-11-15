# =====================================================================================
# Step 03 — Independent validation in GSE3218 (Korkola et al., MSKCC; Affymetrix U133A+B)
#   Same definitions as step 02 (TCGA); tests fixed before looking at GSE3218:
#   V1  GermRisk undiff (SEM+EC) > diff (TER+YST)      V2  seminoma vs EC
#   V3  undiff-vs-diff log2FC vs expression-matched random gene sets
#   V4  developmental block (fetal PGC- vs ProSG-specific genes)
#   V5  GermRisk ~ undiff + Pluripotency + Proliferation
#   sensitivity: one sample per patient (the series contains metastases of the same patients)
# Run in Jupyter (paste into one cell) or:  python 03_validation_V.py      (after 01 and 02)
# =====================================================================================
import os, sys, re, io, gzip, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
# ---------------- shared setup (paths, downloads, gene sets) ----------------
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
# ---------------- end shared setup ----------------

OUT = RES["V"]; log = Logger(os.path.join(OUT, "run_log.txt"))
N_PERM = int(os.environ.get("TGCT_NPERM", 10000)); rng = np.random.default_rng(20261113)
M = load_T_modules(); GERM_RISK, BG_PGC, BG_PROSG = M["GERM_RISK"], M["BG_PGC"], M["BG_PROSG"]

# ---------- GEO series matrices (U133A = GPL96, U133B = GPL97) ----------
def parse_matrix(path):
    meta, rows, intable = {}, [], False
    with gzip.open(path, "rt", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"): intable = True; continue
            if line.startswith("!series_matrix_table_end"): break
            if intable: rows.append(line.split("\t")); continue
            if line.startswith("!Sample_"):
                k, *v = line.split("\t"); meta.setdefault(k, []).append([x.strip('"') for x in v])
    hdr = [h.strip('"') for h in rows[0]]
    df = pd.DataFrame([r for r in rows[1:] if len(r) == len(hdr)], columns=hdr)
    df[hdr[0]] = df[hdr[0]].str.strip('"'); df = df.set_index(hdr[0]).apply(pd.to_numeric, errors="coerce")
    gsm = meta["!Sample_geo_accession"][0]
    desc = [" | ".join(v[i] for k in ["!Sample_title", "!Sample_source_name_ch1", "!Sample_characteristics_ch1"]
                       for v in meta.get(k, []) if i < len(v)) for i in range(len(gsm))]
    ann = pd.DataFrame({"gsm": gsm, "title": meta["!Sample_title"][0], "platform": meta["!Sample_platform_id"][0],
                        "description": desc}).set_index("gsm")
    return df[[c for c in df.columns if c in ann.index]], ann
def to_genes(df, gpl):
    p = fetch(geo_dir(gpl, "platforms") + f"annot/{gpl}.annot.gz", f"{gpl}.annot.gz", log)
    with gzip.open(p, "rt", errors="ignore") as f:
        t = pd.read_csv(io.StringIO("".join(l for l in f if not l.startswith(("!", "#", "^")))), sep="\t", low_memory=False)
    sym = t.set_index("ID")["Gene symbol"].dropna().astype(str).str.split("///").str[0].str.strip()
    x = np.log2(df.clip(lower=1)) if np.nanmax(df.values) > 100 else df.copy()
    x["sym"] = sym.reindex(x.index).values; x = x.dropna(subset=["sym"])
    x["_m"] = x.drop(columns="sym").mean(1)
    return x.sort_values("_m", ascending=False).drop_duplicates("sym").set_index("sym").drop(columns="_m")
def histology(desc):
    d = desc.lower()
    if "normal testis" in d: return "Normal"
    m = re.search(r"(pure|mixed)[^|]*", d)
    if m is None: return "?"
    if m.group(1) == "mixed": return "Mixed"
    for k, v in [("seminoma", "Seminoma"), ("embryonal", "Embryonal"), ("teratoma", "Teratoma"), ("yolk", "YolkSac"), ("chorio", "Chorio")]:
        if k in m.group(0): return v
    return "?"
parts = {}
for gpl in ["GPL96", "GPL97"]:
    fn = f"GSE3218-{gpl}_series_matrix.txt.gz"
    p = fetch(geo_dir("GSE3218", "series") + "matrix/" + fn, fn, log)
    if p is None: continue
    df, ann = parse_matrix(p)
    ann["key"] = ann.title.str.replace(r"\s+[12]$", "", regex=True).str.strip()
    ann["histology"] = ann.description.apply(histology)
    parts[gpl] = (to_genes(df, gpl), ann)
if "GPL96" not in parts: sys.exit("GSE3218 could not be downloaded -> check run_log.txt")
EA, AA = parts["GPL96"]; kA = AA.reset_index().set_index("key")
a = EA[kA.gsm]; a.columns = kA.index
if "GPL97" in parts:
    EB, AB = parts["GPL97"]; kB = AB.reset_index().set_index("key")
    common = kA.index.intersection(kB.index)
    b = EB[kB.loc[common, "gsm"]]; b.columns = common
    EXT = pd.concat([a[common], b.loc[~b.index.isin(a.index)]])
else:
    EXT = a
AEXT = kA.loc[EXT.columns]
AEXT[["gsm", "title", "description", "histology"]].to_csv(os.path.join(OUT, "V0_sample_annotation.csv"))
log(f"GSE3218: {EXT.shape[1]} samples x {EXT.shape[0]} genes (U133A+B merged)")
log("Histology (from GEO characteristics):", AEXT.histology.value_counts().to_dict())
log("Gene coverage:", {k: f"{sum(g in EXT.index for g in v)}/{len(v)}" for k, v in
                       {"GermRisk": GERM_RISK, "BG_PGC": BG_PGC, "BG_ProSG": BG_PROSG, **REF}.items()})

# ---------- scores ----------
tum = AEXT.index[AEXT.histology.isin(["Seminoma", "Embryonal", "Teratoma", "YolkSac", "Chorio", "Mixed"])]
X = EXT[tum]; X = X[(X > np.nanpercentile(X.values, 10)).mean(1) > 0.1]
Z = zscore_rows(X)
S = pd.DataFrame({"histology": AEXT.loc[tum, "histology"]})
for k, gl in {"GermRisk": GERM_RISK, "BG_PGC": BG_PGC, "BG_ProSG": BG_PROSG, **REF}.items():
    g = [x for x in gl if x in Z.index]; S[k] = Z.loc[g].mean(0)
S["patient"] = S.index.to_series().str.extract(r"(\d{3})[A-Z]\s*$")[0].fillna(S.index.to_series())
S.to_csv(os.path.join(OUT, "V0_sample_scores.csv"))
UND, DIF = ["Seminoma", "Embryonal"], ["Teratoma", "YolkSac"]
pure = S[S.histology.isin(UND + DIF)].copy(); pure["undiff"] = pure.histology.isin(UND).astype(float)
tests = []
def add(test, n, effect, p, **kw): tests.append(dict(cohort="GSE3218", test=test, n=n, effect=effect, p=p, **kw))
def mw(a, b, alt="two-sided"): return mannwhitneyu(a, b, alternative=alt).pvalue

for y in ["GermRisk", "BG_PGC", "Pluripotency", "Proliferation"]:
    u, d = pure.loc[pure.undiff == 1, y], pure.loc[pure.undiff == 0, y]
    add(f"V1 {y}: undiff (SEM+EC) > diff (TER+YST)", f"{len(u)} vs {len(d)}", u.median() - d.median(), mw(u, d, "greater"))
    s_, e_ = pure.loc[pure.histology == "Seminoma", y], pure.loc[pure.histology == "Embryonal", y]
    add(f"V2 {y}: seminoma vs EC", f"{len(s_)} vs {len(e_)}", s_.median() - e_.median(), mw(s_, e_))
for a_, b_, lab in [("Embryonal", DIF, "EC > TER+YST"), ("Embryonal", ["Teratoma"], "EC > TER"),
                    ("Embryonal", ["YolkSac"], "EC > YST"), ("Seminoma", DIF, "SEM > TER+YST")]:
    u, d = pure.loc[pure.histology == a_, "GermRisk"], pure.loc[pure.histology.isin(b_), "GermRisk"]
    add(f"V1b GermRisk {lab}", f"{len(u)} vs {len(d)}", u.median() - d.median(), mw(u, d, "greater"))
p1 = pure[~pure.patient.duplicated()]
u, d = p1.loc[p1.undiff == 1, "GermRisk"], p1.loc[p1.undiff == 0, "GermRisk"]
add("V1c GermRisk undiff > diff, one sample per patient", f"{len(u)} vs {len(d)}", u.median() - d.median(), mw(u, d, "greater"))
undI, difI = pure.index[pure.undiff == 1], pure.index[pure.undiff == 0]
lfc = X[undI].mean(1) - X[difI].mean(1)
gr = [g for g in GERM_RISK if g in lfc.index]
null_v3 = matched_null_mean(lfc, gr, X[list(undI) + list(difI)].mean(1), N_PERM, rng); obs_v3 = lfc[gr].mean()
add("V3 GermRisk undiff-vs-diff log2FC vs expression-matched random gene sets", len(gr), obs_v3,
    (1 + (null_v3 >= obs_v3).sum()) / (1 + N_PERM), z=(obs_v3 - null_v3.mean()) / null_v3.std(), null_mean=null_v3.mean())
pct = X.mean(1).rank(pct=True)
pg, ps = [g for g in BG_PGC if g in pct.index], [g for g in BG_PROSG if g in pct.index]
add("V4 block: fetal PGC-specific > ProSG-specific genes (tumour expression percentile)", f"{len(pg)} vs {len(ps)}",
    pct[pg].median() - pct[ps].median(), mw(pct[pg], pct[ps], "greater"))
pd.DataFrame({"gene": pg + ps, "class": ["PGC-specific"] * len(pg) + ["ProSG-specific"] * len(ps),
              "tumour_expr_percentile": list(pct[pg]) + list(pct[ps])}).to_csv(os.path.join(OUT, "V4_block_genes.csv"), index=False)
fit = smf.ols("GermRisk ~ undiff + Pluripotency + Proliferation", data=pure).fit(); ci = fit.conf_int().loc["undiff"]
add("V5 GermRisk ~ undiff + Pluripotency + Proliferation (coef undiff)", len(pure), fit.params["undiff"], fit.pvalues["undiff"],
    ci_low=ci[0], ci_high=ci[1])
semI, ecI = pure.index[pure.histology == "Seminoma"], pure.index[pure.histology == "Embryonal"]
pd.DataFrame([dict(gene=g, log2FC_undiff_vs_diff=lfc[g], p_undiff_vs_diff=mw(X.loc[g, undI], X.loc[g, difI]),
                   log2FC_sem_vs_EC=X.loc[g, semI].mean() - X.loc[g, ecI].mean()) for g in gr]) \
  .sort_values("log2FC_undiff_vs_diff", ascending=False).to_csv(os.path.join(OUT, "V1_per_gene_risk.csv"), index=False)
T = pd.DataFrame(tests); T.to_csv(os.path.join(OUT, "V_tests.csv"), index=False)
np.savez(os.path.join(OUT, "V_nulls.npz"), V3=null_v3, V3_obs=obs_v3)
pd.set_option("display.width", 250)
log("\nTESTS:\n" + T.drop(columns="cohort").round(5).to_string(index=False))
log.close(); print("\nStep 03 done ->", OUT)
