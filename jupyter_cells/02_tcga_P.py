# =====================================================================================
# Step 02 — TCGA-TGCT (UCSC Xena): is the fetal germ-risk programme retained in tumours?
#   P1  GermRisk score across histologies; undifferentiated (SEM+EC) vs differentiated (TER+YST);
#       seminoma vs EC; adjusted models; EC fraction within non-seminomas
#   P2  risk genes vs expression-matched NON-risk fetal germ genes (tumour expression)
#   P3  undiff-vs-diff log2FC of risk genes vs expression-matched random genes   (identical to step 03 V4)
#   P4  developmental block: fetal PGC-specific vs prospermatogonia-specific genes in tumours
#   P5  niche-risk genes in bulk tumours track stromal/Sertoli remnants (Spearman)
# Run in Jupyter (paste into one cell) or:  python 02_tcga_P.py      (after 01_fetal_T.py)
# =====================================================================================
import os, sys, re, gzip, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, kruskal, spearmanr
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

OUT = RES["P"]; log = Logger(os.path.join(OUT, "run_log.txt"))
N_PERM = int(os.environ.get("TGCT_NPERM", 10000)); rng = np.random.default_rng(0)
M = load_T_modules(); GERM_RISK, NICHE_RISK, BG_PGC, BG_PROSG, risk = M["GERM_RISK"], M["NICHE_RISK"], M["BG_PGC"], M["BG_PROSG"], M["risk"]
fetal = pd.read_csv(os.path.join(RES["T"], "T0_mean_CP10k_by_celltype.csv.gz"), index_col=0, compression="gzip")
log(f"GermRisk {len(GERM_RISK)} | NicheRisk {len(NICHE_RISK)} | BG_PGC {len(BG_PGC)} | BG_ProSG {len(BG_PROSG)}")

# ---------- data ----------
XENA = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
fx = fetch(XENA + "TCGA.TGCT.sampleMap%2FHiSeqV2.gz", "TCGA.TGCT.sampleMap__HiSeqV2.gz", log)
fc = fetch(XENA + "TCGA.TGCT.sampleMap%2FTGCT_clinicalMatrix", "TCGA.TGCT.sampleMap__TGCT_clinicalMatrix", log)
if not (fx and fc):
    sys.exit("TCGA files could not be downloaded from UCSC Xena (TCGA TGCT: HiSeqV2 + clinicalMatrix) -> put them in ~/TGCT_paper/raw/")
def read_tab(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as fh: return pd.read_csv(fh, sep="\t", index_col=0, low_memory=False)
expr = read_tab(fx); expr = expr.loc[:, [c for c in expr.columns if re.match(r"TCGA-..-....-01", c)]]
clin = read_tab(fc)
log("Expression:", expr.shape, "| clinical:", clin.shape)

def comp_type(x):
    x = str(x).lower()
    if "seminoma" in x and "non" not in x and "spermatocytic" not in x: return "Seminoma"
    if "embryonal" in x: return "Embryonal"
    if "yolk" in x: return "YolkSac"
    if "teratoma" in x: return "Teratoma"
    if "chorio" in x: return "Chorio"
    if "mixed" in x: return "Mixed"
    return None
def n_hits(col):
    return sum(any(comp_type(p) for p in str(x).split("|")) for x in clin[col].dropna())
cands = [c for c in clin.columns if re.search(r"histolog|diagnosis|tumor_type|morpholog", c, re.I)]
hcol = max(cands, key=n_hits)
pcol = next((c for c in clin.columns if "histolog" in c.lower() and "percent" in c.lower()), None)
log("Histology column:", hcol, "| percentage column:", pcol)
def classify(row):
    types = [comp_type(t) for t in str(row[hcol]).split("|")]
    pct = [float(x) if re.fullmatch(r"\d+(\.\d+)?", x) else np.nan for x in str(row.get(pcol, "")).split("|")] if pcol else []
    comps = {}
    for i, t in enumerate(types):
        if t is None: continue
        w = pct[i] if i < len(pct) and not np.isnan(pct[i]) and pct[i] <= 100 else np.nan
        comps[t] = comps.get(t, 0) + (0 if np.isnan(w) else w)
    if not comps: return pd.Series({"histology": np.nan, "seminoma_pct": np.nan, "EC_pct": np.nan})
    known = [t for t in comps if t != "Mixed"]; tot = sum(comps.values())
    sem = 100 * comps.get("Seminoma", 0) / tot if tot > 0 else (100.0 if known == ["Seminoma"] else np.nan)
    ec = 100 * comps.get("Embryonal", 0) / tot if tot > 0 else (100.0 if known == ["Embryonal"] else np.nan)
    if len(known) == 1 and "Mixed" not in comps: h = known[0]
    elif tot > 0 and max(comps.values()) / tot >= 0.95 and max(comps, key=comps.get) != "Mixed": h = max(comps, key=comps.get)
    else: h = "Mixed"
    return pd.Series({"histology": h, "seminoma_pct": sem, "EC_pct": ec})
meta = clin.reindex(expr.columns).apply(classify, axis=1); meta.index = expr.columns
log("Tumours by histology:", meta.histology.value_counts(dropna=False).to_dict())

# ---------- scores ----------
Z = zscore_rows(expr)
SETS = {"GermRisk": GERM_RISK, "NicheRisk": NICHE_RISK, "BG_PGC": BG_PGC, "BG_ProSG": BG_PROSG, **REF}
for k, gl in SETS.items():
    g = [x for x in gl if x in Z.index]; meta[k] = Z.loc[g].mean(0); log(f"score {k}: {len(g)}/{len(gl)} genes")
meta.to_csv(os.path.join(OUT, "P0_sample_scores.csv"))
UND, DIF = ["Seminoma", "Embryonal"], ["Teratoma", "YolkSac"]
pure = meta[meta.histology.isin(UND + DIF)].copy(); pure["undiff"] = pure.histology.isin(UND).astype(float)
tests = []
def add(test, n, effect, p, **kw): tests.append(dict(cohort="TCGA", test=test, n=n, effect=effect, p=p, **kw))
def mw(a, b, alt="two-sided"): return mannwhitneyu(a, b, alternative=alt).pvalue

# ---------- P1 ----------
order = [h for h in ["Seminoma", "Embryonal", "YolkSac", "Teratoma", "Mixed"] if (meta.histology == h).sum() >= 3]
kw_ = kruskal(*[meta.loc[meta.histology == h, "GermRisk"] for h in order])
add("P1 GermRisk across histologies (Kruskal-Wallis)", int(meta.histology.isin(order).sum()), np.nan, kw_.pvalue)
for y in ["GermRisk", "BG_PGC", "Pluripotency", "Proliferation"]:
    u, d = pure.loc[pure.undiff == 1, y], pure.loc[pure.undiff == 0, y]
    add(f"P1 {y}: undiff (SEM+EC) > diff (TER+YST)", f"{len(u)} vs {len(d)}", u.median() - d.median(), mw(u, d, "greater"))
    s_, e_ = pure.loc[pure.histology == "Seminoma", y], pure.loc[pure.histology == "Embryonal", y]
    add(f"P1 {y}: seminoma vs EC", f"{len(s_)} vs {len(e_)}", s_.median() - e_.median(), mw(s_, e_))
for a_, b_, lab in [("Embryonal", DIF, "EC > TER+YST"), ("Embryonal", ["Teratoma"], "EC > TER"),
                    ("Embryonal", ["YolkSac"], "EC > YST"), ("Seminoma", DIF, "SEM > TER+YST")]:
    u, d = pure.loc[pure.histology == a_, "GermRisk"], pure.loc[pure.histology.isin(b_), "GermRisk"]
    add(f"P1b GermRisk {lab}", f"{len(u)} vs {len(d)}", u.median() - d.median(), mw(u, d, "greater"))
# harmonised adjusted model (identical in GSE3218)
fit = smf.ols("GermRisk ~ undiff + Pluripotency + Proliferation", data=pure).fit()
ci = fit.conf_int().loc["undiff"]
add("P1c GermRisk ~ undiff + Pluripotency + Proliferation (coef undiff)", len(pure), fit.params["undiff"], fit.pvalues["undiff"],
    ci_low=ci[0], ci_high=ci[1])
# full model incl. remnant / stromal covariates, all tumours (reference = mixed)
d = meta.dropna(subset=["histology"]).copy()
d["is_sem"] = (d.histology == "Seminoma").astype(int); d["is_EC"] = (d.histology == "Embryonal").astype(int)
d["is_diff"] = d.histology.isin(["Teratoma", "YolkSac", "Chorio"]).astype(int)
f2 = smf.ols("GermRisk ~ is_sem + is_EC + is_diff + Pluripotency + Proliferation + Sertoli_remnant + Leydig_remnant + Fibroblast + Immune", data=d).fit()
for t in ["is_sem", "is_EC", "is_diff"]:
    c2 = f2.conf_int().loc[t]
    add(f"P1d full model (all tumours, ref = mixed): {t}", len(d), f2.params[t], f2.pvalues[t], ci_low=c2[0], ci_high=c2[1])
ns = meta[meta.histology.notna() & (meta.histology != "Seminoma")].dropna(subset=["EC_pct"])
r = spearmanr(ns.EC_pct, ns.GermRisk); res = smf.ols("GermRisk ~ Proliferation + Pluripotency", data=ns).fit().resid
r2 = spearmanr(ns.EC_pct, res)
add("P1e non-seminomas: GermRisk vs EC% (Spearman rho)", len(ns), r[0], r[1])
add("P1e non-seminomas: GermRisk residual (minus prolif./pluri.) vs EC% (rho)", len(ns), r2[0], r2[1])

# ---------- per-gene table ----------
undI, difI = pure.index[pure.undiff == 1], pure.index[pure.undiff == 0]
semI, ecI = pure.index[pure.histology == "Seminoma"], pure.index[pure.histology == "Embryonal"]
rows = []
for gname in [x for x in GERM_RISK if x in expr.index]:
    v = expr.loc[gname]
    rows.append(dict(gene=gname, module=risk.set_index("gene").module[gname],
                     log2FC_undiff_vs_diff=v[undI].mean() - v[difI].mean(), p_undiff_vs_diff=mw(v[undI], v[difI]),
                     log2FC_sem_vs_EC=v[semI].mean() - v[ecI].mean(), p_sem_vs_EC=mw(v[semI], v[ecI]),
                     rho_proliferation=spearmanr(v, meta.Proliferation)[0]))
GT = pd.DataFrame(rows).sort_values("log2FC_undiff_vs_diff", ascending=False)
GT.to_csv(os.path.join(OUT, "P1_per_gene_risk.csv"), index=False)

# ---------- P2 risk vs matched non-risk fetal germ genes ----------
fl = np.log1p(fetal[["Germ_PGC", "Germ_ProSG"]].mean(1))
cand = [g for g in M["bg"][~M["bg"].is_risk].index if g in expr.index and g in fl.index]
rk = [g for g in GERM_RISK if g in expr.index and g in fl.index]
tumour_mean = expr.mean(1)
# draw only from non-risk germ-specific genes, matched on fetal germ expression decile
bins = pd.qcut(fl.reindex(cand + rk).rank(method="first"), 10, labels=False)
pool = {k: tumour_mean[[g for g in cand if bins[g] == k]].to_numpy() for k in range(10)}
comp = pd.Series([bins[g] for g in rk]).value_counts(); null_p2 = np.zeros(N_PERM)
for k, n in comp.items(): null_p2 += pool[k][rng.integers(0, len(pool[k]), size=(N_PERM, n))].sum(1)
null_p2 /= len(rk); obs_p2 = tumour_mean[rk].mean()
add("P2 tumour expression: risk genes vs matched non-risk fetal germ genes", len(rk), obs_p2,
    (1 + (null_p2 >= obs_p2).sum()) / (1 + N_PERM), z=(obs_p2 - null_p2.mean()) / null_p2.std(), null_mean=null_p2.mean())

# ---------- P3 undiff-vs-diff log2FC vs expression-matched random genes ----------
lfc = expr[undI].mean(1) - expr[difI].mean(1)
gr = [g for g in GERM_RISK if g in lfc.index]
null_p3 = matched_null_mean(lfc, gr, expr[list(undI) + list(difI)].mean(1), N_PERM, rng)
obs_p3 = lfc[gr].mean()
add("P3 GermRisk undiff-vs-diff log2FC vs expression-matched random gene sets", len(gr), obs_p3,
    (1 + (null_p3 >= obs_p3).sum()) / (1 + N_PERM), z=(obs_p3 - null_p3.mean()) / null_p3.std(), null_mean=null_p3.mean())

# ---------- P4 developmental block ----------
pct = expr.mean(1).rank(pct=True)
pg, ps = [g for g in BG_PGC if g in pct.index], [g for g in BG_PROSG if g in pct.index]
add("P4 block: fetal PGC-specific > ProSG-specific genes (tumour expression percentile)", f"{len(pg)} vs {len(ps)}",
    pct[pg].median() - pct[ps].median(), mw(pct[pg], pct[ps], "greater"))
pd.DataFrame({"gene": pg + ps, "class": ["PGC-specific"] * len(pg) + ["ProSG-specific"] * len(ps),
              "tumour_expr_percentile": list(pct[pg]) + list(pct[ps])}).to_csv(os.path.join(OUT, "P4_block_genes.csv"), index=False)

# ---------- P5 niche-risk genes vs compartment scores ----------
rows = []
for gname in NICHE_RISK:
    if gname not in expr.index: continue
    rows.append(dict(gene=gname, **{f"rho_{c}": spearmanr(expr.loc[gname], meta[c])[0]
                                     for c in ["Sertoli_remnant", "Leydig_remnant", "Fibroblast", "Immune", "Pluripotency"]}))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "P5_niche_gene_correlations.csv"), index=False)

T = pd.DataFrame(tests); T.to_csv(os.path.join(OUT, "P_tests.csv"), index=False)
np.savez(os.path.join(OUT, "P_nulls.npz"), P2=null_p2, P2_obs=obs_p2, P3=null_p3, P3_obs=obs_p3)
pd.set_option("display.width", 250)
log("\nTESTS:\n" + T.drop(columns="cohort").round(5).to_string(index=False))
log("\nPer-gene (germ-risk):\n" + GT.round(3).to_string(index=False))
log.close(); print("\nStep 02 done ->", OUT)
