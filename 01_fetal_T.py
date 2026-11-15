# =====================================================================================
# Step 01 — Fetal testis single-cell atlas (GSE86146, Li et al. 2017, Smart-seq2)
#   * male fetal cells, QC, clustering, marker-based annotation incl. two germ states
#     (PGC-state: POU5F1/NANOG/NANOS3+ ; prospermatogonia: PIWIL4/TEX15/MAGEA4/TDRD1+)
#   * T1  cell-type specificity enrichment of TGCT risk genes (expression-matched bootstrap)
#   * T2  lineage-level enrichment (germ vs niche)
#   * T3  ligand–receptor communication (LIANA consensus): niche -> germ channels carrying risk genes
#   * T5  per-gene module assignment (germ_PGC / germ_ProSG / germ_both / niche / ...) + germ-specific
#         non-risk background genes  -> inputs for steps 02 and 03
# Run:  conda activate scvi-env ; python 01_fetal_T.py
# =====================================================================================
import os, sys, re, io, gc, tarfile, gzip, json, warnings, subprocess, importlib.util
warnings.filterwarnings("ignore")
need = [p for p in ["scanpy", "anndata", "liana", "statsmodels", "igraph", "leidenalg"] if importlib.util.find_spec(p) is None]
if need:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + need, check=False)
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tgct_common import RES, Logger, fetch, GENESETS

OUT = RES["T"]
log = Logger(os.path.join(OUT, "run_log.txt"))
N_BOOT = int(os.environ.get("TGCT_NBOOT", 10000)); SEED = 0
rng = np.random.default_rng(SEED)
log("Gene sets:", {k: len(v) for k, v in GENESETS.items()})

# ---------- 1. data ----------
TAR = fetch(["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE86nnn/GSE86146/suppl/GSE86146_RAW.tar",
             "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE86146&format=file"], "GSE86146_RAW.tar", log, min_bytes=1e6)
if TAR is None:
    sys.exit("GSE86146_RAW.tar could not be downloaded. Download it in a browser from GEO (GSE86146) into ~/TGCT_paper/raw/")
HK = {"GAPDH", "ACTB", "POU5F1", "SOX9", "DDX4", "MALAT1"}
def read_table(fobj, name):
    raw = fobj.read()
    if name.endswith(".gz"): raw = gzip.decompress(raw)
    txt = raw.decode("utf-8", errors="replace")
    sep = "\t" if txt[:5000].count("\t") >= txt[:5000].count(",") else ","
    df = pd.read_csv(io.StringIO(txt), sep=sep, low_memory=False)
    nonnum = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if nonnum:
        gcol = max(nonnum, key=lambda c: df[c].astype(str).isin(HK).sum())
        df.index = df[gcol].astype(str); df = df[num]
    if not set(df.index) & HK and set(df.columns.astype(str)) & HK: df = df.T
    return df.groupby(level=0).sum()
frames, metas = [], []
with tarfile.open(TAR) as tf:
    for m in tf.getmembers():
        if not m.isfile(): continue
        name = os.path.basename(m.name)
        sample = re.sub(r"^GSM\d+_|_gene_expression|\.(txt|tsv|csv)(\.gz)?$", "", name)
        if not sample.startswith("M_") or "_and_" in sample:      # male, unmixed samples only
            continue
        df = read_table(tf.extractfile(m), name)
        gsm = re.match(r"(GSM\d+)", name).group(1)
        wk = re.search(r"_(\d+)W_", sample)
        cells = [f"{gsm}:{c}" for c in df.columns.astype(str)]; df.columns = cells
        frames.append(df)
        metas.append(pd.DataFrame({"sample": sample, "gsm": gsm, "week": int(wk.group(1)) if wk else np.nan,
                                   "embryo": re.sub(r"_\d+$", "", sample)}, index=cells))
E = pd.concat(frames, axis=1, join="outer").fillna(0)
A = ad.AnnData(X=sp.csr_matrix(E.T.values.astype(np.float32)), obs=pd.concat(metas).loc[E.columns],
               var=pd.DataFrame(index=E.index.astype(str)))
del E, frames; gc.collect()
A.var_names_make_unique()
A.obs["n_genes"] = np.asarray((A.X > 0).sum(1)).ravel()
n0 = A.n_obs
A = A[A.obs.n_genes >= 1000].copy()
log(f"Male cells: {n0} -> {A.n_obs} after QC (>=1000 detected genes) | weeks: {A.obs.week.value_counts().sort_index().to_dict()}")
IS_COUNTS = np.allclose(A.X.data[:50000], np.round(A.X.data[:50000]))
L = A.copy()
if IS_COUNTS:
    sc.pp.normalize_total(L, target_sum=1e4)
else:
    rs = np.asarray(L.X.sum(1)).ravel(); rs[rs == 0] = 1
    L.X = sp.diags(1e4 / rs) @ L.X
LIN_X = L.X.copy().tocsr()            # linear CP10k
sc.pp.log1p(L)

# ---------- 2. annotation (cluster level; germ states as separate modules) ----------
MODULES = {
    "Germ_PGC":     ["POU5F1", "NANOG", "NANOS3", "TFAP2C", "DPPA3", "SOX17", "PRDM1", "ALPL", "KIT"],
    "Germ_ProSG":   ["PIWIL4", "TEX15", "MAGEA4", "TDRD1", "MAEL", "DDX4", "TDRD9", "MOV10L1", "SIX1"],
    "Sertoli":      ["SOX9", "AMH", "DHH", "FATE1", "INHA", "CITED1", "WT1", "CLDN11"],
    "Leydig":       ["CYP17A1", "INSL3", "HSD3B2", "LHCGR", "STAR"],
    "Interstitial": ["PDGFRA", "TCF21", "ARX", "DCN", "COL1A1", "NR2F2", "LUM"],
    "Peritubular":  ["ACTA2", "MYH11", "CNN1", "DES", "TAGLN"],
    "Endothelial":  ["PECAM1", "CDH5", "EMCN", "CD34", "KDR"],
    "Immune":       ["PTPRC", "CD68", "CD14", "LYZ", "TYROBP"],
}
for k, gs in MODULES.items():
    gs = [g for g in gs if g in L.var_names]
    sc.tl.score_genes(L, gs, score_name=f"sc_{k}", random_state=SEED)
sc.pp.highly_variable_genes(L, n_top_genes=2000)
sc.pp.pca(L, n_comps=30, use_highly_variable=True, random_state=SEED)
sc.pp.neighbors(L, n_neighbors=15, random_state=SEED)
sc.tl.leiden(L, resolution=1.0, random_state=SEED, key_added="cluster")
S = L.obs[[f"sc_{k}" for k in MODULES]]
Sz = (S - S.mean()) / (S.std() + 1e-9)
clu = Sz.groupby(L.obs.cluster).mean()
lab = clu.idxmax(axis=1).str.replace("sc_", "")
lab[clu.max(axis=1) < 0.3] = "Unassigned"
clu.assign(label=lab, n_cells=L.obs.cluster.value_counts()).round(2).to_csv(os.path.join(OUT, "T0_cluster_module_z.csv"))
log("Cluster annotation (module z-scores):\n" + clu.assign(label=lab, n=L.obs.cluster.value_counts()).round(2).to_string())
L.obs["celltype"] = L.obs.cluster.map(lab).astype(str)
g = L.obs.celltype.str.startswith("Germ").to_numpy()
ax_ = np.full(L.n_obs, np.nan)
pz = lambda v: (v - v[g].mean()) / (v[g].std() + 1e-9)
ax_[g] = (pz(L.obs.sc_Germ_ProSG.to_numpy()) - pz(L.obs.sc_Germ_PGC.to_numpy()))[g]
L.obs["germ_axis"] = ax_                                   # low = PGC-like, high = prospermatogonia-like
sc.tl.umap(L, random_state=SEED)
ct = L.obs.celltype.to_numpy()
log("Cell types:", pd.Series(ct).value_counts().to_dict())
log("Cell type x gestational week:\n" + pd.crosstab(L.obs.celltype, L.obs.week).to_string())
pd.crosstab(L.obs.celltype, L.obs.week).to_csv(os.path.join(OUT, "T0_celltype_by_week.csv"))
cm = L.obs[["sample", "week", "embryo", "n_genes", "cluster", "celltype", "germ_axis"]].copy()
cm["UMAP1"], cm["UMAP2"] = L.obsm["X_umap"][:, 0], L.obsm["X_umap"][:, 1]
cm.to_csv(os.path.join(OUT, "T0_cells_umap.csv"))

# ---------- 3. pseudobulk per cell type ----------
MIN_CELLS = 10
types = [c for c, n in pd.Series(ct).value_counts().items() if n >= MIN_CELLS and c != "Unassigned"]
order = [t for t in ["Germ_PGC", "Germ_ProSG", "Sertoli", "Leydig", "Interstitial", "Peritubular", "Endothelial", "Immune"] if t in types]
log("Cell types used (>=%d cells):" % MIN_CELLS, {t: int((ct == t).sum()) for t in order})
Xb = (A.X > 0).astype(np.float32).tocsr()
mean_ct = pd.DataFrame(np.vstack([np.asarray(LIN_X[ct == t].mean(0)).ravel() for t in order]), index=order, columns=A.var_names)
frac_ct = pd.DataFrame(np.vstack([np.asarray(Xb[ct == t].mean(0)).ravel() for t in order]), index=order, columns=A.var_names)
mean_ct.T.to_csv(os.path.join(OUT, "T0_mean_CP10k_by_celltype.csv.gz"), compression="gzip")
frac_ct.T.to_csv(os.path.join(OUT, "T0_frac_expressing_by_celltype.csv.gz"), compression="gzip")
ncell = {t: int((ct == t).sum()) for t in order}
json.dump(ncell, open(os.path.join(OUT, "T0_group_sizes.json"), "w"))
LIN = pd.Series({t: ("germ" if t.startswith("Germ") else "niche_supporting" if t == "Sertoli" else
                     "interstitial" if t in ("Leydig", "Interstitial", "Peritubular") else t.lower()) for t in order})

# ---------- 4. T1 / T2 specificity enrichment (EWCE-style, expression-decile-matched bootstrap) ----------
expr = mean_ct.loc[:, mean_ct.sum(0) > 0]
spec = expr / expr.sum(0); G = spec.columns; SP = spec.to_numpy()
dec = pd.qcut(np.log1p(expr.mean(0)).rank(method="first"), 10, labels=False).to_numpy()
dpool = {k: np.where(dec == k)[0] for k in range(10)}
def boot(gi, M):
    comp = pd.Series(dec[gi]).value_counts(); out = np.zeros((N_BOOT, M.shape[0]))
    for k, c in comp.items():
        pool = dpool[k]; out += M[:, pool[rng.integers(0, len(pool), size=(N_BOOT, c))]].sum(-1).T
    return out / len(gi)
levels = sorted(LIN.unique()); SL = np.vstack([(LIN == l).to_numpy() for l in levels]).astype(float) @ SP
r1, r2 = [], []
for name, gs in GENESETS.items():
    pr = [x for x in gs if x in G]
    log(f"{name}: {len(pr)}/{len(gs)} genes detected; missing: {sorted(set(gs) - set(pr))}")
    gi = np.array([G.get_loc(x) for x in pr])
    o = SP[:, gi].mean(1); nl = boot(gi, SP)
    for i, t in enumerate(spec.index):
        r1.append(dict(gene_set=name, celltype=t, lineage=LIN[t], n_cells=ncell[t], n_genes=len(pr), mean_specificity=o[i],
                       null_mean=nl[:, i].mean(), fold=o[i] / nl[:, i].mean(), z=(o[i] - nl[:, i].mean()) / nl[:, i].std(),
                       p=(1 + (nl[:, i] >= o[i]).sum()) / (1 + N_BOOT)))
    o2 = SL[:, gi].mean(1); nl2 = boot(gi, SL)
    for i, l in enumerate(levels):
        r2.append(dict(gene_set=name, lineage=l, fold=o2[i] / nl2[:, i].mean(), z=(o2[i] - nl2[:, i].mean()) / nl2[:, i].std(),
                       p=(1 + (nl2[:, i] >= o2[i]).sum()) / (1 + N_BOOT)))
T1 = pd.DataFrame(r1); T1["fdr"] = T1.groupby("gene_set").p.transform(lambda x: multipletests(x, method="fdr_bh")[1])
T2 = pd.DataFrame(r2); T2["fdr"] = T2.groupby("gene_set").p.transform(lambda x: multipletests(x, method="fdr_bh")[1])
T1.to_csv(os.path.join(OUT, "T1_celltype_enrichment.csv"), index=False)
T2.to_csv(os.path.join(OUT, "T2_lineage_enrichment.csv"), index=False)
log("\nT1 cell-type enrichment:\n" + T1.sort_values(["gene_set", "p"]).round(4).to_string(index=False))
log("\nT2 lineage enrichment:\n" + T2.round(4).to_string(index=False))

# ---------- 5. T3 ligand–receptor communication (LIANA consensus) ----------
try:
    import liana as li
    S3 = L[L.obs.celltype.isin(order)].copy(); S3.obs["ct"] = pd.Categorical(S3.obs.celltype, categories=order)
    try:
        li.mt.rank_aggregate(S3, groupby="ct", resource_name="consensus", expr_prop=0.1, use_raw=False,
                             n_perms=1000, seed=SEED, verbose=False)
    except TypeError:                                   # older liana versions have no `seed` argument
        li.mt.rank_aggregate(S3, groupby="ct", resource_name="consensus", expr_prop=0.1, use_raw=False,
                             n_perms=1000, verbose=False)
    lr = S3.uns["liana_res"].copy()
    lr.to_csv(os.path.join(OUT, "T3_liana_all.csv.gz"), index=False, compression="gzip")
    lr["src_lin"] = lr.source.map(LIN); lr["tgt_lin"] = lr.target.map(LIN)
    def genes_of(df):
        out = set()
        for c in ["ligand_complex", "receptor_complex"]:
            for x in df[c].astype(str): out.update(x.split("_"))
        return out
    universe = genes_of(lr); sig = lr[lr.specificity_rank <= 0.05]
    res = []
    for direction, (s_, t_) in {"niche->germ": ("niche_supporting", "germ"), "germ->niche": ("germ", "niche_supporting"),
                                "interstitial->germ": ("interstitial", "germ")}.items():
        hits = genes_of(sig[(sig.src_lin == s_) & (sig.tgt_lin == t_)])
        for name, gs in GENESETS.items():
            R = set(gs) & universe
            a_, b_, c_ = len(R & hits), len(R - hits), len(hits - R); d_ = len(universe - hits - R)
            OR, p = fisher_exact([[a_, b_], [c_, d_]], alternative="greater")
            res.append(dict(direction=direction, gene_set=name, risk_genes_in_LR_resource=len(R), risk_genes_in_channel=a_,
                            channel_genes=len(hits), OR=OR, p=p, genes=",".join(sorted(R & hits))))
    pd.DataFrame(res).to_csv(os.path.join(OUT, "T3_LR_channel_enrichment.csv"), index=False)
    log("\nT3 ligand–receptor channel enrichment:\n" + pd.DataFrame(res).round(4).to_string(index=False))
    allrisk = set(GENESETS["Risk_union"])
    has = lambda x: bool(set(str(x).split("_")) & allrisk)
    rk = lr[(lr.ligand_complex.apply(has) | lr.receptor_complex.apply(has)) &
            (lr.source.str.startswith("Germ") | lr.target.str.startswith("Germ"))].sort_values("specificity_rank")
    rk.to_csv(os.path.join(OUT, "T3_risk_gene_interactions_with_germ.csv"), index=False)
    kit = lr[(lr.ligand_complex == "KITLG") & (lr.receptor_complex.astype(str).str.contains("KIT"))].sort_values("specificity_rank")
    kit.to_csv(os.path.join(OUT, "T3_KITLG_KIT.csv"), index=False)
    log("\nKITLG -> KIT (all sender/receiver pairs):\n" + kit[["source", "target", "receptor_complex", "magnitude_rank", "specificity_rank"]].to_string(index=False))
except Exception as e:
    log("T3 (LIANA) failed:", repr(e))

# ---------- 6. T5 per-gene modules + germ-specific background ----------
lfc = np.log2((mean_ct.loc["Germ_ProSG"] + 0.5) / (mean_ct.loc["Germ_PGC"] + 0.5))
rows = []
for x in GENESETS["Risk_union"]:
    if x not in G: continue
    top = spec[x].idxmax(); ms = spec[x].max()
    lin = LIN[top] if ms >= 1.5 / len(spec) else "nonspecific"
    if lin == "germ": mod = "germ_PGC" if lfc[x] < -1 else ("germ_ProSG" if lfc[x] > 1 else "germ_both")
    elif lin in ("niche_supporting", "interstitial"): mod = "niche"
    else: mod = lin if lin == "nonspecific" else "other"
    rows.append(dict(gene=x, module=mod, top_celltype=top, max_spec=round(ms, 3),
                     germ_spec=round(spec.loc[["Germ_PGC", "Germ_ProSG"], x].sum(), 3), log2FC_ProSG_vs_PGC=round(lfc[x], 2),
                     in_GWAS=x in GENESETS["GWAS"], in_TWAS=x in GENESETS["TWAS"],
                     **{f"CP10k_{t}": round(mean_ct.loc[t, x], 2) for t in order}))
T5 = pd.DataFrame(rows).sort_values(["module", "max_spec"], ascending=[True, False])
T5.to_csv(os.path.join(OUT, "T5_risk_gene_modules.csv"), index=False)
log("\nT5 risk-gene modules:\n" + T5[["gene", "module", "top_celltype", "max_spec", "germ_spec", "log2FC_ProSG_vs_PGC"]].to_string(index=False))
log("Module sizes:", T5.module.value_counts().to_dict())
gspec = spec.loc[["Germ_PGC", "Germ_ProSG"]].sum(0); expressed = frac_ct.max(0) >= 0.1
bg = pd.DataFrame({"germ_spec": gspec, "log2FC": lfc.reindex(G)}).loc[(gspec >= 0.6) & expressed.reindex(G).fillna(False)]
bg["module"] = np.where(bg.log2FC < -1, "germ_PGC", np.where(bg.log2FC > 1, "germ_ProSG", "germ_both"))
bg["is_risk"] = bg.index.isin(GENESETS["Risk_union"])
bg.to_csv(os.path.join(OUT, "T5_germ_specific_background.csv"))
log("Germ-specific background genes:", bg[~bg.is_risk].module.value_counts().to_dict())
log.close()
print("\nStep 01 done ->", OUT)
