# =====================================================================================
# Step 04 — publication figures and tables (reads only results/T, results/P, results/V)
#   Fig 1  study design
#   Fig 2  fetal testis: cell types, germ states, risk-gene enrichment, KITLG->KIT, risk-gene expression
#   Fig 3  tumours: GermRisk by histology in TCGA (discovery) and GSE3218 (validation), robustness
#   Fig 4  developmental block: fetal PGC-state genes are retained in tumours, prospermatogonia genes are not
#   Fig S1-S3 supplementary;  Tables: Table1_replication.csv + Supplementary_Tables.xlsx
# Output: ~/TGCT_paper/figures/*.pdf (vector, editable text) + *.png (600 dpi) + *.tiff (600 dpi, LZW)
# Run in Jupyter (paste into one cell) or:  python 04_figures.py
# =====================================================================================
import os, sys, json, zipfile, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
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

# ---------------- style ----------------
from matplotlib import font_manager
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"] if f in avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "mathtext.default": "regular"})
MM = 1 / 25.4; W2 = 180 * MM                       # double-column width
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "purple": "#CC79A7",
         "vermillion": "#D55E00", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#8C8C8C"}
CT_COL = {"Germ_PGC": OKABE["vermillion"], "Germ_ProSG": OKABE["orange"], "Sertoli": OKABE["blue"],
          "Interstitial": OKABE["green"], "Leydig": OKABE["sky"], "Peritubular": OKABE["purple"],
          "Endothelial": OKABE["yellow"], "Immune": OKABE["grey"], "Unassigned": "#D9D9D9"}
CT_LAB = {"Germ_PGC": "Germ: PGC state", "Germ_ProSG": "Germ: prospermatogonia", "Sertoli": "Sertoli",
          "Interstitial": "Interstitial", "Leydig": "Leydig", "Peritubular": "Peritubular",
          "Endothelial": "Endothelial", "Immune": "Immune", "Unassigned": "Unassigned"}
H_COL = {"Seminoma": OKABE["blue"], "Embryonal": OKABE["sky"], "YolkSac": OKABE["orange"],
         "Teratoma": OKABE["vermillion"], "Mixed": OKABE["grey"], "Chorio": OKABE["purple"]}
H_LAB = {"Seminoma": "SEM", "Embryonal": "EC", "YolkSac": "YST", "Teratoma": "TER", "Mixed": "Mixed", "Chorio": "CC"}

def fmt_p(p):
    if p is None or not np.isfinite(p): return "n.a."
    if p >= 0.001: return f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}"
    m, e = f"{p:.1e}".split("e"); return rf"P = {m}$\times$10$^{{{int(e)}}}$"
def stars(q): return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
def letter(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
def save(fig, name):
    for ext, kw in [("pdf", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        try: fig.savefig(os.path.join(FIG, f"{name}.{ext}"), **kw)
        except Exception as e: print("  could not save", name, ext, e)
    plt.close(fig); print("saved", name)
def null_label(ax, nl, obs, r):
    lo, hi = ax.get_xlim(); w = hi - lo
    ax.set_xlim(lo, max(hi, obs + 0.42 * w))                       # room to the right of the observed line
    ax.text(obs + 0.03 * w, ax.get_ylim()[1] * 0.97, f"risk genes\nz = {r.z:.1f}\n{fmt_p(r.p)}",
            color=OKABE["vermillion"], fontsize=5.8, va="top", ha="left")
def box_strip(ax, data, order, colors, ylabel, rng):
    vals = [data.loc[data.histology == h, "GermRisk"].values for h in order]
    bp = ax.boxplot(vals, positions=range(len(order)), widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=0.9), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    boxprops=dict(lw=0.6))
    for patch, h in zip(bp["boxes"], order):
        patch.set_facecolor(colors[h]); patch.set_alpha(0.35); patch.set_edgecolor(colors[h])
    for i, (v, h) in enumerate(zip(vals, order)):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=colors[h], edgecolor="white", lw=0.3, zorder=3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{H_LAB[h]}\n{len(v)}" for h, v in zip(order, vals)])
    ax.set_ylabel(ylabel); ax.axhline(0, color="#BBBBBB", lw=0.5, ls=":", zorder=0)

rng = np.random.default_rng(0)
Tm = load_T_modules(); risk = Tm["risk"]
T = RES["T"]; P = RES["P"]; V = RES["V"]
cells = pd.read_csv(os.path.join(T, "T0_cells_umap.csv"), index_col=0)
mean_ct = pd.read_csv(os.path.join(T, "T0_mean_CP10k_by_celltype.csv.gz"), index_col=0, compression="gzip")
frac_ct = pd.read_csv(os.path.join(T, "T0_frac_expressing_by_celltype.csv.gz"), index_col=0, compression="gzip")
ncell = json.load(open(os.path.join(T, "T0_group_sizes.json")))
T1 = pd.read_csv(os.path.join(T, "T1_celltype_enrichment.csv")); T2 = pd.read_csv(os.path.join(T, "T2_lineage_enrichment.csv"))
kit = pd.read_csv(os.path.join(T, "T3_KITLG_KIT.csv")) if os.path.exists(os.path.join(T, "T3_KITLG_KIT.csv")) else None
lrc = pd.read_csv(os.path.join(T, "T3_LR_channel_enrichment.csv")) if os.path.exists(os.path.join(T, "T3_LR_channel_enrichment.csv")) else None
SP = pd.read_csv(os.path.join(P, "P0_sample_scores.csv"), index_col=0); SV = pd.read_csv(os.path.join(V, "V0_sample_scores.csv"), index_col=0)
TP = pd.read_csv(os.path.join(P, "P_tests.csv")); TV = pd.read_csv(os.path.join(V, "V_tests.csv"))
NP = np.load(os.path.join(P, "P_nulls.npz")); NV = np.load(os.path.join(V, "V_nulls.npz"))
GP = pd.read_csv(os.path.join(P, "P1_per_gene_risk.csv")); GV = pd.read_csv(os.path.join(V, "V1_per_gene_risk.csv"))
BP = pd.read_csv(os.path.join(P, "P4_block_genes.csv")); BV = pd.read_csv(os.path.join(V, "V4_block_genes.csv"))
def tp(df, key):
    r = df[df.test == key]
    if not len(r): r = df[df.test.str.startswith(key + " ")] if (df.test.str.startswith(key + " ")).any() else df[df.test.str.startswith(key)]
    return r.iloc[0] if len(r) else None
order_ct = [c for c in CT_COL if c in mean_ct.columns]
n_cells = len(cells); n_germrisk = len(Tm["GERM_RISK"])

# =====================================================================================
# Figure 1 — study design
# =====================================================================================
fig = plt.figure(figsize=(W2, 62 * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 180); ax.set_ylim(0, 62); ax.axis("off")
def box(x, y, w, h, title, body, col):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=col + "1A", ec=col, lw=0.8))
    ax.text(x + 1.5, y + h - 1.4, title, fontsize=6.8, fontweight="bold", va="top", color="black")
    ax.text(x + 1.5, y + h - 5.6, body, fontsize=6, va="top", linespacing=1.35)
def arrow(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#444444", mutation_scale=7))
nP = int(SP.histology.notna().sum()); nV = int(SV.histology.isin(["Seminoma", "Embryonal", "Teratoma", "YolkSac", "Mixed", "Chorio"]).sum())
wk = f"{int(cells.week.min())}–{int(cells.week.max())} weeks"
for x, t in [(2, "Inherited risk"), (62, "Fetal origin (single-cell)"), (124, "Tumour retention (bulk)")]:
    ax.text(x, 59.5, t, fontsize=7.5, fontweight="bold", color=OKABE["grey"])
box(2, 35, 52, 20, "GWAS + TWAS risk genes",
    "78 GWAS loci (Pluta et al. 2021)\n46 TWAS genes (Ugalde-Morales et al. 2025)\n"
    f"→ {len(set(risk.gene))} risk genes detected in fetal testis", OKABE["purple"])
box(2, 4, 52, 22, "Control analyses",
    "Expression-decile-matched random gene sets\n(10,000 draws); non-risk fetal germ genes;\nadjustment for pluripotency and proliferation", OKABE["grey"])
box(62, 35, 54, 20, "Fetal testis atlas (GSE86146)",
    f"Smart-seq2, {n_cells:,} male cells, {wk}\nGerm cells: PGC state vs prospermatogonia\nNiche: Sertoli, interstitial, immune", OKABE["vermillion"])
box(62, 4, 54, 22, "Risk-gene mapping",
    "Cell-type specificity enrichment\nLigand–receptor channels (LIANA)\n"
    f"→ {n_germrisk} germ-module risk genes (GermRisk)\n→ fetal PGC- vs ProSG-specific genes", OKABE["vermillion"])
box(124, 35, 54, 20, "Discovery: TCGA-TGCT",
    f"RNA-seq, {nP} primary tumours\nSEM / EC / YST / TER / mixed\nGermRisk score, block test", OKABE["blue"])
box(124, 4, 54, 22, "Validation: GSE3218 (MSKCC)",
    f"Affymetrix U133A+B, {nV} tumours\nidentical scores and tests\napplied unchanged", OKABE["blue"])
arrow(54.6, 45, 61.4, 45); arrow(89, 34.4, 89, 26.6); arrow(116.6, 15, 123.4, 40); arrow(116.6, 15, 123.4, 15)
arrow(151, 34.4, 151, 26.6); arrow(54.6, 15, 61.4, 15)
save(fig, "Fig1_study_design")

# =====================================================================================
# Figure 2 — fetal testis
# =====================================================================================
fig = plt.figure(figsize=(W2, 175 * MM))
gs = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.05, 0.85, 0.62], wspace=0.75, hspace=0.95)
SHORT = lambda c: CT_LAB[c].replace("Germ: ", "").replace("prospermatogonia", "ProSG")
# (a) UMAP
ax = fig.add_subplot(gs[0, 0])
for c in [c for c in CT_COL if c in cells.celltype.unique()]:
    m = cells.celltype == c
    ax.scatter(cells.UMAP1[m], cells.UMAP2[m], s=2.2, color=CT_COL[c], lw=0, rasterized=True, label=f"{SHORT(c)} ({m.sum()})")
    if c != "Unassigned" and m.sum() >= 10:
        ax.text(cells.UMAP1[m].median(), cells.UMAP2[m].median(), SHORT(c), fontsize=6, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title(f"Male fetal testis, {n_cells:,} cells"); letter(ax, "a", -0.08)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, markerscale=3, handletextpad=0.2, columnspacing=0.6, fontsize=5.6)
# (b) dot plot of markers
ax = fig.add_subplot(gs[0, 1:])
MK = [("PGC state", ["POU5F1", "NANOG", "NANOS3", "TFAP2C", "SOX17"]), ("Prospermatogonia", ["PIWIL4", "TEX15", "MAGEA4", "TDRD1", "DDX4"]),
      ("Sertoli", ["SOX9", "AMH", "FATE1"]), ("Interstitial", ["PDGFRA", "DCN", "TCF21"]), ("Leydig", ["INSL3", "CYP17A1"]),
      ("Immune", ["PTPRC", "CD68"]), ("Risk genes", ["KITLG", "KIT", "PRDM14", "DMRT1", "TFCP2L1", "TKTL1"])]
genes = [g for _, gl in MK for g in gl if g in mean_ct.index]
mexp = np.log1p(mean_ct.loc[genes, order_ct]); mz = mexp.sub(mexp.min(1), axis=0).div(mexp.max(1) - mexp.min(1) + 1e-9, axis=0)
fr = frac_ct.loc[genes, order_ct]
for i, ctn in enumerate(order_ct):
    sca = ax.scatter(np.arange(len(genes)), np.full(len(genes), i), s=fr[ctn].values * 55 + 0.5, c=mz[ctn].values,
                     cmap="Reds", vmin=0, vmax=1, edgecolor="#555555", lw=0.25)
ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=90, fontstyle="italic")
ax.set_yticks(range(len(order_ct))); ax.set_yticklabels([f"{SHORT(c)} (n={ncell.get(c, '')})" for c in order_ct])
ax.spines[["left", "bottom"]].set_visible(False); ax.tick_params(length=0)
x0 = 0; stagger = False
for lab, gl in MK:
    k = sum(g in mean_ct.index for g in gl)
    if k == 0: continue
    ax.plot([x0 - 0.3, x0 + k - 0.7], [-0.85, -0.85], color="black", lw=0.7, clip_on=False)
    stag = 0.32 if (k <= 2 and stagger) else 0.0
    ax.text(x0 + (k - 1) / 2, -1.05 - stag, lab, ha="center", va="bottom", fontsize=5.8)
    stagger = not stagger if k <= 2 else False
    x0 += k
ax.set_xlim(-0.6, len(genes) - 0.4); ax.set_ylim(len(order_ct) - 0.5, -2.0)
cax = ax.inset_axes([1.015, 0.5, 0.022, 0.45]); cb = fig.colorbar(sca, cax=cax); cb.set_label("scaled mean\nexpression", fontsize=5.8); cb.ax.tick_params(labelsize=5.5)
hs = [ax.scatter([], [], s=f_ * 55 + 0.5, color="white", edgecolor="#555555", lw=0.4) for f_ in [0.25, 0.5, 0.75]]
ax.legend(hs, ["25%", "50%", "75%"], title="cells\nexpressing", loc="upper left", bbox_to_anchor=(1.0, 0.42), fontsize=5.5,
          title_fontsize=5.5, labelspacing=0.8)
ax.set_title("Germ-state and niche markers", pad=16); letter(ax, "b", -0.2, 1.12)
# (c) enrichment heatmap
ax = fig.add_subplot(gs[1, 0])
sets = [s for s in ["GWAS", "TWAS", "Risk_union"] if s in T1.gene_set.unique()]
Zm = T1.pivot(index="celltype", columns="gene_set", values="z").loc[order_ct, sets]
Qm = T1.pivot(index="celltype", columns="gene_set", values="fdr").loc[order_ct, sets]
lim = max(3, np.nanmax(np.abs(Zm.values)))
im = ax.imshow(Zm.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
for i in range(Zm.shape[0]):
    for j in range(Zm.shape[1]):
        ax.text(j, i, f"{Zm.values[i, j]:.1f}{stars(Qm.values[i, j])}", ha="center", va="center", fontsize=6,
                color="white" if abs(Zm.values[i, j]) > lim * 0.6 else "black")
ax.set_xticks(range(len(sets))); ax.set_xticklabels([{"Risk_union": "Union"}.get(s, s) for s in sets])
ax.set_yticks(range(len(order_ct))); ax.set_yticklabels([SHORT(c) for c in order_ct]); ax.tick_params(length=0)
ax.spines[["left", "bottom"]].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.07, pad=0.04); cb.set_label("z", fontsize=6, rotation=0, labelpad=4); cb.ax.tick_params(labelsize=5.5)
ax.annotate("*FDR<0.05  **<0.01  ***<0.001", (0.5, -0.13), xycoords="axes fraction", ha="center", va="top", fontsize=5.4, color="#444444")
ax.set_title("Risk-gene specificity"); letter(ax, "c", -0.55)
# (d) lineage-level enrichment
ax = fig.add_subplot(gs[1, 1])
t2 = T2[T2.gene_set == "Risk_union"].copy(); t2["lab"] = t2.lineage.map({"germ": "Germ", "niche_supporting": "Sertoli (niche)",
                                                                          "interstitial": "Interstitial", "immune": "Immune", "endothelial": "Endothelial"}).fillna(t2.lineage)
t2 = t2.sort_values("fold")
ax.barh(range(len(t2)), t2.fold - 1, left=1, color=[OKABE["vermillion"] if l == "germ" else "#BBBBBB" for l in t2.lineage], height=0.55)
for i, (_, r) in enumerate(t2.iterrows()):
    ax.text(max(r.fold, 1) + 0.015, i, fmt_p(r.p), va="center", fontsize=5.6)
ax.axvline(1, color="#666666", lw=0.6); ax.set_yticks(range(len(t2))); ax.set_yticklabels(t2.lab)
ax.set_xlim(min(0.8, t2.fold.min() - 0.05), max(1.3, t2.fold.max() + 0.25))
ax.set_xlabel("fold enrichment\n(vs expression-matched genes)"); ax.set_title("Germ vs niche lineages"); letter(ax, "d", -0.5)
# (e) KITLG -> KIT communication
ax = fig.add_subplot(gs[1, 2])
if kit is not None and len(kit):
    kk = kit[kit.receptor_complex.astype(str) == "KIT"] if (kit.receptor_complex.astype(str) == "KIT").any() else kit
    snd = [c for c in order_ct if c in kk.source.unique()]; rcv = [c for c in order_ct if c in kk.target.unique()]
    ps_ = kk.pivot_table(index="source", columns="target", values="specificity_rank", aggfunc="min").reindex(index=snd, columns=rcv)
    pm_ = kk.pivot_table(index="source", columns="target", values="magnitude_rank", aggfunc="min").reindex(index=snd, columns=rcv)
    for i, s_ in enumerate(snd):
        for j, t_ in enumerate(rcv):
            sr, mr = ps_.loc[s_, t_], pm_.loc[s_, t_]
            if not np.isfinite(mr): continue
            sz = np.clip(-np.log10(max(sr, 1e-3)), 0, 3) / 3 * 80 + 4
            sc_ = ax.scatter(j, i, s=sz, c=[-np.log10(max(mr, 1e-4))], cmap="Purples", vmin=0, vmax=3, edgecolor="#444444", lw=0.3, zorder=3)
            if sr <= 0.05: ax.scatter(j, i, s=sz + 40, facecolor="none", edgecolor="black", lw=0.8, zorder=2)
    ax.set_xticks(range(len(rcv))); ax.set_xticklabels([SHORT(c) for c in rcv], rotation=40, ha="right")
    ax.set_yticks(range(len(snd))); ax.set_yticklabels([SHORT(c) for c in snd])
    ax.set_xlim(-0.6, len(rcv) - 0.4); ax.set_ylim(len(snd) - 0.4, -0.6)
    ax.set_xlabel("receiver ($KIT$)"); ax.set_ylabel("sender ($KITLG$)")
    cb = fig.colorbar(sc_, ax=ax, fraction=0.07, pad=0.04); cb.set_label("−log$_{10}$ magnitude rank", fontsize=5.8); cb.ax.tick_params(labelsize=5.5)
ax.set_title("$KITLG$ → $KIT$ signalling"); letter(ax, "e", -0.5)
# (f) risk-gene expression strip (genes on x)
ax = fig.add_subplot(gs[2, :])
rk = risk[risk.module.isin(["germ_PGC", "germ_both", "germ_ProSG", "niche"])].copy()
rk["mord"] = rk.module.map({"germ_PGC": 0, "germ_both": 1, "germ_ProSG": 2, "niche": 3})
rk = rk[rk.gene.isin(mean_ct.index)].sort_values(["mord", "log2FC_ProSG_vs_PGC"])
H = np.log1p(mean_ct.loc[rk.gene, order_ct]); H = H.sub(H.mean(1), axis=0).div(H.std(1) + 1e-9, axis=0)
im = ax.imshow(H.T.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
ax.set_xticks(range(len(rk))); ax.set_xticklabels(rk.gene, rotation=90, fontsize=5.2, fontstyle="italic")
ax.set_yticks(range(len(order_ct))); ax.set_yticklabels([SHORT(c) for c in order_ct], fontsize=6)
ax.tick_params(length=0); ax.spines[["left", "bottom"]].set_visible(False)
MODC = {"germ_PGC": OKABE["vermillion"], "germ_both": "#E8A07A", "germ_ProSG": OKABE["orange"], "niche": OKABE["blue"]}
MODL = {"germ_PGC": "germ: PGC-biased", "germ_both": "germ: both states", "germ_ProSG": "germ: ProSG-biased", "niche": "niche"}
for i, m in enumerate(rk.module): ax.add_patch(patches.Rectangle((i - 0.5, -1.3), 1, 0.55, color=MODC[m], clip_on=False, lw=0))
ax.set_ylim(len(order_ct) - 0.5, -0.5)
ax.legend(handles=[Line2D([], [], color=MODC[m], lw=4) for m in MODC if m in set(rk.module)],
          labels=[f"{MODL[m]} (n={int((rk.module == m).sum())})" for m in MODC if m in set(rk.module)],
          loc="lower left", bbox_to_anchor=(0.0, 1.1), ncol=4, fontsize=5.8, handlelength=1.2, columnspacing=1.2,
          title="Risk-gene module (from fetal expression)", title_fontsize=6, alignment="left")
cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01); cb.set_label("row z", fontsize=6); cb.ax.tick_params(labelsize=5.5)
letter(ax, "f", -0.06, 1.12)
save(fig, "Fig2_fetal_testis")

# =====================================================================================
# Figure 3 — tumours: discovery (TCGA) and validation (GSE3218)
# =====================================================================================
fig = plt.figure(figsize=(W2, 120 * MM))
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.05], wspace=0.75, hspace=0.62)
HORD = ["Seminoma", "Embryonal", "YolkSac", "Teratoma", "Mixed"]
for k, (S_, TT, nm, coh) in enumerate([(SP, TP, "TCGA-TGCT (discovery)", "P1 GermRisk: undiff"),
                                       (SV, TV, "GSE3218 (validation)", "V1 GermRisk: undiff")]):
    ax = fig.add_subplot(gs[0, k])
    order = [h for h in HORD if (S_.histology == h).sum() >= 3]
    box_strip(ax, S_, order, H_COL, "GermRisk score (mean z)", rng)
    r = tp(TT, coh)
    yl = ax.get_ylim(); top = yl[1] + 0.05 * (yl[1] - yl[0])
    xu = [order.index(h) for h in ["Seminoma", "Embryonal"] if h in order]; xd = [order.index(h) for h in ["YolkSac", "Teratoma"] if h in order]
    ax.plot([min(xu) - 0.2, max(xu) + 0.2], [top, top], color="black", lw=0.6); ax.plot([min(xd) - 0.2, max(xd) + 0.2], [top, top], color="black", lw=0.6)
    top2 = top + 0.09 * (yl[1] - yl[0])
    ax.plot([np.mean(xu), np.mean(xu), np.mean(xd), np.mean(xd)], [top, top2, top2, top], color="black", lw=0.6)
    ax.text(np.mean([np.mean(xu), np.mean(xd)]), top2, f"undiff. vs diff.\n{fmt_p(r.p)}", ha="center", va="bottom", fontsize=5.8)
    ax.set_ylim(yl[0], top2 + 0.2 * (yl[1] - yl[0]))
    ax.set_title(nm); letter(ax, "ab"[k])
# (c) forest: unadjusted and adjusted undiff effect, both cohorts
ax = fig.add_subplot(gs[0, 2])
rows = []
for S_, coh in [(SP, "TCGA"), (SV, "GSE3218")]:
    d = S_[S_.histology.isin(["Seminoma", "Embryonal", "Teratoma", "YolkSac"])].copy(); d["undiff"] = d.histology.isin(["Seminoma", "Embryonal"]).astype(float)
    for lab, f in [("unadj.", "GermRisk ~ undiff"), ("adjusted", "GermRisk ~ undiff + Pluripotency + Proliferation")]:
        fit = smf.ols(f, data=d).fit(); ci = fit.conf_int().loc["undiff"]
        rows.append(dict(cohort=coh, model=lab, b=fit.params["undiff"], lo=ci[0], hi=ci[1], p=fit.pvalues["undiff"], n=len(d)))
FR = pd.DataFrame(rows); FR.to_csv(os.path.join(FIG, "source_Fig3c_forest.csv"), index=False)
ys = np.arange(len(FR))[::-1]
for y, (_, r) in zip(ys, FR.iterrows()):
    col = OKABE["blue"] if r.cohort == "TCGA" else OKABE["vermillion"]
    ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.1); ax.scatter(r.b, y, s=18, color=col, marker="o" if r.model == "unadj." else "D", zorder=3, edgecolor="white", lw=0.4)
    ax.text(1.02, y, fmt_p(r.p), transform=ax.get_yaxis_transform(), va="center", fontsize=5.6)
ax.axvline(0, color="#999999", lw=0.6, ls="--")
ax.set_yticks(ys); ax.set_yticklabels([f"{r.cohort}, {r.model}" for _, r in FR.iterrows()], fontsize=6)
ax.set_xlim(min(0, FR.lo.min()) - 0.1, FR.hi.max() * 1.1)
ax.set_xlabel("β, undifferentiated histology (95% CI)\nadjusted: + pluripotency, proliferation"); ax.set_title("Adjusted for pluripotency/proliferation"); letter(ax, "c", -0.62)
# (d,e) matched-null
for k, (nl, obs, TT, key, nm) in enumerate([(NP["P3"], float(NP["P3_obs"]), TP, "P3", "TCGA"), (NV["V3"], float(NV["V3_obs"]), TV, "V3", "GSE3218")]):
    ax = fig.add_subplot(gs[1, k]); r = tp(TT, key)
    ax.hist(nl, bins=50, color="#BFBFBF", edgecolor="white", lw=0.2)
    ax.axvline(obs, color=OKABE["vermillion"], lw=1.2)
    ax.set_xlim(min(nl.min(), obs) - 0.05 * np.ptp(nl), max(nl.max(), obs) + 0.1 * np.ptp(nl)); null_label(ax, nl, obs, r)
    ax.set_xlabel("mean log$_2$FC, undiff. vs diff."); ax.set_ylabel("random gene sets (n)")
    ax.set_title(f"{nm}: matched-gene null"); letter(ax, "de"[k])
# (f) per-gene replication
ax = fig.add_subplot(gs[1, 2])
m = GP.merge(GV, on="gene", suffixes=("_TCGA", "_GSE"))
x, y = m.log2FC_undiff_vs_diff_TCGA, m.log2FC_undiff_vs_diff_GSE
ax.axhline(0, color="#BBBBBB", lw=0.5); ax.axvline(0, color="#BBBBBB", lw=0.5)
cols = [{"germ_PGC": OKABE["vermillion"], "germ_ProSG": OKABE["orange"]}.get(mm, "#555555") for mm in m.module]
ax.scatter(x, y, s=12, c=cols, edgecolor="white", lw=0.3, zorder=3)
rho, pr = spearmanr(x, y)
lab = m.assign(s=(x.abs() + y.abs())).sort_values("s", ascending=False).head(5)
for k_, (_, rr) in enumerate(lab.iterrows()):
    ax.annotate(rr.gene, (rr.log2FC_undiff_vs_diff_TCGA, rr.log2FC_undiff_vs_diff_GSE), xytext=(3, 4 if k_ % 2 == 0 else -7),
                textcoords="offset points", fontsize=5.2, fontstyle="italic")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c_, markersize=3.5) for c_ in [OKABE["vermillion"], OKABE["orange"], "#555555"]],
          labels=["PGC-biased", "ProSG-biased", "both states"], loc="lower right", fontsize=5.5, handletextpad=0.1)
ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}\n{fmt_p(pr)}\n{int(((x > 0) & (y > 0)).sum())}/{len(m)} genes up in both", transform=ax.transAxes, va="top", fontsize=5.8)
ax.set_xlabel("TCGA log$_2$FC (undiff. vs diff.)"); ax.set_ylabel("GSE3218 log$_2$FC (undiff. vs diff.)")
ax.set_title("Per-gene replication"); letter(ax, "f")
save(fig, "Fig3_tumour_retention")

# =====================================================================================
# Figure 4 — developmental block
# =====================================================================================
fig = plt.figure(figsize=(W2, 62 * MM))
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1, 1], wspace=0.45)
bg = Tm["bg"]; bgn = bg[~bg.is_risk]
ax = fig.add_subplot(gs[0, 0])
pct = BP.set_index("gene").tumour_expr_percentile
d = bgn.loc[bgn.index.intersection(pct.index)].copy(); d["pct"] = pct.reindex(d.index)
cmap = {"germ_PGC": OKABE["vermillion"], "germ_ProSG": OKABE["orange"], "germ_both": "#BBBBBB"}
for mod in [m_ for m_ in ["germ_both", "germ_ProSG", "germ_PGC"] if (d.module == m_).any()]:
    dd = d[d.module == mod]; ax.scatter(dd.log2FC, dd.pct, s=5, color=cmap[mod], lw=0, alpha=0.8,
                                        label={"germ_PGC": "PGC-specific", "germ_ProSG": "ProSG-specific", "germ_both": "shared"}[mod])
ax.axvline(-1, color="#999999", lw=0.5, ls="--"); ax.axvline(1, color="#999999", lw=0.5, ls="--")
ax.set_xlabel("fetal germ log$_2$FC (ProSG vs PGC state)"); ax.set_ylabel("TCGA tumour expression (percentile rank)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, markerscale=2.2, handletextpad=0.1, columnspacing=0.8)
ax.set_title("Fetal germ genes in tumours", pad=14); letter(ax, "a", -0.16, 1.12)
ax = fig.add_subplot(gs[0, 1])
vals, labs, cs = [], [], []
for B_, coh in [(BP, "TCGA"), (BV, "GSE3218")]:
    for cl, c in [("PGC-specific", OKABE["vermillion"]), ("ProSG-specific", OKABE["orange"])]:
        vals.append(B_.loc[B_["class"] == cl, "tumour_expr_percentile"].values); labs.append(cl.split('-')[0]); cs.append(c)
pos = [0, 1, 2.6, 3.6]
vp = ax.violinplot(vals, positions=pos, widths=0.8, showextrema=False)
for b, c in zip(vp["bodies"], cs): b.set_facecolor(c); b.set_alpha(0.35); b.set_edgecolor(c)
ax.boxplot(vals, positions=pos, widths=0.18, showfliers=False, medianprops=dict(color="black", lw=0.9), whiskerprops=dict(lw=0.6),
           capprops=dict(lw=0), boxprops=dict(lw=0.6))
for i, (TT, key) in enumerate([(TP, "P4"), (TV, "V4")]):
    r = tp(TT, key); x0, x1 = pos[2 * i], pos[2 * i + 1]
    ax.plot([x0, x0, x1, x1], [1.03, 1.06, 1.06, 1.03], color="black", lw=0.6); ax.text((x0 + x1) / 2, 1.07, fmt_p(r.p), ha="center", fontsize=5.8)
ax.set_xticks(pos); ax.set_xticklabels(labs)
for xc, coh in [(0.5, "TCGA"), (3.1, "GSE3218")]:
    ax.text(xc, -0.17, coh, ha="center", va="top", fontsize=6.5, fontweight="bold", transform=ax.get_xaxis_transform())
ax.set_ylim(0, 1.2); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("tumour expression (percentile rank)"); ax.set_title("Developmental block, both cohorts"); letter(ax, "b")
ax = fig.add_subplot(gs[0, 2]); r = tp(TP, "P2"); nl, obs = NP["P2"], float(NP["P2_obs"])
ax.hist(nl, bins=50, color="#BFBFBF", edgecolor="white", lw=0.2); ax.axvline(obs, color=OKABE["vermillion"], lw=1.2)
ax.set_xlim(min(nl.min(), obs) - 0.05 * np.ptp(nl), max(nl.max(), obs) + 0.1 * np.ptp(nl))
null_label(ax, nl, obs, r)
ax.set_xlabel("mean tumour expression (log$_2$)"); ax.set_ylabel("matched non-risk germ gene sets (n)")
ax.set_title("Risk vs non-risk germ genes (TCGA)"); letter(ax, "c")
save(fig, "Fig4_developmental_block")

# =====================================================================================
# Supplementary figures
# =====================================================================================
fig, axs = plt.subplots(1, 3, figsize=(W2, 58 * MM), gridspec_kw=dict(wspace=0.75))
ax = axs[0]; sc_ = ax.scatter(cells.UMAP1, cells.UMAP2, c=cells.week, s=2, cmap="viridis", lw=0, rasterized=True)
fig.colorbar(sc_, ax=ax, fraction=0.05).set_label("gestational week", fontsize=6); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_title("Gestational week"); letter(ax, "a", -0.08)
ax = axs[1]; gm = cells.germ_axis.notna()
ax.scatter(cells.UMAP1[~gm], cells.UMAP2[~gm], s=2, color="#E5E5E5", lw=0, rasterized=True)
sc_ = ax.scatter(cells.UMAP1[gm], cells.UMAP2[gm], c=cells.germ_axis[gm], s=2.5, cmap="RdBu_r", vmin=-3, vmax=3, lw=0, rasterized=True)
fig.colorbar(sc_, ax=ax, fraction=0.05).set_label("germ axis\n(PGC → ProSG)", fontsize=6); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1"); ax.set_title("Germ developmental axis"); letter(ax, "b", -0.08)
ax = axs[2]; tab = pd.crosstab(cells.week, cells.celltype); tab = tab[[c for c in CT_COL if c in tab.columns]]; tab = tab.div(tab.sum(1), axis=0)
bottom = np.zeros(len(tab))
for c in tab.columns:
    ax.bar(range(len(tab)), tab[c], bottom=bottom, color=CT_COL[c], width=0.8, edgecolor="white", lw=0.4, label=CT_LAB[c]); bottom += tab[c].values
ax.set_xticks(range(len(tab))); ax.set_xticklabels([f"{w:.0f}" for w in tab.index]); ax.set_xlabel("gestational week"); ax.set_ylabel("fraction of cells")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=5.5); ax.set_title("Composition by week"); letter(ax, "c")
save(fig, "FigS1_fetal_atlas_qc")

fig, axs = plt.subplots(2, 4, figsize=(W2, 95 * MM), gridspec_kw=dict(wspace=0.45, hspace=0.75))
for i, (S_, coh) in enumerate([(SP, "TCGA"), (SV, "GSE3218")]):
    order = [h for h in HORD if (S_.histology == h).sum() >= 3]
    for j, sc_name in enumerate(["BG_PGC", "BG_ProSG", "Pluripotency", "Proliferation"]):
        ax = axs[i, j]; tmp = S_.rename(columns={sc_name: "GermRisk_tmp"}).copy(); tmp["GermRisk"] = S_[sc_name]
        box_strip(ax, tmp, order, H_COL, f"{sc_name.replace('BG_', 'fetal ')} score" if j == 0 or True else "", rng)
        ax.set_title(f"{coh}: {sc_name.replace('BG_PGC', 'fetal PGC genes').replace('BG_ProSG', 'fetal ProSG genes')}", fontsize=6.5)
        ax.tick_params(axis="x", labelsize=5.2)
        if i == 0 and j == 0: letter(ax, "a")
        if i == 1 and j == 0: letter(ax, "b")
save(fig, "FigS2_reference_programmes")

if lrc is not None:
    fig, ax = plt.subplots(1, 1, figsize=(W2 * 0.45, 40 * MM))
    l2 = lrc[lrc.gene_set == "Risk_union"]
    ax.barh(range(len(l2)), -np.log10(l2.p), color=OKABE["blue"], height=0.6)
    for i, (_, r) in enumerate(l2.iterrows()):
        lab_ = f"{r.genes}  (OR {r.OR:.1f}, P = {r.p:.2f})" if isinstance(r.genes, str) and r.genes else f"no risk genes (P = {r.p:.2f})"
        ax.text(-np.log10(r.p) + 0.02, i, lab_, va="center", fontsize=5.3)
    ax.set_xlim(0, max(1.0, -np.log10(l2.p).max() * 2.4))
    ax.set_yticks(range(len(l2))); ax.set_yticklabels([d.replace("->", " → ").capitalize() for d in l2.direction])
    ax.set_xlabel("−log$_{10}$ P (one-sided Fisher)"); ax.set_title("Union risk genes in specific ligand–receptor interactions")
    save(fig, "FigS3_LR_channels")

# =====================================================================================
# Tables
# =====================================================================================
pairs = [("GermRisk: undiff (SEM+EC) > diff (TER+YST)", "P1 GermRisk: undiff", "V1 GermRisk: undiff"),
         ("GermRisk: seminoma vs EC", "P1 GermRisk: seminoma", "V2 GermRisk: seminoma"),
         ("GermRisk: EC > TER", "P1b GermRisk EC > TER", "V1b GermRisk EC > TER"),
         ("GermRisk: EC > YST", "P1b GermRisk EC > YST", "V1b GermRisk EC > YST"),
         ("vs expression-matched random genes (z)", "P3", "V3"),
         ("Adjusted for pluripotency + proliferation (β)", "P1c", "V5"),
         ("Developmental block: PGC- > ProSG-specific genes", "P4", "V4")]
rows = []
for lab, kp, kv in pairs:
    a, b = tp(TP, kp), tp(TV, kv)
    rows.append({"Test": lab, "TCGA n": a.n, "TCGA effect": round(a.z if "z)" in lab else a.effect, 3), "TCGA P": a.p,
                 "GSE3218 n": b.n, "GSE3218 effect": round(b.z if "z)" in lab else b.effect, 3), "GSE3218 P": b.p})
TAB1 = pd.DataFrame(rows); TAB1.to_csv(os.path.join(FIG, "Table1_replication.csv"), index=False)
print(TAB1.to_string(index=False))
sheets = {"S1_risk_gene_modules": risk, "S2_celltype_enrichment": T1, "S3_lineage_enrichment": T2,
          "S4_KITLG_KIT": kit if kit is not None else pd.DataFrame(), "S5_LR_channels": lrc if lrc is not None else pd.DataFrame(),
          "S6_TCGA_tests": TP, "S7_GSE3218_tests": TV, "S8_TCGA_per_gene": GP, "S9_GSE3218_per_gene": GV}
try:
    with pd.ExcelWriter(os.path.join(FIG, "Supplementary_Tables.xlsx")) as xw:
        for k, v in sheets.items(): v.to_excel(xw, sheet_name=k[:31], index=False)
except Exception as e:
    print("xlsx not written (", e, ") -> CSVs instead")
    for k, v in sheets.items(): v.to_csv(os.path.join(FIG, f"{k}.csv"), index=False)

zp = os.path.join(BASE, "TGCT_paper_results.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(FIG):
        if not f.endswith(".tiff"): z.write(os.path.join(FIG, f), "figures/" + f)
    for k, d in RES.items():
        for f in os.listdir(d):
            if f.endswith((".csv", ".txt", ".json")): z.write(os.path.join(d, f), f"results/{k}/{f}")
print(f"\nDONE -> {zp} ({os.path.getsize(zp) / 1e6:.1f} MB) — send this file back.")
