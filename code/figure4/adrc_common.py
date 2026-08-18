"""
Layout of the Stanford ADRC release, shared by every step of the figure-4
pipeline: the nine modality keys in joint-matrix order, where each modality's
CSV sits inside the real/ and sphere_xk/x{k}/ trees, and the constants the
rotation and the panels are evaluated at.
"""

MIN_SITE_SIZE = 10          # rotation sites smaller than this are merged away
KS_DEFAULT = [1, 2, 3, 4, 5]  # rotation counts of the privacy progression (panel c)
RELEASE_K = 2               # released operating point (panels b, d, e, f, g)

MODALITY_KEYS = ["cognitive", "biomarkers", "imaging_amy", "imaging_tau",
                 "wgs", "scrna", "csf", "plasma", "demographics"]

MODALITY_REL = {
    "cognitive":    "cognitive_scores/cognitive_scores_c2_b4.csv",
    "biomarkers":   "biomarkers/biomarkers.csv",
    "imaging_amy":  "imaging_phenotypes/imaging_amyloid.csv",
    "imaging_tau":  "imaging_phenotypes/imaging_tau.csv",
    "wgs":          "wgs/wgs_genodata.csv",
    "scrna":        "rna_seq/sc_rna_seq_cellexp.csv",
    "csf":          "proteomics/proteomics_somalogic_csf.csv",
    "plasma":       "proteomics/proteomics_somalogic_plasma.csv",
    "demographics": "demographics_diagnosis/demographics_diagnosis.csv",
}

ID_COL = "adrc_id"
SKIP_ID = frozenset({ID_COL})

# ── Figure-side naming ───────────────────────────────────────────────────────
# The panels name the two imaging modalities in full, where the joint matrix
# uses the shorter prefixes above; both point at the same files.
FIG_MOD_FILE = {
    "demographics":    MODALITY_REL["demographics"],
    "cognitive":       MODALITY_REL["cognitive"],
    "biomarkers":      MODALITY_REL["biomarkers"],
    "imaging_amyloid": MODALITY_REL["imaging_amy"],
    "imaging_tau":     MODALITY_REL["imaging_tau"],
    "wgs":             MODALITY_REL["wgs"],
    "scrna":           MODALITY_REL["scrna"],
    "csf":             MODALITY_REL["csf"],
    "plasma":          MODALITY_REL["plasma"],
}
FIG_MOD_ORDER = list(FIG_MOD_FILE)
CROSS_MODS = FIG_MOD_ORDER[1:]   # the eight measured modalities of panels b, f, g

# Bookkeeping columns that are not features and must not enter any metric
SKIP_FIG = frozenset({"adrc_id", "pidn", "year_quarter", "visit_number", "type",
                      "c2_completed", "b4_completed", "additional_race"})

DIAG_ORDER = ["HC", "MCI", "AD", "PD", "PDMCI", "PDD", "LBD", "Other"]
DIAG_SHORT = {
    "Healthy Control": "HC", "Mild Cognitive Impairment": "MCI",
    "Probable Alzheimers Disease": "AD", "Possible Alzheimers Disease": "AD",
    "Possible Alzheimers Disease ; Lewy Body Disease": "AD",
    "Possible Alzheimers Disease ; Other": "AD", "Parkinsons Disease only": "PD",
    "Parkinsons Disease and Mild cognitive impairment": "PDMCI",
    "Parkinsons Disease and Dementia": "PDD", "Lewy Body Disease": "LBD"}

_FULL_CSV_CACHE: dict = {}


def read_full_csv(path):
    """Read a modality CSV once (bookkeeping columns dropped, indexed by
    adrc_id) and cache it, so the cross-modality step can resample columns
    instead of re-reading the 193k-column scRNA file on every seed."""
    import pandas as pd

    key = str(path)
    if key not in _FULL_CSV_CACHE:
        df = pd.read_csv(path, low_memory=False)
        df = df.drop(columns=[c for c in SKIP_FIG if c in df.columns and c != ID_COL],
                     errors="ignore")
        if ID_COL in df.columns:
            df = df.set_index(ID_COL)
        _FULL_CSV_CACHE[key] = df
    return _FULL_CSV_CACHE[key]


def load_csv(path, max_cols=None, seed=0):
    """read_full_csv, optionally down to a random max_cols columns."""
    import numpy as np

    df = read_full_csv(path)
    if max_cols and len(df.columns) > max_cols:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(df.columns), size=max_cols, replace=False)
        return df.iloc[:, sorted(chosen)]
    return df.copy()
