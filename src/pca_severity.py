import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_PATH  = Path("DATA_processed_features.csv")
OUTPUT_PATH = INPUT_PATH.parent / "DATA_processed_severity.csv"
FEATURES    = ["fires_z", "collisions_z"]

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
df = df.sort_values(["country", "date"]).reset_index(drop=True)

nan_mask = df[FEATURES].isna().any(axis=1)
print(f"Rows dropped (NaN in features): {nan_mask.sum()} — first row of each country")
df = df[~nan_mask].copy()

# ─────────────────────────────────────────────
# HELPER: robust sign correction
# ─────────────────────────────────────────────
def fix_sign(loadings: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Ensure the loading vector points in the 'severity-increasing' direction.

    Decision rule (order of priority):
      1. Find the dominant feature (largest |loading|).
      2. If that dominant loading is negative → flip the entire vector.
      3. Tie-break (equal magnitudes, e.g. ±0.707): flip if sum < 0,
         or if sum == 0 flip if the first element is negative.

    This guarantees that at least the most influential feature always
    contributes positively to severity, regardless of PCA sign ambiguity.
    """
    dom_idx = np.argmax(np.abs(loadings))

    if loadings[dom_idx] < 0:
        return -loadings, True

    # Equal-magnitude tie-break (e.g. ±0.7071 case)
    if np.isclose(np.abs(loadings).max(), np.abs(loadings).min()):
        s = loadings.sum()
        if s < 0 or (np.isclose(s, 0) and loadings[0] < 0):
            return -loadings, True

    return loadings, False

# ─────────────────────────────────────────────
# 2. PER-COUNTRY PCA → PC1 LOADINGS → Y_severity
# ─────────────────────────────────────────────
loading_records = []
results = []

for country, group in df.groupby("country"):
    g = group.copy()

    # Standardise per country before PCA
    X        = g[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit 1-component PCA
    pca = PCA(n_components=1)
    pca.fit(X_scaled)

    raw_loadings  = pca.components_[0]
    explained_var = pca.explained_variance_ratio_[0]

    # Apply sign correction
    loadings, flipped = fix_sign(raw_loadings)

    # Normalise to percentage weights
    weights = np.abs(loadings) / np.abs(loadings).sum()

    loading_records.append({
        "country"              : country,
        "raw_fires_z"          : round(raw_loadings[0], 6),
        "raw_collisions_z"     : round(raw_loadings[1], 6),
        "loading_fires_z"      : round(loadings[0], 6),
        "loading_collisions_z" : round(loadings[1], 6),
        "weight_fires_z"       : round(weights[0],  4),
        "weight_collisions_z"  : round(weights[1],  4),
        "PC1_explained_var_pct": round(explained_var * 100, 2),
        "sign_flipped"         : flipped,
    })

    # Project onto corrected PC1 then min-max scale → [0, 1]
    pc1_scores    = X_scaled @ loadings
    g["Y_severity"] = (pc1_scores - pc1_scores.min()) / \
                      (pc1_scores.max() - pc1_scores.min())

    results.append(g)

# ─────────────────────────────────────────────
# 3. REPORT
# ─────────────────────────────────────────────
loadings_df = pd.DataFrame(loading_records).set_index("country")

print("\n" + "=" * 60)
print("RAW vs CORRECTED PC1 LOADINGS per country")
print("=" * 60)
print(loadings_df.to_string())

print("\n── Interpretation ──")
for country, row in loadings_df.iterrows():
    dom = FEATURES[np.argmax(np.abs([row["loading_fires_z"],
                                     row["loading_collisions_z"]]))]
    corr_note = ""
    if np.sign(row["loading_fires_z"]) != np.sign(row["loading_collisions_z"]):
        corr_note = " ⚠ anti-correlated features: PC1 captures contrast, not total severity"
    flip_note = " [sign flipped]" if row["sign_flipped"] else ""
    print(f"  {country:10s} | PC1 explains {row['PC1_explained_var_pct']:.1f}% variance "
          f"| dominant: {dom}{flip_note}{corr_note}")

print("\n" + "=" * 60)
print("Y_severity DISTRIBUTION per country")
print("=" * 60)
df_out = pd.concat(results).sort_values(["country", "date"]).reset_index(drop=True)
print(df_out.groupby("country")["Y_severity"]
      .agg(["mean", "std", "min", "max"]).round(4).to_string())

# ─────────────────────────────────────────────
# 4. SAVE
# ─────────────────────────────────────────────
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved → {OUTPUT_PATH.resolve()}")
print(f"Final shape: {df_out.shape[0]:,} rows × {df_out.shape[1]} columns")
