"""A1+A2+A4 — extended OLS with country fixed effects + residual plot.

Inputs:  contestants_with_youtube_full.csv (1,712 rows, 232 finalists 2016–25)
Outputs:
    slide_assets/J_REGRESSION_FE.png       — coefficient bar chart, with-FE
    slide_assets/J_RESIDUALS.png            — predicted-vs-actual + top sleepers
    regression_results.json                  — all the numbers
    eurovision_residuals.csv                 — every finalist with predicted/residual
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "/sessions/laughing-elegant-goodall/mnt/outputs/slide_assets"
os.makedirs(OUT, exist_ok=True)
NAVY, INK, MAGENTA, GOLD, MUTED = "#0B1B3D", "#0B3D91", "#E6007E", "#F2C94C", "#9AA5B1"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200,
                     "font.family": "DejaVu Sans", "axes.titleweight": "bold"})


def ols(X, y):
    """Closed-form OLS via pseudoinverse (handles collinear country dummies)."""
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return beta, r2, yhat


def boot_coef(X, y, idx_jury, idx_tele, n_boot=1500, seed=42):
    rng = np.random.default_rng(seed); n = len(y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        b = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ yb
        out.append(b)
    out = np.array(out)
    return {
        "jury_lo": float(np.percentile(out[:, idx_jury], 2.5)),
        "jury_hi": float(np.percentile(out[:, idx_jury], 97.5)),
        "tele_lo": float(np.percentile(out[:, idx_tele], 2.5)),
        "tele_hi": float(np.percentile(out[:, idx_tele], 97.5)),
    }


# ============================================================
# Load and filter
# ============================================================
df = pd.read_csv("/sessions/laughing-elegant-goodall/mnt/outputs/contestants_with_youtube_full.csv")
sub = df[(df["year"].between(2016, 2025))
         & df["place_final"].notna()
         & df["points_jury_final"].notna() & df["points_tele_final"].notna()
         & df["yt_views"].notna() & (df["yt_views"] > 0)].copy()
sub["log_yt"] = np.log10(sub["yt_views"].astype(float))
print(f"Analysis subset (A1 expansion): n = {len(sub)} grand finalists")

# ============================================================
# Standardize core predictors
# ============================================================
core = ["points_jury_final", "points_tele_final", "year"]
for c in core:
    sub[f"{c}_z"] = (sub[c] - sub[c].mean()) / sub[c].std()

# ============================================================
# Model 1 — no country FE (baseline; same as before)
# ============================================================
print("\n" + "=" * 70)
print("  MODEL 1 — no country fixed effects")
print("=" * 70)
X1_cols = ["points_jury_final_z", "points_tele_final_z", "year_z"]
X1 = np.c_[np.ones(len(sub)), sub[X1_cols].values]
y = sub["log_yt"].values.astype(float)

beta1, r2_1, yhat1 = ols(X1, y)
ci1 = boot_coef(X1, y, idx_jury=1, idx_tele=2)

print(f"  Intercept      = {beta1[0]:+.3f}")
print(f"  jury_z         = {beta1[1]:+.3f}   95% CI [{ci1['jury_lo']:+.3f}, {ci1['jury_hi']:+.3f}]")
print(f"  tele_z         = {beta1[2]:+.3f}   95% CI [{ci1['tele_lo']:+.3f}, {ci1['tele_hi']:+.3f}]")
print(f"  year_z         = {beta1[3]:+.3f}")
print(f"  R²             = {r2_1:.3f}")
print(f"  tele/jury      = {beta1[2]/beta1[1]:.1f}×")

# ============================================================
# Model 2 — WITH country fixed effects (A2)
# ============================================================
print("\n" + "=" * 70)
print("  MODEL 2 — with country fixed effects (one-hot countries)")
print("=" * 70)

# One-hot encode countries (drop one as the reference level)
country_dummies = pd.get_dummies(sub["to_country"], prefix="c", drop_first=True).astype(float)
print(f"  Country dummies created: {country_dummies.shape[1]} (one country dropped as reference)")

X2 = np.c_[np.ones(len(sub)), sub[X1_cols].values, country_dummies.values]
beta2, r2_2, yhat2 = ols(X2, y)
ci2 = boot_coef(X2, y, idx_jury=1, idx_tele=2)

print(f"  Intercept      = {beta2[0]:+.3f}")
print(f"  jury_z         = {beta2[1]:+.3f}   95% CI [{ci2['jury_lo']:+.3f}, {ci2['jury_hi']:+.3f}]")
print(f"  tele_z         = {beta2[2]:+.3f}   95% CI [{ci2['tele_lo']:+.3f}, {ci2['tele_hi']:+.3f}]")
print(f"  year_z         = {beta2[3]:+.3f}")
print(f"  R²             = {r2_2:.3f}   (Δ vs Model 1: +{r2_2 - r2_1:.3f})")
print(f"  tele/jury      = {beta2[2]/beta2[1]:.1f}×")

# Save residuals from model 2 for the plot
sub["yhat"] = yhat2
sub["resid"] = y - yhat2
sub["expected_views"] = 10 ** yhat2

# ============================================================
# A4 — Predicted vs actual + empirical Sleeper Hits
# ============================================================
print("\n" + "=" * 70)
print("  A4 — Top empirical Sleeper Hits (largest positive residuals)")
print("=" * 70)
top_sleepers = sub.nlargest(10, "resid")[
    ["year", "to_country", "performer", "song", "place_final",
     "yt_views", "expected_views", "resid"]
].copy()
print(top_sleepers.to_string(index=False))

print("\n=== Top empirical Jury Darlings (largest negative residuals) ===")
top_jury_darlings = sub.nsmallest(10, "resid")[
    ["year", "to_country", "performer", "song", "place_final",
     "yt_views", "expected_views", "resid"]
].copy()
print(top_jury_darlings.to_string(index=False))


# ============================================================
# FIG 1 — coefficient bar chart, model 1 vs model 2
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
labels = ["Jury vote", "Televote", "Year"]
m1_vals = [beta1[1], beta1[2], beta1[3]]
m2_vals = [beta2[1], beta2[2], beta2[3]]
xs = np.arange(len(labels)); width = 0.35

ax.bar(xs - width/2, m1_vals, width, color=MUTED, label="No country FE", edgecolor="white", linewidth=1)
ax.bar(xs + width/2, m2_vals, width, color=[INK, MAGENTA, GOLD],
       label="WITH country fixed effects", edgecolor="white", linewidth=1)

# Annotate
for i, (v1, v2) in enumerate(zip(m1_vals, m2_vals)):
    ax.text(i - width/2, v1 + 0.005, f"{v1:+.2f}", ha="center", fontsize=9, color="#666")
    ax.text(i + width/2, v2 + 0.005, f"{v2:+.2f}", ha="center", fontsize=9.5, color="#222", fontweight="bold")

ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=11.5, fontweight="bold")
ax.set_ylabel("Standardized OLS coefficient (effect on log YouTube views)", fontsize=10.5)
ax.set_title("The televote effect survives country fixed effects.\n"
             "Voting blocs explain some variance — but not the jury–audience gap.",
             fontsize=13, color=NAVY, pad=12)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(True, axis="y", lw=0.3, alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylim(-0.05, max(m1_vals + m2_vals) * 1.4)

ax.text(0, -0.15,
        f"n = {len(sub)} finalists, 2016–2025  ·  Model 1 R² = {r2_1:.2f}, Model 2 R² = {r2_2:.2f}  ·  "
        f"Country FE adds {country_dummies.shape[1]} parameters",
        transform=ax.transAxes, fontsize=8.5, color="#666", style="italic")

plt.tight_layout()
plt.savefig(f"{OUT}/J_REGRESSION_FE.png", bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {OUT}/J_REGRESSION_FE.png")


# ============================================================
# FIG 2 — Predicted vs actual residual plot
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")

ax.scatter(yhat2, y, s=50, c=INK, alpha=0.5, edgecolor="white", linewidth=0.8)

# Diagonal reference line
mn, mx = min(y.min(), yhat2.min()), max(y.max(), yhat2.max())
ax.plot([mn, mx], [mn, mx], color="#888", lw=1, ls="--", label="perfect prediction")

# Highlight top 6 sleepers (positive residuals)
top6 = sub.nlargest(6, "resid")
ax.scatter(top6["yhat"], top6["log_yt"], s=140, facecolor="none",
           edgecolor=MAGENTA, linewidth=2.5, label="Empirical Sleeper Hits")
for _, r in top6.iterrows():
    ax.annotate(f"{r['performer']}\n{r['song']} ({int(r['year'])})",
                xy=(r["yhat"], r["log_yt"]),
                xytext=(12, 6), textcoords="offset points", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MAGENTA, lw=1, alpha=0.95))

# Highlight bottom 4 jury darlings (negative residuals)
bot4 = sub.nsmallest(4, "resid")
ax.scatter(bot4["yhat"], bot4["log_yt"], s=140, facecolor="none",
           edgecolor=GOLD, linewidth=2.5, label="Empirical Jury Darlings")
for _, r in bot4.iterrows():
    ax.annotate(f"{r['performer']}\n{r['song']} ({int(r['year'])})",
                xy=(r["yhat"], r["log_yt"]),
                xytext=(12, -28), textcoords="offset points", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GOLD, lw=1, alpha=0.95))

ax.set_xlabel("Predicted log₁₀(YouTube views)  —  from votes + year + country", fontsize=11)
ax.set_ylabel("Actual log₁₀(YouTube views)", fontsize=11)
ax.set_title("Predicted vs actual: songs above the dashed line are Sleeper Hits.\n"
             "Songs below it are Jury Darlings.",
             fontsize=13, color=NAVY, pad=12)
ax.grid(True, lw=0.3, alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=10)
ax.text(0, -0.10,
        f"n = {len(sub)} finalists, 2016–2025  ·  OLS with country fixed effects  ·  "
        f"R² = {r2_2:.2f}  ·  residual = log₁₀(actual) − log₁₀(predicted)",
        transform=ax.transAxes, fontsize=8.5, color="#666", style="italic")
plt.tight_layout()
plt.savefig(f"{OUT}/J_RESIDUALS.png", bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}/J_RESIDUALS.png")


# ============================================================
# Save numbers + residuals
# ============================================================
results = {
    "n_finalists": int(len(sub)),
    "model_1_no_FE": {
        "intercept": float(beta1[0]),
        "jury_z": float(beta1[1]),
        "tele_z": float(beta1[2]),
        "year_z": float(beta1[3]),
        "r2": float(r2_1),
        "tele_jury_ratio": float(beta1[2] / beta1[1]),
        "ci": ci1,
    },
    "model_2_with_FE": {
        "intercept": float(beta2[0]),
        "jury_z": float(beta2[1]),
        "tele_z": float(beta2[2]),
        "year_z": float(beta2[3]),
        "r2": float(r2_2),
        "tele_jury_ratio": float(beta2[2] / beta2[1]),
        "n_country_dummies": int(country_dummies.shape[1]),
        "ci": ci2,
    },
}
with open("/sessions/laughing-elegant-goodall/mnt/outputs/regression_results.json", "w") as f:
    json.dump(results, f, indent=2)

sub_out = sub[["year", "to_country", "performer", "song", "place_final",
               "points_jury_final", "points_tele_final", "yt_views",
               "yhat", "expected_views", "resid"]].copy()
sub_out = sub_out.rename(columns={"to_country": "country", "performer": "artist",
                                  "yhat": "predicted_log_yt"})
sub_out.to_csv("/sessions/laughing-elegant-goodall/mnt/outputs/eurovision_residuals.csv",
               index=False)

print("\nDone.")
PYEOF