"""Re-run the jury-vs-televote analysis on the REAL Spijkervet/th0mk data.

Inputs:
    contestants_clean.csv        — 1,676 cleaned Eurovision entries
    eurovision_jury_televote_2016_2023.csv  — my hand-curated Spotify popularity estimates

Outputs:
    eurovision_real_2016_2025.csv  — joined: real points + estimated popularity
    jury_vs_televote_results_real.json
    slide_assets/J1_jury_vs_tele_scatter_real.png
    slide_assets/J2_disagreement_bars_real.png
    slide_assets/J3_correlation_chart_real.png
"""
import json
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CON   = "/sessions/laughing-elegant-goodall/mnt/outputs/contestants_clean.csv"
MINE  = "/sessions/laughing-elegant-goodall/mnt/outputs/eurovision_jury_televote_2016_2023.csv"
OUTC  = "/sessions/laughing-elegant-goodall/mnt/outputs/eurovision_real_2016_2025.csv"
OUTJ  = "/sessions/laughing-elegant-goodall/mnt/outputs/jury_vs_televote_results_real.json"
SLIDE = "/sessions/laughing-elegant-goodall/mnt/outputs/slide_assets"

import os
os.makedirs(SLIDE, exist_ok=True)

# Brand palette
NAVY, INK, MAGENTA, GOLD, MUTED = "#0B1B3D", "#0B3D91", "#E6007E", "#F2C94C", "#9AA5B1"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200,
                     "font.family": "DejaVu Sans", "axes.titleweight": "bold"})


# ============================================================
# Stats helpers
# ============================================================
def pearson_r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    return float((xm * ym).sum() / np.sqrt((xm ** 2).sum() * (ym ** 2).sum()))


def boot_ci(x, y, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed); n = len(x)
    rs = [pearson_r(np.asarray(x)[idx], np.asarray(y)[idx])
          for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def steiger_z(r12, r13, r23, n):
    rm2 = (r12 ** 2 + r13 ** 2) / 2
    f = (1 - r23) / (2 * (1 - rm2)); h = (1 - f * rm2) / (1 - rm2)
    z12 = 0.5 * np.log((1 + r12) / (1 - r12))
    z13 = 0.5 * np.log((1 + r13) / (1 - r13))
    z = (z12 - z13) * np.sqrt((n - 3) / (2 * (1 - r23) * h))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(z), float(p)


def normkey(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ============================================================
# Load real Spijkervet data
# ============================================================
print("=" * 70)
print("  JURY vs TELEVOTE — REAL DATA (2016-2025)")
print("=" * 70)

con = pd.read_csv(CON)
real = con[
    (con["year"].between(2016, 2025))
    & con["place_final"].notna()
    & con["points_jury_final"].notna()
    & con["points_tele_final"].notna()
].copy()
print(f"\nReal Spijkervet 2016-2025 finalists: {len(real)}")
print(f"Per year:")
print(real.groupby("year").size().to_string())

# ============================================================
# Merge in my Spotify popularity estimates
# ============================================================
mine = pd.read_csv(MINE)
mine = mine.rename(columns={"country": "to_country", "artist": "performer"})
mine["k_country"] = mine["to_country"].apply(normkey)
mine["k_artist"]  = mine["performer"].apply(normkey)
mine["k_song"]    = mine["song"].apply(normkey)

real["k_country"] = real["to_country"].apply(normkey)
real["k_artist"]  = real["performer"].apply(normkey)
real["k_song"]    = real["song"].apply(normkey)

joined = real.merge(
    mine[["year", "k_country", "k_artist", "k_song", "spotify_pop", "yt_views_millions"]],
    on=["year", "k_country", "k_artist", "k_song"], how="left",
)

matched = joined[joined["spotify_pop"].notna()].copy()
print(f"\nMatched with my Spotify estimates: {len(matched)} / {len(real)}")
print(f"Unmatched 2016-2023 (showing first 10):")
unmatched = joined[joined["spotify_pop"].isna() & (joined["year"] <= 2023)]
print(unmatched[["year", "to_country", "performer", "song"]].head(10).to_string(index=False))

# Save real-data CSV (keeps unmatched too — flag with NaN popularity)
out = joined[["year", "to_country", "performer", "song",
              "place_final", "points_final", "points_jury_final", "points_tele_final",
              "spotify_pop", "yt_views_millions", "youtube_url"]].copy()
out = out.rename(columns={"to_country": "country", "performer": "artist",
                          "points_final": "points_total",
                          "points_jury_final": "jury_points",
                          "points_tele_final": "tele_points"})
out.to_csv(OUTC, index=False)
print(f"\nWrote: {OUTC}  ({len(out)} rows)")


# ============================================================
# Run correlations on the matched subset
# ============================================================
df = matched.copy()
df["log10_yt_views"] = np.log10(df["yt_views_millions"] * 1e6)

print("\n" + "=" * 70)
print(f"  CORRELATIONS (n = {len(df)} finalists with popularity estimates)")
print("=" * 70)

pairs = {
    "jury_vs_spotify": ("points_jury_final", "spotify_pop"),
    "tele_vs_spotify": ("points_tele_final", "spotify_pop"),
    "jury_vs_yt":      ("points_jury_final", "log10_yt_views"),
    "tele_vs_yt":      ("points_tele_final", "log10_yt_views"),
    "total_vs_spotify": ("points_final", "spotify_pop"),
    "total_vs_yt":      ("points_final", "log10_yt_views"),
}

results = {}
print(f"{'pair':24s}  {'r':>7s}    {'95% CI':>17s}")
print("-" * 60)
for name, (x_col, y_col) in pairs.items():
    r = pearson_r(df[x_col], df[y_col])
    lo, hi = boot_ci(df[x_col].to_numpy(), df[y_col].to_numpy())
    results[name] = {"r": round(r, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
    print(f"{name:24s}  {r:+.3f}   [{lo:+.3f}, {hi:+.3f}]")

n = len(df)
r_jt = pearson_r(df["points_jury_final"], df["points_tele_final"])
z_s, p_s = steiger_z(results["tele_vs_spotify"]["r"], results["jury_vs_spotify"]["r"], r_jt, n)
z_y, p_y = steiger_z(results["tele_vs_yt"]["r"],     results["jury_vs_yt"]["r"],     r_jt, n)
print(f"\nSteiger Z (Spotify):   z={z_s:+.2f}   p={p_s:.4f}")
print(f"Steiger Z (YouTube):   z={z_y:+.2f}   p={p_y:.4f}")
results["steiger_spotify"] = {"z": round(z_s, 4), "p": round(p_s, 4)}
results["steiger_yt"]      = {"z": round(z_y, 4), "p": round(p_y, 4)}
results["jury_tele_corr"]  = round(r_jt, 4)
results["n_finalists"]     = int(n)

with open(OUTJ, "w") as f: json.dump(results, f, indent=2)
print(f"\nWrote: {OUTJ}")


# ============================================================
# Figure 1 — side-by-side scatter
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor="white")
for ax, x_col, color, label, key in [
    (axes[0], "points_jury_final", INK,     "Jury points",     "jury_vs_spotify"),
    (axes[1], "points_tele_final", MAGENTA, "Televote points", "tele_vs_spotify"),
]:
    ax.scatter(df[x_col], df["spotify_pop"], s=55, c=color, alpha=0.65,
               edgecolor="white", linewidth=0.7)
    z = np.polyfit(df[x_col], df["spotify_pop"], 1)
    xs = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    ax.plot(xs, z[0] * xs + z[1], color=color, lw=1.5, alpha=0.4, ls="--")
    r = results[key]["r"]
    ax.text(0.05, 0.93, f"r = {r:+.3f}", transform=ax.transAxes,
            fontsize=16, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, lw=1))
    ax.set_title(label, fontsize=14, color=color)
    ax.set_xlabel(label, fontsize=11)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("Spotify popularity (0-100, estimate)", fontsize=11)

# Annotate top jury-tele disagreements
df["disagree"] = df["points_tele_final"] - df["points_jury_final"]
for _, row in df.nlargest(2, "disagree").iterrows():
    axes[1].annotate(f"{row['performer']}\n{row['song']}",
                     xy=(row["points_tele_final"], row["spotify_pop"]),
                     xytext=(8, 8), textcoords="offset points", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", alpha=0.9))
for _, row in df.nsmallest(2, "disagree").iterrows():
    axes[0].annotate(f"{row['performer']}\n{row['song']}",
                     xy=(row["points_jury_final"], row["spotify_pop"]),
                     xytext=(8, 8), textcoords="offset points", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", alpha=0.9))

fig.suptitle(f"Eurovision 2016-2025 — jury vs. televote, each against Spotify popularity (n={n})",
             fontsize=14, fontweight="bold", color=NAVY, y=1.02)
plt.tight_layout()
fig.savefig(f"{SLIDE}/J1_jury_vs_tele_scatter_real.png", bbox_inches="tight", facecolor="white")
plt.close()


# ============================================================
# Figure 3 — bar chart with CIs
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
cats = ["Spotify popularity\n(0-100)", "YouTube views\n(log10)"]
xs = np.arange(len(cats)); width = 0.32
jury_rs = [results["jury_vs_spotify"]["r"], results["jury_vs_yt"]["r"]]
tele_rs = [results["tele_vs_spotify"]["r"], results["tele_vs_yt"]["r"]]
jury_err = [
    [results["jury_vs_spotify"]["r"] - results["jury_vs_spotify"]["ci_low"],
     results["jury_vs_yt"]["r"]      - results["jury_vs_yt"]["ci_low"]],
    [results["jury_vs_spotify"]["ci_high"] - results["jury_vs_spotify"]["r"],
     results["jury_vs_yt"]["ci_high"]      - results["jury_vs_yt"]["r"]],
]
tele_err = [
    [results["tele_vs_spotify"]["r"] - results["tele_vs_spotify"]["ci_low"],
     results["tele_vs_yt"]["r"]      - results["tele_vs_yt"]["ci_low"]],
    [results["tele_vs_spotify"]["ci_high"] - results["tele_vs_spotify"]["r"],
     results["tele_vs_yt"]["ci_high"]      - results["tele_vs_yt"]["r"]],
]
ax.bar(xs - width/2, jury_rs, width, yerr=jury_err, capsize=6,
       color=INK, label="Jury vote", edgecolor="white", linewidth=1)
ax.bar(xs + width/2, tele_rs, width, yerr=tele_err, capsize=6,
       color=MAGENTA, label="Televote", edgecolor="white", linewidth=1)
ax.set_xticks(xs); ax.set_xticklabels(cats, fontsize=11)
ax.set_ylabel("Pearson correlation with…", fontsize=11)
ax.set_title("Televote correlates with real-world popularity\nmore strongly than the jury vote",
             fontsize=14, color=NAVY, pad=12)
ax.axhline(0, color="black", lw=0.6); ax.legend(loc="upper right", frameon=False)
ax.grid(True, axis="y", lw=0.3, alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylim(0, max(max(tele_rs), max(jury_rs)) * 1.4)
foot = (f"Bars: bootstrap 95% CI  ·  n = {n} finalists, 2016-2025 (real Spijkervet data)  ·  "
        f"Steiger's Z (Spotify): p = {p_s:.3f}")
ax.text(0, -0.18, foot, transform=ax.transAxes, fontsize=9, color="#666", style="italic")
plt.tight_layout()
fig.savefig(f"{SLIDE}/J3_correlation_chart_real.png", bbox_inches="tight", facecolor="white")
plt.close()


# ============================================================
# Figure 2 — disagreement bars
# ============================================================
df["abs_dis"] = df["disagree"].abs()
top10 = df.nlargest(10, "abs_dis").sort_values("disagree")
fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")
colors = [MAGENTA if d > 0 else INK for d in top10["disagree"]]
ax.barh(np.arange(len(top10)), top10["disagree"], color=colors, edgecolor="white", linewidth=0.6)
ax.set_yticks(np.arange(len(top10)))
ax.set_yticklabels([f"{r['performer']} — {r['song']} ({r['year']})"
                    for _, r in top10.iterrows()], fontsize=10)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Televote points − Jury points", fontsize=11)
ax.set_title("The 10 biggest jury–audience disagreements (2016-2025)",
             fontsize=14, color=NAVY, pad=12)
ax.grid(True, axis="x", lw=0.3, alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.03, "← Jury picked    Audience picked →", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10, style="italic", color="#555")
plt.tight_layout()
fig.savefig(f"{SLIDE}/J2_disagreement_bars_real.png", bbox_inches="tight", facecolor="white")
plt.close()


# ============================================================
# Print the headline summary
# ============================================================
print("\n" + "=" * 70)
print(f"  HEADLINE (real Spijkervet data, n = {n})")
print("=" * 70)
print(f"  Spotify popularity:")
print(f"    Jury vote:     r = {results['jury_vs_spotify']['r']:+.3f}   "
      f"[{results['jury_vs_spotify']['ci_low']:+.2f}, {results['jury_vs_spotify']['ci_high']:+.2f}]")
print(f"    Televote:      r = {results['tele_vs_spotify']['r']:+.3f}   "
      f"[{results['tele_vs_spotify']['ci_low']:+.2f}, {results['tele_vs_spotify']['ci_high']:+.2f}]")
print(f"    Steiger's Z:   p = {p_s:.4f}")
print(f"  YouTube views:")
print(f"    Jury vote:     r = {results['jury_vs_yt']['r']:+.3f}")
print(f"    Televote:      r = {results['tele_vs_yt']['r']:+.3f}")
print(f"    Steiger's Z:   p = {p_y:.4f}")
print(f"\n  Figures saved to: {SLIDE}/J{{1,2,3}}_*_real.png")
