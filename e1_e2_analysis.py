"""E1 + E2 analyses on top of existing real-data files.

E1 — Year-by-year disagreement gap:
    For each year 2016-2025, compute r(tele, log_yt) and r(jury, log_yt) on
    that year's finalists. Plot the gap over time.

E2 — Counterfactual: what if Eurovision used televote only since 2016?
    Re-rank each year's finalists by televote_only, compute the
    counterfactual winner, podium, and key positions changes.
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


def pearson_r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    return float((xm*ym).sum() / np.sqrt((xm**2).sum() * (ym**2).sum()))


# ============================================================
# Load data
# ============================================================
df = pd.read_csv("/sessions/laughing-elegant-goodall/mnt/outputs/contestants_with_youtube_full.csv")
sub = df[(df["year"].between(2016, 2025))
         & df["place_final"].notna()
         & df["points_jury_final"].notna() & df["points_tele_final"].notna()
         & df["yt_views"].notna() & (df["yt_views"] > 0)].copy()
sub["log_yt"] = np.log10(sub["yt_views"].astype(float))


# ============================================================
# E1 — Year-by-year correlation gap
# ============================================================
print("=" * 70)
print("  E1 — Year-by-year disagreement gap")
print("=" * 70)

per_year = []
for y in sorted(sub["year"].unique()):
    yr = sub[sub["year"] == y]
    if len(yr) < 5:
        continue
    rj = pearson_r(yr["points_jury_final"], yr["log_yt"])
    rt = pearson_r(yr["points_tele_final"], yr["log_yt"])
    per_year.append({"year": int(y), "n": len(yr), "r_jury": rj, "r_tele": rt, "gap": rt - rj})

py_df = pd.DataFrame(per_year)
print(py_df.round(3).to_string(index=False))
print(f"\nMean gap across years: {py_df['gap'].mean():+.3f}")
print(f"Range:                  {py_df['gap'].min():+.3f} to {py_df['gap'].max():+.3f}")

# Plot
fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
years = py_df["year"]
ax.plot(years, py_df["r_tele"], "o-", color=MAGENTA, lw=2.5, markersize=10,
        label="r(televote, log YouTube views)", zorder=3)
ax.plot(years, py_df["r_jury"], "o-", color=INK, lw=2.5, markersize=10,
        label="r(jury, log YouTube views)", zorder=3)

# Shade the gap region
ax.fill_between(years, py_df["r_jury"], py_df["r_tele"],
                color=MAGENTA, alpha=0.12, zorder=1)

# Annotate the gap each year
for _, r in py_df.iterrows():
    ax.annotate(f"+{r['gap']:.2f}",
                xy=(r["year"], (r["r_tele"] + r["r_jury"]) / 2),
                fontsize=8.5, color="#444", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.85))

ax.axhline(0, color="black", lw=0.6, alpha=0.5)
ax.set_xlabel("Contest year", fontsize=11)
ax.set_ylabel("Within-year correlation with log YouTube views", fontsize=11)
ax.set_title("The televote out-predicts the jury every single year (2016–2025)\n"
             "Gap labels show televote − jury per year",
             fontsize=13, color=NAVY, pad=12)
ax.legend(loc="upper left", frameon=False, fontsize=10.5)
ax.grid(True, lw=0.3, alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_xticks(years)
ax.set_ylim(-0.2, 1.0)

ax.text(0, -0.13,
        f"n per year ranges from {py_df['n'].min()} to {py_df['n'].max()}; mean gap = +{py_df['gap'].mean():.2f}",
        transform=ax.transAxes, fontsize=9, color="#666", style="italic")

plt.tight_layout()
plt.savefig(f"{OUT}/J_GAP_OVER_TIME.png", bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {OUT}/J_GAP_OVER_TIME.png")


# ============================================================
# E2 — Counterfactual: televote only
# ============================================================
print("\n" + "=" * 70)
print("  E2 — Counterfactual: what if Eurovision used televote only?")
print("=" * 70)

cf = []
for y in sorted(sub["year"].unique()):
    yr = sub[sub["year"] == y].copy()
    # Actual ranking (already there as place_final)
    yr["actual_rank"] = yr["place_final"].astype(int)
    # Counterfactual ranking by televote points only
    yr["cf_rank"] = yr["points_tele_final"].rank(method="min", ascending=False).astype(int)
    yr["place_change"] = yr["actual_rank"] - yr["cf_rank"]   # positive = climbed in CF
    cf.append(yr)

cf_df = pd.concat(cf, ignore_index=True)

# How often did the actual winner stay #1 under televote-only?
winners = cf_df[cf_df["actual_rank"] == 1].copy()
winners["cf_winner_won"] = winners["cf_rank"] == 1
print(f"\nActual winners who would have ALSO won under televote-only:")
print(winners[["year", "to_country", "performer", "song", "cf_rank", "cf_winner_won"]].to_string(index=False))
print(f"\n  → Same winner: {winners['cf_winner_won'].sum()} of {len(winners)} years")

# Per year: who would the new winner be?
print(f"\n--- New winners under televote-only ---")
new_winners = []
for y in sorted(sub["year"].unique()):
    yr = cf_df[cf_df["year"] == y]
    new = yr[yr["cf_rank"] == 1].iloc[0]
    actual = yr[yr["actual_rank"] == 1].iloc[0]
    same = new["to_country"] == actual["to_country"]
    new_winners.append({
        "year": int(y),
        "actual_winner": f"{actual['to_country']} — {actual['performer']} — {actual['song']}",
        "actual_winner_pts": int(actual["points_final"]),
        "cf_winner": f"{new['to_country']} — {new['performer']} — {new['song']}",
        "cf_winner_tele": int(new["points_tele_final"]),
        "same": same,
    })
    flag = "✓" if same else "≠"
    print(f"  {y}  {flag}  Actual: {actual['to_country']:14s} {actual['performer']:25s}  →  "
          f"Tele-only: {new['to_country']:14s} {new['performer']}")

# Biggest movers (climbed in counterfactual)
print(f"\n--- Top 10 'audience-favoured' songs (climbed most in counterfactual) ---")
movers = cf_df.nlargest(10, "place_change")
print(movers[["year", "to_country", "performer", "song", "actual_rank", "cf_rank", "place_change"]].to_string(index=False))

# Iceland specifically
print(f"\n--- ICELAND under televote-only ---")
ice = cf_df[cf_df["to_country"] == "Iceland"]
print(ice[["year", "performer", "song", "actual_rank", "cf_rank", "place_change"]].to_string(index=False))

# Save data for the website
cf_payload = {
    "per_year_winners": new_winners,
    "biggest_climbers": [
        {
            "year": int(r["year"]),
            "country": r["to_country"], "artist": r["performer"], "song": r["song"],
            "actual_rank": int(r["actual_rank"]), "cf_rank": int(r["cf_rank"]),
            "places_climbed": int(r["place_change"]),
            "tele": int(r["points_tele_final"]), "jury": int(r["points_jury_final"]),
        }
        for _, r in cf_df.nlargest(10, "place_change").iterrows()
    ],
    "biggest_fallers": [
        {
            "year": int(r["year"]),
            "country": r["to_country"], "artist": r["performer"], "song": r["song"],
            "actual_rank": int(r["actual_rank"]), "cf_rank": int(r["cf_rank"]),
            "places_dropped": -int(r["place_change"]),
            "tele": int(r["points_tele_final"]), "jury": int(r["points_jury_final"]),
        }
        for _, r in cf_df.nsmallest(10, "place_change").iterrows()
    ],
    "iceland": [
        {
            "year": int(r["year"]),
            "artist": r["performer"], "song": r["song"],
            "actual_rank": int(r["actual_rank"]), "cf_rank": int(r["cf_rank"]),
            "place_change": int(r["place_change"]),
        }
        for _, r in ice.iterrows()
    ],
}
with open("/sessions/laughing-elegant-goodall/mnt/outputs/counterfactual.json", "w") as f:
    json.dump(cf_payload, f, indent=2)

# Plot — counterfactual winners table as figure
fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
ax.axis("off")

table_data = []
for nw in new_winners:
    flag = "✓ same winner" if nw["same"] else "← CHANGE"
    table_data.append([
        str(nw["year"]),
        nw["actual_winner"][:42],
        nw["cf_winner"][:42],
        flag,
    ])

# Build the table manually
y_start = 0.95
ax.text(0.04, y_start + 0.04, "Year", fontweight="bold", fontsize=10, color=NAVY, transform=ax.transAxes)
ax.text(0.13, y_start + 0.04, "Actual winner", fontweight="bold", fontsize=10, color=NAVY, transform=ax.transAxes)
ax.text(0.50, y_start + 0.04, "Televote-only winner", fontweight="bold", fontsize=10, color=NAVY, transform=ax.transAxes)
ax.text(0.86, y_start + 0.04, "", transform=ax.transAxes)

for i, row in enumerate(table_data):
    yp = y_start - (i + 0.4) * 0.10
    color = "#444" if "✓" in row[3] else MAGENTA
    weight = "normal" if "✓" in row[3] else "bold"
    ax.text(0.04, yp, row[0], fontsize=10, color=color, transform=ax.transAxes)
    ax.text(0.13, yp, row[1], fontsize=10, color=color, transform=ax.transAxes)
    ax.text(0.50, yp, row[2], fontsize=10, color=color, fontweight=weight, transform=ax.transAxes)
    ax.text(0.86, yp, row[3], fontsize=10, color=color, transform=ax.transAxes)

ax.set_title("What if Eurovision had used televote only since 2016?",
             fontsize=14, color=NAVY, pad=20)
ax.text(0.04, -0.05,
        f"Of 9 contests, the televote-only winner matches the actual winner in "
        f"{sum(1 for nw in new_winners if nw['same'])}; "
        f"{sum(1 for nw in new_winners if not nw['same'])} would have produced different winners.",
        fontsize=10, color="#666", style="italic", transform=ax.transAxes)

plt.tight_layout()
plt.savefig(f"{OUT}/J_COUNTERFACTUAL.png", bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {OUT}/J_COUNTERFACTUAL.png")


# Save E1 numbers for reuse
e1_payload = py_df.to_dict("records")
with open("/sessions/laughing-elegant-goodall/mnt/outputs/e1_per_year_gap.json", "w") as f:
    json.dump(e1_payload, f, indent=2)

print("\nDone.")
