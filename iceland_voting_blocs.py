"""Compute Iceland's voting allies from votes_clean.csv.

For each country pair (Iceland → X) and (X → Iceland), sum tele/jury points
across all years where data is available, then visualise.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VOTES = "/sessions/laughing-elegant-goodall/mnt/outputs/votes_clean.csv"
OUT_FIG = "/sessions/laughing-elegant-goodall/mnt/outputs/slide_assets/iceland_voting_blocs.png"
OUT_JSON = "/sessions/laughing-elegant-goodall/mnt/outputs/iceland_voting_blocs.json"

NAVY, INK, MAGENTA, GOLD, MUTED = "#0B1B3D", "#0B3D91", "#E6007E", "#F2C94C", "#9AA5B1"
NORDIC = {"Sweden", "Norway", "Denmark", "Finland"}
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200,
                     "font.family": "DejaVu Sans", "axes.titleweight": "bold"})

vot = pd.read_csv(VOTES)
finals = vot[vot["round"] == "final"].copy()
print(f"Finals votes: {len(finals):,} rows, {finals['year'].min()}-{finals['year'].max()}")

# votes.csv stores country codes; map to full names via contestants_clean.csv
con = pd.read_csv("/sessions/laughing-elegant-goodall/mnt/outputs/contestants_clean.csv")
code_to_name = dict(zip(con["to_country_id"], con["to_country"]))
print(f"Country code mapping built: {len(code_to_name)} codes")
print(f"  e.g.  is → {code_to_name.get('is')}, se → {code_to_name.get('se')}, "
      f"dk → {code_to_name.get('dk')}, no → {code_to_name.get('no')}")

finals["from_country"] = finals["from_country_id"].map(code_to_name).fillna(finals["from_country_id"])
finals["to_country"]   = finals["to_country_id"].map(code_to_name).fillna(finals["to_country_id"])

# ============================================================
# Iceland → others (all-time)
# ============================================================
ice_out = finals[finals["from_country"] == "Iceland"]
out_totals = (ice_out.groupby("to_country")
              .agg(total=("total_points", "sum"),
                   tele=("tele_points", "sum"),
                   jury=("jury_points", "sum"),
                   appearances=("year", "nunique"))
              .reset_index())
out_totals = out_totals.sort_values("total", ascending=False)
out_totals["avg_per_year"] = (out_totals["total"] / out_totals["appearances"]).round(2)

# Filter to countries with enough overlap for a reliable estimate
MIN_APPEARANCES = 15
out_totals = out_totals[out_totals["appearances"] >= MIN_APPEARANCES]

print(f"\nTOP 10 — Iceland → them  (filter: ≥{MIN_APPEARANCES} appearances)")
top_out = out_totals.nlargest(10, "avg_per_year")
print(top_out[["to_country", "appearances", "total", "avg_per_year"]].to_string(index=False))

# ============================================================
# Others → Iceland
# ============================================================
ice_in = finals[finals["to_country"] == "Iceland"]
in_totals = (ice_in.groupby("from_country")
             .agg(total=("total_points", "sum"),
                  tele=("tele_points", "sum"),
                  jury=("jury_points", "sum"),
                  appearances=("year", "nunique"))
             .reset_index())
in_totals["avg_per_year"] = (in_totals["total"] / in_totals["appearances"]).round(2)
in_totals = in_totals[in_totals["appearances"] >= MIN_APPEARANCES]

print(f"\nTOP 10 — them → Iceland  (filter: ≥{MIN_APPEARANCES} appearances)")
top_in = in_totals.nlargest(10, "avg_per_year")
print(top_in[["from_country", "appearances", "total", "avg_per_year"]].to_string(index=False))


# ============================================================
# Figure: dual horizontal bars
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="white")

# Iceland → Others
ax = axes[0]
top10_out = out_totals.nlargest(10, "avg_per_year")
y_pos = np.arange(len(top10_out))
colors_out = [MAGENTA if c in NORDIC else INK for c in top10_out["to_country"]]
ax.barh(y_pos, top10_out["avg_per_year"], color=colors_out, edgecolor="white", linewidth=0.8)
ax.set_yticks(y_pos); ax.set_yticklabels(top10_out["to_country"], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Avg points per appearance (Iceland → them)", fontsize=10)
ax.set_title("Iceland's favourite countries\n(who Iceland sends points to)",
             fontsize=13, color=NAVY, pad=12)
ax.grid(True, axis="x", lw=0.3, alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
# Annotate
for i, (c, v) in enumerate(zip(top10_out["to_country"], top10_out["avg_per_year"])):
    ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=9, color="#444")

# Others → Iceland
ax = axes[1]
top10_in = in_totals.nlargest(10, "avg_per_year")
y_pos = np.arange(len(top10_in))
colors_in = [MAGENTA if c in NORDIC else INK for c in top10_in["from_country"]]
ax.barh(y_pos, top10_in["avg_per_year"], color=colors_in, edgecolor="white", linewidth=0.8)
ax.set_yticks(y_pos); ax.set_yticklabels(top10_in["from_country"], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Avg points per appearance (them → Iceland)", fontsize=10)
ax.set_title("Iceland's biggest fans\n(who sends points to Iceland)",
             fontsize=13, color=NAVY, pad=12)
ax.grid(True, axis="x", lw=0.3, alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for i, (c, v) in enumerate(zip(top10_in["from_country"], top10_in["avg_per_year"])):
    ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=9, color="#444")

# Legend (manual)
from matplotlib.patches import Patch
legend = [
    Patch(facecolor=MAGENTA, label="Nordic country (Sweden, Norway, Denmark, Finland)"),
    Patch(facecolor=INK,     label="Non-Nordic country"),
]
fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(f"The Nordic Voting Bloc: Iceland's Eurovision Allies\n(countries with ≥{MIN_APPEARANCES} grand-final co-appearances)",
             fontsize=14, fontweight="bold", color=NAVY, y=1.02)
plt.tight_layout()
fig.savefig(OUT_FIG, bbox_inches="tight", facecolor="white")
plt.close()

# ============================================================
# Save JSON for the website
# ============================================================
def to_records(df, key, n=10):
    return [
        {"country": r[key], "avg": round(float(r["avg_per_year"]), 2),
         "total": int(r["total"]), "appearances": int(r["appearances"]),
         "is_nordic": bool(r[key] in NORDIC)}
        for _, r in df.nlargest(n, "avg_per_year").iterrows()
    ]

payload = {
    "iceland_to_others": to_records(out_totals, "to_country", 12),
    "others_to_iceland": to_records(in_totals,  "from_country", 12),
}
with open(OUT_JSON, "w") as f:
    json.dump(payload, f, indent=2)

print(f"\nWrote: {OUT_FIG}")
print(f"Wrote: {OUT_JSON}")
