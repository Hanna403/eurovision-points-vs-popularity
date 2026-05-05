"""Compute the choropleth data: Iceland → them and them → Iceland avg points,
mapped to ISO-3166-1 alpha-3 country codes for Plotly's choropleth."""
import json
import pandas as pd

VOTES = "/sessions/laughing-elegant-goodall/mnt/outputs/votes_clean.csv"
CONTESTANTS = "/sessions/laughing-elegant-goodall/mnt/outputs/contestants_clean.csv"
OUT = "/sessions/laughing-elegant-goodall/mnt/outputs/choropleth_data.json"

# Eurovision country -> ISO-3 mapping (Plotly choropleth uses these codes)
# Built manually from Spijkervet country names
ISO3 = {
    "Albania": "ALB", "Andorra": "AND", "Armenia": "ARM", "Australia": "AUS",
    "Austria": "AUT", "Azerbaijan": "AZE", "Belarus": "BLR", "Belgium": "BEL",
    "Bosnia & Herzegovina": "BIH", "Bulgaria": "BGR", "Croatia": "HRV",
    "Cyprus": "CYP", "Czechia": "CZE", "Czech Republic": "CZE",
    "Denmark": "DNK", "Estonia": "EST", "Finland": "FIN", "France": "FRA",
    "Georgia": "GEO", "Germany": "DEU", "Greece": "GRC", "Hungary": "HUN",
    "Iceland": "ISL", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
    "Latvia": "LVA", "Lithuania": "LTU", "Luxembourg": "LUX", "Malta": "MLT",
    "Moldova": "MDA", "Monaco": "MCO", "Montenegro": "MNE", "Morocco": "MAR",
    "Netherlands": "NLD", "North Macedonia": "MKD", "Norway": "NOR",
    "Poland": "POL", "Portugal": "PRT", "Romania": "ROU", "Russia": "RUS",
    "San Marino": "SMR", "Serbia": "SRB", "Serbia & Montenegro": "SCG",
    "Slovakia": "SVK", "Slovenia": "SVN", "Spain": "ESP", "Sweden": "SWE",
    "Switzerland": "CHE", "Turkey": "TUR", "Ukraine": "UKR",
    "United Kingdom": "GBR", "Yugoslavia": "YUG",
}

con = pd.read_csv(CONTESTANTS)
code_to_name = dict(zip(con["to_country_id"], con["to_country"]))

vot = pd.read_csv(VOTES)
finals = vot[vot["round"] == "final"].copy()
finals["from_country"] = finals["from_country_id"].map(code_to_name).fillna(finals["from_country_id"])
finals["to_country"]   = finals["to_country_id"].map(code_to_name).fillna(finals["to_country_id"])

# Iceland → them
ice_out = finals[finals["from_country"] == "Iceland"]
out_agg = (ice_out.groupby("to_country")
           .agg(total=("total_points", "sum"),
                appearances=("year", "nunique"))
           .reset_index())
out_agg["avg"] = out_agg["total"] / out_agg["appearances"]

# Them → Iceland
ice_in = finals[finals["to_country"] == "Iceland"]
in_agg = (ice_in.groupby("from_country")
          .agg(total=("total_points", "sum"),
               appearances=("year", "nunique"))
          .reset_index())
in_agg["avg"] = in_agg["total"] / in_agg["appearances"]

# Build the records — only countries with ISO codes + ≥3 co-appearances
records = []
for country, iso in ISO3.items():
    if country == "Iceland":
        continue  # skip self
    out_row = out_agg[out_agg["to_country"] == country]
    in_row  = in_agg[in_agg["from_country"] == country]
    if len(out_row) == 0 and len(in_row) == 0:
        continue

    out_avg = float(out_row.iloc[0]["avg"]) if len(out_row) else None
    out_n   = int(out_row.iloc[0]["appearances"]) if len(out_row) else 0
    in_avg  = float(in_row.iloc[0]["avg"]) if len(in_row) else None
    in_n    = int(in_row.iloc[0]["appearances"]) if len(in_row) else 0

    if max(out_n, in_n) < 3:
        continue   # too thin to show

    records.append({
        "country": country, "iso3": iso,
        "out_avg": round(out_avg, 2) if out_avg is not None else None,
        "out_n":   out_n,
        "in_avg":  round(in_avg, 2) if in_avg is not None else None,
        "in_n":    in_n,
    })

records.sort(key=lambda r: r["out_avg"] or 0, reverse=True)

with open(OUT, "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(records)} country records to {OUT}")
print(f"\nTop 10 (Iceland → them):")
for r in records[:10]:
    nordic = "🇮🇸" if r["country"] in ("Sweden","Denmark","Norway","Finland") else "  "
    print(f"  {nordic} {r['country']:20s} {r['iso3']}  out: {r['out_avg']:.2f} ({r['out_n']:>2}× )  in: {r['in_avg']} ({r['in_n']}×)")
