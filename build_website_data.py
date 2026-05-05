"""Build the JSON data files the website needs:
    - finalists.js   — all 229 triangulated finalists (year, country, artist, song,
                       jury, tele, place, log_yt, log_play, cinderella_score)
    - iceland.js     — every Icelandic Eurovision entry across all years
    - cinderella.js  — sorted by underrated/overrated for the leaderboard
"""
import json
import numpy as np
import pandas as pd

CONTESTANTS = "/sessions/laughing-elegant-goodall/mnt/outputs/contestants_with_lastfm.csv"
TRI         = "/sessions/laughing-elegant-goodall/mnt/outputs/eurovision_analysis_TRIANGULATED.csv"
OUT_JS      = "/sessions/laughing-elegant-goodall/mnt/outputs/website/website_data.js"


# ============================================================
# 1. Triangulated finalists with Cinderella score
# ============================================================
tri = pd.read_csv(TRI)

# Z-score popularity (combined, log-scaled) and placement within the dataset
tri["log_yt"]      = np.log10(tri["yt_views"].astype(float))
tri["log_play"]    = np.log10(tri["lastfm_playcount"].astype(float))
tri["log_listen"]  = np.log10(tri["lastfm_listeners"].astype(float))
tri["pop_combined"] = (
    (tri["log_yt"] - tri["log_yt"].mean()) / tri["log_yt"].std()
    + (tri["log_play"] - tri["log_play"].mean()) / tri["log_play"].std()
    + (tri["log_listen"] - tri["log_listen"].mean()) / tri["log_listen"].std()
) / 3

# Higher placement = worse. We want a "betterness" score = 1 - (place - 1) / max_place
tri["place_score"] = 1 - (tri["place_final"] - 1) / 26
# Z-score the place score so it's comparable to pop_combined
tri["place_z"] = (tri["place_score"] - tri["place_score"].mean()) / tri["place_score"].std()

# Cinderella score = how much MORE popular this song is than its placement would suggest
# Positive = "robbed by the jury", negative = "jury overhyped it"
tri["cinderella"] = tri["pop_combined"] - tri["place_z"]

print(f"Triangulated finalists: {len(tri)}")
print(f"\nTop 12 'robbed' (audience loved, Eurovision underrated):")
top = tri.nlargest(12, "cinderella")
for _, r in top.iterrows():
    print(f"  {int(r['year'])} {r['to_country']:14s} #{int(r['place_final']):>2}  "
          f"{r['performer'][:25]:25s} {r['song'][:25]:25s}  "
          f"score = {r['cinderella']:+.2f}")

print(f"\nBottom 12 'overhyped' (Eurovision loved, audience yawned):")
bot = tri.nsmallest(12, "cinderella")
for _, r in bot.iterrows():
    print(f"  {int(r['year'])} {r['to_country']:14s} #{int(r['place_final']):>2}  "
          f"{r['performer'][:25]:25s} {r['song'][:25]:25s}  "
          f"score = {r['cinderella']:+.2f}")

# Build the finalist records
def to_dict(r):
    return {
        "year":     int(r["year"]),
        "country":  str(r["to_country"]),
        "artist":   str(r["performer"]),
        "song":     str(r["song"]),
        "place":    int(r["place_final"]),
        "jury":     int(r["points_jury_final"]),
        "tele":     int(r["points_tele_final"]),
        "total":    int(r["points_final"]),
        "yt_views": int(r["yt_views"]),
        "lastfm_play":     int(r["lastfm_playcount"]),
        "lastfm_listen":   int(r["lastfm_listeners"]),
        "cinderella":      round(float(r["cinderella"]), 3),
        "yt_id":     str(r["yt_video_id"]) if pd.notna(r.get("yt_video_id")) else None,
    }


finalists_data = [to_dict(r) for _, r in tri.iterrows()]


# ============================================================
# 2. Iceland — every Icelandic Eurovision entry, all years
# ============================================================
con = pd.read_csv(CONTESTANTS)
iceland = con[con["to_country"] == "Iceland"].copy()
print(f"\n\nIceland entries: {len(iceland)} ({iceland['year'].min()}–{iceland['year'].max()})")

iceland_records = []
for _, r in iceland.iterrows():
    rec = {
        "year":     int(r["year"]),
        "artist":   str(r["performer"]),
        "song":     str(r["song"]),
        "place":    int(r["place_final"]) if pd.notna(r["place_final"]) else None,
        "points":   int(r["points_final"]) if pd.notna(r["points_final"]) else None,
        "jury":     int(r["points_jury_final"]) if pd.notna(r["points_jury_final"]) else None,
        "tele":     int(r["points_tele_final"]) if pd.notna(r["points_tele_final"]) else None,
        "yt_views": int(r["yt_views"]) if pd.notna(r["yt_views"]) else None,
        "yt_id":    str(r["yt_video_id"]) if pd.notna(r.get("yt_video_id")) else None,
    }
    iceland_records.append(rec)


# ============================================================
# 3. Triangulated stats summary
# ============================================================
with open("/sessions/laughing-elegant-goodall/mnt/outputs/jury_vs_televote_results_TRIANGULATED.json") as f:
    stats = json.load(f)


# ============================================================
# Write a single bundled JS file the website can import
# ============================================================
js_payload = (
    f"// Generated by build_website_data.py\n"
    f"// {len(finalists_data)} triangulated finalists + {len(iceland_records)} Iceland entries\n\n"
    f"const FINALISTS = {json.dumps(finalists_data, ensure_ascii=False)};\n\n"
    f"const ICELAND = {json.dumps(iceland_records, ensure_ascii=False)};\n\n"
    f"const STATS = {json.dumps(stats, ensure_ascii=False)};\n"
)

with open(OUT_JS, "w") as f:
    f.write(js_payload)

print(f"\nWrote: {OUT_JS}  ({len(js_payload)/1024:.0f} KB)")
print(f"  FINALISTS: {len(finalists_data)} records")
print(f"  ICELAND:   {len(iceland_records)} records")
print(f"  STATS:     triangulated correlations + Steiger Z")
