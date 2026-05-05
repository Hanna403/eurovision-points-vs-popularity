"""Clean the real Spijkervet/th0mk data.

Inputs:
    contestants-6aff225e.csv   — raw scraper output, with mangled performer/song columns
    votes-deca8d75.csv         — raw votes, with self-vote rows

Outputs:
    contestants_clean.csv      — clean, deduplicated, all 21 cols
    votes_clean.csv            — self-votes filtered out
    cleaning_audit.txt         — list of every row we changed (for transparency)

The scraper's bug:
    `performer` column → actually holds the LAST WORD of the song title
    `song` column      → either holds just the artist name (one-word song)
                          OR "[rest of the song title]  [artist name]" with a double-space
"""
import re
import pandas as pd
from pathlib import Path

UPLOADS = Path("/sessions/laughing-elegant-goodall/mnt/uploads")
OUT = Path("/sessions/laughing-elegant-goodall/mnt/outputs")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CONTESTANTS — fix performer/song mangling
# ============================================================
def fix_performer_song(row):
    """Apply the double-space rule to recover the original song title and artist."""
    perf = str(row["performer"]).strip()
    song = str(row["song"]).strip()
    if perf == "nan" or song == "nan":
        return perf, song

    if "  " in song:
        # Multi-word song with artist appended after a double-space
        head, tail = song.split("  ", 1)
        true_song   = f"{head.strip()} {perf}".strip()
        true_artist = tail.strip()
    else:
        # Single-word song; perf IS the song; song IS the artist
        true_song   = perf
        true_artist = song

    # Capitalisation tidy-up: collapse multiple spaces, strip
    true_song   = re.sub(r"\s+", " ", true_song).strip()
    true_artist = re.sub(r"\s+", " ", true_artist).strip()
    return true_artist, true_song


# ============================================================
# 2. COUNTRY name fixes
# ============================================================
COUNTRY_FIXES = {
    "North MacedoniaN.Macedonia": "North Macedonia",
    "F.Y.R. Macedonia":           "North Macedonia",
    "F.Y.R Macedonia":            "North Macedonia",
    "Czech Republic":             "Czechia",                 # newer Spijkervet uses Czechia
    "The Netherlands":            "Netherlands",
}


def fix_country(s):
    s = str(s).strip()
    return COUNTRY_FIXES.get(s, s)


# ============================================================
# Run cleaning
# ============================================================
print("=" * 70)
print("  CLEANING contestants.csv")
print("=" * 70)

con = pd.read_csv(UPLOADS / "contestants-1981cc05.csv")
print(f"Input:  {len(con):,} rows × {con.shape[1]} cols")

# Audit log: capture every row's original values so we can verify the fix later
audit_rows = []

new_perf, new_song = [], []
for _, r in con.iterrows():
    artist, song = fix_performer_song(r)
    new_perf.append(artist)
    new_song.append(song)

con["performer_orig"] = con["performer"]
con["song_orig"] = con["song"]
con["performer"] = new_perf
con["song"] = new_song

# Fix country names
n_country_fixed = 0
for old, new in COUNTRY_FIXES.items():
    n = (con["to_country"] == old).sum()
    if n > 0:
        print(f"  Country fix: '{old}' → '{new}' ({n} rows)")
        con.loc[con["to_country"] == old, "to_country"] = new
        n_country_fixed += n
print(f"  Total country fixes:    {n_country_fixed}")

# Drop the orig columns from the saved file (kept just for the audit pass)
audit = con[["year", "to_country", "performer", "song",
             "performer_orig", "song_orig",
             "place_final", "points_final"]].copy()
audit_path = OUT / "cleaning_audit.csv"
audit.to_csv(audit_path, index=False)
print(f"  Audit saved to: {audit_path}")

con = con.drop(columns=["performer_orig", "song_orig"])

# ============================================================
# 3. Dedup: keep the most-informative row per (year, country, performer, song)
# ============================================================
print("\n  Deduplicating semi-final + final pairs…")
key = ["year", "to_country", "performer", "song"]
n_before = len(con)

# A row's "informativeness" = number of non-null fields, with bonus for place_final
def info_score(row):
    base = row.notna().sum()
    bonus = 0 if pd.isna(row["place_final"]) else 100
    return base + bonus

con["_score"] = con.apply(info_score, axis=1)
con = (con.sort_values("_score", ascending=False)
          .drop_duplicates(subset=key, keep="first")
          .drop(columns=["_score"])
          .sort_values(["year", "to_country"])
          .reset_index(drop=True))

print(f"  {n_before:,} → {len(con):,} rows after dedup")

# Save
clean_con_path = OUT / "contestants_clean.csv"
con.to_csv(clean_con_path, index=False)
print(f"  Wrote: {clean_con_path}  ({clean_con_path.stat().st_size/1024:.0f} KB)")


# ============================================================
# 4. VOTES — filter self-votes
# ============================================================
print("\n" + "=" * 70)
print("  CLEANING votes.csv")
print("=" * 70)

vot = pd.read_csv(UPLOADS / "votes-deca8d75.csv")
print(f"Input:  {len(vot):,} rows")

# Self-votes
self_mask = vot["from_country_id"] == vot["to_country_id"]
print(f"  Self-votes (from==to):  {self_mask.sum():,}  (removing)")
vot = vot[~self_mask].copy()

# Country fixes in votes too
for col in ["from_country", "to_country"]:
    for old, new in COUNTRY_FIXES.items():
        vot.loc[vot[col] == old, col] = new

clean_vot_path = OUT / "votes_clean.csv"
vot.to_csv(clean_vot_path, index=False)
print(f"  Output: {len(vot):,} rows")
print(f"  Wrote:  {clean_vot_path}  ({clean_vot_path.stat().st_size/1024:.0f} KB)")


# ============================================================
# 5. VERIFY — spot-check known winners
# ============================================================
print("\n" + "=" * 70)
print("  VERIFICATION — known winners after cleaning")
print("=" * 70)

known = [
    (2016, "Ukraine",     "Jamala",            "1944",                534),
    (2017, "Portugal",    "Salvador Sobral",   "Amar pelos dois",     758),
    (2018, "Israel",      "Netta",             "Toy",                 529),
    (2019, "Netherlands", "Duncan Laurence",   "Arcade",              498),
    (2021, "Italy",       "Måneskin",          "Zitti e buoni",       524),
    (2022, "Ukraine",     "Kalush Orchestra",  "Stefania",            631),
    (2023, "Sweden",      "Loreen",            "Tattoo",              583),
    (2014, "Austria",     "Conchita Wurst",    "Rise Like a Phoenix", 290),
    (1974, "Sweden",      "ABBA",              "Waterloo",             24),
]
for year, country, expected_artist, expected_song, expected_pts in known:
    row = con[(con["year"] == year) & (con["to_country"] == country)]
    if len(row) == 0:
        print(f"  ✗ {year} {country:14s} → not found"); continue
    if len(row) > 1:
        # Multiple rows for same year/country → unlikely after dedup but defensive
        print(f"  ⚠ {year} {country:14s} → {len(row)} rows — picking first"); row = row.head(1)
    r = row.iloc[0]
    artist_ok = expected_artist.split()[-1].lower() in str(r["performer"]).lower() \
                or expected_artist.split()[0].lower() in str(r["performer"]).lower()
    song_ok = expected_song.split()[0].lower() in str(r["song"]).lower()
    pts_ok = (pd.notna(r["points_final"]) and abs(r["points_final"] - expected_pts) < 5)
    flag = "✓" if (artist_ok and song_ok and pts_ok) else "✗"
    print(f"  {flag} {year} {country:14s} {str(r['performer'])[:18]:18s} | {str(r['song'])[:25]:25s} "
          f"| {int(r['points_final']) if pd.notna(r['points_final']) else '—'} pts")


# ============================================================
# 6. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
finals_2016_23 = con[
    (con["year"].between(2016, 2023))
    & con["place_final"].notna()
    & con["points_jury_final"].notna()
    & con["points_tele_final"].notna()
]
print(f"  Cleaned contestants: {len(con):,} rows")
print(f"  Cleaned votes:       {len(vot):,} rows")
print(f"  2016-2023 finalists with full jury+tele data: {len(finals_2016_23):,}")
print(f"  → ready for Spotify + YouTube enrichment")
