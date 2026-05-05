"""fetch_youtube_ytdlp.py — fill in missing YouTube views using yt-dlp.

No API key needed. No daily quota. Goes after every entry that's still
missing yt_views in contestants_with_youtube.csv.

Setup:
    pip3 install yt-dlp

Run:
    python3 fetch_youtube_ytdlp.py

Resumable: re-run any time, picks up where it left off.
Saves progress every 10 lookups.
"""
import time
import pandas as pd
from pathlib import Path
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

INPUT  = "contestants_with_youtube.csv"
OUTPUT = "contestants_with_youtube.csv"

# Silent yt-dlp config — extract metadata only, never download
ydl_opts = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "extract_flat": False,
    "noplaylist": True,
    "ignoreerrors": True,
    "socket_timeout": 15,
}


# ---------- load data ----------
df = pd.read_csv(INPUT)
have = df["yt_views"].notna().sum()
missing = df[df["yt_views"].isna()]
print(f"Loaded {len(df):,} rows from {INPUT}")
print(f"  Have yt_views:    {have:,}")
print(f"  Missing yt_views: {len(missing):,}")
print(f"\nNo quota limit — yt-dlp can run through all of these in 15–30 min.\n")

if len(missing) == 0:
    print("Nothing to do.")
    raise SystemExit(0)


def lookup(artist, song, year, country):
    """Return (view_count, video_id, url) or (None, None, None)."""
    queries = [
        f'ytsearch1:Eurovision {year} {country} {artist} {song}',
        f'ytsearch1:Eurovision {year} {artist} {song}',
        f'ytsearch1:{artist} {song} Eurovision',
    ]
    with YoutubeDL(ydl_opts) as ydl:
        for q in queries:
            try:
                info = ydl.extract_info(q, download=False)
            except DownloadError:
                continue
            if not info:
                continue
            entries = info.get("entries", [info])
            if not entries:
                continue
            entry = entries[0]
            if not entry:
                continue
            vc = entry.get("view_count")
            vid = entry.get("id")
            if vc is not None and vid:
                url = entry.get("webpage_url") or f"https://youtube.com/watch?v={vid}"
                return int(vc), vid, url
    return None, None, None


# ---------- run ----------
done = 0; filled = 0
t0 = time.time()
for idx in missing.index:
    row = df.loc[idx]
    artist  = str(row["performer"]).strip()
    song    = str(row["song"]).strip()
    year    = int(row["year"])
    country = str(row["to_country"]).strip()

    if not artist or not song or artist == "nan" or song == "nan":
        done += 1
        continue

    try:
        views, vid, url = lookup(artist, song, year, country)
    except KeyboardInterrupt:
        print("\n  Interrupted — saving progress…")
        break
    except Exception as e:
        print(f"   ! lookup error for {artist} - {song}: {e}")
        views, vid, url = None, None, None

    done += 1
    if views is not None:
        df.at[idx, "yt_views"]    = views
        df.at[idx, "yt_video_id"] = vid
        df.at[idx, "youtube_url"] = url
        filled += 1
        status = f"{views:>12,} views"
    else:
        status = "    (no match)"

    rate = done / max(time.time() - t0, 1)
    print(f"  [{done:>3}/{len(missing)}]  {year} {country:14s} {artist[:18]:18s} {song[:25]:25s} → {status}    "
          f"({rate:.1f}/s)")

    if done % 10 == 0:
        df.to_csv(OUTPUT, index=False)


# ---------- final save ----------
df.to_csv(OUTPUT, index=False)
new_have = df["yt_views"].notna().sum()
new_missing = len(df) - new_have
elapsed = time.time() - t0
print(f"\nDone.")
print(f"  Looked up:           {done:,}")
print(f"  Successfully filled: {filled:,}")
print(f"  Total coverage:      {new_have:,} / {len(df):,} ({100*new_have/len(df):.1f}%)")
print(f"  Still missing:       {new_missing:,}")
print(f"  Time elapsed:        {elapsed/60:.1f} min")
if new_missing:
    print(f"\n  Some entries (likely pre-1980s) have no official YouTube upload — re-runs won't find them.")
