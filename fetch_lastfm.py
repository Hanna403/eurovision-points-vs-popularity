"""fetch_lastfm.py — pull Last.fm scrobble + listener counts for every Eurovision song.

Last.fm is a music tracking service with ~50M users. Their public API gives
you per-track playcount (total listens) and listeners (unique listeners) —
both real counts, not Spotify's mysterious 0-100 percentile.

No quota, no daily limit, free unlimited. Rate-limited to 5 req/sec.

Setup:
    1. Sign up at https://www.last.fm/api/account/create
    2. Paste your API key into .env as LASTFM_API_KEY
    3. pip3 install requests pandas

Run:
    python3 fetch_lastfm.py

Output: contestants_with_lastfm.csv (or whatever INPUT was, with two new cols)
"""
import os
import time
from pathlib import Path
import pandas as pd
import requests

INPUT  = "contestants_with_youtube.csv"        # any cleaned contestants file works
OUTPUT = "contestants_with_lastfm.csv"
ENDPOINT = "http://ws.audioscrobbler.com/2.0/"


# ---------- credentials ----------
def load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists(): return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, _, val = line.partition("=")
        key = key.strip().replace("export ", "")
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


load_dotenv()
api_key = os.environ.get("LASTFM_API_KEY")
if not api_key or api_key == "paste_your_lastfm_api_key_here":
    raise SystemExit(
        "ERROR: LASTFM_API_KEY not set in .env\n"
        "Get one at: https://www.last.fm/api/account/create"
    )


def lastfm_lookup(artist, song):
    """Return (playcount, listeners) for a track, or (None, None)."""
    params = {
        "method": "track.getInfo",
        "api_key": api_key,
        "artist": artist,
        "track": song,
        "format": "json",
        "autocorrect": 1,         # let Last.fm fix small typos
    }
    try:
        r = requests.get(ENDPOINT, params=params, timeout=15)
        if r.status_code != 200:
            return None, None
        data = r.json()
        track = data.get("track")
        if not track:
            return None, None
        playcount = track.get("playcount")
        listeners = track.get("listeners")
        return (int(playcount) if playcount else None,
                int(listeners) if listeners else None)
    except Exception as e:
        print(f"     ! error: {e}")
        return None, None


# ---------- load + resume ----------
df = pd.read_csv(INPUT)
print(f"Loaded {len(df):,} rows from {INPUT}")

if Path(OUTPUT).exists():
    prev = pd.read_csv(OUTPUT)
    if "lastfm_playcount" in prev.columns:
        key = ["year", "to_country", "performer", "song"]
        df = df.merge(prev[key + ["lastfm_playcount", "lastfm_listeners"]],
                      on=key, how="left")
        already = df["lastfm_playcount"].notna().sum()
        print(f"Resume: {already:,} rows already fetched.")
    else:
        df["lastfm_playcount"] = pd.Series(dtype="float")
        df["lastfm_listeners"] = pd.Series(dtype="float")
else:
    df["lastfm_playcount"] = pd.Series(dtype="float")
    df["lastfm_listeners"] = pd.Series(dtype="float")


# ---------- fetch ----------
print("\nFetching Last.fm playcounts (rate-limited to 5/sec, no daily quota)…\n")
todo = df[df["lastfm_playcount"].isna()]
done = 0; filled = 0
for idx in todo.index:
    row = df.loc[idx]
    artist = str(row["performer"]).strip()
    song   = str(row["song"]).strip()
    if not artist or not song or artist == "nan" or song == "nan":
        done += 1; continue

    playcount, listeners = lastfm_lookup(artist, song)
    df.at[idx, "lastfm_playcount"] = playcount
    df.at[idx, "lastfm_listeners"] = listeners

    done += 1
    if playcount is not None:
        filled += 1
        status = f"{playcount:>10,} plays · {listeners:>8,} listeners"
    else:
        status = "    (not on Last.fm)"
    print(f"  [{done:>4}/{len(todo)}]  {int(row['year'])} {str(row['to_country']):14s} "
          f"{artist[:18]:18s} {song[:25]:25s} → {status}")

    if done % 25 == 0:
        df.to_csv(OUTPUT, index=False)

    time.sleep(0.2)            # Last.fm allows 5 req/s; we go at 5/s


# ---------- final save ----------
df.to_csv(OUTPUT, index=False)
total_filled = df["lastfm_playcount"].notna().sum()
print(f"\nDone.")
print(f"  Looked up:           {done:,}")
print(f"  Successfully filled: {filled:,}")
print(f"  Total coverage:      {total_filled:,} of {len(df):,} ({100*total_filled/len(df):.1f}%)")
print(f"  Wrote: {OUTPUT}")
