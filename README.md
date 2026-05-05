# Eurovision: Points vs. Popularity

A 70-year look at whether the Eurovision Song Contest's scoreboard predicts real-world popularity.

**Group 18** · Arnar Thor Bjornsson & Hanna Margrét Pétursdóttir
**Course:** 02806 Social Data Analysis & Visualization · DTU · Spring 2026

🌐 **[View the website →](https://hanna403.github.io/eurovision-points-vs-popularity/website/)**
📓 **[View the explainer notebook →](https://github.com/Hanna403/eurovision-points-vs-popularity/blob/main/Assignment_B_Explainer.ipynb)**

---

## What's in this repo

```
eurovision_clean/
├── website/ ← public-facing data story
│ ├── index.html main page
│ ├── website_data.js all the data + analysis results
│ └── assets/ static figures (PNG)
│
├── Assignment_B_Explainer.ipynb technical companion (analysis + ML)
│
├── contestants_clean.csv 1,713 Eurovision entries 1956–2025
├── contestants_with_youtube_full.csv real YouTube views via yt-dlp (1,712/1,713)
├── contestants_with_lastfm.csv Last.fm playcount + listeners (1,668/1,713)
├── eurovision_real_2016_2025.csv 232 grand-final analysis subset
├── eurovision_residuals.csv OLS predictions + residuals (n=232)
├── eurovision_analysis_TRIANGULATED.csv triangulated analysis (n=229)
├── jury_vs_televote_results_TRIANGULATED.json headline correlations
├── jury_vs_televote_results_FULL.json
├── regression_results.json OLS coefficients (Model 1 + Model 2)
├── counterfactual.json E2: televote-only winners
├── e1_per_year_gap.json E1: per-year correlation gap
├── iceland_voting_blocs.json Iceland's reciprocal allies
└── cleaning_audit.csv before/after for every cleaned row
```

## How to reproduce

The pipeline runs in order:

```bash
# 1. Clean Spijkervet/th0mk raw data
python3 clean_real_data.py
# → contestants_clean.csv, votes_clean.csv

# 2. Pull YouTube views (no API key, uses yt-dlp)
pip3 install yt-dlp pandas
python3 fetch_youtube_ytdlp.py
# → contestants_with_youtube.csv

# 3. Pull Last.fm playcount + listeners (free API key)
# Add LASTFM_API_KEY to .env
python3 fetch_lastfm.py
# → contestants_with_lastfm.csv

# 4. Run all analyses + generate figures
python3 jury_vs_tele_real.py # triangulated correlations + Steiger Z
python3 regression_with_country_fe.py # OLS with country FE + residuals
python3 e1_e2_analysis.py # year-by-year gap + counterfactual
python3 iceland_voting_blocs.py # Iceland's voting allies
python3 build_website_data.py # bundle everything into website_data.js
```

## Headline findings

1. **The televote correlates with YouTube views at r = +0.68; the jury at r = +0.39.** Steiger's Z = +5.43, p < 10⁻⁷, n = 229 grand finalists, 2016–2025.
2. **For Last.fm playcount and listeners, jury and televote correlate equally** (both ≈ +0.20, p > 0.15). YouTube captures *virality*; Last.fm captures *sustained listening*. the jury and audience are answering different questions.
3. **OLS regression with country fixed effects** confirms the result is robust to voting-bloc variance. Standardized televote coefficient is 10.2× the jury coefficient. R² = 0.65.
4. **Counterfactual analysis**: 5 of 9 post-2016 contests would have produced different winners under televote-only (Käärijä in 2023, Baby Lasagna in 2024, etc.).
5. **Iceland's televote ally is overwhelmingly Sweden** (avg 9.1 points/appearance); Iceland's strongest fan is Denmark (avg 5.9).

## Acknowledgements

- **Spijkervet Eurovision Dataset**. Burgoyne, Spijkervet & Baker (2023), ISMIR. We use the th0mk fork for current-year support.
- **YouTube Data API v3** + **yt-dlp** for view counts.
- **Last.fm Web Services** for playcount and listener data.
- **Segel & Heer (2010)**, IEEE TVCG, narrative visualization framework (Interactive Slideshow genre).

---

*Last updated: May 2026.*
