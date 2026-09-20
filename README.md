# Auspify Technologies — Data Science Internship Submission
**Intern:** Awais Ahmed
**Dataset:** Netflix Titles (8,790 raw records)
**Tasks completed:** All 6 of 6

## Folder contents

task1_cleaning.py            Task 1: Data Cleaning & Preprocessing
task2_eda.py                 Task 2: Exploratory Data Analysis
task3_recommendation.py      Task 3: Recommendation System Analysis
task4_trend_prediction.py    Task 4: Trend Prediction Analysis
task5_ml_classification.py   Task 5: ML Classification Model
task6_dashboard.py           Task 6: Business Insights Dashboard

data/netflix_raw.csv         Original dataset
outputs/netflix_cleaned.csv  Cleaned dataset (Task 1 output, used by all later tasks)
outputs/task*_summary.txt    Write-up for each task
outputs/task6_business_dashboard.html   Final dashboard (open in a browser)
figures/*.png                All charts generated across the tasks

## How to reproduce
Run in order:
python3 task1_cleaning.py
python3 task2_eda.py
python3 task3_recommendation.py
python3 task4_trend_prediction.py
python3 task5_ml_classification.py
python3 task6_dashboard.py

## Task-by-task summary

**Task 1 — Cleaning:** Removed 3 duplicate rows, standardized missing markers, split `duration` into value + unit, parsed dates, exploded genres. Caught a bug where a movie literally titled "Unknown" was almost wiped by naive missing-value cleanup.

**Task 2 — EDA:** Movies outnumber TV Shows ~2.3:1. US, India, UK are top content-producing countries. "International Movies" is the top genre. Additions peaked in 2019.

**Task 3 — Recommendations:** TF-IDF + cosine-similarity content-based recommender, tested on sample titles with strong genre-overlap results.

**Task 4 — Trend Forecasting:** Linear regression forecast for 2021-2023, flagged as an optimistic upper bound.

**Task 5 — Classification:** ~99.8% accuracy model, investigated and found data leakage (duration units + genre label text both reveal the answer) — documented honestly.

**Task 6 — Dashboard:** Combines everything into one HTML report with KPIs, charts, and business recommendations.
