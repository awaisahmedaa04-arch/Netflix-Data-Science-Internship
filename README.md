# Auspify Technologies — Data Science Internship Submission
**Intern:** Awais Ahmed
**Dataset:** Netflix Titles (8,790 raw records)
**Tasks completed:** All 6 of 6

## Folder contents

```
task1_cleaning.py            Task 1: Data Cleaning & Preprocessing
task2_eda.py                 Task 2: Exploratory Data Analysis
task3_recommendation.py      Task 3: Recommendation System Analysis
task4_trend_prediction.py    Task 4: Trend Prediction Analysis
task5_ml_classification.py   Task 5: ML Classification Model
task6_dashboard.py           Task 6: Business Insights Dashboard (builds the HTML report)

data/netflix_raw.csv         Original dataset
outputs/netflix_cleaned.csv  Cleaned dataset (output of Task 1, used by all later tasks)
outputs/task*_summary.txt    Console-style write-up for each task
outputs/task6_business_dashboard.html   Final presentable dashboard (open in a browser)
figures/*.png                All charts generated across the tasks
```

## How to reproduce
Run in order (each script reads the previous task's output):
```
python3 task1_cleaning.py
python3 task2_eda.py
python3 task3_recommendation.py
python3 task4_trend_prediction.py
python3 task5_ml_classification.py
python3 task6_dashboard.py
```

## Task-by-task summary

**Task 1 — Cleaning:** Removed 3 duplicate rows, standardized missing markers,
split `duration` into a numeric value + unit, parsed `date_added`, and exploded
`listed_in` into a genre list. One pitfall caught: a title literally named
"Unknown" was almost wiped out by naive placeholder-replacement — fixed by
excluding the `title` column from that step.

**Task 2 — EDA:** Movies outnumber TV Shows ~2.3:1. The US, India and the UK
are the top content-producing countries. "International Movies" is the single
largest genre. Content additions peaked in 2019.

**Task 3 — Recommendations:** A TF-IDF + cosine-similarity model over genre/type/
rating gives strong content-based recommendations (near-perfect genre overlap
for the sample titles tested).

**Task 4 — Trend Forecasting:** A linear regression on yearly additions
(2015-2020) forecasts continued growth, but the writeup flags this as an
optimistic upper bound since real catalogs plateau.

**Task 5 — Classification:** A model to predict Movie vs TV Show hit ~99.8%
accuracy — investigated and found this is mostly due to data leakage
(duration units and genre label text both implicitly reveal the answer),
documented as a key learning rather than presented as a "real" result.

**Task 6 — Dashboard:** Combines all of the above into a single HTML report
with KPIs, charts, and five business recommendations for a hypothetical
Netflix content strategy team.
