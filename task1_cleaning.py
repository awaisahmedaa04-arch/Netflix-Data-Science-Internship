"""
Task 1 (Easy) - Data Cleaning & Preprocessing
Auspify Technologies - Data Science Internship

Prepares the Netflix dataset for analysis by cleaning, transforming,
and organizing the raw data.
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/netflix_raw.csv"
CLEAN_PATH = "outputs/netflix_cleaned.csv"
REPORT_PATH = "outputs/task1_cleaning_report.txt"

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


# Step 1: Import the dataset
df = pd.read_csv(RAW_PATH)
log("TASK 1 - DATA CLEANING & PREPROCESSING")
log("=" * 50)
log(f"Raw dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
log(f"Columns: {list(df.columns)}")

# Step 2: Identify missing and duplicate records
log("\n--- Missing values before cleaning ---")
missing_before = df.isna().sum()
log(missing_before.to_string())

# Some datasets store missing info as literal strings instead of NaN.
# NOTE: we deliberately do NOT apply "Unknown" to the "title" column below,
# since "Unknown" is an actual movie title in this dataset (show_id s1468) -
# blindly replacing it would wrongly erase real data.
placeholder_strings = ["", "nan", "NaN", "N/A", "n/a", "Unknown", "unknown"]
cols_to_standardize = [c for c in df.columns if c != "title"]
df[cols_to_standardize] = df[cols_to_standardize].replace(placeholder_strings, np.nan)

log("\n--- Missing values after standardizing placeholder text ---")
missing_after_std = df.isna().sum()
log(missing_after_std.to_string())

dup_full = df.duplicated().sum()
dup_key = df.duplicated(subset=["title", "type", "country"]).sum()
log(f"\nFully duplicated rows: {dup_full}")
log(f"Duplicate rows by (title, type, country): {dup_key}")

# Step 3: Handle null values and inconsistent formats
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["title", "type", "country"], keep="first")

# director / country: many real-world Netflix rows have no director listed.
# We keep the row (it's still valid content) but flag it clearly instead of
# silently dropping data.
df["director"] = df["director"].fillna("Not Specified")
df["country"] = df["country"].fillna("Not Specified")

# Some rows list multiple countries; keep the first as "primary_country"
df["primary_country"] = df["country"].apply(lambda x: x.split(",")[0].strip())

# rating: fill any missing with the most common rating
if df["rating"].isna().any():
    df["rating"] = df["rating"].fillna(df["rating"].mode()[0])

# Step 4: Transform categorical and date-related columns
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month_name()

# Split "duration" into a numeric value + unit (min for movies, seasons for TV)
def split_duration(row):
    dur = str(row["duration"])
    parts = dur.split(" ")
    try:
        value = int(parts[0])
    except (ValueError, IndexError):
        value = np.nan
    return value

df["duration_value"] = df.apply(split_duration, axis=1)
df["duration_unit"] = np.where(df["type"] == "Movie", "minutes", "seasons")

# listed_in: split the comma-separated genre string into a clean list column
df["genres"] = df["listed_in"].apply(lambda x: [g.strip() for g in str(x).split(",")])
df["primary_genre"] = df["genres"].apply(lambda g: g[0] if g else "Not Specified")

# type / rating as proper categorical dtypes
df["type"] = df["type"].astype("category")
df["rating"] = df["rating"].astype("category")

# Step 5: Create a clean dataset for further analysis
final_cols = [
    "show_id", "type", "title", "director", "country", "primary_country",
    "date_added", "year_added", "month_added", "release_year", "rating",
    "duration", "duration_value", "duration_unit", "listed_in",
    "genres", "primary_genre",
]
df_clean = df[final_cols].reset_index(drop=True)

log("\n--- Missing values after cleaning ---")
log(df_clean.isna().sum().to_string())

log(f"\nFinal cleaned dataset shape: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
log(f"Rows removed (duplicates): {8790 - df_clean.shape[0] + df_clean['date_added'].isna().sum()*0}")

df_clean.to_csv(CLEAN_PATH, index=False)

with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nSaved cleaned dataset -> {CLEAN_PATH}")
print(f"Saved report -> {REPORT_PATH}")
