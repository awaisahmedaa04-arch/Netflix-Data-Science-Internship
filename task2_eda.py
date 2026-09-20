"""
Task 2 (Easy) - Exploratory Data Analysis (EDA)
Explores the cleaned Netflix dataset to uncover trends, patterns, and insights.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import ast

sns.set_theme(style="whitegrid")

df = pd.read_csv("outputs/netflix_cleaned.csv")
df["genres"] = df["genres"].apply(ast.literal_eval)

report = []


def log(line=""):
    print(line)
    report.append(line)


log("TASK 2 - EXPLORATORY DATA ANALYSIS")
log("=" * 50)

# Step 1: Analyze dataset structure and statistics
log(f"Dataset: {df.shape[0]} titles, {df.shape[1]} columns")
log(f"Release years span: {df['release_year'].min()} - {df['release_year'].max()}")

# Step 2: Study content distribution by type
type_counts = df["type"].value_counts()
log("\nContent type distribution:")
log(type_counts.to_string())

fig, ax = plt.subplots(figsize=(6, 4))
type_counts.plot(kind="bar", color=["#E50914", "#221f1f"], ax=ax)
ax.set_title("Netflix Content: Movies vs TV Shows")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("figures/eda_type_distribution.png", dpi=150)
plt.close()

# Step 3: Identify top countries and categories
top_countries = df["primary_country"].value_counts().head(10)
log("\nTop 10 countries by content count:")
log(top_countries.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
top_countries.sort_values().plot(kind="barh", color="#E50914", ax=ax)
ax.set_title("Top 10 Countries by Netflix Content Count")
ax.set_xlabel("Number of Titles")
plt.tight_layout()
plt.savefig("figures/eda_top_countries.png", dpi=150)
plt.close()

all_genres = df.explode("genres")["genres"].str.strip()
top_genres = all_genres.value_counts().head(10)
log("\nTop 10 genres:")
log(top_genres.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
top_genres.sort_values().plot(kind="barh", color="#221f1f", ax=ax)
ax.set_title("Top 10 Genres on Netflix")
ax.set_xlabel("Number of Titles")
plt.tight_layout()
plt.savefig("figures/eda_top_genres.png", dpi=150)
plt.close()

# Step 4: Create visualizations for key metrics (content growth over time)
content_by_year = df.groupby("year_added").size()
log("\nContent added per year (top of range):")
log(content_by_year.tail(8).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
content_by_year.plot(kind="line", marker="o", color="#E50914", ax=ax)
ax.set_title("Netflix Content Added to Platform per Year")
ax.set_xlabel("Year Added")
ax.set_ylabel("Number of Titles Added")
plt.tight_layout()
plt.savefig("figures/eda_content_by_year.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, y="rating", order=df["rating"].value_counts().index, color="#E50914", ax=ax)
ax.set_title("Content Distribution by Rating")
plt.tight_layout()
plt.savefig("figures/eda_ratings.png", dpi=150)
plt.close()

# Step 5: Summarize findings
log("\n--- KEY FINDINGS ---")
log(f"1. Movies dominate the catalog ({type_counts.get('Movie', 0)} vs "
    f"{type_counts.get('TV Show', 0)} TV Shows).")
log(f"2. '{top_countries.index[0]}' produces the most content "
    f"({top_countries.iloc[0]} titles), followed by "
    f"'{top_countries.index[1]}' and '{top_countries.index[2]}'.")
log(f"3. '{top_genres.index[0]}' is the most common genre "
    f"({top_genres.iloc[0]} titles).")
log(f"4. Content additions peaked around {content_by_year.idxmax()} "
    f"with {content_by_year.max()} titles added that year.")
log(f"5. Most common content rating is '{df['rating'].mode()[0]}'.")

with open("outputs/task2_eda_summary.txt", "w") as f:
    f.write("\n".join(report))

print("\nSaved 5 charts to figures/ and summary to outputs/task2_eda_summary.txt")
