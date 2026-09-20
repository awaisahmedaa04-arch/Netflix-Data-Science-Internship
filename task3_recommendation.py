"""
Task 3 (Medium) - Recommendation System Analysis
Builds a basic content-based recommendation model using Netflix titles
and categories (genre similarity).
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("outputs/netflix_cleaned.csv").reset_index(drop=True)

report = []


def log(line=""):
    print(line)
    report.append(line)


log("TASK 3 - RECOMMENDATION SYSTEM ANALYSIS")
log("=" * 50)

# Step 1: Extract relevant content features (genres + type + rating as "content profile")
df["content_profile"] = (
    df["listed_in"].fillna("") + " " + df["type"].fillna("") + " " + df["rating"].fillna("")
)

# Step 2: Perform text preprocessing (TF-IDF handles lowercasing/tokenizing)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["content_profile"])
log(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Step 3: Calculate content similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

title_to_index = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()


def recommend(title, n=5):
    key = title.lower()
    if key not in title_to_index:
        return None
    idx = title_to_index[key]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]
    rec_indices = [i for i, _ in sim_scores]
    scores = [round(s, 3) for _, s in sim_scores]
    result = df.loc[rec_indices, ["title", "type", "listed_in"]].copy()
    result["similarity_score"] = scores
    return result


# Step 4: Generate recommendations for a few selected titles
sample_titles = ["Dick Johnson Is Dead", "Midnight Mass", "Ganglands"]
all_recs = {}
for t in sample_titles:
    recs = recommend(t, n=5)
    all_recs[t] = recs
    log(f"\nBecause you watched '{t}', you might also like:")
    if recs is not None:
        log(recs.to_string(index=False))
    else:
        log("  (title not found)")

# Step 5: Evaluate recommendation quality (genre overlap as a proxy metric)
log("\n--- Evaluation: genre overlap of recommendations vs seed title ---")
for t, recs in all_recs.items():
    if recs is None:
        continue
    seed_genres = set(g.strip() for g in df.loc[title_to_index[t.lower()], "listed_in"].split(","))
    overlaps = []
    for _, row in recs.iterrows():
        rec_genres = set(g.strip() for g in row["listed_in"].split(","))
        overlap = len(seed_genres & rec_genres) / len(seed_genres | rec_genres)
        overlaps.append(overlap)
    avg_overlap = sum(overlaps) / len(overlaps)
    log(f"'{t}': average genre-overlap (Jaccard) with its top-5 recommendations = {avg_overlap:.2f}")

with open("outputs/task3_recommendation_summary.txt", "w") as f:
    f.write("\n".join(report))

# save a reusable function usage example + all recs to CSV
combined = []
for t, recs in all_recs.items():
    if recs is not None:
        r = recs.copy()
        r.insert(0, "seed_title", t)
        combined.append(r)
pd.concat(combined).to_csv("outputs/task3_sample_recommendations.csv", index=False)

print("\nSaved outputs/task3_recommendation_summary.txt and task3_sample_recommendations.csv")
