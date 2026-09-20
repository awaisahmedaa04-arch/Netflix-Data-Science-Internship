"""
Task 6 (Advanced) - Data Science Business Insights Dashboard
Combines analytics, ML, and business recommendations from Tasks 1-5
into a single presentable HTML report.
"""
import pandas as pd
import base64

df = pd.read_csv("outputs/netflix_cleaned.csv")


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


type_counts = df["type"].value_counts()
top_country = df["primary_country"].value_counts().index[0]
top_genre = df.explode(df["genres"].apply(eval).name if False else "genres")  # placeholder, unused

figs = {
    "type": img_b64("figures/eda_type_distribution.png"),
    "countries": img_b64("figures/eda_top_countries.png"),
    "genres": img_b64("figures/eda_top_genres.png"),
    "by_year": img_b64("figures/eda_content_by_year.png"),
    "ratings": img_b64("figures/eda_ratings.png"),
    "forecast": img_b64("figures/task4_trend_forecast.png"),
    "confusion": img_b64("figures/task5_confusion_matrix.png"),
}

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Netflix Data Science Insights Dashboard | Auspify Internship</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #141414; color: #eee; margin: 0; padding: 0; }}
  header {{ background: #000; padding: 30px 40px; border-bottom: 4px solid #E50914; }}
  header h1 {{ color: #E50914; margin: 0; font-size: 28px; }}
  header p {{ color: #aaa; margin: 6px 0 0; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 30px 40px; }}
  .kpi-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 30px; }}
  .kpi {{ background: #1f1f1f; border-radius: 10px; padding: 20px; flex: 1; min-width: 180px; text-align: center; border-top: 3px solid #E50914; }}
  .kpi .num {{ font-size: 28px; font-weight: bold; color: #fff; }}
  .kpi .label {{ color: #aaa; font-size: 13px; margin-top: 6px; }}
  section {{ margin-bottom: 40px; }}
  h2 {{ border-left: 4px solid #E50914; padding-left: 12px; color: #fff; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .chart-card {{ background: #1f1f1f; border-radius: 10px; padding: 15px; }}
  .chart-card img {{ width: 100%; border-radius: 6px; }}
  .insights {{ background: #1f1f1f; border-radius: 10px; padding: 20px 25px; }}
  .insights li {{ margin-bottom: 10px; line-height: 1.5; }}
  footer {{ text-align: center; padding: 20px; color: #666; font-size: 13px; }}
</style>
</head>
<body>
<header>
  <h1>Netflix Content &amp; Business Insights Dashboard</h1>
  <p>Data Science Internship &middot; Auspify Technologies &middot; Prepared by Awais Ahmed</p>
</header>
<div class="container">

  <div class="kpi-row">
    <div class="kpi"><div class="num">{len(df):,}</div><div class="label">Total Titles Analyzed</div></div>
    <div class="kpi"><div class="num">{type_counts.get('Movie', 0):,}</div><div class="label">Movies</div></div>
    <div class="kpi"><div class="num">{type_counts.get('TV Show', 0):,}</div><div class="label">TV Shows</div></div>
    <div class="kpi"><div class="num">{top_country}</div><div class="label">Top Content Origin</div></div>
  </div>

  <section>
    <h2>1. Content Overview (Tasks 1-2)</h2>
    <div class="chart-grid">
      <div class="chart-card"><img src="data:image/png;base64,{figs['type']}"></div>
      <div class="chart-card"><img src="data:image/png;base64,{figs['by_year']}"></div>
      <div class="chart-card"><img src="data:image/png;base64,{figs['countries']}"></div>
      <div class="chart-card"><img src="data:image/png;base64,{figs['genres']}"></div>
    </div>
  </section>

  <section>
    <h2>2. Growth Forecast (Task 4)</h2>
    <div class="chart-card"><img src="data:image/png;base64,{figs['forecast']}"></div>
  </section>

  <section>
    <h2>3. Content Type Classification Model (Task 5)</h2>
    <div class="chart-card" style="max-width:450px;"><img src="data:image/png;base64,{figs['confusion']}"></div>
  </section>

  <section>
    <h2>4. Business Recommendations</h2>
    <div class="insights">
      <ul>
        <li><b>Double down on international originals:</b> International Movies and TV Shows are the single largest genre group, and India, the UK and South Korea already rank in the top content-producing countries alongside the US &mdash; continued investment in regional-language originals lines up with where the catalog is already strongest.</li>
        <li><b>TV Shows are under-represented relative to viewer demand trends:</b> Movies outnumber TV Shows more than 2:1, yet serialized content tends to drive longer engagement per subscriber &mdash; this gap is worth testing against retention data.</li>
        <li><b>Catalog growth is decelerating:</b> additions peaked in 2019 and slightly declined in 2020-2021; a simple linear model still projects continued growth, but that likely overstates reality &mdash; content acquisition budgets should plan for a plateau, not unchecked growth.</li>
        <li><b>Recommendation quality depends on genre tagging:</b> the content-based recommender performs well when genre tags are specific and consistent; investing in richer, more granular tagging (mood, theme, cast) would directly improve recommendation relevance.</li>
        <li><b>Be cautious with "obvious" ML features:</b> a naive classifier for Movie vs TV Show hit ~99.8% accuracy almost entirely because duration and genre labels implicitly reveal the answer &mdash; a reminder to sanity-check any model that looks "too good," especially before using it for real business decisions.</li>
      </ul>
    </div>
  </section>

</div>
<footer>Generated as part of the Auspify Technologies Data Science Internship &middot; Task 6</footer>
</body>
</html>
"""

with open("outputs/task6_business_dashboard.html", "w") as f:
    f.write(html)

print("Saved outputs/task6_business_dashboard.html")
