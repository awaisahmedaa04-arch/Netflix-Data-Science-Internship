"""
Task 4 (Medium) - Trend Prediction Analysis
Analyzes historical content-addition trends and forecasts future growth.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("outputs/netflix_cleaned.csv")

report = []


def log(line=""):
    print(line)
    report.append(line)


log("TASK 4 - TREND PREDICTION ANALYSIS")
log("=" * 50)

# Step 1: Prepare release-year data (use year_added = when content joined the platform)
yearly = df.groupby("year_added").size().reset_index(name="titles_added")
# Drop the first/last partial years which distort a simple trend (e.g. 2021 cutoff mid-year)
yearly_full = yearly[(yearly["year_added"] >= 2015) & (yearly["year_added"] <= 2020)]
log("\nYearly content additions used for modeling (2015-2020, excluding partial years):")
log(yearly_full.to_string(index=False))

# Step 2: Analyze yearly content trends
growth = yearly_full["titles_added"].pct_change().mean() * 100
log(f"\nAverage year-over-year growth rate: {growth:.1f}%")

# Step 3: Build forecasting model (simple linear regression on year -> titles_added)
X = yearly_full["year_added"].values.reshape(-1, 1)
y = yearly_full["titles_added"].values

model = LinearRegression()
model.fit(X, y)

y_pred_train = model.predict(X)
mae = mean_absolute_error(y, y_pred_train)
r2 = r2_score(y, y_pred_train)
log(f"\nLinear regression fit: R^2 = {r2:.2f}, MAE = {mae:.1f} titles/year")

future_years = np.array([2021, 2022, 2023]).reshape(-1, 1)
future_preds = model.predict(future_years)

log("\nForecast for future years:")
for yr, pred in zip(future_years.flatten(), future_preds):
    log(f"  {yr}: ~{max(0, round(pred))} titles predicted to be added")

# Step 4: Visualize future predictions
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(yearly_full["year_added"], yearly_full["titles_added"], "o-", color="#221f1f", label="Actual")
all_years = np.concatenate([X.flatten(), future_years.flatten()])
all_preds = model.predict(all_years.reshape(-1, 1))
ax.plot(all_years, all_preds, "--", color="#E50914", label="Linear trend / forecast")
ax.axvline(2020.5, color="gray", linestyle=":", alpha=0.6)
ax.set_title("Netflix Content Additions: Historical Trend & Forecast")
ax.set_xlabel("Year")
ax.set_ylabel("Titles Added")
ax.legend()
plt.tight_layout()
plt.savefig("figures/task4_trend_forecast.png", dpi=150)
plt.close()

# Step 5: Interpret results
log("\n--- INTERPRETATION ---")
log("The linear model captures the broad growth trend in content additions from")
log("2015-2020, but a simple straight-line fit likely overstates future growth,")
log("since real catalogs tend to plateau. The forecast should be read as an")
log("optimistic upper bound rather than a precise prediction.")

with open("outputs/task4_trend_summary.txt", "w") as f:
    f.write("\n".join(report))

print("\nSaved figures/task4_trend_forecast.png and outputs/task4_trend_summary.txt")
