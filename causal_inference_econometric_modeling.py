# ============================================================================
# CAUSAL INFERENCE & STATISTICAL MODELING WORKFLOW
# Python / Google Colab
# Focus: Difference-in-Differences (DiD) & Regression Discontinuity (RDD)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from rdrobust import rdrobust, rdplot

# Setting chart styling to look clean
sns.set_theme(style="whitegrid")

# ============================================================================
# 1. DIFFERENCE-IN-DIFFERENCES (DID)
# ============================================================================
# Load the hotel spending dataset from the project repository
DID_URL = 'https://raw.githubusercontent.com/dansacks/gb740/refs/heads/main/hotel_spend.csv'
did_df = pd.read_csv(DID_URL)

# Define city codes for our two target markets
TREAT_CBSA = 16980   # Chicago (The city that got the concert)
CONTROL_CBSA = 18140  # Columbus (The comparison city with no concert)

# Find the exact week the concert happened in Chicago
chicago_cw = did_df.loc[did_df['cbsa_code'] == TREAT_CBSA, 'concert_week_num'].iloc[0]

# Convert total spending into logs to keep the scale stable
did_df['log_spend'] = np.log(did_df['total_spend'])

# Pull the log spending for Chicago (before and after the concert)
treat_post = did_df.loc[(did_df['cbsa_code'] == TREAT_CBSA) & (did_df['week_num'] == chicago_cw), 'log_spend'].item()
treat_pre = did_df.loc[(did_df['cbsa_code'] == TREAT_CBSA) & (did_df['week_num'] == chicago_cw - 1), 'log_spend'].item()

# Pull the log spending for Columbus during those same exact weeks
control_post = did_df.loc[(did_df['cbsa_code'] == CONTROL_CBSA) & (did_df['week_num'] == chicago_cw), 'log_spend'].item()
control_pre = did_df.loc[(did_df['cbsa_code'] == CONTROL_CBSA) & (did_df['week_num'] == chicago_cw - 1), 'log_spend'].item()

# Calculate the direct difference-in-differences value by hand
did_estimate = (treat_post - treat_pre) - (control_post - control_pre)
print("--- Baseline Difference-in-Differences Summary ---")
print(f"Calculated Impact Estimate: {did_estimate:.4f}\n")

# Create flags to mark if a row is 'after the concert' and if it is 'Chicago'
did_df['post'] = (did_df['week_num'] == chicago_cw).astype(int)
did_df['treat'] = (did_df['cbsa_code'] == TREAT_CBSA).astype(int)

# Filter the data to focus only on these two specific cities
est_data = did_df.loc[(did_df['cbsa_code'] == TREAT_CBSA) | (did_df['cbsa_code'] == CONTROL_CBSA)]

# Run a standard regression model using our tracking flags to get statistical significance
did_formula = 'log_spend ~ post + treat + post:treat'
did_model = smf.ols(formula=did_formula, data=est_data).fit()

print("--- Regression Model Analysis Summary ---")
print(did_model.summary())

# Plot a line chart tracking both cities over time to compare their paths
plt.figure(figsize=(9, 5))
plt.plot(did_df.loc[did_df['cbsa_code'] == TREAT_CBSA, 'week_num'], did_df.loc[did_df['cbsa_code'] == TREAT_CBSA, 'log_spend'], label='Chicago (Concert City)', linewidth=2)
plt.plot(did_df.loc[did_df['cbsa_code'] == CONTROL_CBSA, 'week_num'], did_df.loc[did_df['cbsa_code'] == CONTROL_CBSA, 'log_spend'], label='Columbus (Control City)', linewidth=2)
plt.axvline(chicago_cw, color='grey', linestyle='--', label='Concert Week Cutoff')

plt.title('Tracking Hotel Spending Trends: Chicago vs Columbus', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Week Number', fontsize=11)
plt.ylabel('Log of Hotel Spending', fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()


# ============================================================================
# 2. REGRESSION DISCONTINUITY (RDD)
# ============================================================================
# Load the ride tipping dataset from the project repository
RDD_URL = 'https://raw.githubusercontent.com/dansacks/gb740/main/tips.csv'
rdd_df = pd.read_csv(RDD_URL)

# Calculate the tip percentage based on the base fare price
rdd_df['tip_pct'] = rdd_df['tip'] / rdd_df['fare']

# Center our fare numbers directly around the $15 cutoff point
rdd_df['margin'] = rdd_df['fare'] - 15

# Group by the fare distance from the cutoff and find the average tip percent
rdd_agg = rdd_df.groupby('margin')['tip_pct'].mean().reset_index()

# Run the formal RDD model to see if crossing the $15 mark creates a clean jump in tips
print("\n--- Regression Discontinuity Design (RDD) Summary ---")
print(rdrobust(y=rdd_agg['tip_pct'], x=rdd_agg['margin'], c=0))

# Group the fare margins into 1-dollar bins so the scatter plot looks clean
binsize = 1
rdd_agg['margin_r'] = (rdd_agg['margin'] // binsize) * binsize

# Limit our view to rides that are within 10 dollars of the cutoff line
bandwidth_limit = 10
rdd_narrow = rdd_agg[np.abs(rdd_agg['margin_r']) < bandwidth_limit]

# Plot the binned data points around the $15 line
plt.figure(figsize=(9, 5))
plt.scatter(rdd_narrow['margin_r'], rdd_narrow['tip_pct'], color='steelblue', alpha=0.7, label='Average Tip per Bin')
plt.axvline(0, color='crimson', linestyle='--', linewidth=1.5, label='$15 Fare Cutoff Line')

plt.xlabel('Fare Price Relative to $15 Cutoff', fontsize=11)
plt.ylabel('Average Tip (as % of Fare)', fontsize=11)
plt.title('Customer Tipping Behavior Near the Cutoff Line', fontsize=14, fontweight='bold', pad=15)
plt.legend()
plt.tight_layout()
plt.show()

# Separate the rows into two groups: those above the cutoff and those below it
data_above = rdd_narrow[rdd_narrow['margin_r'] >= 0]
data_below = rdd_narrow[rdd_narrow['margin_r'] < 0]

# Draw a trendline on the left side and a trendline on the right side to look for a break
plt.figure(figsize=(9, 5))
sns.regplot(x=data_below['margin_r'], y=data_below['tip_pct'], scatter=True, ci=None, line_kws={"color": "darkorange", "linewidth": 2.5}, label='Trend Below Cutoff')
sns.regplot(x=data_above['margin_r'], y=data_above['tip_pct'], scatter=True, ci=None, line_kws={"color": "seagreen", "linewidth": 2.5}, label='Trend Above Cutoff')
plt.axvline(0, color='crimson', linestyle='--', linewidth=1.5, label='$15 Fare Cutoff Line')

plt.xlabel('Fare Price Relative to $15 Cutoff', fontsize=11)
plt.ylabel('Average Tip (as % of Fare)', fontsize=11)
plt.title('Linear Trend Discontinuity at the Cutoff', fontsize=14, fontweight='bold', pad=15)
plt.legend()
plt.tight_layout()
plt.show()

# Build a distribution histogram to make sure data isn't being artificially manipulated at the cutoff
plt.figure(figsize=(9, 5))
plt.hist(rdd_df['margin'], range=[-15, 30], bins=61, color='lightgray', edgecolor='black', alpha=0.8)
plt.axvline(0, color='crimson', linestyle='--', linewidth=1.5, label='$15 Fare Cutoff Line')

plt.xlabel('Fare Price Relative to $15 Cutoff', fontsize=11)
plt.ylabel('Number of Rides Logged', fontsize=11)
plt.title('Ride Frequency Distribution: Cutoff Integrity Check', fontsize=14, fontweight='bold', pad=15)
plt.legend()
plt.tight_layout()
plt.show()
