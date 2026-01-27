import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

# Project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# ============================================================================
# 1. Load data
# ============================================================================
data_path = os.path.join(project_root, 'data', 'raw', 'household_power_consumption.csv')
data = pd.read_csv(data_path)
print(f"Original shape: {data.shape}")
print("\nFirst 5 rows (original):")
print(data.head())

# ============================================================================
# 2. Create datetime index
# ============================================================================
print("\n" + "=" * 60)
print("2. Creating datetime index...")

data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
data = data.set_index('datetime')
data = data.drop(['Date', 'Time'], axis=1)
data = data.sort_index()

original_len = len(data)
data = data[~data.index.duplicated(keep='first')]
duplicates_removed = original_len - len(data)
if duplicates_removed > 0:
    print(f"⚠️  Removed {duplicates_removed} duplicate timestamps")

print(f"Date range: {data.index.min()} to {data.index.max()}")
print(f"Total days: {(data.index.max() - data.index.min()).days}")
print(f"Merged shape: {data.shape}")
print(data.head())

# ============================================================================
# 3. Convert to numeric
# ============================================================================
print("\n" + "=" * 60)
print("3. Converting all numeric columns to numeric type...")

numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
print(f"✓ Converted {len(numeric_columns)} columns to numeric")
print("\nDtypes (after conversion):")
print(data[numeric_columns].dtypes)

# ============================================================================
# 4. Missing values before resample
# ============================================================================
print("\n" + "=" * 60)
print("4. Missing values count:")
missing_counts = data.isnull().sum()
print(missing_counts)
print(f"\nTotal missing values: {missing_counts.sum()}")
print(f"Missing percentage: {(missing_counts.sum() / (data.shape[0] * data.shape[1]) * 100):.2f}%")

missing_before_resample = missing_counts.to_dict()

# ============================================================================
# 5. Hourly resampling
# ============================================================================
print("\n" + "=" * 60)
print("5. Resampling to hourly data...")

hourly_data = data.resample('h').mean()

print(f"Hourly data shape: {hourly_data.shape}")
print(f"Original rows: {len(data)}, Hourly rows: {len(hourly_data)}")

print("\nFirst 5 rows (hourly):")
print(hourly_data.head())

# ============================================================================
# 6. Fill missing values (with limit)
# ============================================================================
print("\n" + "=" * 60)
print("6. Filling missing values...")

print("\nMissing values in hourly data:")
hourly_missing_counts = hourly_data.isnull().sum()
print(hourly_missing_counts)

missing_after_resample = hourly_missing_counts.to_dict()

hourly_filled = hourly_data.interpolate(method="time", limit=24)
hourly_filled = hourly_filled.ffill(limit=3).bfill(limit=3)

missing_after = hourly_filled.isna().sum().sum()
print(f"Missing cells AFTER fill (hourly): {missing_after}")

missing_after_fill = hourly_filled.isnull().sum().to_dict()

# Missing report
missing_report_df = pd.DataFrame({
    'Column': list(missing_before_resample.keys()),
    'Before_Resample': list(missing_before_resample.values()),
    'After_Resample': [missing_after_resample.get(col, 0) for col in missing_before_resample.keys()],
    'After_Fill': [missing_after_fill.get(col, 0) for col in missing_before_resample.keys()]
})
missing_report_path = os.path.join(project_root, 'data', 'processed', 'missing_report.csv')
missing_report_df.to_csv(missing_report_path, index=False)
print(f"\n✓ Missing values report saved to: {missing_report_path}")

# ============================================================================
# 7. Outlier analysis (IQR method)
# ============================================================================
print("\n" + "=" * 60)
print("7. Outlier analysis (IQR method)...")

series = hourly_filled['Global_active_power'].dropna()

Q1 = series.quantile(0.25)
Q3 = series.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_mask = (hourly_filled['Global_active_power'] < lower_bound) | (hourly_filled['Global_active_power'] > upper_bound)
outliers_df = hourly_filled[outliers_mask][['Global_active_power']].copy()
outliers_count = len(outliers_df)
outliers_pct = (outliers_count / len(series)) * 100

print(f"Outliers detected: {outliers_count} ({outliers_pct:.2f}%)")
print(f"IQR: {IQR:.3f}, Lower bound: {lower_bound:.3f}, Upper bound: {upper_bound:.3f}")

outliers_path = os.path.join(project_root, 'data', 'processed', 'outliers_iqr.csv')
outliers_df = outliers_df.reset_index()
outliers_df.to_csv(outliers_path, index=False)
print(f"✓ Outliers saved to: {outliers_path}")

# Outlier boxplot (zoom to 1-99 percentile)
print("Creating outlier boxplot (zoomed 1-99 percentile)...")
p01 = series.quantile(0.01)
p99 = series.quantile(0.99)

fig_outlier, ax_outlier = plt.subplots(figsize=(12, 6))
ax_outlier.boxplot(hourly_filled['Global_active_power'].dropna(), vert=True, patch_artist=True)
ax_outlier.set_ylim(p01, p99)
ax_outlier.set_ylabel('Global Active Power (kW)', fontsize=12)
ax_outlier.set_title('Outlier Analysis: Global Active Power (Zoomed to 1-99% Range)', fontsize=14, fontweight='bold')
ax_outlier.grid(True, alpha=0.3, axis='y')
outlier_boxplot_path = os.path.join(project_root, 'figures', 'eda', 'outlier_boxplot.png')
fig_outlier.tight_layout()
fig_outlier.savefig(outlier_boxplot_path, dpi=200)
plt.close(fig_outlier)
print(f"✓ Saved outlier boxplot: outlier_boxplot.png")

# ============================================================================
# 8. Time features
# ============================================================================
print("\n" + "=" * 60)
print("8. Creating time features...")

df = hourly_filled.copy()

df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day
df["Hour"] = df.index.hour
df["DayOfWeek"] = df.index.dayofweek

df["MonthName"] = df.index.month_name()
df["DayName"] = df.index.day_name()

print("✓ Time columns added.")
print("\nFirst 5 rows with time columns:")
print(df.head())

# ============================================================================
# 9. Lag and rolling features
# ============================================================================
print("\n" + "=" * 60)
print("9. Creating lag and rolling features for ML...")

df_features = df.copy()
y = df_features['Global_active_power']

# Lag feature
df_features['lag_24'] = y.shift(24)

# Rolling features (leakage-free)
y_shift = y.shift(1)

df_features['rolling_mean_24'] = y_shift.rolling(window=24, min_periods=24).mean()
df_features['rolling_std_24'] = y_shift.rolling(window=24, min_periods=24).std()
df_features['rolling_min_24'] = y_shift.rolling(window=24, min_periods=24).min()
df_features['rolling_max_24'] = y_shift.rolling(window=24, min_periods=24).max()

df_features['rolling_mean_168'] = y_shift.rolling(window=168, min_periods=168).mean()
df_features['rolling_std_168'] = y_shift.rolling(window=168, min_periods=168).std()

print(f"✓ Created 7 lag/rolling features (leakage-free)")

feature_cols = ['lag_24', 'rolling_mean_24', 'rolling_std_24', 
                'rolling_min_24', 'rolling_max_24', 'rolling_mean_168', 'rolling_std_168']
feature_missing = df_features[feature_cols].isnull().sum()
print(f"\nMissing values in engineered features:")
print(feature_missing)

# ============================================================================
# 10. Clean ML dataset
# ============================================================================
print("\n" + "=" * 60)
print("10. Cleaning NaN values for ML dataset...")

required_cols = [
    'Global_active_power',
    'Year', 'Month', 'Day', 'Hour', 'DayOfWeek',
    'lag_24',
    'rolling_mean_24', 'rolling_std_24', 'rolling_min_24', 'rolling_max_24',
    'rolling_mean_168', 'rolling_std_168'
]

print(f"Original shape: {df_features.shape}")

df_ml = df_features[required_cols].dropna().copy()

print(f"After dropping NaN: {df_ml.shape}")
print(f"Rows removed: {len(df_features) - len(df_ml)}")

nan_check = df_ml[required_cols].isnull().sum().sum()
print(f"\nNaN count in ML dataset: {nan_check}")
if nan_check == 0:
    print("✓ ML dataset is clean and ready for modeling!")

# ============================================================================
# 11. Save datasets
# ============================================================================
print("\n" + "=" * 60)
print("11. Saving processed data...")

output_path = os.path.join(project_root, 'data', 'processed', 'processed_hourly.csv')
df.to_csv(output_path)
print(f"✓ Saved basic processed data to: {output_path}")
print(f"  → Shape: {df.shape}")

features_path = os.path.join(project_root, 'data', 'processed', 'hourly_features.csv')
df_ml.to_csv(features_path)
print(f"✓ Saved ML-ready features to: {features_path}")
print(f"  → Shape: {df_ml.shape}")

# ============================================================================
# 12. EDA Plots
# ============================================================================
def save_plot(fig, filename: str):
    path = os.path.join(project_root, 'figures', 'eda', filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved figure: {os.path.basename(path)}")

print("\n" + "=" * 60)
print("12. Creating EDA plots...")

# Time-series overview
print("\nCreating time-series overview plot...")
fig_ts, ax_ts = plt.subplots(figsize=(16, 6))
ax_ts.plot(df.index, df['Global_active_power'], linewidth=0.5, alpha=0.7)
ax_ts.set_xlabel('Date', fontsize=12)
ax_ts.set_ylabel('Global Active Power (kW)', fontsize=12)
ax_ts.set_title('Time-Series Overview: Global Active Power (Full Dataset)', fontsize=14, fontweight='bold')
ax_ts.grid(True, alpha=0.3)
save_plot(fig_ts, 'timeseries_overview.png')

# Time-series zoom
print("Creating time-series zoom plot (1 month)...")
zoom_start = '2007-01-01'
zoom_end = '2007-02-01'
df_zoom = df.loc[zoom_start:zoom_end]

fig_zoom, ax_zoom = plt.subplots(figsize=(16, 6))
ax_zoom.plot(df_zoom.index, df_zoom['Global_active_power'], linewidth=1, color='#FF6B6B')
ax_zoom.set_xlabel('Date', fontsize=12)
ax_zoom.set_ylabel('Global Active Power (kW)', fontsize=12)
ax_zoom.set_title('Time-Series Zoom: January 2007 (Daily/Hourly Patterns Visible)', fontsize=14, fontweight='bold')
ax_zoom.grid(True, alpha=0.3)
save_plot(fig_zoom, 'timeseries_zoom_1month.png')

# Monthly average trend
print("\nCreating monthly average trend plot...")
monthly_avg = df.groupby('Month')['Global_active_power'].mean()

fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Average Global Active Power (kW)', fontsize=12)
ax1.set_title('Monthly Average Energy Consumption Trend', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
save_plot(fig1, 'monthly_average_trend.png')

# Monthly distribution boxplot
print("Creating monthly distribution boxplot...")
fig2, ax2 = plt.subplots(figsize=(14, 6))
df.boxplot(column='Global_active_power', by='Month', ax=ax2)
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Global Active Power (kW)', fontsize=12)
ax2.set_title('Monthly Distribution of Energy Consumption (Volatility Analysis)', fontsize=14, fontweight='bold')
plt.suptitle('')
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
save_plot(fig2, 'monthly_distribution_boxplot.png')

# Day-of-week average
print("Creating day-of-week average bar chart...")
dow_avg = df.groupby('DayOfWeek')['Global_active_power'].mean()

fig3, ax3 = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B' if i >= 5 else '#4ECDC4' for i in range(7)]
ax3.bar(dow_avg.index, dow_avg.values, color=colors, edgecolor='black', linewidth=1.2)
ax3.set_xlabel('Day of Week', fontsize=12)
ax3.set_ylabel('Average Global Active Power (kW)', fontsize=12)
ax3.set_title('Average Energy Consumption by Day of Week', fontsize=14, fontweight='bold')
ax3.set_xticks(range(7))
ax3.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax3.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(dow_avg.values):
    ax3.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

save_plot(fig3, 'dayofweek_average_bar.png')

# Day-of-month average
print("Creating day-of-month average line plot...")
dom_avg = df.groupby('Day')['Global_active_power'].mean()

fig4, ax4 = plt.subplots(figsize=(14, 6))
ax4.plot(dom_avg.index, dom_avg.values, marker='o', linewidth=2, markersize=6, color='#FF6B6B')
ax4.set_xlabel('Day of Month', fontsize=12)
ax4.set_ylabel('Average Global Active Power (kW)', fontsize=12)
ax4.set_title('Average Energy Consumption by Day of Month (Intra-month Pattern)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(1, 32))

ax4.axvspan(1, 5, alpha=0.1, color='green', label='Early month')
ax4.axvspan(15, 20, alpha=0.1, color='blue', label='Mid-month')
ax4.axvspan(25, 31, alpha=0.1, color='orange', label='End of month')
ax4.legend()

save_plot(fig4, 'dayofmonth_average_line.png')

# Hour-of-day pattern
print("Creating hour-of-day boxplot...")
fig5, ax5 = plt.subplots(figsize=(14, 6))
df.boxplot(column='Global_active_power', by='Hour', ax=ax5)
ax5.set_xlabel('Hour of Day', fontsize=12)
ax5.set_ylabel('Global Active Power (kW)', fontsize=12)
ax5.set_title('Hourly Distribution of Energy Consumption (0-23h Pattern)', fontsize=14, fontweight='bold')
plt.suptitle('')
ax5.set_xticklabels(range(24))
save_plot(fig5, 'hour_of_day_boxplot.png')

print("\n" + "=" * 60)
print("✓ All EDA plots created successfully!")
