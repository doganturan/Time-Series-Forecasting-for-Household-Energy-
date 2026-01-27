import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

print("=" * 60)
print("Day-2: Train ML Models for Energy Consumption Forecasting")
print("=" * 60)

# ============================================================================
# 1. Load data
# ============================================================================
print("\n1. Loading hourly_features.csv...")
data_path = os.path.join(project_root, 'data', 'processed', 'hourly_features.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df = df.sort_index()

print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================================
# 2. Prepare X and y
# ============================================================================
print("\n" + "=" * 60)
print("2. Preparing features and target...")

y = df['Global_active_power'].copy()
X = df.drop('Global_active_power', axis=1).copy()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nFeatures: {list(X.columns)}")

# ============================================================================
# 3. Train/Test split (chronological)
# ============================================================================
print("\n" + "=" * 60)
print("3. Train/Test split (80/20, chronological)...")

split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

train_end_date = X_train.index[-1]
test_start_date = X_test.index[0]

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
print(f"Train end: {train_end_date}")
print(f"Test start: {test_start_date}")

# ============================================================================
# 4. Helper function for metrics
# ============================================================================
def calculate_metrics(y_true, y_pred, dataset_name=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    if dataset_name:
        print(f"\n{dataset_name} Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# ============================================================================
# 5. Model A: Linear Regression
# ============================================================================
print("\n" + "=" * 60)
print("5a. Training Linear Regression...")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

print("\nLinear Regression Results:")
lr_train_metrics = calculate_metrics(y_train, lr_train_pred, "Train")
lr_test_metrics = calculate_metrics(y_test, lr_test_pred, "Test")

# ============================================================================
# 6. Model B: Random Forest Regressor
# ============================================================================
print("\n" + "=" * 60)
print("5b. Training Random Forest Regressor...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_split=10,
    min_samples_leaf=5,   
    max_depth=15
)
rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

print("\nRandom Forest Results:")
rf_train_metrics = calculate_metrics(y_train, rf_train_pred, "Train")
rf_test_metrics = calculate_metrics(y_test, rf_test_pred, "Test")

# Feature Importance Analysis
print("\nFeature Importance Analysis:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))

# ============================================================================
# 7. Model C: Support Vector Regressor (with scaling)
# ============================================================================
print("\n" + "=" * 60)
print("5c. Training SVR (with StandardScaler)...")

# Scale features for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr_model = SVR(kernel="rbf")
svr_model.fit(X_train_scaled, y_train)

svr_train_pred = svr_model.predict(X_train_scaled)
svr_test_pred = svr_model.predict(X_test_scaled)

print("\nSVR Results:")
svr_train_metrics = calculate_metrics(y_train, svr_train_pred, "Train")
svr_test_metrics = calculate_metrics(y_test, svr_test_pred, "Test")

# ============================================================================
# 8. Compile results
# ============================================================================
print("\n" + "=" * 60)
print("6. Compiling model results...")

results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'SVR'],
    'Train_MAE': [lr_train_metrics['MAE'], rf_train_metrics['MAE'], svr_train_metrics['MAE']],
    'Train_RMSE': [lr_train_metrics['RMSE'], rf_train_metrics['RMSE'], svr_train_metrics['RMSE']],
    'Train_R2': [lr_train_metrics['R2'], rf_train_metrics['R2'], svr_train_metrics['R2']],
    'Test_MAE': [lr_test_metrics['MAE'], rf_test_metrics['MAE'], svr_test_metrics['MAE']],
    'Test_RMSE': [lr_test_metrics['RMSE'], rf_test_metrics['RMSE'], svr_test_metrics['RMSE']],
    'Test_R2': [lr_test_metrics['R2'], rf_test_metrics['R2'], svr_test_metrics['R2']]
})

print("\nModel Comparison:")
print(results_df.to_string(index=False))

results_path = os.path.join(project_root, 'data', 'processed', 'model_results_day2.csv')
results_df.to_csv(results_path, index=False)
print(f"\n✓ Results saved to: {results_path}")

# ============================================================================
# 9. Select best model
# ============================================================================
print("\n" + "=" * 60)
print("7. Selecting best model (lowest Test RMSE)...")

best_idx = results_df['Test_RMSE'].idxmin()
best_model_name = results_df.loc[best_idx, 'Model']
best_test_rmse = results_df.loc[best_idx, 'Test_RMSE']

print(f"\n✓ Best Model: {best_model_name}")
print(f"  Test RMSE: {best_test_rmse:.4f}")

# Get best model predictions
if best_model_name == 'Linear Regression':
    best_test_pred = lr_test_pred
elif best_model_name == 'Random Forest':
    best_test_pred = rf_test_pred
else:  # SVR
    best_test_pred = svr_test_pred

# Save best model predictions
pred_df = pd.DataFrame({
    'datetime': y_test.index,
    'y_true': y_test.values,
    'y_pred': best_test_pred
})
pred_path = os.path.join(project_root, 'data', 'processed', 'test_predictions_best_model.csv')
pred_df.to_csv(pred_path, index=False)
print(f"✓ Best model predictions saved to: {pred_path}")

# ============================================================================
# 10. Create results directory if needed
# ============================================================================
results_dir = os.path.join(project_root, 'figures', 'results')
os.makedirs(results_dir, exist_ok=True)

# ============================================================================
# 11. Plot 1: Predicted vs Actual (last 7 days)
# ============================================================================
print("\n" + "=" * 60)
print("8. Creating visualizations...")

print("\nPlot 1: Predicted vs Actual (last 7 days)...")

# Last 7 days = 168 hours
last_168_idx = -168 if len(y_test) >= 168 else 0

y_test_last = y_test.iloc[last_168_idx:]
pred_last = best_test_pred[last_168_idx:]

fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(y_test_last.index, y_test_last.values, label='Actual', linewidth=2, alpha=0.8)
ax1.plot(y_test_last.index, pred_last, label='Predicted', linewidth=2, alpha=0.8)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Global Active Power (kW)', fontsize=12)
ax1.set_title(f'Predicted vs Actual: Last 7 Days (Best Model: {best_model_name})', 
              fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()

plot1_path = os.path.join(results_dir, 'pred_vs_actual_line.png')
fig1.savefig(plot1_path, dpi=200)
plt.close(fig1)
print(f"✓ Saved: pred_vs_actual_line.png")

# ============================================================================
# 12. Plot 2: Residuals histogram
# ============================================================================
print("\nPlot 2: Residuals histogram...")

residuals = y_test.values - best_test_pred

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title(f'Residuals Distribution (Best Model: {best_model_name})', 
              fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero line')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()
fig2.tight_layout()

plot2_path = os.path.join(results_dir, 'residuals_hist.png')
fig2.savefig(plot2_path, dpi=200)
plt.close(fig2)
print(f"✓ Saved: residuals_hist.png")

# ============================================================================
# 12.5 Plot 3: Feature Importance (for Random Forest)
# ============================================================================
if best_model_name == 'Random Forest':
    print("\nPlot 3: Feature Importance...")
    
    feature_imp_sorted = feature_importance.sort_values('Importance', ascending=True)
    
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.barh(feature_imp_sorted['Feature'], feature_imp_sorted['Importance'], 
             color='#4ECDC4', edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Importance Score', fontsize=12)
    ax3.set_ylabel('Feature', fontsize=12)
    ax3.set_title('Random Forest: Feature Importance Ranking', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (feature, importance) in enumerate(zip(feature_imp_sorted['Feature'], feature_imp_sorted['Importance'])):
        ax3.text(importance + 0.005, i, f'{importance:.3f}', 
                va='center', fontsize=9, fontweight='bold')
    
    fig3.tight_layout()
    plot3_path = os.path.join(results_dir, 'feature_importance.png')
    fig3.savefig(plot3_path, dpi=200)
    plt.close(fig3)
    print(f"✓ Saved: feature_importance.png")

# ============================================================================
# 13. Done
# ============================================================================
print("\n" + "=" * 60)
print("✓ Day-2 completed successfully!")
print("=" * 60)

print("\n📊 Summary:")
print(f"  - Models trained: 3 (Linear Regression, Random Forest, SVR)")
print(f"  - Best model: {best_model_name} (Test RMSE: {best_test_rmse:.4f})")
print(f"  - Results saved: model_results_day2.csv")
print(f"  - Predictions saved: test_predictions_best_model.csv")
figure_count = 3 if best_model_name == 'Random Forest' else 2
figure_names = "pred_vs_actual_line.png, residuals_hist.png, feature_importance.png" if best_model_name == 'Random Forest' else "pred_vs_actual_line.png, residuals_hist.png"
print(f"  - Figures created: {figure_count} ({figure_names})")


# ============================================================================
# Bu script ne yapıyor? (6 madde)
# ============================================================================
# 
# 1) hourly_features.csv dosyasını okur ve datetime index'i geri yükler
# 
# 2) Train/test split yapar (80/20, kronolojik - shuffle yok)
# 
# 3) 3 farklı model eğitir:
#    - Linear Regression (default parametreler)
#    - Random Forest (200 ağaç)
#    - SVR (RBF kernel, StandardScaler ile)
# 
# 4) Her model için MAE, RMSE, R2 metriklerini hesaplar ve karşılaştırır
# 
# 5) En iyi modeli seçer (Test RMSE en düşük olan) ve tahminlerini kaydeder
# 
# 6) 2 görselleştirme üretir:
#    - Son 7 günün actual vs predicted grafiği
#    - Residuals histogramı
