Day-1 (hazır olanlar)

Dataset yükleme, datetime index, duplicate timestamp temizliği

Numeric cast (errors=coerce)

Hourly resample (resample('h').mean()) ve gerekçesi (computationally manageable + smoothing)

Missing handling: interpolate(time, limit=24) + ffill/bfill limit

Outlier analizi: IQR bounds + outlier sayısı/yüzdesi + CSV export + boxplot (1–99% zoom)

EDA figürleri (timeseries, monthly trend, monthly boxplot, DOW, DOM, hour-of-day)

Feature engineering (leakage-free): lag_24 + rolling stats (24 ve 168) + time features

Day-2 (hazır olanlar)

Chronological split (80/20)

3 model: Linear Regression, Random Forest, SVR(+StandardScaler)

Metrics: MAE, RMSE, R²

Best model seçimi: Random Forest (Test RMSE 0.6001) + test prediction CSV

2 sonuç plotu: predicted vs actual (last 7 days), residuals histogram

Raporda “Implementation” ve “Evaluation & Results” bölümleri bu çıktılarla çok rahat doldurulacak.