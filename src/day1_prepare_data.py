import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

# ============================================================================
# ENERGY CONSUMPTION DATA PREPROCESSING PIPELINE
# ============================================================================
# Bu script şunları yapar:
# 1. Ham veriyi yükler (minute-level data)
# 2. Datetime index oluşturur ve sıralar
# 3. Sayısal kolonları düzenler ve missing values analizi yapar
# 4. Hourly resampling yapar (dakikalıktan saatliğe geçiş)
# 5. Missing values'ları interpolation ile doldurur
# 6. Time features ekler (Year, Month, Day, Hour, DayOfWeek)
# 7. EDA grafikleri oluşturur
# 8. Lag ve rolling features ekler (ML için)
# 9. Clean ML-ready dataset oluşturur
# ============================================================================

# ADIM 0: Project root path'i dinamik olarak bul
# Bu sayede scripti nereden çalıştırırsak çalıştıralım, dosya yolları doğru çalışır
current_dir = os.path.dirname(os.path.abspath(__file__))  # src klasörü
project_root = os.path.dirname(current_dir)  # MLF_Energy_Forecasting klasörü

# ============================================================================
# ADIM 1: HAM VERİYİ YÜKLE
# ============================================================================
# Ham veri: Dakika bazında elektrik tüketim verileri
# Her satır: 1 dakikalık ölçüm (Date, Time, ve 7 ölçüm kolonu)
data_path = os.path.join(project_root, 'data', 'raw', 'household_power_consumption.csv')
data = pd.read_csv(data_path)
print(f"Original shape: {data.shape}")
print("\nFirst 5 rows (original):")
print(data.head())

# ============================================================================
# ADIM 2: DATETIME INDEX OLUŞTUR VE DÜZENLE
# ============================================================================
# Neden önemli?
# - Date ve Time kolonlarını birleştirip tek bir datetime index yapmak
# - Time-series analizleri için index'in datetime olması gerekli
# - Resample, rolling, lag gibi işlemler datetime index ister

print("\n" + "=" * 60)
print("2. Creating datetime index...")

# Date (16/12/2006) ve Time (17:24:00) kolonlarını birleştir
data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

# Datetime'ı index yap (artık her satırın zamanı index olarak kullanılıyor)
data = data.set_index('datetime')

# Artık Date ve Time kolonlarına ihtiyacımız yok, sil
data = data.drop(['Date', 'Time'], axis=1)

# ÇOK ÖNEMLİ: Index'i sırala (time-series için kritik!)
# Eğer veri karışık sıradaysa, resample ve lag işlemleri hatalı çalışır
data = data.sort_index()

# ÇOK ÖNEMLİ: Duplicate (tekrarlayan) zaman damgalarını temizle
# Bazı time-series'lerde aynı timestamp 2+ kez görünebilir
# Biz her timestamp'in tek olmasını istiyoruz (ilkini tutuyoruz)
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
# ADIM 3: SAYISAL KOLONLARI DÜZENLE
# ============================================================================
# Neden gerekli?
# - CSV'den okunan bazı sayısal değerler "string" olarak gelebilir
# - Örnek: "1.234" bir string ise matematiksel işlem yapamazsın
# - pd.to_numeric() ile tüm sayısal kolonları float'a çeviriyoruz
# - errors='coerce' => eğer dönüştürülemezse (örn: "?"), NaN yap

print("\n" + "=" * 60)
print("3. Converting all numeric columns to numeric type...")

numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
print(f"✓ Converted {len(numeric_columns)} columns to numeric")
# Quick sanity check
print("\nDtypes (after conversion):")
print(data[numeric_columns].dtypes)

# ============================================================================
# ADIM 4: MISSING VALUES ANALİZİ (RESAMPLE ÖNCESI)
# ============================================================================
# Neden kontrol ediyoruz?
# - Missing values (NaN) modellerde hata yaratır
# - Resample sonrası NaN sayısı değişebilir
# - Bu adımda "before resample" durumunu kaydediyoruz

print("\n" + "=" * 60)
print("4. Missing values count:")
missing_counts = data.isnull().sum()
print(missing_counts)
print(f"\nTotal missing values: {missing_counts.sum()}")
print(f"Missing percentage: {(missing_counts.sum() / (data.shape[0] * data.shape[1]) * 100):.2f}%")

# Bu bilgiyi daha sonra CSV raporu için saklıyoruz
missing_before_resample = missing_counts.to_dict()

# ============================================================================
# ADIM 5: HOURLY RESAMPLING (DAKİKALIKTAN SAATLİĞE GEÇİŞ)
# ============================================================================
# Neden resample yapıyoruz?
# - Ham veri dakika bazında (her 1 dakika = 1 satır)
# - Bu çok fazla veri (60x daha fazla satır)
# - Makine öğrenmesi için saatlik veri yeterli ve daha hızlı
# - resample('h') => 'h' = hour (saatlik)
# - .mean() => her saat içindeki 60 dakikalık değerlerin ortalamasını al

print("\n" + "=" * 60)
print("5. Resampling to hourly data...")

# Dakikalık veriyi saatliğe çevir (her saatin ortalaması)
hourly_data = data.resample('h').mean()

print(f"Hourly data shape: {hourly_data.shape}")
print(f"Original rows: {len(data)}, Hourly rows: {len(hourly_data)}")

print("\nFirst 5 rows (hourly):")
print(hourly_data.head())

# ============================================================================
# ADIM 6: MISSING VALUES'LARI DOLDUR (INTERPOLATION)
# ============================================================================
# Neden dolduruyoruz?
# - Resample sonrası bazı saatlerde NaN olabilir (ölçüm yapılmamış)
# - Makine öğrenmesi algoritmaları NaN'ları sevmez
# - İki yöntem kullanıyoruz:

print("\nMissing values in hourly data:")
hourly_missing_counts = hourly_data.isnull().sum()
print(hourly_missing_counts)

# Bu bilgiyi rapor için saklıyoruz
missing_after_resample = hourly_missing_counts.to_dict()

# (A) Time-based interpolation:
# - Zaman bazlı doğrusal interpolasyon
# - Örnek: Saat 10'da NaN varsa, saat 9 ve 11'in ortalamasını kullan
# - Kısa boşluklar için çok iyi çalışır
hourly_filled = hourly_data.interpolate(method="time")

# (B) Forward/Backward fill (sınırlı):
# - ffill => önceki değeri kopyala (maksimum 3 satır)
# - bfill => sonraki değeri kopyala (maksimum 3 satır)
# - limit=3 sayesinde çok uzun boşlukları körü körüne doldurmuyor
hourly_filled = hourly_filled.ffill(limit=3).bfill(limit=3)

missing_after = hourly_filled.isna().sum().sum()
print(f"Missing cells AFTER fill (hourly): {missing_after}")

# Bu bilgiyi rapor için saklıyoruz
missing_after_fill = hourly_filled.isnull().sum().to_dict()

# ============================================================================
# MISSING VALUES RAPORU OLUŞTUR (CSV'ye kaydet)
# ============================================================================
# Bu rapor şunları gösterir:
# - Before resample: Ham veri durumu
# - After resample: Saatlik veri durumu
# - After fill: Interpolation sonrası durum
# Raporu rapora ekleyebilirsiniz!

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
# ADIM 7: TIME FEATURES OLUŞTUR (ZAMAN ÖZELLİKLERİ)
# ============================================================================
# Neden time features ekliyoruz?
# - Makine öğrenmesi modelleri "datetime" objesini anlayamaz
# - Ama "Month=12" veya "Hour=18" gibi sayısal değerleri anlar
# - Bu features'lar patternleri yakalamaya yardımcı olur:
#   * Month => Kış aylarında tüketim fazla olabilir
#   * DayOfWeek => Hafta sonu tüketim farklı olabilir
#   * Hour => Gece/gündüz pattern'i

df = hourly_filled.copy()

# Sayısal time features (MODEL İÇİN BUNLAR KULLANILIR)
df["Year"] = df.index.year           # Örnek: 2007
df["Month"] = df.index.month         # 1-12 (Ocak=1, Aralık=12)
df["Day"] = df.index.day             # 1-31 (ayın günü)
df["Hour"] = df.index.hour           # 0-23 (gece 00:00 = 0, öğlen 12:00 = 12)
df["DayOfWeek"] = df.index.dayofweek # 0-6 (Pazartesi=0, Pazar=6)

# String features (SADECE EDA İÇİN, MODEL İÇİN KULLANILMAZ!)
# Neden kullanılmaz? Çünkü modeller string kabul etmez, sadece sayı ister
df["MonthName"] = df.index.month_name()  # "January", "February", ...
df["DayName"] = df.index.day_name()      # "Monday", "Tuesday", ...

print("✓ Time columns added.")
print("⚠️  Note: MonthName/DayName are string features - exclude from model input!")
print("\nFirst 5 rows with time columns:")
print(df.head())

# ============================================================================
# ADIM 8: LAG VE ROLLING FEATURES (MAKİNE ÖĞRENMESİ İÇİN)
# ============================================================================
# Neden lag ve rolling features?
# - Time-series modellerinde "geçmiş" önemlidir
# - Lag features: Geçmişteki değerleri feature olarak kullan
#   * lag_1 = 1 saat önceki tüketim ne kadardı?
#   * lag_24 = 24 saat önceki (dün aynı saat) tüketim ne kadardı?
#   * lag_168 = 1 hafta önceki (geçen hafta aynı saat) tüketim ne kadardı?
# - Rolling features: Hareketli pencere istatistikleri
#   * rolling_mean_24 = Son 24 saatin ortalaması ne?
#   * rolling_std_24 = Son 24 saatin standart sapması ne? (volatilite)

print("\n" + "=" * 60)
print("8. Creating lag and rolling features for ML...")

# Kopyasını al (orijinal df'i bozmamak için)
df_features = df.copy()
y = df_features['Global_active_power']  # Hedef değişken (tahmin edeceğimiz)

# ----------------------------------------------------------------------------
# LAG FEATURES (GEÇMİŞTEKİ DEĞERLER)
# ----------------------------------------------------------------------------
# .shift(n) => n satır yukarı kaydır
# lag_24 = 24 saat önceki değer (dün aynı saat)
print("Creating lag features...")
df_features['lag_24'] = y.shift(24)    # 24 saat önce (dün aynı saat)

# ----------------------------------------------------------------------------
# ROLLING FEATURES (HAREKETLİ PENCERE İSTATİSTİKLERİ)
# ----------------------------------------------------------------------------
# ÇOK ÖNEMLİ: DATA LEAKAGE'ı ÖNLE!
# - Yanlış yöntem: .rolling(24).mean() => şu anki değeri de dahil eder (LEAK!)
# - Doğru yöntem: önce shift(1) yap, sonra rolling => gelecek bilgisi sızmaz

print("Creating rolling window features (leakage-free)...")
y_shift = y.shift(1)  # Önce 1 saat kaydır (şu anki değeri dahil etme!)

# 24 saatlik pencere (günlük pattern)
# min_periods=24 => ilk 24 saat için NaN olur (yeterli veri yok)
df_features['rolling_mean_24'] = y_shift.rolling(window=24, min_periods=24).mean()
df_features['rolling_std_24'] = y_shift.rolling(window=24, min_periods=24).std()
df_features['rolling_min_24'] = y_shift.rolling(window=24, min_periods=24).min()
df_features['rolling_max_24'] = y_shift.rolling(window=24, min_periods=24).max()

# 168 saatlik pencere (haftalık pattern)
df_features['rolling_mean_168'] = y_shift.rolling(window=168, min_periods=168).mean()
df_features['rolling_std_168'] = y_shift.rolling(window=168, min_periods=168).std()

print(f"✓ Created 7 lag/rolling features (leakage-free)")

# Missing values kontrolü (lag/rolling başlarda NaN yaratır, bu normal!)
feature_cols = ['lag_24', 'rolling_mean_24', 'rolling_std_24', 
                'rolling_min_24', 'rolling_max_24', 'rolling_mean_168', 'rolling_std_168']
feature_missing = df_features[feature_cols].isnull().sum()
print(f"\nMissing values in engineered features (expected at the beginning):")
print(feature_missing)

# ============================================================================
# ADIM 9: NaN TEMİZLİĞİ (ML-READY DATASET)
# ============================================================================
# Neden NaN temizliği?
# - Lag ve rolling features başlarda NaN yaratır (örn: lag_168 için ilk 168 saat)
# - Makine öğrenmesi algoritmaları NaN'lı satırları kullanamaz
# - Bu adımda: ML için gerekli kolonlarda NaN olan satırları atıyoruz
#
# Not: MonthName ve DayName'i dahil ETMİYORUZ çünkü:
# - Bunlar string (model string kabul etmez)
# - Zaten Month ve DayOfWeek sayısal versiyonları var

print("\n" + "=" * 60)
print("9. Cleaning NaN values for ML dataset...")

# Model için gerekli kolonlar (string features hariç)
required_cols = [
    'Global_active_power',  # Tahmin edeceğimiz hedef (target)
    'Year', 'Month', 'Day', 'Hour', 'DayOfWeek',  # Time features
    'lag_24',  # Lag features (24 saat önceki değer)
    'rolling_mean_24', 'rolling_std_24', 'rolling_min_24', 'rolling_max_24',  # 24h rolling
    'rolling_mean_168', 'rolling_std_168'  # 168h rolling
]

print(f"Original shape: {df_features.shape}")

# Bu kolonlarda NaN olan satırları at (dropna)
df_ml = df_features.dropna(subset=required_cols).copy()

print(f"After dropping NaN in required columns: {df_ml.shape}")
print(f"Rows removed: {len(df_features) - len(df_ml)} (due to lag/rolling initial NaN)")

# Son kontrol: Hiç NaN kaldı mı?
nan_check = df_ml[required_cols].isnull().sum().sum()
print(f"\nNaN count in ML dataset (required columns): {nan_check}")
if nan_check == 0:
    print("✓ ML dataset is clean and ready for modeling!")

# ============================================================================
# ADIM 10: DOSYALARI KAYDET
# ============================================================================
# İki farklı dosya kaydediyoruz:
# 1. processed_hourly.csv: Temel işlenmiş veri (lag/rolling YOK, EDA için)
# 2. hourly_features.csv: ML-ready veri (lag/rolling VAR, temiz, modelleme için)

print("\n" + "=" * 60)
print("10. Saving processed data...")

# DOSYA 1: Temel işlenmiş veri (EDA + genel analiz için)
output_path = os.path.join(project_root, 'data', 'processed', 'processed_hourly.csv')
df.to_csv(output_path)
print(f"✓ Saved basic processed data to: {output_path}")
print(f"  → Shape: {df.shape}")
print(f"  → Contains: Time features, no lag/rolling")

# DOSYA 2: ML-ready dataset (modelleme için)
# - Lag ve rolling features dahil
# - NaN'lar temizlenmiş
# - Direkt olarak train/test split'e hazır
features_path = os.path.join(project_root, 'data', 'processed', 'hourly_features.csv')
df_ml.to_csv(features_path)
print(f"✓ Saved ML-ready features to: {features_path}")
print(f"  → Features: lag_24, rolling_mean_24, rolling_std_24, rolling_min_24, rolling_max_24, etc.")
print(f"  → Shape: {df_ml.shape}")
print(f"  → No NaN in required columns - ready for train/test split!")

# 9 EDA Plots - Analysis of Global Active Power
def save_plot(fig, filename: str):
    """Save figure safely with explicit figure reference"""
    path = os.path.join(project_root, 'figures', 'eda', filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved figure: {os.path.basename(path)}")

# 7. EDA Plots - Analysis of Global Active Power
print("\n" + "=" * 60)
print("7. Creating EDA plots...")

# ----------------------------------------------------------------------------
# GRAFİK 1: TIME-SERIES OVERVIEW (TÜM SERİ)
# ----------------------------------------------------------------------------
# Ne gösterir? Tüm veri setinin zaman içindeki görünümü
# Neden önemli? Uzun dönemli trend, seasonality, anomaliler görülebilir
print("\nCreating time-series overview plot...")
fig_ts, ax_ts = plt.subplots(figsize=(16, 6))
ax_ts.plot(df.index, df['Global_active_power'], linewidth=0.5, alpha=0.7)
ax_ts.set_xlabel('Date', fontsize=12)
ax_ts.set_ylabel('Global Active Power (kW)', fontsize=12)
ax_ts.set_title('Time-Series Overview: Global Active Power (Full Dataset)', fontsize=14, fontweight='bold')
ax_ts.grid(True, alpha=0.3)
save_plot(fig_ts, 'timeseries_overview.png')

# ----------------------------------------------------------------------------
# GRAFİK 2: TIME-SERIES ZOOM (1 AYLIK DETAY)
# ----------------------------------------------------------------------------
# Ne gösterir? Kısa bir dönemin detaylı görünümü (Ocak 2007)
# Neden önemli? Günlük ve saatlik pattern'ler net görülebilir
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

# ----------------------------------------------------------------------------
# GRAFİK 3: AYLIK ORTALAMA TREND (MONTHLY AVERAGE)
# ----------------------------------------------------------------------------
# Ne gösterir? Her ayın ortalama tüketimi (1-12 arası)
# Neden önemli? Kış aylarında tüketim artışı vb. görülebilir
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

# 2. Monthly Distribution (Boxplot) - Shows volatility across months
print("Creating monthly distribution boxplot...")
fig2, ax2 = plt.subplots(figsize=(14, 6))
df.boxplot(column='Global_active_power', by='Month', ax=ax2)
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Global Active Power (kW)', fontsize=12)
ax2.set_title('Monthly Distribution of Energy Consumption (Volatility Analysis)', fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove default title
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
save_plot(fig2, 'monthly_distribution_boxplot.png')

# 3. Day-of-Week Average (Bar Chart) - Weekday/Weekend Pattern
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

# Add value labels on bars
for i, v in enumerate(dow_avg.values):
    ax3.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

save_plot(fig3, 'dayofweek_average_bar.png')

# 4. Day-of-Month Average (Line Plot) - Within-month pattern
print("Creating day-of-month average line plot...")
dom_avg = df.groupby('Day')['Global_active_power'].mean()

fig4, ax4 = plt.subplots(figsize=(14, 6))
ax4.plot(dom_avg.index, dom_avg.values, marker='o', linewidth=2, markersize=6, color='#FF6B6B')
ax4.set_xlabel('Day of Month', fontsize=12)
ax4.set_ylabel('Average Global Active Power (kW)', fontsize=12)
ax4.set_title('Average Energy Consumption by Day of Month (Intra-month Pattern)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(1, 32))

# Highlight interesting periods
ax4.axvspan(1, 5, alpha=0.1, color='green', label='Early month')
ax4.axvspan(15, 20, alpha=0.1, color='blue', label='Mid-month')
ax4.axvspan(25, 31, alpha=0.1, color='orange', label='End of month')
ax4.legend()

save_plot(fig4, 'dayofmonth_average_line.png')

# 5. Hour-of-Day Pattern (Boxplot) - Hourly consumption pattern
print("Creating hour-of-day boxplot...")
fig5, ax5 = plt.subplots(figsize=(14, 6))
df.boxplot(column='Global_active_power', by='Hour', ax=ax5)
ax5.set_xlabel('Hour of Day', fontsize=12)
ax5.set_ylabel('Global Active Power (kW)', fontsize=12)
ax5.set_title('Hourly Distribution of Energy Consumption (0-23h Pattern)', fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove default title
ax5.set_xticklabels(range(24))
save_plot(fig5, 'hour_of_day_boxplot.png')

print("\n" + "=" * 60)
print("✓ All EDA plots created successfully!")