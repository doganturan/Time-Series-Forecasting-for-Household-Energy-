1) Kod tarafında (Gün 1’de yapılanlar)
A) Dosya ve yol yönetimi

Projenin kök klasörünü (project_root) dinamik buluyorsun.

Böylece scripti nereden çalıştırırsan çalıştır, şu yollar hep doğru:

data/raw/household_power_consumption.csv

data/processed/*.csv

figures/eda/*.png

B) Ham veriyi okuma ve zaman indeksine çevirme

CSV okunuyor (pd.read_csv).

Date + Time birleştirilip datetime oluşturuluyor.

datetime index yapılıyor.

Date ve Time kolonları siliniyor.

Veri tarihe göre sıralanıyor.

Aynı timestamp birden fazla ise duplicate timestamp’ler kaldırılıyor.

C) Sayısal kolonları düzeltme

Aşağıdaki kolonları numeric’e çeviriyorsun (errors='coerce'):

Global_active_power

Global_reactive_power

Voltage

Global_intensity

Sub_metering_1/2/3

Eğer veri içinde “?” gibi sayıya çevrilemeyen bir değer varsa NaN’a dönüşüyor (doğru yaklaşım).

D) Missing values analizi (resample öncesi)

Ham dakika verisinde her kolon için NaN sayısı hesaplanıyor.

Toplam missing ve yüzde missing yazdırılıyor.

Bu “before resample” değerleri rapora girecek şekilde saklanıyor.

E) Dakikalıktan saatliğe çevirme (resampling)

resample('h').mean() ile:

1 saat içindeki 60 ölçümün ortalaması alınıyor.

Böylece veri boyutu küçülüyor ve ML için daha yönetilebilir oluyor.

F) Missing values doldurma (limitli)

Saatlik verideki eksikler:

interpolate(method='time', limit=24)
→ en fazla 24 saatlik boşluğu lineer doldurur

ffill(limit=3).bfill(limit=3)
→ küçük boşluklar için son düzeltme

Sonrasında kalan NaN sayısı tekrar kontrol ediliyor.

G) Missing report dosyası üretimi

data/processed/missing_report.csv dosyası oluşuyor.

İçinde 3 aşama var:

Before_Resample

After_Resample

After_Fill

H) Outlier analizi (sadece analiz)

Hedef seri: Global_active_power

IQR yöntemiyle:

Q1, Q3, IQR

Lower/Upper bound

Outlier sayısı ve yüzdesi

Outlier kayıtları:

data/processed/outliers_iqr.csv

Ek olarak görsel:

figures/eda/outlier_boxplot.png (1–99 percentile zoom)

I) Time features ekleme (EDA için + ML’de numeric olarak kullanmak için)

Numeric zaman feature’ları:

Year, Month, Day, Hour, DayOfWeek

String zaman feature’ları (sadece EDA için):

MonthName, DayName
(Bunlar model dataset’ine girmiyor.)

J) Feature Engineering (ML için)

Target: Global_active_power

Lag:

lag_24 (sadece 24 saat)

Rolling (leakage-free):

rolling_mean_24

rolling_std_24

rolling_min_24

rolling_max_24

rolling_mean_168

rolling_std_168

Leakage-free olmasının sebebi:

rolling hesaplamadan önce y.shift(1) yapıyorsun (çok doğru).

K) ML dataset oluşturma (temiz, sadece gerekli kolonlar)

required_cols listesi var (target + feature’lar).

df_ml = df_features[required_cols].dropna().copy() ile:

Sadece gerekli kolonlar seçiliyor

NaN’lı satırlar atılıyor

Sonuç: modellemeye hazır tablo.

L) Dataset kayıtları

data/processed/processed_hourly.csv
→ EDA için (stringler de olabilir)

data/processed/hourly_features.csv
→ Model için (sadece numeric + lag_24 + rolling + time numeric)

2) Data tarafında (Gün 1 çıktısı olarak elinde ne var?)
A) Veri dönüşümü

Başlangıç: dakika seviyesinde (çok satır)

Sonuç: saatlik seviyede (çok daha az satır)

Bu sayede modelleme daha hızlı ve daha stabil olur.

B) Missing values süreci

Ham veride missing’ler vardı (CSV’den numeric dönüşümde “?” → NaN olabilir).

Saatlik resample sonrası bazı saatlerde veri eksik olabiliyor.

Sen bunu:

zaman interpolasyonu + limit

ffill/bfill ile
yönetmiş oldun.

Üç aşamalı raporun var: missing_report.csv

C) Modellemeye hazır tablo

hourly_features.csv içinde artık şunlar var:

Target:

Global_active_power

Time features:

Year, Month, Day, Hour, DayOfWeek

Lag:

lag_24

Rolling:

rolling_mean_24, rolling_std_24, rolling_min_24, rolling_max_24

rolling_mean_168, rolling_std_168

NaN yok (required_cols içinde)

Bu dosya Gün-2’de direkt train/test split’e hazır.

3) EDA tarafında (Gün 1’de ürettiğin analizler)

Elindeki EDA görselleri şunları cevaplıyor:

1) Genel seri (trend + dönemsel yapı)

timeseries_overview.png

Tüm zaman aralığında tüketim nasıl değişmiş?

Uzun dönem trend/seasonality/anomali hissi verir.

2) Yakınlaştırılmış seri (kısa dönem pattern)

timeseries_zoom_1month.png

Günlük/saatlik dalgalanma daha net görünür.

3) Aylık trend

monthly_average_trend.png

Hangi aylarda ortalama tüketim daha yüksek/düşük?

4) Aylara göre dağılım (volatilite + outlier hissi)

monthly_distribution_boxplot.png

Ay bazında dağılım geniş mi?

Bazı aylarda outlier/volatilite artıyor mu?

5) Haftanın günlerine göre tüketim

dayofweek_average_bar.png

Hafta içi/hafta sonu farkı var mı?

6) Ayın günlerine göre pattern

dayofmonth_average_line.png

Ay içinde bazı günlerde artış var mı? (maaş dönemi vb. gibi yorumlanabilir)

Bu grafik bazen “ayın kaç çektiği” etkisi nedeniyle yanıltıcı olabilir; raporda 1 cümle not düşebilirsin.

7) Günün saatine göre dağılım

hour_of_day_boxplot.png

Hangi saatlerde tüketim daha yüksek?

Saatlik dağılım geniş mi?

8) Outlier için özel boxplot

outlier_boxplot.png

Genel dağılımı daha okunaklı göstermek için 1–99 aralığına zoom yapıyor.

Outlier’lar ayrıca CSV’de listeleniyor.

Outlier CSV

outliers_iqr.csv

Outlier kabul edilen timestamp’ler ve değerleri var.

Raporun appendix’ine koymak veya 3-5 örneği Results/EDA kısmında anmak için ideal.

Gün 1’in “somut deliverable” listesi (elindeki ürünler)

✅ data/processed/missing_report.csv

✅ data/processed/outliers_iqr.csv

✅ data/processed/processed_hourly.csv

✅ data/processed/hourly_features.csv

✅ figures/eda/*.png (7–8 grafik)

✅ Gün-2 için hazır feature set + temiz dataset