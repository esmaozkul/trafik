import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Örnek veri setini kontrol edin
data = {
    'trafik_yogunlugu': ['hafif', 'normal', 'yogun', 'cok_yogun'],
    'kirmizi_isik': [60, 50, 40, 30],
    'sari_isik': [3, 3, 3, 3],
    'yesil_isik': [60, 70, 80, 90],
    
}
df = pd.DataFrame(data)

# Veri setinde yeterince değişkenlik olup olmadığını kontrol edin
print("Veri setinin dağılımı:")
print(df.describe())  # Sayısal sütunlar için özet istatistikler
print("Benzersiz değerler:")
print(df.nunique())  # Sütun başına benzersiz değer sayısı

# Trafik yoğunluğunu sayısal hale getirin
le = LabelEncoder()
df['trafik_yogunlugu_encoded'] = le.fit_transform(df['trafik_yogunlugu'])

# Bağımsız ve bağımlı değişkenleri belirleyin
X = df[['kirmizi_isik', 'sari_isik', 'yesil_isik', 'trafik_yogunlugu_encoded']]
y = df['yesil_isik']

# Eğitim ve test setlerini oluşturun
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli oluşturun ve eğitin
model = LinearRegression()
model.fit(X_train, y_train)

# Modelin performansını değerlendirin
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r_squared = model.score(X_test, y_test)
r_squared_alt = r2_score(y_test, y_pred)

print("Ortalama Kare Hatası (MSE):", mse)
print("R-kare:", r_squared)
print("R-kare (alternatif):", r_squared_alt)

# Modeli kullanarak yeni bir tahmin yapın
# Farklı bir tahmin yapmak için farklı bir veri sağlayın
yeni_veri = pd.DataFrame({
    'kirmizi_isik': [60],  # Farklı değerler kullanın
    'sari_isik': [3],
    'yesil_isik': [60],
    'trafik_yogunlugu_encoded': [3]  # 'normal' trafikten gelen değer
})

# Modeli kullanarak yeni bir tahmin yapın
tahmin = model.predict(yeni_veri)

print("Tahmini Işık Süresi:", tahmin[0])

y_pred = model.predict(X_test)

# Gerçek değerlere karşı tahminleri çizin
plt.scatter(y_test, y_pred, color='blue')  # Gerçek değerlere karşı tahminleri çizin
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # İdeal bir uyum çizgisi ekleyin
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek Değerler vs. Tahminler')
plt.show()