# =============================================
# KREDİ KARTI DOLANDIRICILIK TESPİTİ PROJESİ
# =============================================

# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# 1. Veri setini yükle
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# Orijinal veri çok büyük olduğu için, %10'luk bir örneklem alıyoruz.
df_sample = df.sample(frac=0.1, random_state=42) # aynı sonuç için sabit rastgelelik

#  Özellikler ve hedef değişken
X = df_sample.drop('Class', axis=1) #Modelin tahmin edeceği özellikler (Zaman vb.)
y = df_sample['Class']

#  Özellikleri normalize ediyoruz
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi eğitim (%80) ve test (%20) olarak ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Azınlık sınıfı (Class=1, dolandırıcılık) örneklerini kopyalayarak çoğaltıyoruz.
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

# 7. Modeli optimize edilmiş parametrelerle kur
model = RandomForestClassifier(
    n_estimators=500, # daha güçlü bir model olması için 500 ağaç yaptım
    max_depth=15,  # ağaç derinliğini 15 ile sınırladım
    min_samples_split=5, # dallanmalar için minimum 5 örnek gerekli
    min_samples_leaf=2, # her yaprakta 2 örnek olmalı
    class_weight='balanced', # modeli azınlık sınıfına daha duyarlı yaptım
    random_state=42
)

#  Modeli dengelenmiş veri ile eğittim
model.fit(X_res, y_res)

# Her test örneği için dolandırıcılık olasılıklarını (Class=1) alıyoruz.
y_proba = model.predict_proba(X_test)[:,1]

#  Eşik değerini düşürerek tahmin yap (default 0.5 yerine 0.3)
y_pred = (y_proba >= 0.3).astype(int)

#  Performans metriklerini hesapla
confusion = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# 12. Sonuçları yazdır
print("Confusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification)
print(f"Accuracy: {accuracy:.4f}")
