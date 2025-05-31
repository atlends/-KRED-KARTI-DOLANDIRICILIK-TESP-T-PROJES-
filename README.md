# Kredi Kartı Dolandırıcılık Tespiti

Bu proje, kredi kartı işlemleri üzerinde dolandırıcılık (fraud) tespiti yapmak için hazırlanmıştır.  
Amaç: Dolandırıcılık olan işlemleri (Class=1) doğru bir şekilde tahmin eden bir model geliştirmek.



# Kullanılan Yöntemler

- **Veri seti**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Ön İşleme**: StandardScaler ile özellikleri normalleştirme
- **Veri Dengesi**: RandomOverSampler ile sınıf 1 (fraud) örneklerini çoğaltma
- **Model**: RandomForestClassifier (500 ağaç, 15 max derinlik, dengeli sınıf ağırlığı)

---

# Sonuçlar

- **Accuracy**: %99.96
- **Not**: Yüksek accuracy’ye rağmen, dengesiz verilerde recall ve precision gibi metriklere dikkat etmek gerekir.

---

# Nasıl Çalıştırılır?

1. Gerekli kütüphaneleri yükleyin:
    ```
    pip install pandas numpy scikit-learn imbalanced-learn
    ```
2. Kod dosyasını çalıştırın:
    ```
    python fraud_detection.py
    ```


