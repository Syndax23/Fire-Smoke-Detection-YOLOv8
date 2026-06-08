# 🔥 Yangın ve Duman Algılama Sistemi

YOLOv8s tabanlı gerçek zamanlı yangın ve duman tespit uygulaması.

## 📋 Proje Hakkında

Bu proje, görüntülerdeki yangın ve dumanı tespit etmek için YOLOv8s modelini kullanmaktadır. Model, 800+ görüntüden oluşan bir veri seti ile 191 epoch boyunca eğitilmiştir.

### Tespit Edilen Sınıflar
- 🔥 **Fire** (Yangın)
- 💨 **Smoke** (Duman)

## 🚀 Kurulum

### 1. Gerekli kütüphaneleri yükle

```bash
pip install -r requirements.txt
```

### 2. Uygulamayı başlat

```bash
streamlit run demo_app.py
```

Tarayıcıda `http://localhost:8501` adresine git.

## 📁 Dosya Yapısı

```
├── best.pt          # Eğitilmiş YOLOv8s modeli
├── demo_app.py      # Streamlit arayüzü
├── train.py         # Model eğitim kodu
├── data.yaml        # Dataset konfigürasyonu
└── requirements.txt # Gerekli kütüphaneler
```

## 🛠️ Kullanım

1. Uygulamayı başlat
2. Sol sidebar'dan **Güven Eşiği** ve **IOU Eşiği** ayarlarını yap
3. Bir resim yükle
4. Model otomatik olarak yangın ve dumanı tespit eder

> 💡 **İpucu:** Yangın tespit edilemiyorsa Güven Eşiğini 0.10'a düşürün.

## 📊 Model Performansı

| Sınıf | mAP50 |
|-------|-------|
| Fire  | 0.378 |
| Smoke | 0.644 |
| Ortalama | 0.511 |

## 🔧 Teknolojiler

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io)
- [PyTorch](https://pytorch.org)
- Python 3.11
