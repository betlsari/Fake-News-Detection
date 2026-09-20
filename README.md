# 🔍 Sahte Haber Tespiti (Fake News Detection)

**NLP Dersi Projesi** - BERT Tabanlı Derin Öğrenme ile Sahte Haber Sınıflandırma

---

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Teknolojiler](#teknolojiler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Veri Seti](#veri-seti)
- [Model Mimarisi](#model-mimarisi)
- [Sonuçlar](#sonuçlar)
- [Proje Yapısı](#proje-yapısı)

---

## 🎯 Proje Hakkında

Bu proje, doğal dil işleme (NLP) ve derin öğrenme teknikleri kullanarak haberlerin gerçek mi sahte mi olduğunu tespit eden bir sistemdir. BERT (Bidirectional Encoder Representations from Transformers) modeli kullanılarak metinlerin anlamsal özelliklerini çıkarır ve sınıflandırma yapar.

### Amaç
- Sosyal medya ve haber platformlarındaki sahte haberleri otomatik olarak tespit etmek
- NLP ve derin öğrenme tekniklerini pratikte uygulamak
- BERT gibi modern transformer modellerini kullanmayı öğrenmek

---

## ✨ Özellikler

- ✅ **Otomatik Metin Temizleme**: URL, emoji, özel karakterlerin kaldırılması
- ✅ **BERT Tabanlı Sınıflandırma**: State-of-the-art transformer modeli
- ✅ **Kapsamlı Ön İşleme**: NLTK ve SpaCy ile metin normalizasyonu
- ✅ **Görselleştirme**: Eğitim grafikleri ve confusion matrix
- ✅ **Kolay Kullanım**: Komut satırı arayüzü ile basit eğitim ve tahmin
- ✅ **Esnek Yapı**: Farklı BERT varyantları kullanabilme (BERTurk, DistilBERT, vb.)

---

## 🛠 Teknolojiler

### Python Kütüphaneleri

| Kütüphane | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| **PyTorch** | 2.0.1 | Derin öğrenme framework'ü |
| **Transformers** | 4.31.0 | BERT model ve tokenizer'lar |
| **NLTK** | 3.8.1 | Metin ön işleme |
| **SpaCy** | 3.6.0 | İleri seviye NLP |
| **Pandas** | 2.0.3 | Veri manipülasyonu |
| **NumPy** | 1.24.3 | Nümerik hesaplamalar |
| **Scikit-learn** | 1.3.0 | Metrikler ve veri bölme |
| **Matplotlib/Seaborn** | 3.7.2/0.12.2 | Görselleştirme |

### Kullanılan Modeller

- **bert-base-uncased**: Temel İngilizce BERT modeli
- **BERTurk** (opsiyonel): Türkçe metinler için
- **DistilBERT** (opsiyonel): Daha hızlı ve hafif versiyon

---

## 🚀 Kurulum

### 1. Repoyu İndir veya Klasör Oluştur

```bash
# Proje klasörünü oluştur
mkdir fake-news-detection
cd fake-news-detection

# Proje yapısını oluştur
mkdir -p data/raw data/processed models notebooks src
```

### 2. Virtual Environment Oluştur (Önerilen)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Gerekli Kütüphaneleri Yükle

```bash
pip install -r requirements.txt
```

### 4. NLTK Verilerini İndir

Python konsolunda:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 5. Kaggle API Kurulumu

**a) Kaggle hesabı oluştur:** https://www.kaggle.com

**b) API Token al:**
- Profil → Settings → API → "Create New API Token"
- `kaggle.json` dosyası inecek

**c) Token'ı doğru yere koy:**
```bash
# Windows
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\

# Mac/Linux
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**d) Veri setini indir:**
```bash
# WELFake Dataset - 72,134 haber içeriyor
kaggle datasets download -d saurabhshahane/fake-news-classification
unzip fake-news-classification.zip -d data/raw/
```

---

## 💻 Kullanım

### Hızlı Başlangıç - Tam Eğitim

```bash
# Temel eğitim (3 epoch)
python main.py --mode train

# Özelleştirilmiş eğitim
python main.py --mode train \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --max_length 256
```

### Hızlı Test İçin Küçük Örnekle Eğitim

```bash
# İlk test için sadece 1000 örnek kullan
python main.py --mode train \
    --sample_size 1000 \
    --epochs 2 \
    --batch_size 8
```

### Model Testi

```bash
# Eğitilmiş modeli test et
python main.py --mode test
```

### Tek Metin Tahmini

```bash
# Komut satırından
python main.py --mode predict \
    --text "Breaking news: Scientists discover new planet!"

# İnteraktif mod
python main.py --mode predict
# Ardından metni gir
```

### Farklı BERT Modelleri Kullanma

```bash
# DistilBERT (daha hızlı)
python main.py --mode train --model_name distilbert-base-uncased

# BERTurk (Türkçe için)
python main.py --mode train --model_name dbmdz/bert-base-turkish-cased

# BERT Large (daha güçlü ama yavaş)
python main.py --mode train --model_name bert-large-uncased
```

---

## 📊 Veri Seti

### WELFake Dataset

**Kaynak**: [Kaggle - Fake News Classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

**İstatistikler:**
- **Toplam Örnek**: 72,134 haber
- **Sınıflar**: 
  - 0: Gerçek haberler (Real)
  - 1: Sahte haberler (Fake)
- **Dil**: İngilizce
- **Özellikler**:
  - `text`: Haber metni
  - `title`: Başlık
  - `label`: Etiket (0/1)

**Veri Bölme:**
- Eğitim: %70
- Validasyon: %15
- Test: %15

---

## 🧠 Model Mimarisi

### BERT Mimarisi

```
Input Text → Tokenization → BERT Encoder → [CLS] Token → Dropout → Linear Layer → Softmax → Prediction
```

### Detaylı Yapı

1. **Input Layer**: Metin tokenize edilir (max 512 token)
2. **BERT Encoder**: 
   - 12 transformer katmanı (base) veya 24 (large)
   - 768 (base) veya 1024 (large) hidden size
   - Multi-head attention mechanism
3. **Pooling**: [CLS] token'ının çıktısı kullanılır
4. **Dropout**: %30 dropout (overfitting önleme)
5. **Classification Head**: Linear layer (768 → 2)
6. **Output**: Softmax ile olasılıklar

### Hiperparametreler

```python
{
    "model": "bert-base-uncased",
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "optimizer": "AdamW",
    "loss": "CrossEntropyLoss",
    "dropout": 0.3
}
```

---

## 📈 Sonuçlar

### Beklenen Performans

| Metrik | Değer |
|--------|-------|
| **Accuracy** | ~94-96% |
| **Precision** | ~93-95% |
| **Recall** | ~94-96% |
| **F1-Score** | ~94-95% |

### Çıktı Dosyaları

Eğitim sonrası oluşacak dosyalar:

```
models/
├── best_model.pt              # En iyi model ağırlıkları

./ (proje kök dizini)
├── training_history.png       # Loss ve accuracy grafikleri
├── confusion_matrix.png       # Karışıklık matrisi
```

### Örnek Grafikler

**Training History:**
- Epoch bazında loss ve accuracy değişimi
- Overfitting kontrolü için train vs validation karşılaştırması

**Confusion Matrix:**
```
              Gerçek    Sahte
Gerçek    [   TP         FP   ]
Sahte     [   FN         TN   ]
```

---

## 📁 Proje Yapısı

```
fake-news-detection/
│
├── data/
│   ├── raw/
│   │   └── WELFake_Dataset.csv        # Ham veri seti
│   └── processed/                      # İşlenmiş veriler
│
├── models/
│   └── best_model.pt                   # Eğitilmiş model
│
├── notebooks/
│   └── exploratory_analysis.ipynb      # (Opsiyonel) Veri analizi
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Veri ön işleme
│   ├── model.py                        # Model tanımı
│   └── train.py                        # Eğitim fonksiyonları
│
├── main.py                             # Ana çalıştırma dosyası
├── requirements.txt                    # Python bağımlılıkları
├── README.md                           # Bu dosya
├── training_history.png                # Eğitim grafikleri
└── confusion_matrix.png                # Confusion matrix
```

---

## 🔧 İleri Seviye Özellikler

### 1. Özel Veri Seti Kullanma

Kendi veri setini kullanmak için:

```python
# CSV formatında veri hazırla
# Sütunlar: text, label

python main.py --mode train \
    --data_path path/to/your/data.csv \
    --text_column your_text_column \
    --label_column your_label_column
```

### 2. Model Fine-Tuning

```python
# Daha fazla epoch ile ince ayar
python main.py --mode train \
    --epochs 10 \
    --learning_rate 1e-5
```

### 3. GPU Kullanımı

Model otomatik olarak GPU tespit eder. CUDA kuruluysa otomatik kullanır.

```python
import torch
print(torch.cuda.is_available())  # True ise GPU kullanılacak
```

---

## 📝 Kod Örnekleri

### Python'dan Kullanım

```python
from src.model import get_model_and_tokenizer, predict
import torch

# Model yükle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = get_model_and_tokenizer('bert-base-uncased')
model.load_state_dict(torch.load('models/best_model.pt'))

# Tahmin yap
text = "Scientists discover new cure for cancer!"
prediction, confidence = predict(model, tokenizer, text, device)

print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
print(f"Confidence: {confidence:.2%}")
```

---

## 🐛 Sorun Giderme

### Yaygın Hatalar ve Çözümleri

**1. CUDA out of memory**
```bash
# Batch size'ı küçült
python main.py --mode train --batch_size 8
```

**2. NLTK data hatası**
```python
import nltk
nltk.download('all')  # Tüm NLTK verilerini indir
```

**3. Kaggle API hatası**
```bash
# Token'ı doğru yere koyduğundan emin ol
ls ~/.kaggle/kaggle.json  # Mac/Linux
dir %USERPROFILE%\.kaggle\kaggle.json  # Windows
```

**4. Transformers kütüphane hatası**
```bash
# Transformers'ı güncelle
pip install --upgrade transformers
```

---

## 📚 Kaynaklar

### Öğrenme Materyalleri

- [BERT Paper](https://arxiv.org/abs/1810.04805) - Orijinal BERT makalesi
- [Hugging Face Documentation](https://huggingface.co/docs) - Transformers dokümantasyonu
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - PyTorch öğretici
- [NLTK Documentation](https://www.nltk.org/) - NLTK rehberi

### Faydalı Linkler

- [Kaggle Competitions](https://www.kaggle.com/competitions?search=fake+news) - Fake news yarışmaları
- [Papers With Code](https://paperswithcode.com/task/fake-news-detection) - SOTA modeller
- [Medium Articles](https://medium.com/search?q=fake%20news%20detection%20bert) - Blog yazıları

---

## 🤝 Katkıda Bulunma

Bu bir eğitim projesidir. Geliştirme önerileri:

1. Farklı BERT modelleri deneyin (RoBERTa, ALBERT, etc.)
2. Ensemble yöntemleri uygulayın
3. Açıklanabilir AI (XAI) teknikleri ekleyin
4. Çoklu dil desteği ekleyin
5. Web arayüzü oluşturun (Streamlit, Flask)

---

## 📄 Lisans

Bu proje eğitim amaçlıdır. Ticari kullanım için ilgili kütüphanelerin lisanslarını kontrol edin.

---

## 👨‍💻 Yazar

**3. Sınıf Bilgisayar Mühendisliği Öğrencisi**  
NLP Dersi Projesi - 2024

---

## 📞 İletişim ve Destek

Sorularınız için:
- GitHub Issues kullanın
- NLP dersi hocası ile iletişime geçin
- Kaggle discussion forumlarını ziyaret edin

---

## ⚡ Hızlı Komutlar Özeti

```bash
# Kurulum
pip install -r requirements.txt
kaggle datasets download -d saurabhshahane/fake-news-classification
unzip fake-news-classification.zip -d data/raw/

# Hızlı test (küçük veri)
python main.py --mode train --sample_size 1000 --epochs 2

# Tam eğitim
python main.py --mode train --epochs 3

# Test
python main.py --mode test

# Tahmin
python main.py --mode predict --text "Your news text here"
```

---

