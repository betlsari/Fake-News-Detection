# 🔍 Fake News Detection



Bu proje, **Doğal Dil İşleme (NLP)** ve **Transformer tabanlı derin öğrenme** teknikleri kullanılarak haber metinlerinin gerçek veya sahte olup olmadığını sınıflandırmak amacıyla geliştirilmiştir.

Projenin temelinde **BERT (Bidirectional Encoder Representations from Transformers)** mimarisi ve **PyTorch** kullanılmaktadır.

---

## 📌 Proje Hakkında

Sahte haberler, özellikle sosyal medya ve çevrim içi haber platformlarının yaygınlaşmasıyla önemli bir problem haline gelmiştir. Bu projede, haber metinlerinden yararlanarak otomatik bir **Fake News Detection** sistemi geliştirilmiştir.

Sistem;

* Haber metnini alır
* Metni ön işler
* BERT tokenizer ile metni tokenize eder
* BERT modelinden anlamsal özellikleri çıkarır
* Sınıflandırma katmanı üzerinden tahmin üretir
* Haberi **Real (Gerçek)** veya **Fake (Sahte)** olarak sınıflandırır

### 🎯 Projenin Amaçları

* NLP tekniklerini uygulamalı olarak öğrenmek
* Transformer mimarisini ve BERT modelini kullanmak
* Metin sınıflandırma problemi üzerinde çalışmak
* Derin öğrenme ile gerçek/sahte haber ayrımı yapmak
* Model performansını farklı metriklerle değerlendirmek

---

## ✨ Özellikler

* 🧹 **Metin Ön İşleme**
  URL, özel karakter ve gereksiz metin öğelerinin temizlenmesi

* 🤖 **BERT Tabanlı Sınıflandırma**
  Transformer tabanlı BERT modeli ile metin sınıflandırma

* 📊 **Model Değerlendirme**
  Accuracy, Precision, Recall ve F1-Score metrikleri

* 📈 **Eğitim Görselleştirmesi**
  Training/validation loss ve accuracy grafiklerinin oluşturulması

* 🔲 **Confusion Matrix**
  Modelin doğru ve yanlış sınıflandırmalarının görselleştirilmesi

* 🎯 **Tek Metin Tahmini**
  Kullanıcının girdiği bir haber metni üzerinde tahmin yapılabilmesi

* ⚙️ **Esnek Eğitim Yapısı**
  Epoch, batch size, learning rate ve max sequence length gibi parametrelerin değiştirilebilmesi

* 🖥️ **GPU Desteği**
  CUDA destekli sistemlerde GPU kullanımının otomatik olarak algılanması

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji / Kütüphane         | Kullanım Alanı                            |
| ----------------------------- | ----------------------------------------- |
| **Python**                    | Ana programlama dili                      |
| **PyTorch**                   | Derin öğrenme ve model eğitimi            |
| **Hugging Face Transformers** | BERT ve tokenizer                         |
| **NLTK**                      | Doğal dil işleme ve metin ön işleme       |
| **spaCy**                     | NLP işlemleri                             |
| **Pandas**                    | Veri işleme                               |
| **NumPy**                     | Sayısal işlemler                          |
| **Scikit-learn**              | Veri bölme ve değerlendirme metrikleri    |
| **Matplotlib**                | Veri görselleştirme                       |
| **Seaborn**                   | Grafik ve confusion matrix görselleştirme |

### 🤖 Kullanılabilen Modeller

Proje yapısı farklı Transformer modelleriyle çalışabilecek şekilde tasarlanmıştır.

Örnek modeller:

* `bert-base-uncased`
* `distilbert-base-uncased`
* `dbmdz/bert-base-turkish-cased`
* `bert-large-uncased`

---

# 🚀 Kurulum

## 1. Repoyu Klonlama

```bash
git clone https://github.com/betlsari/Fake-News-Detection.git
cd Fake-News-Detection
```

## 2. Virtual Environment Oluşturma

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Gerekli Kütüphaneleri Yükleme

```bash
pip install -r requirements.txt
```

## 4. NLTK Verilerini İndirme

Python konsolunda:

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

---

# 📂 Veri Seti

Projede **WELFake Dataset** kullanılmıştır.

Veri seti Kaggle üzerinden temin edilmiştir.

**Veri seti:** WELFake Dataset
**Toplam haber:** 72.134
**Dil:** İngilizce

### Sınıflar

| Label | Sınıf         |
| ----: | ------------- |
|   `0` | Real — Gerçek |
|   `1` | Fake — Sahte  |

### Temel Özellikler

* `text` — Haber metni
* `title` — Haber başlığı
* `label` — Sınıf etiketi

### Veri Bölme

Veri seti eğitim, doğrulama ve test olmak üzere üç bölüme ayrılmıştır:

* **%70** — Training
* **%15** — Validation
* **%15** — Test

> ⚠️ Veri seti boyutu nedeniyle `data/` klasörü GitHub repository'sine dahil edilmemiştir. Projeyi çalıştırmak için veri setinin ayrıca indirilmesi gerekir.

---

# 🧠 Model Mimarisi

Projenin temelinde BERT tabanlı bir metin sınıflandırma mimarisi bulunmaktadır.

```text
                Input Text
                    │
                    ▼
              Tokenization
                    │
                    ▼
              BERT Encoder
                    │
                    ▼
               [CLS] Token
                    │
                    ▼
                 Dropout
                    │
                    ▼
            Classification Layer
                    │
                    ▼
             Real / Fake
```

### Model Akışı

1. Haber metni sisteme girilir.
2. Metin BERT tokenizer tarafından tokenize edilir.
3. Tokenlar BERT encoder'a gönderilir.
4. BERT'in `[CLS]` token çıktısı sınıflandırma için kullanılır.
5. Dropout katmanı uygulanır.
6. Linear classification layer ile iki sınıflı tahmin yapılır.
7. Sonuç **Real** veya **Fake** olarak döndürülür.

### Temel Hiperparametreler

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

# 💻 Kullanım

## 🔹 Eğitim

Temel eğitim:

```bash
python main.py --mode train
```

Özelleştirilmiş eğitim:

```bash
python main.py --mode train \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --max_length 256
```

### Küçük Veriyle Test

Sistemi hızlıca test etmek için daha küçük bir veri örneği kullanılabilir:

```bash
python main.py --mode train \
    --sample_size 1000 \
    --epochs 2 \
    --batch_size 8
```

---

## 🔹 Model Testi

Eğitilmiş modeli test etmek için:

```bash
python main.py --mode test
```

---

## 🔹 Tek Haber Tahmini

Komut satırından doğrudan haber metni göndermek için:

```bash
python main.py --mode predict \
    --text "Breaking news: Scientists discover new planet!"
```

İnteraktif tahmin modu:

```bash
python main.py --mode predict
```

Daha sonra terminal üzerinden haber metni girilebilir.

---

## 🔹 Farklı Modellerle Eğitim

### DistilBERT

Daha küçük ve hızlı bir model:

```bash
python main.py --mode train \
    --model_name distilbert-base-uncased
```

### BERTurk

Türkçe metinler üzerinde çalışmak için:

```bash
python main.py --mode train \
    --model_name dbmdz/bert-base-turkish-cased
```

### BERT Large

Daha büyük model:

```bash
python main.py --mode train \
    --model_name bert-large-uncased
```

---

# 📊 Model Değerlendirme

Model performansı aşağıdaki metrikler üzerinden değerlendirilmektedir:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

Projede ayrıca model davranışını incelemek için:

* Training History
* Confusion Matrix

oluşturulmaktadır.

### 📈 Eğitim Grafiği

`training_history.png` dosyası eğitim sırasında loss ve accuracy değerlerinin epoch'lara göre değişimini gösterir.

### 🔲 Confusion Matrix

`confusion_matrix.png` dosyası modelin gerçek ve tahmin edilen sınıflar arasındaki dağılımını gösterir.

```text
                  Tahmin
              Real      Fake

Gerçek Real     TP        FN
Gerçek Fake     FP        TN
```

> Not: Buradaki değerlerin yorumlanması, kullanılan label kodlamasına göre yapılmalıdır.

---

# 📁 Proje Yapısı

```text
Fake-News-Detection/
│
├── data/
│   └── raw/
│       └── WELFake_Dataset.csv
│
├── models/
│   └── best_model.pt
│
├── notebooks/
│   └── exploratory_analysis.py
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   └── train.py
│
├── main.py
├── requirements.txt
├── README.md
├── training_history.png
├── confusion_matrix.png
└── .gitignore
```

> `data/`, `models/` ve sanal ortam gibi büyük veya yerel dosyalar `.gitignore` ile GitHub dışında tutulmaktadır.

---

# ⚙️ GPU Kullanımı

PyTorch, sistemde CUDA destekli bir GPU bulunması durumunda GPU kullanımına izin verir.

Kontrol etmek için:

```python
import torch

print(torch.cuda.is_available())
```

Sonuç:

```text
True
```

ise CUDA destekli GPU kullanılabilir.

---

# 🧩 Python ile Kullanım

Model fonksiyonları Python içerisinden de kullanılabilir.

```python
from src.model import get_model_and_tokenizer, predict
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model, tokenizer = get_model_and_tokenizer(
    "bert-base-uncased"
)

model.load_state_dict(
    torch.load("models/best_model.pt")
)

text = "Scientists discover new cure for cancer!"

prediction, confidence = predict(
    model,
    tokenizer,
    text,
    device
)

print(
    f"Prediction: {'Fake' if prediction == 1 else 'Real'}"
)

print(
    f"Confidence: {confidence:.2%}"
)
```

---

# 🐛 Sorun Giderme

## CUDA Out of Memory

Batch size değerini azaltmayı deneyin:

```bash
python main.py --mode train --batch_size 8
```

Gerekirse:

```bash
python main.py --mode train --batch_size 4
```

## NLTK Veri Hatası

Gerekli NLTK paketlerini tekrar indirin:

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

## Transformers Hatası

Kütüphaneyi güncellemek için:

```bash
pip install --upgrade transformers
```

## Kaggle Veri Seti Sorunu

Veri setinin `data/raw/` klasöründe bulunduğundan emin olun.

---

# 🔬 Geliştirme Fikirleri

Proje daha ileri seviyeye taşınmak istenirse:

* [ ] RoBERTa ve ALBERT gibi farklı Transformer modellerini denemek
* [ ] BERTurk ile Türkçe sahte haber sınıflandırması yapmak
* [ ] Hyperparameter tuning uygulamak
* [ ] Explainable AI (XAI) yöntemleri eklemek
* [ ] Model karşılaştırma çalışması yapmak
* [ ] Web tabanlı kullanıcı arayüzü geliştirmek
* [ ] REST API oluşturmak
* [ ] Docker ile uygulamayı containerize etmek
* [ ] Modeli bir cloud ortamında deploy etmek

---

# 📚 Kaynaklar

* [BERT — Original Paper](https://arxiv.org/abs/1810.04805)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
* [PyTorch Documentation](https://pytorch.org/docs/)
* [NLTK Documentation](https://www.nltk.org/)
* [Kaggle](https://www.kaggle.com/)

---

# 👩‍💻 Proje Bilgileri



**Betül Sarı**
Computer Engineering Student
İzmir Katip Çelebi University

---

# ⚡ Hızlı Başlangıç

```bash
# Repository'yi klonla
git clone https://github.com/betlsari/Fake-News-Detection.git

# Proje klasörüne gir
cd Fake-News-Detection

# Virtual environment oluştur
python -m venv venv

# Windows'ta aktif et
venv\Scripts\activate

# Kütüphaneleri yükle
pip install -r requirements.txt

# Küçük veriyle test
python main.py --mode train --sample_size 1000 --epochs 2

# Tam eğitim
python main.py --mode train --epochs 3

# Modeli test et
python main.py --mode test

# Tahmin yap
python main.py --mode predict --text "Your news text here"
```

---

## 📌 Not

Bu proje **eğitim ve akademik çalışma amacıyla** geliştirilmiştir. Model çıktıları gerçek dünyadaki haberlerin doğruluğunu kesin olarak belirleyen bir kaynak olarak değerlendirilmemelidir.
