# Jupyter Notebook - Veri Keşfi ve Analiz
# Bu dosyayı notebooks/ klasörüne kaydet

"""
SAHTE HABER TESPİTİ - VERİ ANALİZİ
Bu notebook veri setini keşfetmek için kullanılır
"""

# %% [markdown]
# # Sahte Haber Tespiti - Veri Analizi
# 
# Bu notebook'ta:
# 1. Veri setini yükleyip inceleyeceğiz
# 2. Metin özelliklerini analiz edeceğiz
# 3. Görselleştirmeler yapacağız
# 4. Basit bir model denemesi yapacağız

# %% Kütüphaneleri İmport Et
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Grafik ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Kütüphaneler yüklendi")

# %% Veriyi Yükle
df = pd.read_csv('../data/raw/WELFake_Dataset.csv')
print(f"✓ {len(df)} satır veri yüklendi")
df.head()

# %% [markdown]
# ## 1. Temel Veri İnceleme

# %% Veri Bilgisi
print("Veri Seti Bilgisi:")
print("="*50)
df.info()
print("\n")
print("İstatistikler:")
print("="*50)
df.describe()

# %% Eksik Değerler
print("Eksik Değerler:")
print(df.isnull().sum())

# Eksik değerleri doldur
df = df.fillna('')

# %% [markdown]
# ## 2. Sınıf Dağılımı

# %% Sınıf Dağılımı Analizi
plt.figure(figsize=(10, 6))

# Count plot
ax = sns.countplot(data=df, x='label')
plt.title('Sınıf Dağılımı', fontsize=16, fontweight='bold')
plt.xlabel('Sınıf (0: Gerçek, 1: Sahte)', fontsize=12)
plt.ylabel('Sayı', fontsize=12)

# Sayıları ekle
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# Yüzdeler
print("\nSınıf Dağılımı (Yüzde):")
print(df['label'].value_counts(normalize=True) * 100)

# %% [markdown]
# ## 3. Metin Uzunluğu Analizi

# %% Metin Uzunluklarını Hesapla
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print("Metin Uzunluğu İstatistikleri:")
print("="*50)
print(df.groupby('label')[['text_length', 'word_count']].describe())

# %% Uzunluk Dağılımı Grafikleri
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Karakter uzunluğu
for label in [0, 1]:
    data = df[df['label'] == label]['text_length']
    axes[0].hist(data, bins=50, alpha=0.6, 
                label=f"{'Gerçek' if label == 0 else 'Sahte'}")
axes[0].set_xlabel('Karakter Sayısı', fontsize=12)
axes[0].set_ylabel('Frekans', fontsize=12)
axes[0].set_title('Metin Uzunluğu Dağılımı', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].set_xlim(0, 10000)

# Kelime sayısı
for label in [0, 1]:
    data = df[df['label'] == label]['word_count']
    axes[1].hist(data, bins=50, alpha=0.6,
                label=f"{'Gerçek' if label == 0 else 'Sahte'}")
axes[1].set_xlabel('Kelime Sayısı', fontsize=12)
axes[1].set_ylabel('Frekans', fontsize=12)
axes[1].set_title('Kelime Sayısı Dağılımı', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].set_xlim(0, 2000)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Kelime Bulutu (Word Cloud)

# %% Gerçek Haberler İçin Word Cloud
real_text = ' '.join(df[df['label'] == 0]['text'].head(1000))

plt.figure(figsize=(15, 8))
wordcloud = WordCloud(width=800, height=400, 
                     background_color='white',
                     colormap='Greens',
                     max_words=100).generate(real_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Gerçek Haberlerde En Çok Kullanılan Kelimeler', 
         fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% Sahte Haberler İçin Word Cloud
fake_text = ' '.join(df[df['label'] == 1]['text'].head(1000))

plt.figure(figsize=(15, 8))
wordcloud = WordCloud(width=800, height=400,
                     background_color='white',
                     colormap='Reds',
                     max_words=100).generate(fake_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Sahte Haberlerde En Çok Kullanılan Kelimeler',
         fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. En Sık Kullanılan Kelimeler

# %% En Sık Kelimeler Analizi
from collections import Counter
import re

def get_top_words(text_series, n=20):
    """En sık kullanılan kelimeleri bul"""
    # Tüm metinleri birleştir
    all_text = ' '.join(text_series.astype(str))
    # Küçük harfe çevir ve temizle
    all_text = all_text.lower()
    all_text = re.sub(r'[^a-z\s]', '', all_text)
    # Kelimelere ayır
    words = all_text.split()
    # Stopwords filtrele (basit liste)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}
    words = [w for w in words if w not in stopwords and len(w) > 3]
    # Say
    return Counter(words).most_common(n)

# Her sınıf için en sık kelimeler
real_top = get_top_words(df[df['label'] == 0]['text'])
fake_top = get_top_words(df[df['label'] == 1]['text'])

# Görselleştir
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gerçek haberler
words_real, counts_real = zip(*real_top)
axes[0].barh(words_real, counts_real, color='green', alpha=0.7)
axes[0].set_xlabel('Frekans', fontsize=12)
axes[0].set_title('Gerçek Haberlerde En Sık Kelimeler', 
                 fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# Sahte haberler
words_fake, counts_fake = zip(*fake_top)
axes[1].barh(words_fake, counts_fake, color='red', alpha=0.7)
axes[1].set_xlabel('Frekans', fontsize=12)
axes[1].set_title('Sahte Haberlerde En Sık Kelimeler',
                 fontsize=14, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Örnek Metinler

# %% Örnek Metinleri Göster
print("GERÇEK HABER ÖRNEĞİ:")
print("="*80)
print(df[df['label'] == 0]['text'].iloc[0][:500])
print("\n")

print("SAHTE HABER ÖRNEĞİ:")
print("="*80)
print(df[df['label'] == 1]['text'].iloc[0][:500])

# %% [markdown]
# ## 7. Basit Model Denemesi (TF-IDF + Logistic Regression)

# %% Hızlı Baseline Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Küçük bir örnek al (hızlı test için)
sample_df = df.sample(n=5000, random_state=42)

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(
    sample_df['text'], 
    sample_df['label'],
    test_size=0.2,
    random_state=42
)

print(f"Eğitim: {len(X_train)}, Test: {len(X_test)}")

# TF-IDF özellik çıkarımı
print("\n🔄 TF-IDF özellikleri çıkarılıyor...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✓ Özellik boyutu: {X_train_tfidf.shape}")

# Logistic Regression modeli
print("\n🎓 Model eğitiliyor...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Tahmin
y_pred = lr_model.predict(X_test_tfidf)

# Sonuçlar
print("\n" + "="*50)
print("BASELINE MODEL SONUÇLARI")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetaylı Rapor:")
print(classification_report(y_test, y_pred, 
                           target_names=['Gerçek', 'Sahte']))

# %% [markdown]
# ## 8. Sonuç ve Öneriler
# 
# ### Gözlemler:
# 1. Veri seti dengeli mi? (Sınıf dağılımına bak)
# 2. Sahte ve gerçek haberlerin uzunlukları farklı mı?
# 3. Hangi kelimeler ayırt edici?
# 
# ### Sonraki Adımlar:
# 1. BERT modeli baseline'dan daha iyi performans gösterecek
# 2. Metin temizleme işlemi önemli
# 3. Daha fazla epoch ve büyük veri ile accuracy artacak
# 
# **Şimdi ana projeye geç:** `python main.py --mode train`

# %%
print("\n" + "="*50)
print("✅ ANALİZ TAMAMLANDI!")
print("="*50)
print("\nBir sonraki adım: BERT modeli ile eğitim")
print("Komut: python main.py --mode train")