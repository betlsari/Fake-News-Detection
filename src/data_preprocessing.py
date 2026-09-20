"""
Veri Ön İşleme Modülü
Bu dosya metin verilerini temizler ve modele hazır hale getirir
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:
    """Metin verilerini temizleyen ve hazırlayan sınıf"""
    
    def __init__(self, language='english'):
        """
        Args:
            language: Kullanılacak dil (stopwords için)
        """
        self.language = language
        # NLTK verilerini indir
        self._download_nltk_data()
        self.stop_words = set(stopwords.words(language))
    
    def _download_nltk_data(self):
        """Gerekli NLTK verilerini indir"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            print("✓ NLTK verileri indirildi")
        except:
            print("⚠ NLTK verilerinde sorun olabilir")
    
    def clean_text(self, text):
        """
        Metni temizle: URL, özel karakterler, fazla boşluklar vb.
        
        Args:
            text: Temizlenecek metin
        Returns:
            Temizlenmiş metin
        """
        if pd.isna(text):
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # URL'leri kaldır
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Email adreslerini kaldır
        text = re.sub(r'\S+@\S+', '', text)
        
        # Mention ve hashtag'leri temizle
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # HTML etiketlerini kaldır
        text = re.sub(r'<.*?>', '', text)
        
        # Özel karakterleri ve sayıları kaldır (sadece harfler ve boşluk kalsın)
        text = re.sub(r'[^a-zA-ZğüşıöçĞÜŞİÖÇ\s]', '', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Stopwords (gereksiz kelimeler) kaldır
        
        Args:
            text: İşlenecek metin
        Returns:
            Stopwords'siz metin
        """
        words = word_tokenize(text)
        filtered_words = [w for w in words if w not in self.stop_words and len(w) > 2]
        return ' '.join(filtered_words)
    
    def preprocess(self, text, remove_stops=False):
        """
        Tam ön işleme pipeline
        
        Args:
            text: İşlenecek metin
            remove_stops: Stopwords kaldırılsın mı?
        Returns:
            İşlenmiş metin
        """
        text = self.clean_text(text)
        if remove_stops:
            text = self.remove_stopwords(text)
        return text


def load_and_prepare_data(data_path, sample_size=None):
    """
    Veri setini yükle ve hazırla
    
    Args:
        data_path: Veri dosyasının yolu
        sample_size: Test için küçük bir örnek kullanmak istersen
    Returns:
        Pandas DataFrame
    """
    print(f"📂 Veri yükleniyor: {data_path}")
    
    # Veriyi oku
    df = pd.read_csv(data_path)
    
    print(f"✓ {len(df)} satır veri yüklendi")
    print(f"📊 Sütunlar: {df.columns.tolist()}")
    
    # Örnek boyutu varsa uygula
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"⚡ {sample_size} satırlık örnek alındı (hızlı test için)")
    
    # Eksik değerleri kontrol et
    print(f"\n📋 Eksik değerler:")
    print(df.isnull().sum())
    
    # Eksik değerleri doldur
    df = df.fillna('')
    
    # Sınıf dağılımını göster
    print(f"\n📊 Sınıf Dağılımı:")
    print(df['label'].value_counts())
    
    return df


def prepare_dataset_for_model(df, text_column='text', label_column='label', 
                               clean=True, remove_stops=False):
    """
    Modele hazır dataset oluştur
    
    Args:
        df: DataFrame
        text_column: Metin sütunu adı
        label_column: Etiket sütunu adı
        clean: Temizlik yapılsın mı?
        remove_stops: Stopwords kaldırılsın mı?
    Returns:
        İşlenmiş texts ve labels
    """
    print("\n🔄 Veri hazırlanıyor...")
    
    preprocessor = TextPreprocessor()
    
    # Metinleri işle
    if clean:
        print("🧹 Metinler temizleniyor...")
        df['processed_text'] = df[text_column].apply(
            lambda x: preprocessor.preprocess(x, remove_stops=remove_stops)
        )
    else:
        df['processed_text'] = df[text_column]
    
    # Boş metinleri filtrele
    df = df[df['processed_text'].str.len() > 10].reset_index(drop=True)
    
    print(f"✓ {len(df)} satır işlendi")
    
    # Etiketleri sayısal forma çevir (0: gerçek, 1: sahte)
    label_map = {0: 0, 1: 1, 'real': 0, 'fake': 1, 'REAL': 0, 'FAKE': 1}
    df['label_numeric'] = df[label_column].map(label_map)
    
    # Eğer etiket sayısal değilse otomatik mapping yap
    if df['label_numeric'].isnull().any():
        df['label_numeric'] = pd.Categorical(df[label_column]).codes
    
    print(f"✓ Etiketler hazır")
    
    return df['processed_text'].tolist(), df['label_numeric'].tolist()


if __name__ == "__main__":
    # Test için çalıştır
    print("=" * 50)
    print("VERİ ÖN İŞLEME TEST")
    print("=" * 50)
    
    # Örnek metin temizleme
    preprocessor = TextPreprocessor()
    
    test_text = """
    Check out this AMAZING article!!! 
    https://fake-news.com/story123
    Contact us at fake@email.com #FakeNews @FakeAccount
    <b>This is HTML</b> with numbers 12345 and symbols $%^&*
    """
    
    print("\nÖrnek Metin:")
    print(test_text)
    print("\nTemizlenmiş Metin:")
    print(preprocessor.preprocess(test_text))
    print("\nStopwords Kaldırılmış:")
    print(preprocessor.preprocess(test_text, remove_stops=True))