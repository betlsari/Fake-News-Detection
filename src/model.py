"""
BERT Tabanlı Sahte Haber Sınıflandırma Modeli
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FakeNewsDataset(Dataset):
    """PyTorch Dataset sınıfı - BERT için veriyi hazırlar"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Args:
            texts: Metin listesi
            labels: Etiket listesi
            tokenizer: BERT tokenizer
            max_length: Maksimum token uzunluğu
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Bir örneği döndür"""
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Metni BERT tokenize et
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,      # [CLS] ve [SEP] tokenları ekle
            max_length=self.max_length,
            padding='max_length',          # Max uzunluğa kadar padding
            truncation=True,               # Uzun metinleri kes
            return_attention_mask=True,
            return_tensors='pt'            # PyTorch tensor döndür
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTFakeNewsClassifier(nn.Module):
    """BERT tabanlı sınıflandırma modeli"""
    
    def __init__(self, model_name='bert-base-uncased', n_classes=2, dropout=0.3):
        """
        Args:
            model_name: Kullanılacak BERT modeli
            n_classes: Sınıf sayısı (2: gerçek/sahte)
            dropout: Dropout oranı
        """
        super(BERTFakeNewsClassifier, self).__init__()
        
        # BERT modelini yükle
        if 'bert' in model_name.lower():
            self.bert = BertModel.from_pretrained(model_name)
        else:
            self.bert = AutoModel.from_pretrained(model_name)
        
        # BERT'in hidden size'ını al
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout katmanı (overfitting'i önler)
        self.dropout = nn.Dropout(dropout)
        
        # Sınıflandırma katmanı
        self.classifier = nn.Linear(self.hidden_size, n_classes)
        
        print(f"✓ Model oluşturuldu: {model_name}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Çıkış sınıfları: {n_classes}")
    
    def forward(self, input_ids, attention_mask):
        """
        İleri geçiş (forward pass)
        
        Args:
            input_ids: Token ID'leri
            attention_mask: Attention mask
        Returns:
            Sınıf olasılıkları (logits)
        """
        # BERT'ten çıktı al
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token'ının çıktısını kullan (ilk token)
        pooled_output = outputs.pooler_output
        
        # Dropout uygula
        output = self.dropout(pooled_output)
        
        # Sınıflandırma
        logits = self.classifier(output)
        
        return logits


def create_data_loader(texts, labels, tokenizer, max_length=512, 
                       batch_size=16, shuffle=True):
    """
    DataLoader oluştur
    
    Args:
        texts: Metin listesi
        labels: Etiket listesi
        tokenizer: BERT tokenizer
        max_length: Maksimum token uzunluğu
        batch_size: Batch boyutu
        shuffle: Veriyi karıştır mı?
    Returns:
        DataLoader
    """
    dataset = FakeNewsDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Windows'ta hata alırsan 0 yap
    )


def get_model_and_tokenizer(model_name='bert-base-uncased', n_classes=2):
    """
    Model ve tokenizer'ı beraber yükle
    
    Args:
        model_name: Model adı
        n_classes: Sınıf sayısı
    Returns:
        model, tokenizer
    """
    print(f"\n📦 Model yükleniyor: {model_name}")
    
    # Tokenizer yükle
    if 'bert' in model_name.lower():
        tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"✓ Tokenizer yüklendi")
    
    # Model oluştur
    model = BERTFakeNewsClassifier(model_name=model_name, n_classes=n_classes)
    
    # GPU varsa kullan
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✓ Model {device} üzerinde hazır")
    
    return model, tokenizer


def predict(model, tokenizer, text, device):
    """
    Tek bir metin için tahmin yap
    
    Args:
        model: Eğitilmiş model
        tokenizer: Tokenizer
        text: Tahmin yapılacak metin
        device: CPU veya GPU
    Returns:
        Tahmin (0: gerçek, 1: sahte) ve olasılık
    """
    model.eval()
    
    # Metni tokenize et
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Tahmin yap
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probs, dim=1).cpu().numpy()[0]
        confidence = probs[0][prediction].cpu().numpy()
    
    return prediction, confidence


if __name__ == "__main__":
    # Test
    print("=" * 50)
    print("MODEL TEST")
    print("=" * 50)
    
    # Küçük bir model test et
    model, tokenizer = get_model_and_tokenizer('bert-base-uncased')
    
    # Örnek veri
    texts = ["This is a test sentence.", "Another example text here."]
    labels = [0, 1]
    
    # DataLoader oluştur
    loader = create_data_loader(texts, labels, tokenizer, batch_size=2)
    
    print(f"\n✓ DataLoader oluşturuldu: {len(loader)} batch")
    
    # Bir batch test et
    batch = next(iter(loader))
    print(f"\n📦 Batch içeriği:")
    print(f"  - input_ids shape: {batch['input_ids'].shape}")
    print(f"  - attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  - labels shape: {batch['label'].shape}")