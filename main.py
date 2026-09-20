"""
SAHTE HABER TESPİTİ PROJESİ
Ana Çalıştırma Dosyası

Kullanım:
    python main.py --mode train    # Model eğitimi
    python main.py --mode test     # Model testi
    python main.py --mode predict  # Tek metin tahmini
"""

import argparse
import torch
import os
import sys
from sklearn.model_selection import train_test_split

# Kendi modüllerimizi import et
from src.data_preprocessing import load_and_prepare_data, prepare_dataset_for_model
from src.model import get_model_and_tokenizer, create_data_loader, predict
from src.train import ModelTrainer, evaluate_model


def ensure_directories():
    """Gerekli klasörleri oluştur"""
    dirs = ['data/raw', 'data/processed', 'models', 'notebooks']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def train_model(args):
    """Model eğitimi"""
    print("\n" + "="*60)
    print("🚀 SAHTE HABER TESPİTİ - MODEL EĞİTİMİ")
    print("="*60 + "\n")
    
    # 1. VERİ YÜKLEME
    print("📂 ADIM 1: Veri Yükleme")
    df = load_and_prepare_data(
        data_path=args.data_path,
        sample_size=args.sample_size
    )
    
    # 2. VERİ ÖN İŞLEME
    print("\n🔄 ADIM 2: Veri Ön İşleme")
    texts, labels = prepare_dataset_for_model(
        df,
        text_column=args.text_column,
        label_column=args.label_column,
        clean=True,
        remove_stops=False
    )
    
    # 3. VERİYİ BÖL (Train/Val/Test)
    print("\n✂️  ADIM 3: Veriyi Böl")
    # Önce train+val ve test olarak böl
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    # Sonra train ve val olarak böl
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Eğitim seti: {len(X_train)} örnek")
    print(f"✓ Validasyon seti: {len(X_val)} örnek")
    print(f"✓ Test seti: {len(X_test)} örnek")
    
    # 4. MODEL VE TOKENIZER YÜKLEME
    print("\n🤖 ADIM 4: Model Hazırlama")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Cihaz: {device}")
    
    model, tokenizer = get_model_and_tokenizer(
        model_name=args.model_name,
        n_classes=2
    )
    
    # 5. DATALOADER OLUŞTURMA
    print("\n📦 ADIM 5: DataLoader Oluşturma")
    train_loader = create_data_loader(
        X_train, y_train, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = create_data_loader(
        X_val, y_val, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_loader = create_data_loader(
        X_test, y_test, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # 6. MODEL EĞİTİMİ
    print("\n🎓 ADIM 6: Model Eğitimi")
    trainer = ModelTrainer(model, device, learning_rate=args.learning_rate)
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=args.model_save_path
    )
    
    # 7. EĞİTİM GRAFİKLERİ
    print("\n📈 ADIM 7: Grafikleri Kaydet")
    trainer.plot_history('training_history.png')
    
    # 8. TEST SETİ DEĞERLENDİRMESİ
    print("\n🧪 ADIM 8: Test Seti Değerlendirmesi")
    # En iyi modeli yükle
    model.load_state_dict(torch.load(args.model_save_path))
    
    evaluate_model(model, test_loader, device)
    
    print("\n" + "="*60)
    print("✅ EĞİTİM SÜRECİ TAMAMLANDI!")
    print(f"💾 Model kaydedildi: {args.model_save_path}")
    print("="*60 + "\n")


def test_model(args):
    """Kaydedilmiş modeli test et"""
    print("\n" + "="*60)
    print("🧪 MODEL TEST")
    print("="*60 + "\n")
    
    # Veriyi yükle
    df = load_and_prepare_data(args.data_path)
    texts, labels = prepare_dataset_for_model(df, args.text_column, args.label_column)
    
    # Test seti için böl
    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42
    )
    
    # Model ve tokenizer yükle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = get_model_and_tokenizer(args.model_name, n_classes=2)
    
    # Kaydedilmiş ağırlıkları yükle
    if not os.path.exists(args.model_save_path):
        print(f"❌ Model bulunamadı: {args.model_save_path}")
        print("Önce modeli eğitmelisin: python main.py --mode train")
        return
    
    model.load_state_dict(torch.load(args.model_save_path))
    print(f"✓ Model yüklendi: {args.model_save_path}")
    
    # Test DataLoader
    test_loader = create_data_loader(
        X_test, y_test, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Değerlendir
    evaluate_model(model, test_loader, device)


def predict_text(args):
    """Tek bir metin için tahmin yap"""
    print("\n" + "="*60)
    print("🔮 TEK METİN TAHMİNİ")
    print("="*60 + "\n")
    
    # Model yükle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = get_model_and_tokenizer(args.model_name, n_classes=2)
    
    if not os.path.exists(args.model_save_path):
        print(f"❌ Model bulunamadı: {args.model_save_path}")
        return
    
    model.load_state_dict(torch.load(args.model_save_path))
    print(f"✓ Model yüklendi\n")
    
    # Tahmin yap
    if args.text:
        text = args.text
    else:
        text = input("📝 Kontrol edilecek haberi gir:\n> ")
    
    prediction, confidence = predict(model, tokenizer, text, device)
    
    print("\n" + "-"*60)
    print("📊 SONUÇ:")
    print("-"*60)
    print(f"Metin: {text[:200]}...")
    print(f"\nTahmin: {'🔴 SAHTE HABER' if prediction == 1 else '✅ GERÇEK HABER'}")
    print(f"Güven: %{confidence*100:.2f}")
    print("-"*60 + "\n")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Sahte Haber Tespiti')
    
    # Temel argümanlar
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'predict'],
                       help='Çalışma modu')
    
    # Veri argümanları
    parser.add_argument('--data_path', type=str,
                       default='data/raw/WELFake_Dataset.csv',
                       help='Veri seti yolu')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Metin sütunu adı')
    parser.add_argument('--label_column', type=str, default='label',
                       help='Etiket sütunu adı')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Hızlı test için örnek sayısı (None=tümü)')
    
    # Model argümanları
    parser.add_argument('--model_name', type=str,
                       default='bert-base-uncased',
                       help='BERT model adı')
    parser.add_argument('--model_save_path', type=str,
                       default='models/best_model.pt',
                       help='Model kayıt yolu')
    
    # Eğitim argümanları
    parser.add_argument('--epochs', type=int, default=3,
                       help='Epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch boyutu')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Öğrenme oranı')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maksimum token uzunluğu')
    
    # Tahmin argümanı
    parser.add_argument('--text', type=str, default=None,
                       help='Tahmin için metin')
    
    args = parser.parse_args()
    
    # Klasörleri oluştur
    ensure_directories()
    
    # Modu çalıştır
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'predict':
        predict_text(args)


if __name__ == "__main__":
    main()