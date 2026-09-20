"""
Model Eğitim ve Değerlendirme
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """Model eğitim sınıfı"""
    
    def __init__(self, model, device, learning_rate=2e-5):
        """
        Args:
            model: Eğitilecek model
            device: CPU veya GPU
            learning_rate: Öğrenme oranı
        """
        self.model = model
        self.device = device
        
        # Loss fonksiyonu (Cross Entropy)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer (AdamW - BERT için önerilen)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # Eğitim geçmişi
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"✓ Trainer hazır (Learning rate: {learning_rate})")
    
    def train_epoch(self, data_loader):
        """
        Bir epoch eğitim
        
        Args:
            data_loader: Eğitim DataLoader'ı
        Returns:
            Ortalama loss ve accuracy
        """
        self.model.train()
        
        losses = []
        correct_predictions = 0
        total_predictions = 0
        
        # Progress bar
        progress_bar = tqdm(data_loader, desc='Eğitim')
        
        for batch in progress_bar:
            # Batch'i device'a taşı
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Gradyanları sıfırla
            self.optimizer.zero_grad()
            
            # İleri geçiş
            outputs = self.model(input_ids, attention_mask)
            
            # Loss hesapla
            loss = self.criterion(outputs, labels)
            losses.append(loss.item())
            
            # Geri yayılım
            loss.backward()
            
            # Gradient clipping (patlayan gradyanları önle)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Parametreleri güncelle
            self.optimizer.step()
            
            # Accuracy hesapla
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            
            # Progress bar güncelle
            progress_bar.set_postfix({
                'loss': np.mean(losses),
                'acc': correct_predictions.double().item() / total_predictions
            })
        
        avg_loss = np.mean(losses)
        avg_acc = correct_predictions.double().item() / total_predictions
        
        return avg_loss, avg_acc
    
    def evaluate(self, data_loader):
        """
        Modeli değerlendir
        
        Args:
            data_loader: Değerlendirme DataLoader'ı
        Returns:
            Loss, accuracy, predictions, true labels
        """
        self.model.eval()
        
        losses = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Değerlendirme'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # İleri geçiş
                outputs = self.model(input_ids, attention_mask)
                
                # Loss hesapla
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())
                
                # Tahminleri topla
                _, preds = torch.max(outputs, dim=1)
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = np.mean(losses)
        avg_acc = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, avg_acc, all_predictions, all_labels
    
    def fit(self, train_loader, val_loader, epochs=3, save_path='models/best_model.pt'):
        """
        Modeli eğit
        
        Args:
            train_loader: Eğitim DataLoader
            val_loader: Validasyon DataLoader
            epochs: Epoch sayısı
            save_path: Modelin kaydedileceği yol
        """
        print(f"\n{'='*50}")
        print(f"EĞİTİM BAŞLIYOR - {epochs} Epoch")
        print(f"{'='*50}\n")
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            start_time = time.time()
            
            # Eğitim
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validasyon
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            # Geçmişi kaydet
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Süre hesapla
            elapsed = time.time() - start_time
            
            # Sonuçları yazdır
            print(f"\n⏱️  Süre: {elapsed:.2f}s")
            print(f"📊 Eğitim    - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"📊 Validasyon - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # En iyi modeli kaydet
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"💾 En iyi model kaydedildi! (Acc: {val_acc:.4f})")
        
        print(f"\n{'='*50}")
        print(f"✅ EĞİTİM TAMAMLANDI!")
        print(f"🏆 En iyi validasyon accuracy: {best_val_acc:.4f}")
        print(f"{'='*50}\n")
    
    def plot_history(self, save_path='training_history.png'):
        """Eğitim grafiklerini çiz"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss grafiği
        axes[0].plot(self.history['train_loss'], label='Eğitim Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Validasyon Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy grafiği
        axes[1].plot(self.history['train_acc'], label='Eğitim Acc', marker='o')
        axes[1].plot(self.history['val_acc'], label='Validasyon Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Grafikler kaydedildi: {save_path}")
        plt.close()


def evaluate_model(model, data_loader, device, class_names=['Gerçek', 'Sahte']):
    """
    Detaylı model değerlendirmesi
    
    Args:
        model: Değerlendirilecek model
        data_loader: Test DataLoader
        device: CPU veya GPU
        class_names: Sınıf isimleri
    """
    print(f"\n{'='*50}")
    print("DETAYLI MODEL DEĞERLENDİRME")
    print(f"{'='*50}\n")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Test'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Metrikler
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    print(f"📊 PERFORMANS METRİKLERİ:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report
    print(f"\n📋 DETAYLI RAPOR:")
    print(classification_report(all_labels, all_predictions, 
                                target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Confusion matrix kaydedildi: confusion_matrix.png")
    plt.close()
    
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    print("Bu dosya main.py tarafından kullanılır.")
    print("Doğrudan çalıştırmak için: python main.py")