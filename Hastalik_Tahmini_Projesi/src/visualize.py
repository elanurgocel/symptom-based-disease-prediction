import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_importance():
    # Dosya yolları
    model_path = "models/random_forest_model.pkl"
    train_path = "data/Training.csv"

    # Kontroller
    if not os.path.exists(model_path) or not os.path.exists(train_path):
        print("Hata: Model veya eğitim verisi bulunamadı. Lütfen önce main.py'yi çalıştırın.")
        return

    # 1. Modeli ve veriyi yükle
    model = joblib.load(model_path)
    train_data = pd.read_csv(train_path)
    X = train_data.drop('prognosis', axis=1)
    
    # 2. Özellik önem derecelerini al
    importances = model.feature_importances_
    feature_names = X.columns
    
    # Verileri bir DataFrame'de topla ve en önemli 15 tanesini seç
    feature_df = pd.DataFrame({'Semptom': feature_names, 'Önem': importances})
    feature_df = feature_df.sort_values(by='Önem', ascending=False).head(15)
    
    # 3. Görselleştirme
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Önem', y='Semptom', data=feature_df, palette='magma')
    plt.title('Hastalık Teşhisinde En Önemli 15 Belirti', fontsize=15)
    plt.xlabel('Önem Skoru', fontsize=12)
    plt.ylabel('Semptomlar', fontsize=12)
    plt.tight_layout()
    
    # Grafiği göster
    print("Grafik oluşturuluyor...")
    plt.show()

if __name__ == "__main__":
    plot_feature_importance()