import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_analysis_charts():
    # Dosya yolları
    data_path = "data/Training.csv"
    output_dir = "reports"
    
    # Rapor klasörü yoksa oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Veriyi Yükle
    if not os.path.exists(data_path):
        print(f"Hata: {data_path} bulunamadı!")
        return
    
    df = pd.read_csv(data_path)

    # --- GRAFİK 1: Hastalık Dağılımı ---
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Hastalıkların sayılarını hesapla ve sıralı çizdir
    order = df['prognosis'].value_counts().index
    sns.countplot(y='prognosis', data=df, palette='viridis', order=order)
    
    plt.title('Veri Setindeki Hastalık Kayıt Sayıları', fontsize=15, fontweight='bold')
    plt.xlabel('Kayıt Sayısı')
    plt.ylabel('Hastalık İsmi')
    plt.tight_layout()
    
    # Grafiği Kaydet
    plt.savefig(f'{output_dir}/hastalik_dagilimi.png')
    print(f"1. Grafik kaydedildi: {output_dir}/hastalik_dagilimi.png")

    # --- GRAFİK 2: Veri Yoğunluğu (Heatmap - İlk 20 Semptom) ---
    plt.figure(figsize=(14, 8))
    # Sadece ilk 20 semptomu ve ilk 50 satırı örnek olarak görselleştirelim
    subset = df.iloc[:50, :20] 
    sns.heatmap(subset, cmap="YlGnBu", cbar_kws={'label': 'Semptom Var(1) / Yok(0)'})
    
    plt.title('Semptom Varlığı Isı Haritası (Örnek Kesit)', fontsize=15, fontweight='bold')
    plt.xlabel('Semptomlar')
    plt.ylabel('Hasta Örnekleri')
    plt.tight_layout()
    
    # Grafiği Kaydet
    plt.savefig(f'{output_dir}/semptom_isi_haritasi.png')
    print(f"2. Grafik kaydedildi: {output_dir}/semptom_isi_haritasi.png")
    
    plt.show()

if __name__ == "__main__":
    generate_analysis_charts()