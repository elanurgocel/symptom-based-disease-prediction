from sklearn.ensemble import RandomForestClassifier
import joblib

def train_save_model(X_train, y_train, model_path):
    # 100 farklı karar ağacından oluşan bir orman kuruyoruz
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Modeli eğitim verisiyle besle
    model.fit(X_train, y_train)
    
    # Eğitilen modeli daha sonra kullanmak üzere models/ klasörüne kaydet
    joblib.dump(model, model_path)
    print(f"--- Model başarıyla eğitildi ve kaydedildi: {model_path} ---")
    return model