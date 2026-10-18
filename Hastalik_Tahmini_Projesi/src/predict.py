import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def make_prediction(symptoms_list):
    # Modeli yükle
    model = joblib.load("models/random_forest_model.pkl")
    
    # Sütun isimlerini al
    df_train = pd.read_csv("data/Training.csv")
    train_cols = df_train.drop('prognosis', axis=1).columns
    
    # Giriş verisini hazırla
    input_data = pd.DataFrame(0, index=[0], columns=train_cols)
    for s in symptoms_list:
        if s in train_cols:
            input_data[s] = 1
            
    # Tahmin yap
    prediction_idx = model.predict(input_data)[0]
    
    # Etiketleri çevir
    le = LabelEncoder()
    le.fit(df_train['prognosis'])
    result = le.inverse_transform([prediction_idx])[0]
    return result

# --- STREAMLIT ARAYÜZÜ ---
st.set_page_config(page_title="Hastalık Teşhis Portalı", layout="centered")

st.title("🏥 Hastalık Teşhis Portalı")
st.write("Makine öğrenmesi tabanlı semptom analizi yapmak için semptomlarınızı seçiniz.")

# Eğitim verisindeki tüm semptom listesini çekelim (Kullanıcının seçmesi için)
df_train = pd.read_csv("data/Training.csv")
all_symptoms = df_train.drop('prognosis', axis=1).columns.tolist()

# Kullanıcı arayüzü bileşenleri
selected_symptoms = st.multiselect(
    "Hangi semptomları yaşıyorsunuz?",
    options=all_symptoms,
    help="Birden fazla seçim yapabilirsiniz."
)

if st.button("Sisteme Giriş Yap / Tahmin Et"):
    if len(selected_symptoms) > 0:
        try:
            tahmin = make_prediction(selected_symptoms)
            st.success(f"**[TEŞHİS]:** Belirtileriniz **'{tahmin}'** hastalığına işaret ediyor olabilir.")
            st.warning("⚠️ Lütfen kesin tanı için bir hekime danışınız.")
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
    else:
        st.info("Lütfen en az bir semptom seçiniz.")