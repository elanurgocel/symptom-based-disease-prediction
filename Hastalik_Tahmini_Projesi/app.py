import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# --- HASTALIK SÖZLÜĞÜ (İNGİLİZCE -> TÜRKÇE) ---
hastalik_sozlugu = {
    "Fungal infection": "Mantar Enfeksiyonu", "Allergy": "Alerji", "GERD": "Gastroözofageal Reflü",
    "Chronic cholestasis": "Kronik Kolestaz", "Drug Reaction": "İlaç Reaksiyonu", "Peptic ulcer diseae": "Peptik Ülser",
    "AIDS": "AIDS", "Diabetes ": "Diyabet", "Gastroenteritis": "Gastroenterit (Mide Üşütmesi)",
    "Bronchial Asthma": "Bronşiyal Astım", "Hypertension ": "Hipertansiyon (Yüksek Tansiyon)",
    "Migraine": "Migren", "Cervical spondylosis": "Servikal Spondiloz (Boyun Kireçlenmesi)",
    "Paralysis (brain hemorrhage)": "Felç (Beyin Kanaması)", "Jaundice": "Sarılık",
    "Malaria": "Sıtma", "Chicken pox": "Su Çiçeği", "Dengue": "Dang Humması", "Typhoid": "Tifo",
    "hepatitis A": "Hepatit A", "Hepatitis B": "Hepatit B", "Hepatitis C": "Hepatit C",
    "Hepatitis D": "Hepatit D", "Hepatitis E": "Hepatit E", "Alcoholic hepatitis": "Alkolik Hepatit",
    "Tuberculosis": "Tüberküloz (Verem)", "Common Cold": "Grip / Soğuk Algınlığı", "Pneumonia": "Zatürre",
    "Dimorphic hemmorhoids(piles)": "Hemoroid (Basur)", "Heart attack": "Kalp Krizi",
    "Varicose veins": "Varisli Damarlar", "Hypothyroidism": "Hipotiroidizm", "Hyperthyroidism": "Hipertiroidizm",
    "Hypoglycemia": "Hipoglisemi (Düşük Kan Şekeri)", "Osteoarthristis": "Osteoartrit (Kireçlenme)",
    "Arthritis": "Artrit (Eklem İltihabı)", "(vertigo) Paroymsal  Positional Vertigo": "Vertigo (Baş Dönmesi)",
    "Acne": "Akne / Sivilce", "Urinary tract infection": "İdrar Yolu Enfeksiyonu", "Psoriasis": "Sedef Hastalığı",
    "Impetigo": "İmpetigo (Deri Enfeksiyonu)"
}

# --- SEMPTOM SÖZLÜĞÜ (İNGİLİZCE -> TÜRKÇE) ---
semptom_sozlugu = {
    "itching": "Kaşıntı", "skin_rash": "Cilt Döküntüsü", "nodal_skin_eruptions": "Ciltte Kabartılar",
    "continuous_sneezing": "Sürekli Hapşırma", "shivering": "Titreme", "chills": "Üşüme",
    "joint_pain": "Eklem Ağrısı", "stomach_pain": "Karın Ağrısı", "acidity": "Mide Yanması",
    "ulcers_on_tongue": "Dil Yaraları", "muscle_wasting": "Kas Erimesi", "vomiting": "Kusma",
    "burning_micturition": "İdrarda Yanma", "spotting_urination": "İdrarda Lekelenme", "fatigue": "Halsizlik",
    "weight_gain": "Kilo Alımı", "anxiety": "Kaygı / Anksiyete", "cold_hands_and_feets": "El/Ayak Soğukluğu",
    "mood_swings": "Ruh Hali Değişimi", "weight_loss": "Kilo Kaybı", "restlessness": "Huzursuzluk",
    "lethargy": "Uyuşukluk", "patches_in_throat": "Boğazda Lekeler", "irregular_sugar_level": "Düzensiz Şeker",
    "cough": "Öksürük", "high_fever": "Yüksek Ateş", "sunken_eyes": "Çökük Gözler", "breathlessness": "Nefes Darlığı",
    "sweating": "Terleme", "dehydration": "Susuz Kalma", "indigestion": "Hazımsızlık", "headache": "Baş Ağrısı",
    "yellowish_skin": "Ciltte Sararma", "dark_urine": "Koyu Renkli İdrar", "nausea": "Bulantı",
    "loss_of_appetite": "İştah Kaybı", "pain_behind_the_eyes": "Göz Arkasında Ağrı", "back_pain": "Sırt Ağrısı",
    "constipation": "Kabızlık", "abdominal_pain": "Karın Bölgesi Ağrısı", "diarrhoea": "İshal",
    "mild_fever": "Hafif Ateş", "yellow_urine": "Sarı İdrar", "yellowing_of_eyes": "Gözlerde Sararma",
    "acute_liver_failure": "Karaciğer Yetmezliği", "swelling_of_stomach": "Mide Şişkinliği",
    "swelled_lymph_nodes": "Şişmiş Lenf Bezleri", "malaise": "Kırgınlık", "phlegm": "Balgam",
    "chest_pain": "Göğüs Ağrısı", "dizziness": "Baş Dönmesi", "excessive_hunger": "Aşırı Açlık",
    "muscle_pain": "Kas Ağrısı", "red_spots_over_body": "Vücutta Kırmızı Lekeler"
}

st.set_page_config(page_title="Yapay Zeka Teşhis", page_icon="🩺")

st.title("🩺 Yapay Zeka Destekli Hastalık Teşhis Sistemi")
st.markdown("---")

@st.cache_resource
def load_resources():
    model = joblib.load("models/random_forest_model.pkl")
    train_data = pd.read_csv("data/Training.csv")
    train_cols = train_data.drop('prognosis', axis=1).columns
    le = LabelEncoder()
    le.fit(train_data['prognosis'])
    return model, train_cols, le

try:
    model, train_cols, le = load_resources()

    st.subheader("Lütfen Belirtilerinizi Seçiniz")
    
    # Semptomları Türkçeleştir
    turkce_liste = {semptom_sozlugu.get(col, col.replace('_', ' ').title()): col for col in train_cols}
    secilen_turkce = st.multiselect("Belirtiler:", list(turkce_liste.keys()))

    if st.button("Teşhis Koy"):
        if secilen_turkce:
            selected_eng = [turkce_liste[s] for s in secilen_turkce]
            input_df = pd.DataFrame(0, index=[0], columns=train_cols)
            for s in selected_eng:
                input_df[s] = 1
            
            prediction_idx = model.predict(input_df)[0]
            hastalik_eng = le.inverse_transform([prediction_idx])[0]
            
            # Hastalık sonucunu Türkçeleştir
            hastalik_tr = hastalik_sozlugu.get(hastalik_eng.strip(), hastalik_eng)

            st.success(f"### Olası Teşhis: **{hastalik_tr}**")
            st.warning("⚠️ **Uyarı:** Bu sonuç sadece bilgilendirme amaçlıdır. Kesin tanı için bir doktora başvurunuz.")
        else:
            st.info("Lütfen en az bir belirti seçin.")

except Exception as e:
    st.error("Model yüklenemedi. Lütfen önce terminalden 'main.py' çalıştırın.")