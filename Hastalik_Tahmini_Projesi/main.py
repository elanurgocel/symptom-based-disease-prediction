import os
import streamlit as st
import pandas as pd
import numpy as np
from src.data_preprocessing import load_and_clean_data
from src.train_model import train_save_model

# 1. Sayfa Yapılandırması
st.set_page_config(page_title="Hastalık Teşhis Sistemi", layout="wide", page_icon="🩺")

# --- TÜRKÇELEŞTİRME SÖZLÜKLERİ ---

TR_SYMPTOMS = {
    "itching": "Kaşıntı", "skin_rash": "Cilt Döküntüsü", "nodal_skin_eruptions": "Düğümlü Cilt Kabarcıkları",
    "continuous_sneezing": "Sürekli Hapşırma", "shivering": "Titreme", "chills": "Üşüme",
    "joint_pain": "Eklem Ağrısı", "stomach_pain": "Karın Ağrısı", "acidity": "Mide Yanması",
    "ulcers_on_tongue": "Dilde Yaralar", "muscle_wasting": "Kas Erimesi", "vomiting": "Kusma",
    "burning_micturition": "İdrarda Yanma", "spotting_ urination": "İdrarda Leke", "fatigue": "Halsizlik",
    "weight_gain": "Kilo Alımı", "anxiety": "Anksiyete", "cold_hands_and_feets": "El ve Ayak Soğukluğu",
    "mood_swings": "Ruh Hali Değişimleri", "weight_loss": "Kilo Kaybı", "restlessness": "Huzursuzluk",
    "lethargy": "Uyuşukluk", "patches_in_throat": "Boğazda Lekeler", "irregular_sugar_level": "Düzensiz Şeker",
    "cough": "Öksürük", "high_fever": "Yüksek Ateş", "sunken_eyes": "Çökük Gözler", "breathlessness": "Nefes Darlığı",
    "sweating": "Terleme", "dehydration": "Sıvı Kaybı", "indigestion": "Hazımsızlık", "headache": "Baş Ağrısı",
    "yellowish_skin": "Ciltte Sararma", "dark_urine": "Koyu Renkli İdrar", "nausea": "Bulantı",
    "loss_of_ appetite": "İştah Kaybı", "pain_behind_the_eyes": "Göz Arkası Ağrısı", "back_pain": "Sırt Ağrısı",
    "constipation": "Kabızlık", "abdominal_pain": "Karın Ağrısı", "diarrhoea": "İshal", "mild_fever": "Hafif Ateş",
    "yellow_urine": "Sarı İdrar", "yellowing_of_eyes": "Gözlerde Sararma", "acute_liver_failure": "Akut Karaciğer Yetmezliği",
    "fluid_overload": "Sıvı Birikmesi", "swelling_of_stomach": "Mide Şişkinliği", "swelled_lymph_nodes": "Şişmiş Lenf Bezleri",
    "malaise": "Kırgınlık", "blurred_and_distorted_vision": "Bulanık Görme", "phlegm": "Balgam",
    "throat_irritation": "Boğaz Tahrişi", "redness_of_eyes": "Gözlerde Kızarıklık", "sinus_pressure": "Sinüs Basıncı",
    "runny_nose": "Burun Akıntısı", "congestion": "Burun Tıkanıklığı", "chest_pain": "Göğüs Ağrısı",
    "weakness_in_limbs": "Uzuvlarda Güçsüzlük", "fast_heart_rate": "Hızlı Kalp Atışı",
    "pain_during_bowel_movements": "Dışkılama Sırasında Ağrı", "pain_in_anal_region": "",
    "bloody_stool": "Kanlı Dışkı", "irritation_in_anus": "Anüste Tahriş", "neck_pain": "Boyun Ağrısı",
    "dizziness": "Baş Dönmesi", "cramps": "Kramplar", "bruising": "Morarma", "obesity": "Obezite",
    "swollen_legs": "Şişmiş Bacaklar", "swollen_blood_vessels": "Şişmiş Kan Damarları", "puffy_face_and_eyes": "Şişmiş Yüz ve Gözler",
    "enlarged_thyroid": "Tiroid Büyümesi", "brittle_nails": "Kırılgan Tırnaklar", "swollen_extremeties": "Şişmiş Ekstremiteler",
    "excessive_hunger": "Aşırı Açlık", "extra_marital_contacts": "Evlilik Dışı Temas", "drying_and_tingling_lips": "Dudak Kuruması",
    "slurred_speech": "Pektek Konuşma", "knee_pain": "Diz Ağrısı", "hip_joint_pain": "Kalça Eklemi Ağrısı",
    "muscle_weakness": "Kas Güçsüzlüğü", "stiff_neck": "Boyun Tutulması", "swelling_joints": "Eklem Şişmesi",
    "movement_stiffness": "Hareket Zorluğu", "spinning_movements": "Dönme Hissi", "loss_of_balance": "Denge Kaybı",
    "unsteadiness": "Dengesizlik", "weakness_of_one_body_side": "Vücudun Bir Yanında Güçsüzlük", "loss_of_smell": "Koku Kaybı",
    "bladder_discomfort": "Mesane Rahatsızlığı", "foul_smell_of urine": "İdrarda Kötü Koku", "continuous_feel_of_urine": "Sürekli İdrar Hissi",
    "passage_of_gases": "Gaz Çıkarma", "internal_itching": "İç Kaşıntı", "toxic_look_(typhos)": "Toksik Görünüm",
    "depression": "Depresyon", "irritability": "Sinirlilik", "muscle_pain": "Kas Ağrısı", "altered_sensorium": "Bilinç Değişikliği",
    "red_spots_over_body": "Vücutta Kırmızı Lekeler", "belly_pain": "Karın Ağrısı", "abnormal_menstruation": "Anormal Adet Döngesi",
    "dischromic _patches": "Ciltte Renk Değişimi", "watering_from_eyes": "Göz Sulanması", "increased_appetite": "İştah Artışı",
    "polyuria": "Aşırı İdrar", "family_history": "", "mucoid_sputum": "Mukuslu Balgam",
    "rusty_sputum": "Paslı Balgam", "lack_of_concentration": "Konsantrasyon Eksikliği", "visual_disturbances": "Görsel Bozukluklar",
    "receiving_blood_transfusion": "Kan Nakli Alımı", "receiving_unsterile_injections": "Steril Olmayan Enjeksiyon",
    "coma": "Koma", "stomach_bleeding": "Mide Kanaması", "distention_of_abdomen": "Karın Şişkinliği",
    "history_of_alcohol_consumption": "Alkol Tüketim ", "fluid_overload.1": "Sıvı Birikmesi",
    "blood_in_sputum": "Balgamda Kan", "prominent_veins_on_calf": "Baldırda Belirgin Damarlar", "palpitations": "Çarpıntı",
    "painful_walking": "Ağrılı Yürüyüş", "pus_filled_pimples": "İrinli Sivilceler", "blackheads": "Siyah Noktalar",
    "scurring": "İz Kalması", "skin_peeling": "Cilt Soyulması", "silver_like_dusting": "Gümüşsu Tozlanma",
    "small_dents_in_nails": "Tırnaklarda Küçük Çukurlar", "inflammatory_nails": "İltihaplı Tırnaklar", "blister": "Kabarcık",
    "red_sore_around_nose": "Burun Çevresinde Kırmızı Yaralar", "yellow_crust_ooze": "Sarı Kabuklanma"
}

TR_DISEASES = {
    "Fungal infection": "Mantar Enfeksiyonu", "Allergy": "Alerji", "GERD": "GÖRH (Reflü)", "Chronic cholestasis": "Kronik Kolestaz",
    "Drug Reaction": "İlaç Reaksiyonu", "Peptic ulcer diseae": "Peptik Ülser", "AIDS": "AIDS", "Diabetes ": "Diyabet",
    "Gastroenteritis": "Gastroenterit (Mide Gribi)", "Bronchial Asthma": "Bronşiyal Astım", "Hypertension ": "Hipertansiyon",
    "Migraine": "Migren", "Cervical spondylosis": "Servikal Spondiloz", "Paralysis (brain hemorrhage)": "Felç (Beyin Kanaması)",
    "Jaundice": "Sarılık", "Malaria": "Sıtma", "Chicken pox": "Su Çiçeği", "Dengue": "Dang Humması", "Typhoid": "Tifo",
    "hepatitis A": "Hepatit A", "Hepatitis B": "Hepatit B", "Hepatitis C": "Hepatit C", "Hepatitis D": "Hepatit D",
    "Hepatitis E": "Hepatit E", "Alcoholic hepatitis": "Alkolik Hepatit", "Tuberculosis": "Tüberküloz (Verem)",
    "Common Cold": "Grip / Soğuk Algınlığı", "Pneumonia": "Zatürre", "Dimorphic hemmorhoids(piles)": "Hemoroid (Basur)",
    "Heart attack": "Kalp Krizi", "Varicose veins": "Varisli Damarlar", "Hypothyroidism": "Hipotiroidizm",
    "Hyperthyroidism": "Hipertiroidizm", "Hypoglycemia": "Hipoglisemi", "Osteoarthristis": "Osteoartrit",
    "Arthritis": "Artrit", "(vertigo) Paroymsal  Positional Vertigo": "Vertigo", "Acne": "Akne / Sivilce",
    "Urinary tract infection": "İdrar Yolu Enfeksiyonu", "Psoriasis": "Sedef Hastalığı", "Impetigo": "İmpetigo"
}

INV_TR_SYMPTOMS = {v: k for k, v in TR_SYMPTOMS.items()}

@st.cache_resource
def initialize_system():
    TRAIN_PATH = "data/Training.csv"
    TEST_PATH = "data/Testing.csv"
    MODEL_PATH = "models/random_forest_model.pkl"
    if not os.path.exists("models"): os.makedirs("models")
    X_train, X_test, y_train, y_test, le = load_and_clean_data(TRAIN_PATH, TEST_PATH)
    model = train_save_model(X_train, y_train, MODEL_PATH)
    symptoms_en = list(X_train.columns)
    
    symptoms_tr_all = sorted([TR_SYMPTOMS.get(s, s) for s in symptoms_en if TR_SYMPTOMS.get(s, s) != "Aile Öyküsü"])
    if "Ciltte Renk Değişimi" in symptoms_tr_all:
        symptoms_tr_all.remove("Ciltte Renk Değişimi")
        symptoms_tr_all.insert(0, "Ciltte Renk Değişimi")
    
    return model, le, symptoms_en, symptoms_tr_all

def main():
    try:
        model, le, symptoms_en, symptoms_tr = initialize_system()
    except Exception as e:
        st.error(f"Hata: {e}"); return

    # Başlıktan AI yazısı kaldırıldı
    st.title(" Klinik Teşhis Paneli")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Semptom Seçimi")
        selected_tr = st.multiselect("Belirtileri Seçin:", options=symptoms_tr)

    with col2:
        st.subheader("🔍 Teşhis Sonucu")
        if st.button("Teşhis Koy", type="primary", use_container_width=True):
            if selected_tr:
                with st.spinner('Analiz ediliyor...'):
                    input_vec = np.zeros(len(symptoms_en))
                    for s in selected_tr:
                        s_en = INV_TR_SYMPTOMS.get(s, s)
                        if s_en in symptoms_en: input_vec[symptoms_en.index(s_en)] = 1
                    
                    pred_en = le.inverse_transform(model.predict([input_vec]))[0]
                    pred_tr = TR_DISEASES.get(pred_en, pred_en)
                    
                    st.success(f"### Olası Teşhis: **{pred_tr}**")
                    st.info("💡 **Not:** Bu araç bir tahminidir, profesyonel tıbbi muayene yerine geçmez.")
            else:
                st.warning("Lütfen analiz için en az bir belirti seçiniz.")

if __name__ == "__main__":
    main()