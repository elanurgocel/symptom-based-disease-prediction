import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(train_path, test_path):
    # Verileri yükle
    train = pd.read_csv(train_path).dropna(axis=1, how='all')
    test = pd.read_csv(test_path).dropna(axis=1, how='all')
    
    # Özellik ve Hedef ayrımı
    X_train = train.drop('prognosis', axis=1)
    y_train = train['prognosis']
    X_test = test.drop('prognosis', axis=1)
    y_test = test['prognosis']
    
    # Hastalık isimlerini sayısal verilere dönüştür (Label Encoding)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    return X_train, X_test, y_train, y_test, le