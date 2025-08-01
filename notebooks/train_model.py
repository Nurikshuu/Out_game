import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

def train_model():
    """Обучение модели классификации рейтингов игр"""
    
    # Загрузка обработанных данных
    print("Загрузка обработанных данных...")
    df = pd.read_csv('data/games_processed.csv')
    
    # Подготовка данных
    X = df['combined_text']
    y = df['rating_encoded']
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    # Создание pipeline
    print("Обучение модели...")
    
    # TF-IDF векторизация
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    # Логистическая регрессия
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    # Обучение векторизатора и модели
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model.fit(X_train_tfidf, y_train)
    
    # Оценка модели
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    
    print("\\nРезультаты на тестовой выборке:")
    print(f"Точность: {accuracy_score(y_test, y_pred):.4f}")
    print("\\nОтчет по классификации:")
    
    # Mapping для читаемых названий классов
    class_names = {0: 'Very Negative', 1: 'Mostly Negative', 2: 'Mixed', 
                   3: 'Mostly Positive', 4: 'Very Positive'}
    
    # Определяем какие классы реально присутствуют в данных
    unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names = [class_names[i] for i in unique_classes]
    
    print(f"Найденные классы в данных: {unique_classes}")
    print(f"Соответствующие названия: {target_names}")
    
    print(classification_report(y_test, y_pred, 
                              labels=unique_classes,
                              target_names=target_names))
    
    # Сохранение модели и векторизатора
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.joblib')
    joblib.dump(vectorizer, 'model/vectorizer.joblib')
    
    print("\\nМодель и векторизатор сохранены в папку model/")
    
    return model, vectorizer

if __name__ == "__main__":
    train_model()