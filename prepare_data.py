import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import os

def clean_text(text):
    """Очистка текста для лучшей обработки"""
    if pd.isna(text):
        return ""
    # Удаляем HTML теги
    text = re.sub(r'<[^>]+>', '', str(text))
    # Удаляем лишние пробелы
    text = re.sub(r'\\s+', ' ', text)
    # Приводим к нижнему регистру
    text = text.lower().strip()
    return text

def prepare_dataset():
    """Подготовка датасета для обучения модели"""
    
    # Загрузка данных
    print("Загрузка games_description.csv...")
    df = pd.read_csv('games_description.csv')
    
    print(f"Загружено {len(df)} игр")
    
    # Очистка данных
    df = df.dropna(subset=['overall_player_rating'])
    
    # Создание текстовых признаков из описаний
    df['combined_text'] = (
        df['short_description'].fillna('').astype(str) + ' ' +
        df['long_description'].fillna('').astype(str) + ' ' +
        df['genres'].fillna('').astype(str)
    )
    
    # Очистка текста
    df['combined_text'] = df['combined_text'].apply(clean_text)
    
    # Фильтрация коротких текстов
    df = df[df['combined_text'].str.len() > 10]
    
    # Кодирование целевой переменной
    rating_mapping = {
        'Overwhelmingly Positive': 5,
        'Very Positive': 4,
        'Overwhelmingly Positive': 4,
        'Mostly Positive': 3,
        'Mixed': 2,
        'Mostly Negative': 1,
        'Very Negative': 0
    }
    
    df['rating_encoded'] = df['overall_player_rating'].map(rating_mapping)
    df = df.dropna(subset=['rating_encoded'])
    
    # Сохранение обработанных данных
    processed_df = df[['name', 'combined_text', 'overall_player_rating', 'rating_encoded']].copy()
    
    os.makedirs('data', exist_ok=True)
    processed_df.to_csv('data/games_processed.csv', index=False)
    
    print(f"Обработано {len(processed_df)} игр")
    print("Распределение рейтингов:")
    print(processed_df['overall_player_rating'].value_counts())
    
    return processed_df

if __name__ == "__main__":
    prepare_dataset()