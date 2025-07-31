import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os

# Настройка страницы
st.set_page_config(
    page_title="🎮 Game Rating Classifier",
    page_icon="🎮",
    layout="wide"
)

# Загрузка модели и векторизатора
@st.cache_resource
def load_model():
    """Загрузка обученной модели и векторизатора"""
    try:
        model = joblib.load('model/model.joblib')
        vectorizer = joblib.load('model/vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Модель не найдена! Сначала запустите обучение модели.")
        return None, None

def predict_rating(text, model, vectorizer):
    """Предсказание рейтинга для текста"""
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    rating_mapping = {
        0: 'Very Negative',
        1: 'Mostly Negative',
        2: 'Mixed',
        3: 'Mostly Positive',
        4: 'Very Positive'
    }

    predicted_rating = rating_mapping[prediction]
    confidence = float(np.max(probabilities))

    return predicted_rating, confidence, probabilities

def log_prediction(game_title, comment, predicted_rating, confidence):
    """Логирование предсказаний"""
    os.makedirs('logs', exist_ok=True)

    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'game_title': game_title,
        'comment': comment,
        'predicted_rating': predicted_rating,
        'confidence': confidence
    }

    log_df = pd.DataFrame([log_data])

    if os.path.exists('logs/logs.csv'):
        log_df.to_csv('logs/logs.csv', mode='a', header=False, index=False)
    else:
        log_df.to_csv('logs/logs.csv', index=False)

def main():
    st.title("🎮 Game Rating Classifier")
    st.markdown("### Классификация отзывов игр по overall player rating")

    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Введите данные об игре")

        game_title = st.text_input(
            "Название игры (опционально):",
            placeholder="например: Counter-Strike 2"
        )

        comment = st.text_area(
            "Комментарий или описание игры:",
            placeholder="Введите описание игры, жанры, особенности...",
            height=150
        )

        if st.button("🔮 Предсказать рейтинг", type="primary"):
            if comment.strip():
                with st.spinner("Анализируем..."):
                    predicted_rating, confidence, probabilities = predict_rating(
                        comment, model, vectorizer
                    )

                log_prediction(game_title or "Unnamed Game", comment,
                               predicted_rating, confidence)

                with col2:
                    st.subheader("🎯 Результат")

                    color_mapping = {
                        'Overwhelmingly Positive': 'blue',
                        'Very Positive': 'green',
                        'Mostly Positive': 'lightgreen',
                        'Mixed': 'orange',
                        'Mostly Negative': 'lightcoral',
                        'Very Negative': 'red'
                    }

                    color = color_mapping.get(predicted_rating, 'gray')

                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; 
                                background-color: {color}; color: white; text-align: center;">
                        <h3>Предсказанный рейтинг:</h3>
                        <h2>{predicted_rating}</h2>
                        <p>Уверенность: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.subheader("📊 Распределение вероятностей")

                    rating_names = ['Very Negative', 'Mostly Negative', 'Mixed',
                                    'Mostly Positive', 'Very Positive', 'Overwhelmingly Positive']

                    prob_df = pd.DataFrame({
                        'Рейтинг': rating_names,
                        'Вероятность': probabilities
                    })

                    st.bar_chart(prob_df.set_index('Рейтинг'))
            else:
                st.warning("Пожалуйста, введите описание игры!")

    # Секция статистики логов
    st.markdown("---")
    st.subheader("📊 Статистика логов")

    if os.path.exists('logs/logs.csv'):
        try:
            logs_df = pd.read_csv('logs/logs.csv', names=[
                'timestamp', 'game_title', 'comment', 'predicted_rating', 'confidence'
            ], header=None)

            st.metric("Всего предсказаний", len(logs_df))

            rating_counts = logs_df['predicted_rating'].value_counts()
            st.bar_chart(rating_counts)

            st.subheader("🕒 Последние предсказания")
            recent_logs = logs_df.tail(5)[['timestamp', 'game_title', 'predicted_rating', 'confidence']]
            st.dataframe(recent_logs, use_container_width=True)

        except Exception as e:
            st.warning(f"Ошибка при обработке логов: {e}")
    else:
        st.info("Лог-файл не найден.")

if __name__ == "__main__":
    main()
