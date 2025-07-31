import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

st.set_page_config(
    page_title="📊 Model Dashboard",
    page_icon="📊",
    layout="wide"
)

def load_data():
    """Загрузка данных для анализа"""
    # Обработанные данные игр
    games_df = pd.read_csv('data/games_processed.csv') if os.path.exists('data/games_processed.csv') else None
    
    # Логи предсказаний
    logs_df = pd.read_csv('logs/logs.csv') if os.path.exists('logs/logs.csv') else None
    
    return games_df, logs_df

def main():
    st.title("📊 Game Rating Classifier Dashboard")
    
    games_df, logs_df = load_data()
    
    if games_df is None:
        st.error("Данные не найдены! Сначала подготовьте датасет.")
        return
    
    # Общая статистика
    st.subheader("🎮 Статистика датасета")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Всего игр", len(games_df))
    
    with col2:
        avg_text_length = games_df['combined_text'].str.len().mean()
        st.metric("Средняя длина текста", f"{avg_text_length:.0f}")
    
    with col3:
        unique_ratings = games_df['overall_player_rating'].nunique()
        st.metric("Уникальных рейтингов", unique_ratings)
    
    with col4:
        if logs_df is not None:
            st.metric("Всего предсказаний", len(logs_df))
        else:
            st.metric("Всего предсказаний", 0)
    
    # Распределение рейтингов в датасете
    st.subheader("📈 Распределение рейтингов в датасете")
    
    col5, col6 = st.columns(2)
    
    with col5:
        rating_counts = games_df['overall_player_rating'].value_counts()
        fig_pie = px.pie(values=rating_counts.values, names=rating_counts.index,
                        title="Распределение рейтингов")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col6:
        fig_bar = px.bar(x=rating_counts.index, y=rating_counts.values,
                        title="Количество игр по рейтингам")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Анализ предсказаний
    if logs_df is not None and len(logs_df) > 0:
        st.subheader("🔮 Анализ предсказаний модели")
        
        col7, col8 = st.columns(2)
        
        with col7:
            # Распределение предсказанных рейтингов
            pred_counts = logs_df['predicted_rating'].value_counts()
            fig_pred = px.bar(x=pred_counts.index, y=pred_counts.values,
                            title="Предсказанные рейтинги")
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col8:
            # Распределение уверенности модели
            fig_conf = px.histogram(logs_df, x='confidence', nbins=20,
                                  title="Распределение уверенности модели")
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Таблица последних предсказаний
        st.subheader("🕒 Последние предсказания")
        recent_predictions = logs_df.tail(10)
        st.dataframe(recent_predictions, use_container_width=True)
    
    # Примеры игр по рейтингам
    st.subheader("🎯 Примеры игр по рейтингам")
    
    for rating in games_df['overall_player_rating'].unique():
        with st.expander(f"Примеры игр с рейтингом: {rating}"):
            rating_games = games_df[games_df['overall_player_rating'] == rating].head(3)
            for _, game in rating_games.iterrows():
                st.write(f"**{game['name']}**")
                st.write(f"Описание: {game['combined_text'][:200]}...")
                st.write("---")

if __name__ == "__main__":
    main()