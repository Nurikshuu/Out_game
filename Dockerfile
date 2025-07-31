FROM python:3.9-slim

WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта
COPY . .

# Создаем необходимые папки
RUN mkdir -p logs data model

# Экспозируем порт для Streamlit
EXPOSE 8501

# Команда для запуска приложения
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]