🎮 Out_game

Проект для предсказания overall player rating игр Steam на основе их описаний и характеристик.

## 📁 Структура проекта

```
project/
├── app/
│   ├── streamlit_app.py   # Основное веб-приложение
│   └── dashboard.py       # Дашборд анализа модели
├── data/
│   └── games_processed.csv # Обработанные данные (создается автоматически)
├── logs/
│   └── logs.csv           # Сохраненные предсказания пользователей
├── model/
│   ├── model.joblib       # Обученная модель Logistic Regression
│   └── vectorizer.joblib  # TF-IDF векторизатор
├── notebooks/
│   └── train_model.py     # Скрипт для обучения модели
├── prepare_data.py        # Скрипт обработки исходного датасета
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── EDA.ipynb              # Анализ датасета
├── docker-compose.yml
├── games_description.csv  # Исходный датасет 
└── README.md
```

## 🛠 Инструкции по установке

**1. Клонируйте репозиторий**

**2. Создайте виртуальное окружение**

```bash
python -m venv venv
source venv/bin/activate        # На Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

**3. Подготовьте данные**

```bash
python prepare_data.py
```

**4. Обучите модель**

```bash
python notebooks/train_model.py
```

**5. Запустите приложение**

```bash
streamlit run app/streamlit_app.py
```

Откройте в браузере: http://localhost:8501

## 📊 Дополнительно: # Docker запуск

```bash
docker-compose build
docker-compose up -d
```
## 📊 Дополнительно: # EDA.ipynb

```bash
pip install jupyterlab
jupyter lab
# Или можете в вс коде разницы нет
```

Дашборд показывает:
* Общую статистику датасета
* Распределение рейтингов
* Анализ предсказаний модели
* Распределение уверенности
* Примеры игр по рейтингам

## 🎯 Классы для классификации

Модель предсказывает следующие рейтинги:
* **Very Positive**
* **Overwhelmingly Positive**
* **Mostly Positive**
* **Mixed**
* **Mostly Negative**
* **Very Negative**

## 🔧 Технические детали

* **Векторизация**: TF-IDF с биграммами
* **Модель**: Logistic Regression с balanced class weights
* **Features**: Объединение short_description + long_description + genres
* **Метрики**: Accuracy, Classification Report, Confidence Score

## 📈 Использование

1. Введите название игры (опционально)
2. Добавьте описание игры, жанры, особенности
3. Получите предсказанный рейтинг с оценкой уверенности
4. Просматривайте статистику и историю предсказаний

Все предсказания автоматически сохраняются в `logs/logs.csv` для дальнейшего анализа.
'''
