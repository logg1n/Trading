# 🤖💰 Идея - Создание торгового бота на основе обученных LSTM моделей 

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### 🧮 Модели обучаются с помощью торговых данных полученных с биржи, в поиске заданных паттернов
***Данный этап:*** 
1. настройка, калибровка и доработка моделей
2. Проверка обучения на жестко созданных синтетических данных
3. Подключение модели в режим real-time для проверки её работы 

📂 Структура проекта
```commandline
📦 Trading
├── 📂 Get Data Learn # Сбор и подготовка данных
├── 📂 Pattern Analyser # Анализ рыночных паттернов
├── 📂 Patterns # База эталонных паттернов
├── 📂 Tools # Вспомогательные утилиты
├── 📂 Trend Analyser # Анализ трендов
├── 📜 README.md # Документация
├── 📜 requirements.txt # Зависимости
```

📜 Лицензия
MIT © logg1n