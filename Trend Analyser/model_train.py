# model_train.py

import sys
import os
import argparse
import logging
import glob
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import warnings
from pathlib import Path
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import seaborn as sns

from trend_lstm import TrendLSTMModel, MultiTimeframeTrendModel, model_summary

# Новый импорт (добавьте):

# Добавляем путь к папке Tools в sys.path
tools_path = os.path.join(os.path.dirname(__file__), '..', 'Tools')
sys.path.insert(0, tools_path)

from indicator_manager import IndicatorManager
from trend_analyzer import TrendAnalyzer

# Определение функции логирования

def setup_logging(log_file: str = "training.log"):
    """Настройка продвинутой системы логирования"""

    # Создаем папку для логов если не существует
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)

    # Настраиваем логгер
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # Раскомментируйте для вывода в консоль тоже
        ]
    )

# Теперь используйте стандартные методы logging
def log_step(message, level: str = "INFO"):
    """Функция логирования с использованием стандартного модуля logging"""
    level = level.upper()

    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    elif level == "DEBUG":
        logging.debug(message)
    else:
        logging.info(message)




def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Trend LSTM Model Trainer')

    parser.add_argument(
        '--junior_tf',
        type=str,
        default='30',
        help='Младший таймфрейм (15, 30, 60, 120, 240 минут)'
    )

    parser.add_argument(
        '--senior_tf',
        type=str,
        default='240',
        help='Старший таймфрейм (15, 30, 60, 120, 240 минут)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Торговая пара (BTCUSDT, ETHUSDT, ADAUSDT, etc)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'test', 'analyze'],
        help='Режим работы: train, test, analyze'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Количество эпох обучения'
    )

    return parser.parse_args()


# Получаем аргументы
args = parse_arguments()

# Конфигурационные константы
MODEL_PATH = Path('models/lstm_model_TrendAnalys.pth')
MODEL_PATH.parent.mkdir(exist_ok=True)
DATA_PATH = 'Get Data Learn/data/train/*/*.csv'
TEST_DATA = "Get Data Learn/data/test/sintetic/trend_data.csv"

JUNIOR_DATA_PATH = f"Get Data Learn/data/train/{args.symbol}/{args.symbol}_{args.junior_tf}.csv"
SENIOR_DATA_PATH = f"Get Data Learn/data/train/{args.symbol}/{args.symbol}_{args.senior_tf}.csv"

class AdvancedModelTrainer:
    """
    Улучшенный тренер модели с расширенной функциональностью.

    Улучшения:
    - Многоуровневое логирование
    - Расширенная обработка ошибок
    - Продвинутые техники обучения
    - Мониторинг и визуализация
    - Поддержка multi-GPU
    """

    def __init__(self, sequence_length=60, config: Dict = None):
        self.model = None
        self.device = self._setup_device()
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_trend_acc': [], 'val_trend_acc': [],
            'train_correction_acc': [], 'val_correction_acc': [],
            'learning_rates': []
        }

        # Конфигурация по умолчанию
        self.config = {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'use_attention': True,
            'use_residual': True,
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 10,
            'min_delta': 1e-4
        }

        if config:
            self.config.update(config)

        # Определяем фичи в правильном порядке
        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma50', 'sma200', 'rsi', 'volatility',
            'trend', 'correction',
            'macd', 'price_vs_sma50', 'price_vs_sma200', 'volume_spike'
        ]

        self.sequence_length = sequence_length
        self.target_columns = ['trend_label', 'correction_label']
        self.timeframe_ratio = 8
        self.confirmation_window = 3

        self._initialize_model()

    def _setup_device(self) -> torch.device:
        """Настройка устройства с поддержкой multi-GPU"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log_step(f"Using GPU: {torch.cuda.get_device_name()}")
            if torch.cuda.device_count() > 1:
                log_step(f"Multiple GPUs available: {torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
            log_step("Using CPU")
        return device

    def ensure_float32(self, tensor: torch.Tensor) -> torch.Tensor:
        """Гарантирует, что тензор имеет тип float32"""
        return tensor.float() if tensor.dtype != torch.float32 else tensor

    def ensure_long(self, tensor: torch.Tensor) -> torch.Tensor:
        """Гарантирует, что тензор имеет тип long (int64)"""
        return tensor.long() if tensor.dtype != torch.long else tensor

    def find_data_files(self):
        """Поиск файлов данных на основе конфигурации"""
        train_files = []

        # Проверяем младший таймфрейм
        if os.path.exists(JUNIOR_DATA_PATH):
            train_files.append(JUNIOR_DATA_PATH)
            log_step(f"Найден файл младшего ТФ: {JUNIOR_DATA_PATH}")
        else:
            log_step(f"Файл не найден: {JUNIOR_DATA_PATH}")

        # Проверяем старший таймфрейм
        if os.path.exists(SENIOR_DATA_PATH):
            train_files.append(SENIOR_DATA_PATH)
            log_step(f"Найден файл старшего ТФ: {SENIOR_DATA_PATH}")
        else:
            log_step(f"Файл не найден: {SENIOR_DATA_PATH}")

        if not train_files:
            raise FileNotFoundError(f"Не найдены файлы данных. Проверьте пути в конфигурации.")

        return train_files

    def detect_trend_label(self, window: pd.DataFrame) -> int:
        """
        Упрощенное определение тренда для избежания ошибок.
        """
        if len(window) < 200:
            return 2  # Недостаточно данных

        try:
            # Упрощенная проверка на недопустимые значения
            if window.empty or window.isnull().values.any():
                return 2

            # Используем УПРОЩЕННЫЙ метод вместо комплексного анализа
            prices = window['close'].tolist()
            trend = TrendAnalyzer.detect_trend(prices)

            # Маппинг текстовых трендов в числовые метки
            trend_mapping = {'uptrend': 0, 'downtrend': 1, 'range': 2, 'insufficient_data': 2}
            return trend_mapping.get(trend, 2)

        except Exception as e:
            log_step(f"Ошибка определения тренда: {str(e)}", "ERROR")
            return 2

    def detect_correction_label(self, window: pd.DataFrame) -> int:
        """Упрощенное определение коррекции"""
        if len(window) < 10:
            return 0

        try:
            # Упрощенная проверка на недопустимые значения
            if window.empty or window.isnull().values.any():
                return 0

            # Простая логика определения коррекции
            recent_high = window['high'].max()
            current_low = window['low'].iloc[-1]

            # Если текущая цена на 2% ниже максимума - считаем коррекцией
            if current_low < recent_high * 0.98:
                return 1
            else:
                return 0

        except Exception as e:
            log_step(f"Ошибка определения коррекции: {str(e)}", "ERROR")
            return 0

    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Улучшенная загрузка и предобработка данных"""
        try:
            log_step(f"Загрузка данных из: {file_path}")

            # Загрузка с обработкой различных форматов
            data = pd.read_csv(file_path)

            # Проверка необходимых колонок
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

            log_step(f"Загружено {len(data)} строк, колонки: {data.columns.tolist()}")

            # Создание фич
            data = self.create_features(data)

            # Добавление меток с прогресс-баром
            data['trend_label'] = 0
            data['correction_label'] = 0

            total_rows = len(data)
            log_step("Создание меток...")

            for i in range(200, total_rows):
                if i % 1000 == 0:
                    log_step(f"Обработано {i}/{total_rows} ({i / total_rows:.1%})")

                window = data.iloc[i - 200:i]
                data.at[data.index[i], 'trend_label'] = self.detect_trend_label(window)
                data.at[data.index[i], 'correction_label'] = self.detect_correction_label(window)

            # Очистка данных
            initial_size = len(data)
            data = data.dropna()
            final_size = len(data)

            log_step(f"Данные очищены: {initial_size} -> {final_size} строк")

            return data

        except Exception as e:
            log_step(f"Ошибка загрузки данных: {str(e)}", "ERROR")
            raise

    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Улучшенная подготовка последовательностей с валидацией"""
        X, y_trend, y_correction = [], [], []

        total_sequences = len(data) - self.sequence_length
        log_step(f"Подготовка последовательностей... Всего: {total_sequences}")

        for i in range(self.sequence_length, len(data)):
            if i % 1000 == 0:
                log_step(f"Подготовлено {i - self.sequence_length}/{total_sequences} последовательностей")

            # Последовательность от i-sequence_length до i-1
            seq = data.iloc[i - self.sequence_length:i]

            # Проверка наличия всех фич
            if not all(feature in seq.columns for feature in self.features):
                continue

            # Целевые метки для момента времени i
            target_trend = data['trend_label'].iloc[i]
            target_correction = data['correction_label'].iloc[i]

            X.append(seq[self.features].values)
            y_trend.append(target_trend)
            y_correction.append(target_correction)

        # Преобразование в тензоры с улучшенной обработкой
        if len(X) == 0:
            raise ValueError("Не создано ни одной последовательности")

        X_array = np.array(X, dtype=np.float32)
        y_trend_array = np.array(y_trend, dtype=np.int64)
        y_correction_array = np.array(y_correction, dtype=np.int64)

        # Валидация данных
        if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
            raise ValueError("Обнаружены NaN или Inf значения в данных")

        X_tensor = torch.tensor(X_array, dtype=torch.float32)
        y_tensor = torch.tensor(np.column_stack((y_trend_array, y_correction_array)), dtype=torch.long)

        log_step(f"Подготовлено последовательностей: {len(X)}")
        log_step(f"Форма X: {X_tensor.shape}, форма y: {y_tensor.shape}")

        return X_tensor, y_tensor

    def enhance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расширенные фичи для улучшения модели с обработкой ошибок"""
        try:
            log_step("Создание расширенных фич...")

            # MACD и производные
            ema_12 = IndicatorManager.EMA_series(data['close'].tolist(), 12)
            ema_26 = IndicatorManager.EMA_series(data['close'].tolist(), 26)
            data['macd'] = [e12 - e26 if e12 is not None and e26 is not None else 0
                            for e12, e26 in zip(ema_12, ema_26)]
            data['macd_signal'] = data['macd'].rolling(9).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']

            # Относительное положение цены с защитой от деления на ноль
            data['price_vs_sma50'] = (data['close'] / data['sma50'].replace(0, 1e-10)) - 1
            data['price_vs_sma200'] = (data['close'] / data['sma200'].replace(0, 1e-10)) - 1

            # Волатильностные фичи
            data['atr'] = IndicatorManager.ATR(data, period=14)

            bb_result = IndicatorManager.BollingerBands(data['close'].tolist())
            if bb_result:
                data['bb_upper'], data['bb_middle'], data['bb_lower'], _ = bb_result.values()
                # Защита от деления на ноль
                bb_range = (data['bb_upper'] - data['bb_lower']).replace(0, 1e-10)
                data['bb_position'] = (data['close'] - data['bb_lower']) / bb_range
            else:
                data['bb_upper'] = data['bb_middle'] = data['bb_lower'] = data['bb_position'] = 0

            # Объемный анализ
            data['volume_ma'] = data['volume'].rolling(20, min_periods=1).mean()
            data['volume_spike'] = ((data['volume'] / data['volume_ma'].replace(0, 1e-10)) - 1) > 0.5
            data['volume_spike'] = data['volume_spike'].astype(int)
            data['volume_trend'] = data['volume'].pct_change(5).fillna(0)

            # Моментум фичи
            data['price_momentum'] = data['close'].pct_change(10).fillna(0)
            data['high_low_range'] = (data['high'] - data['low']) / data['close'].replace(0, 1e-10)

            log_step("Расширенные фичи успешно созданы")
            return data

        except Exception as e:
            log_step(f"Ошибка создания расширенных фич: {str(e)}", "WARNING")
            return data

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Улучшенное создание фич с обработкой ошибок"""
        try:
            log_step("Создание базовых фич...")

            # Базовые индикаторы
            data['sma50'] = IndicatorManager.SMA_series(data['close'].tolist(), 50)
            data['sma200'] = IndicatorManager.SMA_series(data['close'].tolist(), 200)
            data['rsi'] = IndicatorManager.RSI_series(data['close'].tolist(), 14)
            data['volatility'] = IndicatorManager.VOLATILITY(data, period=30)

            # Расширенные фичи
            data = self.enhance_features(data)

            # Аналитические фичи через TrendAnalyzer
            trends = []
            corrections = []
            strengths = []

            total_rows = len(data)
            log_step("Создание аналитических фич...")

            for i in range(len(data)):
                if i % 1000 == 0:
                    log_step(f"Обработано {i}/{total_rows} строк")

                if i < 200:
                    trends.append("insufficient_data")
                    corrections.append(0)
                    strengths.append(0.0)
                    continue

                window = data.iloc[i - 200:i]

                # Пропускаем окна с NaN или Inf значениями
                if window.isnull().any().any() or np.isinf(window.select_dtypes(include=[np.number])).any().any():
                    trends.append("insufficient_data")
                    corrections.append(0)
                    strengths.append(0.0)
                    continue

                try:
                    # Используем TrendAnalyzer для анализа
                    analysis = TrendAnalyzer.analyze_market(window)

                    trends.append(analysis['trend'])
                    corrections.append(int(analysis['correction']))
                    strengths.append(analysis.get('trend_strength', 0.0))
                except Exception as e:
                    log_step(f"Ошибка анализа окна {i}: {str(e)}", "WARNING")
                    trends.append("insufficient_data")
                    corrections.append(0)
                    strengths.append(0.0)

            # Кодирование категориальных фич
            trend_mapping = {'uptrend': 0, 'downtrend': 1, 'range': 2, 'insufficient_data': 2}
            data['trend'] = [trend_mapping.get(t, 2) for t in trends]
            data['correction'] = corrections
            data['trend_strength'] = strengths

            # Нормализация числовых фич
            numeric_features = [
                f for f in self.features
                if f in data.columns and f not in ['trend', 'correction', 'volume_spike']
            ]

            for feature in numeric_features:
                if feature in data.columns:
                    # Заменяем Inf и -Inf на NaN
                    data[feature] = data[feature].replace([np.inf, -np.inf], np.nan)
                    mean = data[feature].mean()
                    std = data[feature].std()
                    if std > 0:
                        data[feature] = (data[feature] - mean) / std
                    else:
                        data[feature] = 0

            # Финализация данных - более агрессивная очистка
            initial_size = len(data)
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.dropna()
            final_size = len(data)

            log_step(f"Фичи созданы. Данные очищены: {initial_size} -> {final_size} строк")

            return data

        except Exception as e:
            log_step(f"Ошибка создания фич: {str(e)}", "ERROR")
            raise

    def create_dataloaders(self, X: torch.Tensor, y: torch.Tensor,
                           val_split: float = 0.2, test_split: float = 0.1) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        """Создание DataLoader'ов с улучшенной валидацией"""
        try:
            dataset_size = len(X)
            test_size = int(test_split * dataset_size)
            val_size = int(val_split * dataset_size)
            train_size = dataset_size - val_size - test_size

            # Стратифицированное разделение
            train_dataset, val_dataset, test_dataset = random_split(
                TensorDataset(X, y),
                [train_size, val_size, test_size]
            )

            # Создание DataLoader'ов
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'] * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['batch_size'] * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            log_step(f"DataLoader'ы созданы: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

            return train_loader, val_loader, test_loader

        except Exception as e:
            log_step(f"Ошибка создания DataLoader'ов: {str(e)}", "ERROR")
            raise

    def calculate_class_weights(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Улучшенный расчет весов классов"""
        try:
            # Для трендов (3 класса)
            trend_counts = data['trend_label'].value_counts().sort_index()
            trend_weights = 1.0 / (trend_counts + 1e-6)
            trend_weights = trend_weights / trend_weights.sum()

            # Для коррекций (2 класса)
            correction_counts = data['correction_label'].value_counts().sort_index()
            correction_weights = 1.0 / (correction_counts + 1e-6)
            correction_weights = correction_weights / correction_weights.sum()

            # Собираем веса в тензоры
            weights = {
                'trend': torch.tensor(trend_weights.values, dtype=torch.float32).to(self.device),
                'correction': torch.tensor(correction_weights.values, dtype=torch.float32).to(self.device)
            }

            log_step(f"Распределение классов трендов: {dict(trend_counts)}")
            log_step(f"Веса трендов: {trend_weights.values.round(3)}")
            log_step(f"Распределение коррекций: {dict(correction_counts)}")
            log_step(f"Веса коррекций: {correction_weights.values.round(3)}")

            return weights

        except Exception as e:
            log_step(f"Ошибка расчета весов классов: {str(e)}", "ERROR")
            # Возвращаем равные веса при ошибке
            return {
                'trend': torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(self.device),
                'correction': torch.tensor([1.0, 1.0], dtype=torch.float32).to(self.device)
            }

    def _initialize_model(self):
        """Инициализация модели с улучшенной обработкой"""
        try:
            if self.model is None:
                log_step("Инициализация новой модели...")

                self.model = TrendLSTMModel(
                    input_size=len(self.features),
                    hidden_size=self.config['hidden_size'],
                    num_layers=self.config['num_layers'],
                    dropout=self.config['dropout'],
                    use_attention=self.config['use_attention'],
                    use_residual=self.config['use_residual']
                ).to(self.device)

                # Multi-GPU поддержка
                if torch.cuda.device_count() > 1:
                    self.model = nn.DataParallel(self.model)
                    log_step(f"Модель распределена на {torch.cuda.device_count()} GPU")

                log_step(f"Архитектура модели:\n{model_summary(self.model)}")

            # Загрузка весов если файл существует
            if MODEL_PATH.exists():
                try:
                    state_dict = torch.load(MODEL_PATH, map_location=self.device)

                    # Обработка multi-GPU state dict
                    if isinstance(self.model, nn.DataParallel):
                        self.model.module.load_state_dict(state_dict)
                    else:
                        self.model.load_state_dict(state_dict)

                    log_step(f"Модель загружена из {MODEL_PATH}")
                except Exception as e:
                    log_step(f"Ошибка загрузки модели: {str(e)}", "WARNING")
                    log_step("Инициализирована новая модель")
            else:
                log_step("Файл модели не найден, инициализирована новая модель")

        except Exception as e:
            log_step(f"Критическая ошибка инициализации модели: {str(e)}", "ERROR")
            raise

    def validate_model(self, val_loader: DataLoader, criterion: Tuple) -> Tuple[float, Dict]:
        """Улучшенная валидация модели"""
        self.model.eval()
        total_loss = 0.0
        trend_correct = 0
        correction_correct = 0
        total = 0

        # Для детального анализа
        all_trend_true = []
        all_trend_pred = []
        all_correction_true = []
        all_correction_pred = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = self.ensure_float32(batch_x).to(self.device)
                y_trend = self.ensure_long(batch_y[:, 0]).to(self.device)
                y_correction = batch_y[:, 1].float().to(self.device)

                trend_pred, correction_pred, _ = self.model(batch_x)

                # Расчет лосса
                loss_trend = criterion[0](trend_pred, y_trend)
                loss_correction = criterion[1](correction_pred, y_correction)
                total_loss += (loss_trend + loss_correction).item()

                # Точность для тренда
                _, trend_pred_class = torch.max(trend_pred, 1)
                trend_correct += (trend_pred_class == y_trend).sum().item()

                # Точность для коррекции
                correction_pred_class = (correction_pred > 0.5).float()
                correction_correct += (correction_pred_class == y_correction).sum().item()

                # Сохранение для детального анализа
                all_trend_true.extend(y_trend.cpu().numpy())
                all_trend_pred.extend(trend_pred_class.cpu().numpy())
                all_correction_true.extend(y_correction.cpu().numpy())
                all_correction_pred.extend(correction_pred_class.cpu().numpy())

                total += y_trend.size(0)

        # Детальная статистика
        trend_accuracy = trend_correct / total
        correction_accuracy = correction_correct / total
        avg_loss = total_loss / len(val_loader)

        return avg_loss, {
            'trend': trend_accuracy,
            'correction': correction_accuracy,
            'trend_true': all_trend_true,
            'trend_pred': all_trend_pred,
            'correction_true': all_correction_true,
            'correction_pred': all_correction_pred
        }

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                    class_weights: Dict, epochs: int = 100) -> Dict:
        """Улучшенное обучение модели с расширенным мониторингом"""
        log_step(f"Начало обучения на {epochs} эпох")

        # Инициализация модели
        self.model.train()

        # Функции потерь
        criterion = (
            nn.CrossEntropyLoss(weight=class_weights['trend']),
            nn.BCEWithLogitsLoss(pos_weight=class_weights['correction'][1].unsqueeze(0))
        )

        # Оптимизатор и планировщик
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        best_loss = float('inf')
        patience = 0
        best_epoch = 0

        for epoch in range(epochs):
            # Фаза обучения
            self.model.train()
            epoch_train_loss = 0.0
            train_trend_correct = 0
            train_correction_correct = 0
            train_total = 0

            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                y_trend = batch_y[:, 0].to(self.device)
                y_correction = batch_y[:, 1].float().to(self.device)

                optimizer.zero_grad()

                trend_pred, correction_pred, aux_outputs = self.model(batch_x)

                # Расчет лосса
                loss_trend = criterion[0](trend_pred, y_trend)
                loss_correction = criterion[1](correction_pred, y_correction)
                batch_loss = loss_trend + 0.7 * loss_correction  # Взвешенный лосс

                batch_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_train_loss += batch_loss.item()

                # Метрики обучения
                _, trend_pred_class = torch.max(trend_pred, 1)
                train_trend_correct += (trend_pred_class == y_trend).sum().item()

                correction_pred_class = (correction_pred > 0.5).float()
                train_correction_correct += (correction_pred_class == y_correction).sum().item()

                train_total += y_trend.size(0)

                if batch_idx % 100 == 0:
                    log_step(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, "
                             f"Loss: {batch_loss.item():.4f}")

            # Фаза валидации
            val_loss, val_metrics = self.validate_model(val_loader, criterion)

            # Обновление планировщика
            scheduler.step()
            # scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']

            # Расчет метрик
            train_loss = epoch_train_loss / len(train_loader)
            train_trend_acc = train_trend_correct / train_total
            train_correction_acc = train_correction_correct / train_total

            val_trend_acc = val_metrics['trend']
            val_correction_acc = val_metrics['correction']

            # Логирование
            log_step(f"Epoch {epoch + 1}/{epochs} | "
                     f"LR: {current_lr:.2e} | "
                     f"Train Loss: {train_loss:.4f} | "
                     f"Val Loss: {val_loss:.4f} | "
                     f"Train Trend Acc: {train_trend_acc:.2%} | "
                     f"Val Trend Acc: {val_trend_acc:.2%} | "
                     f"Train Correction Acc: {train_correction_acc:.2%} | "
                     f"Val Correction Acc: {val_correction_acc:.2%}")

            # Сохранение истории
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_trend_acc'].append(train_trend_acc)
            self.training_history['val_trend_acc'].append(val_trend_acc)
            self.training_history['train_correction_acc'].append(train_correction_acc)
            self.training_history['val_correction_acc'].append(val_correction_acc)
            self.training_history['learning_rates'].append(current_lr)

            # Early stopping и сохранение лучшей модели
            if val_loss < best_loss - self.config['min_delta']:
                best_loss = val_loss
                best_epoch = epoch
                patience = 0

                # Сохранение модели
                if isinstance(self.model, nn.DataParallel):
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                torch.save(state_dict, MODEL_PATH)
                log_step(f"Модель сохранена (Val Loss: {val_loss:.4f})")
            else:
                patience += 1
                if patience >= self.config['patience']:
                    log_step(f"Early stopping после {epoch + 1} эпох. "
                             f"Лучшая эпоха: {best_epoch + 1} с loss: {best_loss:.4f}")
                    break

        # Финальный отчет
        log_step(f"Обучение завершено. Лучшая эпоха: {best_epoch + 1}, "
                 f"Лучший Val Loss: {best_loss:.4f}")

        return {
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'training_history': self.training_history,
            'final_metrics': {
                'train_trend_acc': train_trend_acc,
                'val_trend_acc': val_trend_acc,
                'train_correction_acc': train_correction_acc,
                'val_correction_acc': val_correction_acc
            }
        }

    def train_on_all_data(self, initial_epochs: int = 100, incremental_epochs: int = 50) -> Dict:
        """Улучшенный полный цикл обучения"""
        try:
            data_files = self.find_data_files()
            if not data_files:
                log_step("Не найдено файлов для обучения", "ERROR")
                return {}

            results = {}

            # Первичное обучение
            if not MODEL_PATH.exists():
                log_step("Инициализация нового обучения...")
                initial_data = self.load_and_preprocess_data(data_files[0])

                # Расчет весов классов
                class_weights = self.calculate_class_weights(initial_data)

                # Подготовка данных
                X, y = self.prepare_sequences(initial_data)
                train_loader, val_loader, test_loader = self.create_dataloaders(X, y)

                # Обучение
                initial_results = self.train_model(
                    train_loader, val_loader, class_weights, initial_epochs
                )
                results['initial'] = initial_results
                remaining_files = data_files[1:]
            else:
                log_step("Загрузка существующей модели...")
                self._initialize_model()
                initial_data = self.load_and_preprocess_data(data_files[0])
                class_weights = self.calculate_class_weights(initial_data)
                remaining_files = data_files
                results['initial'] = {'status': 'loaded_existing'}

            # Дообучение на остальных файлах
            incremental_results = []
            for i, file in enumerate(remaining_files):
                try:
                    log_step(f"Дообучение на файле {i + 1}/{len(remaining_files)}: {file}")

                    new_data = self.load_and_preprocess_data(file)
                    X_new, y_new = self.prepare_sequences(new_data)
                    train_loader, val_loader, _ = self.create_dataloaders(X_new, y_new)

                    epoch_results = self.train_model(
                        train_loader, val_loader, class_weights, incremental_epochs
                    )
                    incremental_results.append(epoch_results)

                except Exception as e:
                    log_step(f"Ошибка при обработке {file}: {str(e)}", "WARNING")
                    continue

            results['incremental'] = incremental_results

            # Финальный анализ
            if initial_data is not None:
                log_step("Финальная статистика данных:")
                log_step(f"Тренды: {initial_data['trend_label'].value_counts().to_dict()}")
                log_step(f"Коррекции: {initial_data['correction_label'].mean():.2%}")

            return results

        except Exception as e:
            log_step(f"Критическая ошибка обучения: {str(e)}", "ERROR")
            log_step(traceback.format_exc())
            return {}

    def plot_training_history(self, save_path: str = "training_history.png"):
        """Визуализация истории обучения"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Loss
            ax1.plot(self.training_history['train_loss'], label='Train Loss')
            ax1.plot(self.training_history['val_loss'], label='Val Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # Trend Accuracy
            ax2.plot(self.training_history['train_trend_acc'], label='Train Trend Acc')
            ax2.plot(self.training_history['val_trend_acc'], label='Val Trend Acc')
            ax2.set_title('Trend Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)

            # Correction Accuracy
            ax3.plot(self.training_history['train_correction_acc'], label='Train Correction Acc')
            ax3.plot(self.training_history['val_correction_acc'], label='Val Correction Acc')
            ax3.set_title('Correction Accuracy')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy')
            ax3.legend()
            ax3.grid(True)

            # Learning Rate
            ax4.plot(self.training_history['learning_rates'], label='Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.legend()
            ax4.grid(True)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            log_step(f"Графики обучения сохранены в {save_path}")

        except Exception as e:
            log_step(f"Ошибка построения графиков: {str(e)}", "WARNING")

    def test_model(self, test_path: str) -> Dict:
        """Расширенное тестирование модели"""
        try:
            log_step("Начало расширенного тестирования модели...")

            # Загрузка и подготовка данных
            test_data = self.load_and_preprocess_data(test_path)
            X_test, y_test = self.prepare_sequences(test_data)
            _, _, test_loader = self.create_dataloaders(X_test, y_test)

            if len(test_loader.dataset) == 0:
                raise ValueError("Тестовые данные не содержат примеров")

            # Загрузка лучшей модели
            self._initialize_model()
            self.model.eval()

            # Прогнозирование
            all_preds = []
            all_true = []
            all_probabilities = []
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(self.device)
                    y_trend = batch_y[:, 0].numpy()
                    y_correction = batch_y[:, 1].numpy()

                    # Предсказание с вероятностями
                    trend_pred, correction_pred, _ = self.model(batch_x)

                    trend_probs = torch.softmax(trend_pred, dim=1).cpu().numpy()
                    correction_probs = torch.sigmoid(correction_pred).cpu().numpy()

                    trend_pred_class = trend_pred.argmax(dim=1).cpu().numpy()
                    correction_pred_class = (correction_probs > 0.5).astype(int)

                    all_preds.extend(zip(trend_pred_class, correction_pred_class))
                    all_true.extend(zip(y_trend, y_correction))
                    all_probabilities.extend(zip(trend_probs, correction_probs))
                    total += len(y_trend)

            # Анализ результатов
            true_trend, true_correction = zip(*all_true)
            pred_trend, pred_correction = zip(*all_preds)
            prob_trend, prob_correction = zip(*all_probabilities)

            # Детальные отчеты
            trend_report = classification_report(
                true_trend, pred_trend,
                target_names=['Uptrend', 'Downtrend', 'Sideways'],
                output_dict=True,
                zero_division=0
            )

            correction_report = classification_report(
                true_correction, pred_correction,
                target_names=['No Correction', 'Correction'],
                output_dict=True,
                zero_division=0
            )

            # Матрицы ошибок
            trend_cm = confusion_matrix(true_trend, pred_trend)
            correction_cm = confusion_matrix(true_correction, pred_correction)

            # Визуализация матриц ошибок
            self._plot_confusion_matrices(trend_cm, correction_cm)

            # Вывод результатов
            log_step("\n" + "=" * 50)
            log_step("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
            log_step("=" * 50)
            log_step(f"Общее количество примеров: {total:,}")

            log_step("\nТРЕНДЫ:")
            log_step(f"Accuracy: {trend_report['accuracy']:.2%}")
            for class_name in ['Uptrend', 'Downtrend', 'Sideways']:
                stats = trend_report[class_name]
                log_step(f"{class_name:>10}: "
                         f"Precision={stats['precision']:.2%} | "
                         f"Recall={stats['recall']:.2%} | "
                         f"F1={stats['f1-score']:.2%} | "
                         f"Support={stats['support']:,}")

            log_step("\nКОРРЕКЦИИ:")
            log_step(f"Accuracy: {correction_report['accuracy']:.2%}")
            for class_name in ['No Correction', 'Correction']:
                stats = correction_report[class_name]
                log_step(f"{class_name:>15}: "
                         f"Precision={stats['precision']:.2%} | "
                         f"Recall={stats['recall']:.2%} | "
                         f"F1={stats['f1-score']:.2%} | "
                         f"Support={stats['support']:,}")

            return {
                'trend_report': trend_report,
                'correction_report': correction_report,
                'predictions': all_preds,
                'probabilities': all_probabilities,
                'confusion_matrices': {
                    'trend': trend_cm,
                    'correction': correction_cm
                }
            }

        except Exception as e:
            log_step(f"Ошибка тестирования: {str(e)}", "ERROR")
            return {}

    def _plot_confusion_matrices(self, trend_cm, correction_cm, save_path: str = "confusion_matrices.png"):
        """Визуализация матриц ошибок"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Матрица ошибок для трендов
            sns.heatmap(trend_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                        xticklabels=['Uptrend', 'Downtrend', 'Sideways'],
                        yticklabels=['Uptrend', 'Downtrend', 'Sideways'])
            ax1.set_title('Confusion Matrix - Trends')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')

            # Матрица ошибок для коррекций
            sns.heatmap(correction_cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                        xticklabels=['No Correction', 'Correction'],
                        yticklabels=['No Correction', 'Correction'])
            ax2.set_title('Confusion Matrix - Corrections')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            log_step(f"Матрицы ошибок сохранены в {save_path}")

        except Exception as e:
            log_step(f"Ошибка построения матриц ошибок: {str(e)}", "WARNING")


# Основной блок выполнения
if __name__ == "__main__":
    setup_logging()


    # Выводим информацию о конфигурации
    log_step(f"Конфигурация запуска:")
    log_step(f"  Символ: {args.symbol}")
    log_step(f"  Таймфреймы: {args.junior_tf}min -> {args.senior_tf}min")
    log_step(f"  Режим: {args.mode}")
    log_step(f"  Эпох: {args.epochs}")
    log_step(f"  Junior data: {JUNIOR_DATA_PATH}")
    log_step(f"  Senior data: {SENIOR_DATA_PATH}")

    # Создаем конфигурацию
    config = {
        'symbol': args.symbol,
        'junior_tf': args.junior_tf,
        'senior_tf': args.senior_tf,
        'epochs': args.epochs,
        'model_path': f"models/{args.symbol}_{args.junior_tf}_{args.senior_tf}/",
        'train_data_path': {
            'junior': JUNIOR_DATA_PATH,
            'senior': SENIOR_DATA_PATH
        },
        'val_data_path': {
            'junior': f"Get Data Learn/data/val/{args.symbol}/{args.symbol}_{args.junior_tf}.csv",
            'senior': f"Get Data Learn/data/val/{args.symbol}/{args.symbol}_{args.senior_tf}.csv"
        }
    }

    # Создаем тренера с конфигом
    trainer = AdvancedModelTrainer(sequence_length=60, config=config)

    # Выбор режима работы
    if args.mode == "train":
        results = trainer.train_on_all_data(
            initial_epochs=args.epochs,
            incremental_epochs=args.epochs // 2
        )
        trainer.plot_training_history()

    elif args.mode == "test":
        test_results = trainer.test_model(TEST_DATA)

    elif args.mode == "analyze":
        junior_data = trainer.load_and_preprocess_data(JUNIOR_DATA_PATH)
        analysis = TrendAnalyzer.analyze_market(junior_data.tail(100))
        log_step(json.dumps(analysis, indent=2, default=str), "INFO")