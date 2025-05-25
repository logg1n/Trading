import os
import glob
# Добавить в начало файла
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from datetime import datetime

from trend_lstm import TrendLSTMModel
from Tools.indicator_manager import IndicatorManager


# Определение функции логирования
def log_step(message):
	timestamp = datetime.now().strftime('%H:%M:%S')
	print(f"[{timestamp}] {message}")


MODEL_PATH = 'lstm_model.pth'
DATA_PATH = '../Get Data Learn/data/train/*/*.csv'
TEST_DATA = "../Get Data Learn/data/test/sintetic/trend_data.csv"
JUNIOR_DATA_PATH = "../Get Data Learn/data/train/BTCUSDT/BTCUSDT_30.csv"
SENIOR_DATA_PATH = "../Get Data Learn/data/train/BTCUSDT/BTCUSDT_240.csv"


class ModelTrainer:
	def __init__(self, sequence_length=60):
		self.model = None
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.best_loss = float('inf')
		self.features = [
			'open', 'high', 'low', 'close', 'volume',
			'sma50', 'sma200', 'rsi', 'volatility',
			'trend', 'correction'  # Добавляем недостающие фичи
		]
		self.sequence_length = sequence_length
		self.target_columns = ['trend_label', 'correction_label']  # Новые целевые переменные
		self.timeframe_ratio = 8  # Соотношение 4h/30m = 8
		self.confirmation_window = 3  # Количество свечей для подтверждения

		self.initialize_model()

	def ensure_float32(self, tensor):
		"""Гарантирует, что тензор имеет тип float32"""
		if tensor.dtype != torch.float32:
			return tensor.float()
		return tensor

	def ensure_long(self, tensor):
		"""Гарантирует, что тензор имеет тип long (int64)"""
		if tensor.dtype != torch.long:
			return tensor.long()
		return tensor

	def find_data_files(self):
		global DATA_PATH
		"""Поиск файлов с данными"""
		data_files = glob.glob(DATA_PATH)
		if not data_files:
			raise FileNotFoundError(f"Не найдено файлов по шаблону: {PATH_DATA}")

		log_step(f"Найдено {len(data_files)} файлов для обработки:")
		for file in data_files:
			log_step(f" - {file}")

		return data_files

	def detect_trend_label(self, window):
		"""Определение тренда с использованием 5 факторов"""
		if len(window) < 200:
			return 2  # Недостаточно данных

		try:
			# Инициализация весов и счетчиков
			weights = {
				'sma': 0.3,
				'momentum': 0.2,
				'volume': 0.15,
				'volatility': 0.2,
				'price_action': 0.15
			}

			scores = {
				0: 0.0,  # uptrend
				1: 0.0,  # downtrend
				2: 0.0  # sideways
			}

			# Базовые индикаторы
			prices = window['close'].values
			sma50 = IndicatorManager.SMA(prices, 50)
			sma200 = IndicatorManager.SMA(prices, 200)
			last_price = prices[-1]

			# 1. Фактор SMA
			if sma50 > sma200:
				scores[0] += weights['sma']
			elif sma50 < sma200:
				scores[1] += weights['sma']

			# 2. Моментум (изменение цены)
			price_change = window['close'].iloc[-1] - window['close'].iloc[0]
			if price_change > 0:
				scores[0] += weights['momentum']
			else:
				scores[1] += weights['momentum']

			# 3. Объемы
			volume_change = window['volume'].pct_change().mean()
			if volume_change > 0:
				if price_change > 0:
					scores[0] += weights['volume']
				else:
					scores[1] += weights['volume']

			# 4. Волатильность
			volatility = window['high'].sub(window['low']).mean()
			avg_volatility = window['high'].sub(window['low']).rolling(50).mean().iloc[-1]
			if volatility > avg_volatility * 1.2:
				scores[2] += weights['volatility']

			# 5. Ценовое действие
			higher_highs = (window['high'].iloc[-3:-1] < window['high'].iloc[-1]).all()
			lower_lows = (window['low'].iloc[-3:-1] > window['low'].iloc[-1]).all()

			if higher_highs and not lower_lows:
				scores[0] += weights['price_action']
			elif lower_lows and not higher_highs:
				scores[1] += weights['price_action']
			else:
				scores[2] += weights['price_action']

			# Определение результата
			max_score = max(scores.values())
			if max_score < 0.4:  # Порог уверенности
				return 2

			return max(scores, key=scores.get)

		except Exception as e:
			log_step(f"Ошибка определения тренда: {str(e)}")
			return 2

	def detect_correction_label(self, window):
		"""Определение метки коррекции (исправленная версия)"""
		# Проверяем, что окно не пустое
		if len(window) == 0:
			return 0

		# Получаем значения через iloc
		try:
			# Исправление:
			if len(window) >= 10:
				high = window['high'].iloc[-10:].max()
				low = window['low'].iloc[-10:].min()
			else:
				return 0  # Недостаточно данных
		except IndexError:
			return 0

		# Проверяем наличие данных для расчета
		if pd.isna(high) or pd.isna(low) or pd.isna(current):
			return 0

		# Расчет коррекции
		if (high - low) == 0:  # Защита от деления на ноль
			return 0

		if current < high - 0.38 * (high - low):
			return 1  # Коррекция вниз
		if current > low + 0.38 * (high - low):
			return 1  # Коррекция вверх
		return 0  # Нет коррекции

	def load_multiple_timeframes(self, junior_path, senior_path):
		"""Загрузка и синхронизация двух таймфреймов"""
		# Загрузка данных
		df_junior = pd.read_csv(junior_path, parse_dates=['date'])
		df_senior = pd.read_csv(senior_path, parse_dates=['date'])

		# Синхронизация временных меток
		df_junior = df_junior.set_index('date')
		df_senior = df_senior.set_index('date').reindex(df_junior.index, method='ffill')

		# Удаление пропущенных значений
		return df_junior.dropna(), df_senior.dropna()

	def load_and_preprocess_data(self, file_path):
		try:
			data = pd.read_csv(file_path)
			log_step(f"Колонки в файле: {data.columns.tolist()}")
			log_step(f"Первые 5 строк:\n{data.head()}")

			data = self.create_features(data)

			# Добавляем метки
			data['trend_label'] = 0
			data['correction_label'] = 0

			for i in range(200, len(data)):
				window = data.iloc[i - 200:i]
				data.at[data.index[i], 'trend_label'] = self.detect_trend_label(window)
				data.at[data.index[i], 'correction_label'] = self.detect_correction_label(window)

			return data.dropna()

		except Exception as e:
			log_step(f"Ошибка загрузки данных: {str(e)}")
			raise

	def prepare_sequences(self, data: pd.DataFrame):
		"""Подготовка последовательностей для LSTM"""
		X, y_trend, y_correction = [], [], []

		for i in range(len(data) - self.sequence_length):
			seq = data.iloc[i - self.sequence_length:i]

			# Особенности
			X.append(seq[self.features].values)

			# Целевые переменные
			y_trend.append(seq['trend_label'].iloc[-1])
			y_correction.append(seq['correction_label'].iloc[-1])

		X_tensor = torch.FloatTensor(np.array(X))
		y_tensor = torch.LongTensor(np.column_stack((y_trend, y_correction)))

		return X_tensor, y_tensor

	def create_features(self, data):
		"""Исправленный метод создания фич"""
		# Добавляем расчет индикаторов
		data['sma50'] = IndicatorManager.SMA_series(data['close'].tolist(), 50)
		data['sma200'] = IndicatorManager.SMA_series(data['close'].tolist(), 200)
		data['rsi'] = IndicatorManager.RSI_series(data['close'].tolist())
		data['volatility'] = IndicatorManager.VOLATILITY(data, period=30)
		"""Расширенный метод создания фич"""

		# Вычисляем тренд и коррекцию для каждой точки данных
		trends = []
		corrections = []

		for i in range(len(data)):
			if i < 200:  # Пропускаем первые 200 точек для расчета SMA200
				trends.append("insufficient_data")
				corrections.append(0)
				continue

			window = data.iloc[i - 200:i + 1]

			# Анализ тренда
			trend = IndicatorManager.detect_trend(
				window['close'].tolist(),
				window_short=50,
				window_long=200
			)

			# Анализ коррекции
			correction = IndicatorManager.detect_correction(
				window['high'].tolist(),
				window['low'].tolist(),
				window['close'].tolist()
			)

			trends.append(trend)
			corrections.append(int(correction))  # Конвертируем bool в int здесь

		# Кодируем категориальные фичи
		data['trend'] = pd.Categorical(trends).codes
		data['correction'] = corrections  # Уже конвертировано в int

		# Нормализация
		features_to_normalize = [f for f in self.features if f in data.columns]
		data[features_to_normalize] = data[features_to_normalize].apply(
			lambda x: (x - x.mean()) / x.std()
		)

		return data.dropna()

	def create_dataloaders(self, X, y, val_split=0.2):
		"""Создание DataLoader'ов с проверкой типов"""
		# Разделение данных
		# В DataLoader:

		split_idx = int(len(X) * (1 - val_split))
		X_train, X_val = X[:split_idx], X[split_idx:]
		y_train, y_val = y[:split_idx], y[split_idx:]

		# Гарантируем правильные типы
		if X_train.dtype != torch.float32:
			X_train = X_train.float()
			X_val = X_val.float()
		if y_train.dtype != torch.long:
			y_train = y_train.long()
			y_val = y_val.long()

		log_step(f"Типы данных - X_train: {X_train.dtype}, y_train: {y_train.dtype}")

		train_dataset = TensorDataset(X_train, y_train)
		val_dataset = TensorDataset(X_val, y_val)

		return (DataLoader(train_dataset, batch_size=128, shuffle=True),
				DataLoader(val_dataset, batch_size=128))

	def calculate_class_weights(self, data: pd.DataFrame):
		"""Расчет весов классов для несбалансированных данных"""
		# Для трендов (3 класса)
		trend_counts = data['trend_label'].value_counts().sort_index()
		trend_weights = 1.0 / (trend_counts + 1e-6)  # Защита от деления на ноль
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

	def initialize_model(self):
		global MODEL_PATH

		# Инициализация модели только если она не создана
		if self.model is None:
			# Создаем модель с правильными параметрами
			self.model = TrendLSTMModel(
				input_size=len(self.features),
				hidden_size=128,
				num_trend_classes=3
			).to(self.device)

			# Загрузка весов если файл существует
			if os.path.exists(MODEL_PATH):
				try:
					self.model.load_state_dict(
						torch.load(MODEL_PATH, map_location=self.device))
					log_step(f"Модель загружена из {MODEL_PATH}")
				except Exception as e:
					log_step(f"Ошибка загрузки модели: {str(e)}")
					log_step("Инициализирована новая модель")
			else:
				log_step("Предупреждение: файл модели не найден")
				log_step("Инициализирована новая модель")

	def validate_model(self, val_loader, criterion):
		self.model.eval()
		total_loss = 0.0
		trend_correct = 0
		correction_correct = 0
		total = 0

		with torch.no_grad():
			for batch_x, batch_y in val_loader:
				batch_x = self.ensure_float32(batch_x).to(self.device)
				y_trend = self.ensure_long(batch_y[:, 0]).to(self.device)
				y_correction = batch_y[:, 1].float().to(self.device)

				trend_pred, correction_pred = self.model(batch_x)

				# Рассчитываем лосс
				loss_trend = criterion[0](trend_pred, y_trend)
				loss_correction = criterion[1](correction_pred.squeeze(-1), y_correction)
				total_loss += (loss_trend + loss_correction).item()

				# Точность для тренда
				_, trend_pred = torch.max(trend_pred, 1)
				trend_correct += (trend_pred == y_trend).sum().item()

				# Точность для коррекции
				correction_pred = torch.sigmoid(correction_pred) > 0.5
				correction_correct += (correction_pred.squeeze() == y_correction).sum().item()

				total += y_trend.size(0)

		return (
			total_loss / len(val_loader),
			{
				'trend': trend_correct / total,
				'correction': correction_correct / total
			}
		)

	def train_on_all_data(self, initial_epochs=50, incremental_epochs=10):
		"""Полный цикл обучения с улучшенной обработкой"""
		try:
			data_files = self.find_data_files()
			if not data_files:
				log_step("Не найдено файлов для обучения")
				return None

			# Первичное обучение
			if not os.path.exists(MODEL_PATH):
				log_step("Инициализация нового обучения...")
				initial_data = self.load_and_preprocess_data(data_files[0])

				# Расчет весов классов
				class_weights = self.calculate_class_weights(initial_data)
				# Исправленный вывод
				log_step(f"Веса классов:")
				log_step(f"Тренды: {class_weights['trend'].cpu().numpy().round(3)}")
				log_step(f"Коррекции: {class_weights['correction'].cpu().numpy().round(3)}")

				X, y = self.prepare_sequences(initial_data)
				train_loader, val_loader = self.create_dataloaders(X, y)

				self.initialize_model()
				self.train_model(
					train_loader,
					val_loader,
					class_weights,
					initial_epochs
				)
				remaining_files = data_files[1:]
			else:
				log_step("Загрузка существующей модели...")
				self.initialize_model()
				initial_data = self.load_and_preprocess_data(data_files[0])
				class_weights = self.calculate_class_weights(initial_data)
				remaining_files = data_files

			# Дообучение
			for file in remaining_files:
				try:
					log_step(f"\nДообучение на файле: {file}")
					new_data = self.load_and_preprocess_data(file)

					X_new, y_new = self.prepare_sequences(new_data)
					train_loader, val_loader = self.create_dataloaders(X_new, y_new)

					self.train_model(
						train_loader,
						val_loader,
						class_weights
					)

				except Exception as e:
					log_step(f"Ошибка при обработке {file}: {str(e)}")
					continue
				# Первичный анализ данных
				if initial_data is not None:  # Добавляем проверку
					log_step("Статистика новых фич:")
					log_step(f"Тренды: {initial_data['trend_label'].value_counts().to_dict()}")
					log_step(f"Коррекции: {initial_data['correction_label'].mean():.2%} наблюдений")

		except Exception as e:
			log_step(f"Критическая ошибка: {str(e)}")
			import traceback
			log_step(traceback.format_exc())
			return None

	def train_model(self, train_loader, val_loader, class_weights, epochs=50):
		"""Обновленный метод обучения с исправлениями"""
		self.model = self.model.to(self.device)

		# Двойной лосс: CrossEntropy для тренда, BCEWithLogits для коррекции
		criterion = (
			nn.CrossEntropyLoss(weight=class_weights['trend']),
			nn.BCEWithLogitsLoss(pos_weight=class_weights['correction'][1].unsqueeze(0))
		)

		optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
		best_loss = float('inf')
		patience = 0

		for epoch in range(epochs):
			self.model.train()
			total_loss = 0.0

			for batch_x, batch_y in train_loader:
				batch_x = batch_x.to(self.device)
				y_trend = batch_y[:, 0].to(self.device)
				y_correction = batch_y[:, 1].float().to(self.device)

				optimizer.zero_grad()

				trend_pred, correction_pred = self.model(batch_x)

				# Исправление размерностей
				loss_trend = criterion[0](trend_pred, y_trend)
				loss_correction = criterion[1](correction_pred.squeeze(-1), y_correction)

				# Исправление:
				total_loss = loss_trend + 0.5 * loss_correction  # Добавлены веса

				total_loss.backward()
				optimizer.step()

			# Валидация
			val_loss, val_acc = self.validate_model(val_loader, criterion)

			log_step(f"Epoch {epoch + 1} | "
					 f"Train Loss: {total_loss.item():.4f} | "
					 f"Val Loss: {val_loss:.4f} | "
					 f"Trend Acc: {val_acc['trend']:.2%} | "
					 f"Correction F1: {val_acc['correction']:.2%}")

			# Early stopping
			if val_loss < best_loss:
				best_loss = val_loss
				patience = 0
				torch.save(self.model.state_dict(), MODEL_PATH)
				log_step(f"Модель сохранена (Val Loss: {val_loss:.4f})")
			else:
				patience += 1
				if patience >= 5:
					log_step("Early stopping: нет улучшений 5 эпох подряд")
					break

		def test_multiple_timeframes(self, junior_path, senior_path):
			"""Анализ с улучшенным логированием"""
			try:
				self.initialize_model()
				log_step("Инициализация модели завершена")

				# Проверка существования файлов
				if not os.path.exists(junior_path):
					raise FileNotFoundError(f"Файл не найден: {junior_path}")
				if not os.path.exists(senior_path):
					raise FileNotFoundError(f"Файл не найден: {senior_path}")

				log_step("Загрузка данных...")
				junior = pd.read_csv(junior_path)
				senior = pd.read_csv(senior_path)
				log_step(f"Загружено данных: Junior={len(junior)}, Senior={len(senior)}")

				# Проверка минимального размера данных
				min_bars = max(self.sequence_length * 2, 200)
				if len(junior) < min_bars or len(senior) < min_bars:
					msg = (f"Недостаточно данных. Требуется минимум {min_bars} баров. "
						   f"Junior: {len(junior)}, Senior: {len(senior)}")
					log_step(msg)
					raise ValueError(msg)

				# Генерация фичей
				log_step("Создание фичей для младшего ТФ...")
				junior = self.create_features(junior)
				log_step("Создание фичей для старшего ТФ...")
				senior = self.create_features(senior)
				log_step("Фичи успешно созданы")

				# Анализ
				log_step("Начало анализа...")
				results = self.analyze_timeframes(junior, senior)
				log_step(f"Анализ завершен. Найдено событий: {len(results)}")

				# Фильтрация и сохранение
				confirmed = [r for r in results if r['confirmed'] != 'no_confirmation']
				log_step(f"Подтвержденных сигналов: {len(confirmed)}")

				if confirmed:
					with open("signals.json", "w") as f:
						json.dump(confirmed, f, indent=2)
					log_step("Результаты сохранены в signals.json")
				else:
					log_step("Нет подтвержденных сигналов для сохранения")

				return confirmed

			except Exception as e:
				log_step(f"Критическая ошибка: {str(e)}")
				log_step("Трассировка ошибки:")
				log_step(traceback.format_exc())
				return None

	def predict(self, data):
		"""Предсказание для одного таймфрейма"""
		# Преобразование данных в тензор
		seq_tensor = torch.FloatTensor(data[self.features].values)

		# Прогноз модели
		with torch.no_grad():
			trend_pred, correction_pred = self.model(seq_tensor.unsqueeze(0))

		return {
			'trend': trend_pred.argmax().item(),
			'correction': (torch.sigmoid(correction_pred) > 0.5).item()
		}

	def analyze_senior_tf(self, data):
		"""Улучшенный анализ старшего таймфрейма"""
		try:
			if len(data) < self.confirmation_window:
				return {'trend': 2, 'correction': 0}

			# Используем скользящее среднее для анализа
			trend_values = data['trend'].rolling(
				window=self.confirmation_window,
				min_periods=1
			).mean().fillna(2)

			correction_values = data['correction'].rolling(
				window=self.confirmation_window,
				min_periods=1
			).mean().fillna(0)

			current_trend = trend_values.iloc[-1]
			current_correction = correction_values.iloc[-1]

			return {
				'trend': 0 if current_trend < 0.4 else 1 if current_trend > 0.6 else 2,
				'correction': 1 if current_correction > 0.5 else 0
			}
		except Exception as e:
			log_step(f"Ошибка анализа старшего ТФ: {str(e)}")
			return {'trend': 2, 'correction': 0}

	def check_confirmation(self, junior, senior):
		"""Многофакторное подтверждение с учетом волатильности"""
		try:
			# Проверка базовых условий подтверждения
			trend_confirm = (
					junior['trend'] == senior['trend'] and
					junior['trend'] in [0, 1]  # Только для восходящего/нисходящего тренда
			)

			correction_confirm = (
					junior['correction'] == 1 and
					senior['correction'] == 1
			)

			if trend_confirm:
				return 'trend_confirmed'
			elif correction_confirm:
				return 'correction_confirmed'
			return 'no_confirmation'
		except Exception as e:
			log_step(f"Confirmation error: {str(e)}")
			return {'trend': False, 'correction': False, 'confidence': 0}

	def calculate_levels(self, junior_seq, senior_seq):
		"""Расчет ключевых уровней с проверкой данных"""
		try:
			junior_support = junior_seq['low'].min() if not junior_seq.empty else 0
			junior_resistance = junior_seq['high'].max() if not junior_seq.empty else 0

			# Расчет пивота старшего ТФ с проверкой
			senior_pivot = None
			if not senior_seq.empty and 'close' in senior_seq:
				senior_pivot = senior_seq['close'].mean()

			return {
				'junior_support': float(junior_support),
				'junior_resistance': float(junior_resistance),
				'senior_pivot': float(senior_pivot) if senior_pivot else None
			}
		except Exception as e:
			log_step(f"Ошибка расчета уровней: {str(e)}")
			return {
				'junior_support': 0,
				'junior_resistance': 0,
				'senior_pivot': None
			}

	def calculate_simple_levels(self, junior, senior):
		"""Упрощенный расчет уровней"""
		return {
			'support': junior['low'].min(),
			'resistance': junior['high'].max(),
			'senior_pivot': senior['close'].mean() if not senior.empty else None
		}

	def _get_trend_label(self, code):
		"""Преобразование кода тренда в текст"""
		return ['uptrend', 'downtrend', 'sideways'][int(code)]

	def _format_levels(self, row):
		"""Форматирование уровней"""
		return {
			'support': round(float(row.get('junior_support', 0))),
			'resistance': round(float(row.get('junior_resistance', 0))),
			'senior_pivot': round(float(row['senior_pivot']), 4) if not pd.isna(row.get('senior_pivot')) else None
		}

	def _save_results(self, formatted):
		"""Сохранение результатов в файл"""
		if formatted:
			with open("confirmed_signals.json", 'w') as f:
				json.dump(formatted, f, indent=2, default=str)
			log_step(f"Сохранено {len(formatted)} сигналов")
		else:
			log_step("Нет подтвержденных сигналов для сохранения")

		# Исправленный метод test_multiple_timeframes
	# Добавить в начало файла
	import traceback

	# Исправленный метод test_multiple_timeframes
	def test_multiple_timeframes(self, junior_path, senior_path):
		"""Запуск анализа и сохранение результатов"""
		try:
			self.initialize_model()
			log_step("Инициализация модели завершена")

			# Проверка существования файлов
			if not os.path.exists(junior_path):
				raise FileNotFoundError(f"Файл не найден: {junior_path}")
			if not os.path.exists(senior_path):
				raise FileNotFoundError(f"Файл не найден: {senior_path}")

			log_step("Загрузка данных...")
			junior = pd.read_csv(junior_path)
			senior = pd.read_csv(senior_path)
			log_step(f"Загружено данных: Junior={len(junior)}, Senior={len(senior)}")

			# Проверка минимального размера данных
			min_bars = max(self.sequence_length * 2, 200)
			if len(junior) < min_bars or len(senior) < min_bars:
				msg = (f"Недостаточно данных. Требуется минимум {min_bars} баров. "
					   f"Junior: {len(junior)}, Senior: {len(senior)}")
				log_step(msg)
				raise ValueError(msg)

			# Генерация фичей
			log_step("Создание фичей для младшего ТФ...")
			junior = self.create_features(junior)
			log_step("Создание фичей для старшего ТФ...")
			senior = self.create_features(senior)
			log_step("Фичи успешно созданы")

			# Анализ
			log_step("Начало анализа...")
			results = self.analyze_timeframes(junior, senior)
			log_step(f"Анализ завершен. Найдено событий: {len(results)}")

			# Фильтрация и сохранение
			if not results.empty:
				confirmed_results = results[
					(results['confirmed'] == 'trend_confirmed') |
					(results['confirmed'] == 'correction_confirmed')
					]
				formatted = confirmed_results.to_dict('records')
			else:
				formatted = []

			# Сохранение
			if formatted:
				with open("confirmed_signals.json", 'w') as f:
					json.dump(formatted, f, indent=2)
				log_step(f"Сохранено {len(formatted)} подтвержденных сигналов")
			else:
				log_step("Нет подтвержденных сигналов")

			return formatted

		except Exception as e:
			log_step(f"Критическая ошибка: {str(e)}")
			log_step("Трассировка ошибки:")
			log_step(traceback.format_exc())
			return None

	# Исправленный метод analyze_timeframes
	def analyze_timeframes(self, junior, senior):
		"""Анализ с пошаговым логированием"""
		results = []
		total = len(junior) - self.sequence_length

		for i in range(self.sequence_length, len(junior)):
			try:
				if i % 100 == 0:
					log_step(f"Обработано {i}/{total} ({i / total:.1%})")

				# Получаем окно данных
				j_window = junior.iloc[i - self.sequence_length:i]
				s_index = max(0, i // self.timeframe_ratio)
				s_window = senior.iloc[s_index - self.sequence_length:s_index]

				# Предсказания
				j_trend = self.predict(j_window)['trend']
				s_trend = self.analyze_senior_tf(s_window)['trend']

				# Проверка подтверждения
				confirmation = self.check_confirmation(
					{'trend': j_trend, 'correction': 0},
					{'trend': s_trend, 'correction': 0}
				)

				# Сохраняем результат
				results.append({
					'junior_trend': j_trend,
					'senior_trend': s_trend,
					'confirmed': confirmation,
					'timestamp': i
				})

			except Exception as e:
				log_step(f"Ошибка обработки индекса {i}: {str(e)}")

		return pd.DataFrame(results)

	def test_model(self, test_path):
		"""Метод для тестирования модели с расширенной аналитикой"""
		try:
			log_step("Начало тестирования модели...")

			# Загрузка и подготовка данных
			test_data = self.load_and_preprocess_data(test_path)
			X_test, y_test = self.prepare_sequences(test_data)
			test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128)

			if len(test_loader.dataset) == 0:
				raise ValueError("Тестовые данные не содержат примеров")

			# Загрузка модели
			self.initialize_model()
			self.model.load_state_dict(torch.load(MODEL_PATH))
			self.model.eval()
			log_step(f"Модель загружена с {MODEL_PATH}")

			# Инициализация хранилища результатов
			all_preds = []
			all_true = []
			total = 0

			# Прогнозирование
			with torch.no_grad():
				for batch_x, batch_y in test_loader:
					batch_x = batch_x.to(self.device)
					y_trend = batch_y[:, 0].numpy()
					y_correction = batch_y[:, 1].numpy()

					# Предсказание
					trend_pred, correction_pred = self.model(batch_x)

					# Обработка результатов
					trend_pred = trend_pred.argmax(dim=1).cpu().numpy()
					correction_pred = (torch.sigmoid(correction_pred) > 0.5).cpu().numpy().astype(int)

					all_preds.extend(zip(trend_pred, correction_pred))
					all_true.extend(zip(y_trend, y_correction))
					total += len(y_trend)

			# Извлечение меток
			true_trend, true_correction = zip(*all_true) if all_true else ([], [])
			pred_trend, pred_correction = zip(*all_preds) if all_preds else ([], [])

			# Безопасное создание отчетов
			def safe_classification_report(y_true, y_pred, target_names):
				try:
					report = classification_report(
						y_true, y_pred,
						target_names=target_names,
						output_dict=True,
						zero_division=0
					)
					# Добавляем недостающие классы
					for name in target_names:
						if name not in report:
							report[name] = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
					return report
				except Exception as e:
					log_step(f"Ошибка создания отчета: {str(e)}")
					return {}

			# Генерация отчетов
			trend_report = safe_classification_report(
				true_trend, pred_trend,
				target_names=['Sideways', 'Uptrend', 'Downtrend']
			)

			correction_report = safe_classification_report(
				true_correction, pred_correction,
				target_names=['No Correction', 'Correction']
			)

			# Вывод статистики
			log_step("\nРезультаты тестирования:")
			log_step(f"Общее количество примеров: {total}")

			# Вывод для трендов
			if trend_report:
				log_step("\nТренды:")
				log_step(f"Accuracy: {trend_report.get('accuracy', 0):.2%}")
				for name in ['Uptrend', 'Downtrend', 'Sideways']:
					stats = trend_report.get(name, {})
					log_step(
						f"{name}: "
						f"Precision={stats.get('precision', 0):.2%} | "
						f"Recall={stats.get('recall', 0):.2%} | "
						f"F1={stats.get('f1-score', 0):.2%}"
					)

			# Вывод для коррекций
			if correction_report:
				log_step("\nКоррекции:")
				log_step(f"Accuracy: {correction_report.get('accuracy', 0):.2%}")
				for name in ['Correction', 'No Correction']:
					stats = correction_report.get(name, {})
					log_step(
						f"{name}: "
						f"Precision={stats.get('precision', 0):.2%} | "
						f"Recall={stats.get('recall', 0):.2%} | "
						f"F1={stats.get('f1-score', 0):.2%}"
					)

			return {
				'trend_report': trend_report,
				'correction_report': correction_report,
				'predictions': all_preds
			}

		except Exception as e:
			log_step(f"Критическая ошибка тестирования: {str(e)}")
			import traceback
			log_step(traceback.format_exc())
			return None


# Основной блок выполнения
if __name__ == "__main__":
	trainer = ModelTrainer()

	# Режим работы (train, test, test_multiple)
	mode = "test_multiple"

	if mode == "train":
		# Обычное обучение
		trainer.train_on_all_data(
			initial_epochs=50,
			incremental_epochs=10
		)
	# Исправляем тестовую визуализацию
	if mode == "test":
		test_predictions = trainer.test_model(TEST_DATA)
		if test_predictions is not None:
			test_data = pd.read_csv(TEST_DATA)
			test_data = test_data.iloc[trainer.sequence_length:]
	if mode == "test_multiple":
		# Пример использования
		junior_tf = JUNIOR_DATA_PATH
		senior_tf = SENIOR_DATA_PATH

		results = trainer.test_multiple_timeframes(junior_tf, senior_tf)

		if results:
			log_step("Пример события:")
			print(json.dumps(results[0], indent=2))
