# indicator_manager.py
import pandas as pd
import numpy as np
from typing import List, Union, Optional
import warnings


class IndicatorManager:
	"""
	Улучшенный менеджер технических индикаторов с обработкой ошибок и расширенной функциональностью.

	Улучшения:
	- Единообразная обработка граничных случаев
	- Расширенные типы возвращаемых значений
	- Дополнительные индикаторы для ML-моделей
	- Улучшенная обработка ошибок
	- Оптимизация производительности
	"""

	@staticmethod
	def SMA(prices: List[float], window: int) -> Optional[float]:
		"""
		Возвращает последнее значение простой скользящей средней (SMA).

		Улучшения:
		- Проверка валидности входных данных
		- Защита от окон больше размера данных
		- Возврат None вместо np.nan для лучшей совместимости
		"""
		if not prices or len(prices) == 0:
			return None

		if window <= 0:
			raise ValueError("Window must be positive")

		try:
			# Используем min_periods=1 для постепенного заполнения
			window = min(window, len(prices))
			series = pd.Series(prices)
			sma = series.rolling(window=window, min_periods=1).mean()
			return float(sma.iloc[-1])
		except Exception as e:
			warnings.warn(f"SMA calculation error: {str(e)}")
			return None

	@staticmethod
	def SMA_series(prices: List[float], window: int) -> pd.Series:
		"""
		Возвращает серию значений SMA с улучшенной обработкой ошибок.
		"""
		if not prices or len(prices) == 0:
			return pd.Series([], dtype=float)

		if window <= 0:
			raise ValueError("Window must be positive")

		try:
			window = min(window, len(prices))
			series = pd.Series(prices)
			return series.rolling(window=window, min_periods=1).mean()
		except Exception as e:
			warnings.warn(f"SMA series calculation error: {str(e)}")
			return pd.Series([], dtype=float)

	@staticmethod
	def EMA(prices: List[float], window: int) -> Optional[float]:
		"""
		Возвращает последнее значение экспоненциальной скользящей средней (EMA).

		Улучшения:
		- Проверка минимального количества данных
		- Обработка edge cases
		"""
		if not prices or len(prices) == 0:
			return None

		if window <= 0:
			raise ValueError("Window must be positive")

		try:
			# Для EMA нужно минимум 2 точки для разумного расчета
			if len(prices) < 2:
				return float(prices[-1]) if prices else None

			series = pd.Series(prices)
			ema = series.ewm(span=window, adjust=False, min_periods=1).mean()
			return float(ema.iloc[-1])
		except Exception as e:
			warnings.warn(f"EMA calculation error: {str(e)}")
			return None

	@staticmethod
	def EMA_series(prices: List[float], window: int) -> pd.Series:
		"""
		Возвращает серию значений EMA с улучшенной стабильностью.
		"""
		if not prices or len(prices) == 0:
			return pd.Series([], dtype=float)

		if window <= 0:
			raise ValueError("Window must be positive")

		try:
			series = pd.Series(prices)
			return series.ewm(span=window, adjust=False, min_periods=1).mean()
		except Exception as e:
			warnings.warn(f"EMA series calculation error: {str(e)}")
			return pd.Series([], dtype=float)

	@staticmethod
	def RSI(prices: List[float], window: int = 14) -> Optional[float]:
		"""
		Улучшенный расчет RSI с расширенной обработкой ошибок.

		Улучшения:
		- Защита от деления на ноль
		- Граничные случаи для малых наборов данных
		- Валидация входных параметров
		"""
		if not prices or len(prices) < 2:
			return 50.0  # Нейтральное значение при недостатке данных

		if window <= 0:
			raise ValueError("RSI window must be positive")

		try:
			series = pd.Series(prices)
			delta = series.diff()

			# Separate gains and losses
			gain = delta.where(delta > 0, 0.0)
			loss = -delta.where(delta < 0, 0.0)

			# Calculate rolling averages
			avg_gain = gain.rolling(window=window, min_periods=1).mean()
			avg_loss = loss.rolling(window=window, min_periods=1).mean()

			# Avoid division by zero
			if avg_loss.iloc[-1] == 0:
				return 100.0 if avg_gain.iloc[-1] > 0 else 50.0

			rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
			rsi = 100 - (100 / (1 + rs))

			# Ensure valid range
			return max(0, min(100, rsi))

		except Exception as e:
			warnings.warn(f"RSI calculation error: {str(e)}")
			return 50.0  # Возврат нейтрального значения при ошибке

	@staticmethod
	def RSI_series(prices: List[float], window: int = 14) -> pd.Series:
		"""
		Улучшенный расчет серии RSI с защитой от ошибок.
		"""
		if not prices or len(prices) < 2:
			return pd.Series([50.0] * len(prices) if prices else [])

		if window <= 0:
			raise ValueError("RSI window must be positive")

		try:
			series = pd.Series(prices)
			delta = series.diff()

			gain = delta.where(delta > 0, 0.0)
			loss = -delta.where(delta < 0, 0.0)

			avg_gain = gain.rolling(window=window, min_periods=1).mean()
			avg_loss = loss.rolling(window=window, min_periods=1).mean()

			# Безопасное вычисление RS
			rs = avg_gain / avg_loss.replace(0, np.nan)
			rs = rs.fillna(0)  # Когда avg_loss = 0, RS = 0

			rsi = 100 - (100 / (1 + rs))
			rsi = rsi.fillna(50)  # Заполняем NaN нейтральным значением

			# Обработка случая, когда avg_loss = 0 и avg_gain > 0
			zero_loss_mask = (avg_loss == 0) & (avg_gain > 0)
			rsi = rsi.where(~zero_loss_mask, 100.0)

			return rsi

		except Exception as e:
			warnings.warn(f"RSI series calculation error: {str(e)}")
			return pd.Series([50.0] * len(prices))

	@staticmethod
	def VOLATILITY(df: pd.DataFrame,
				   period: int = 100,
				   price_col: str = "close",
				   annualized: bool = True,
				   min_volatility: float = None) -> pd.Series:
		"""
		Улучшенный расчет волатильности с расширенными опциями.

		Улучшения:
		- Проверка наличия колонок
		- Обработка недостаточных данных
		- Гибкие опции нормализации
		"""
		if price_col not in df.columns:
			raise ValueError(f"Колонка {price_col} не найдена в DataFrame")

		if len(df) < 2:
			return pd.Series([], dtype=float)

		try:
			# Вычисляем логарифмические доходности
			log_returns = np.log(df[price_col] / df[price_col].shift(1))

			# Рассчет стандартного отклонения
			volatility = log_returns.rolling(window=period, min_periods=1).std()

			if annualized:
				# Масштабирование до годового значения (252 торговых дня)
				volatility *= np.sqrt(252)

			# Конвертация в проценты
			volatility = volatility * 100

			# Применение минимального порога
			if min_volatility is not None:
				if not isinstance(min_volatility, (int, float)) or min_volatility < 0:
					raise ValueError("min_volatility должен быть положительным числом")
				volatility = volatility.where(volatility >= min_volatility, np.nan)

			return volatility.fillna(0)  # Заменяем NaN на 0 для совместимости

		except Exception as e:
			warnings.warn(f"Volatility calculation error: {str(e)}")
			return pd.Series([0.0] * len(df))

	@staticmethod
	def VOLATILITY_value(df: pd.DataFrame,
						 period: int = 100,
						 price_col: str = "close",
						 annualized: bool = True,
						 min_volatility: float = None) -> float:
		"""
		Возвращает последнее значение волатильности с обработкой ошибок.
		"""
		try:
			vol_series = IndicatorManager.VOLATILITY(df, period, price_col, annualized, min_volatility)
			return float(vol_series.iloc[-1]) if len(vol_series) > 0 else 0.0
		except Exception as e:
			warnings.warn(f"Volatility value calculation error: {str(e)}")
			return 0.0

	@staticmethod
	def MACD(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
		"""
		Расчет MACD (Moving Average Convergence Divergence).
		Возвращает словарь с MACD, signal line и histogram.
		"""
		if not prices or len(prices) < slow:
			return {'macd': None, 'signal': None, 'histogram': None}

		try:
			ema_fast = IndicatorManager.EMA_series(prices, fast)
			ema_slow = IndicatorManager.EMA_series(prices, slow)

			macd_line = ema_fast - ema_slow
			signal_line = macd_line.ewm(span=signal, adjust=False).mean()
			histogram = macd_line - signal_line

			return {
				'macd': float(macd_line.iloc[-1]) if not macd_line.empty else None,
				'signal': float(signal_line.iloc[-1]) if not signal_line.empty else None,
				'histogram': float(histogram.iloc[-1]) if not histogram.empty else None
			}
		except Exception as e:
			warnings.warn(f"MACD calculation error: {str(e)}")
			return {'macd': None, 'signal': None, 'histogram': None}

	@staticmethod
	def BollingerBands(prices: List[float], window: int = 20, num_std: float = 2) -> dict:
		"""
		Расчет полос Боллинджера.
		"""
		if not prices or len(prices) < window:
			return {'upper': None, 'middle': None, 'lower': None, 'bandwidth': None}

		try:
			series = pd.Series(prices)
			middle = series.rolling(window=window, min_periods=1).mean()
			std = series.rolling(window=window, min_periods=1).std()

			upper = middle + (std * num_std)
			lower = middle - (std * num_std)

			# Bandwidth calculation
			bandwidth = ((upper - lower) / middle) * 100

			return {
				'upper': float(upper.iloc[-1]),
				'middle': float(middle.iloc[-1]),
				'lower': float(lower.iloc[-1]),
				'bandwidth': float(bandwidth.iloc[-1]) if not bandwidth.empty else None
			}
		except Exception as e:
			warnings.warn(f"Bollinger Bands calculation error: {str(e)}")
			return {'upper': None, 'middle': None, 'lower': None, 'bandwidth': None}

	@staticmethod
	def ATR(df: pd.DataFrame, period: int = 14) -> Union[float, pd.Series]:
		"""
		Average True Range - индикатор волатильности.
		"""
		if len(df) < 2:
			return 0.0 if isinstance(df, pd.DataFrame) else pd.Series([], dtype=float)

		try:
			high = df['high']
			low = df['low']
			close = df['close']

			# Calculate True Range
			tr1 = high - low
			tr2 = abs(high - close.shift(1))
			tr3 = abs(low - close.shift(1))

			true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
			atr = true_range.rolling(window=period, min_periods=1).mean()

			return float(atr.iloc[-1]) if isinstance(df, pd.DataFrame) else atr

		except Exception as e:
			warnings.warn(f"ATR calculation error: {str(e)}")
			return 0.0 if isinstance(df, pd.DataFrame) else pd.Series([], dtype=float)

	@staticmethod
	def VolumeProfile(volume: List[float], prices: List[float], bins: int = 20) -> dict:
		"""
		Анализ объемного профиля.
		"""
		if not volume or not prices or len(volume) != len(prices):
			return {'poc_price': None, 'poc_volume': None, 'value_area': None}

		try:
			# Создаем гистограмму объемов по ценам
			price_min, price_max = min(prices), max(prices)
			bin_edges = np.linspace(price_min, price_max, bins + 1)

			volume_profile = {}
			for i in range(len(bin_edges) - 1):
				mask = (prices >= bin_edges[i]) & (prices < bin_edges[i + 1])
				bin_volume = sum(np.array(volume)[mask])
				volume_profile[f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}"] = bin_volume

			# Point of Control (POC) - цена с максимальным объемом
			poc_bin = max(volume_profile, key=volume_profile.get)
			poc_price = (float(poc_bin.split('-')[0]) + float(poc_bin.split('-')[1])) / 2

			return {
				'poc_price': poc_price,
				'poc_volume': volume_profile[poc_bin],
				'value_area': volume_profile
			}
		except Exception as e:
			warnings.warn(f"Volume profile calculation error: {str(e)}")
			return {'poc_price': None, 'poc_volume': None, 'value_area': None}

	@staticmethod
	def get_technical_indicators(df: pd.DataFrame) -> dict:
		"""
		Комплексный расчет всех технических индикаторов для DataFrame.
		Идеально для подготовки фич для ML-моделей.
		"""
		if df.empty:
			return {}

		try:
			prices = df['close'].tolist() if 'close' in df.columns else []
			highs = df['high'].tolist() if 'high' in df.columns else []
			lows = df['low'].tolist() if 'low' in df.columns else []
			volumes = df['volume'].tolist() if 'volume' in df.columns else []

			indicators = {
				# Трендовые индикаторы
				'sma_20': IndicatorManager.SMA(prices, 20),
				'sma_50': IndicatorManager.SMA(prices, 50),
				'sma_200': IndicatorManager.SMA(prices, 200),
				'ema_12': IndicatorManager.EMA(prices, 12),
				'ema_26': IndicatorManager.EMA(prices, 26),

				# Моментум
				'rsi': IndicatorManager.RSI(prices),
				'macd': IndicatorManager.MACD(prices),

				# Волатильность
				'volatility': IndicatorManager.VOLATILITY_value(df),
				'atr': IndicatorManager.ATR(df),
				'bollinger_bands': IndicatorManager.BollingerBands(prices),

				# Объемный анализ
				'volume_profile': IndicatorManager.VolumeProfile(volumes, prices) if volumes and prices else {},

				# Производные метрики
				'price_vs_sma20': (prices[-1] / IndicatorManager.SMA(prices, 20) - 1) * 100 if prices and len(
					prices) >= 20 else None,
				'price_vs_sma50': (prices[-1] / IndicatorManager.SMA(prices, 50) - 1) * 100 if prices and len(
					prices) >= 50 else None,
			}

			# Очистка от None значений
			return {k: v for k, v in indicators.items() if v is not None}

		except Exception as e:
			warnings.warn(f"Technical indicators calculation error: {str(e)}")
			return {}