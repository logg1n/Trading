# Tools/trend_analyzer.py
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from indicator_manager import IndicatorManager


class TrendAnalyzer(IndicatorManager):
	@staticmethod
	def detect_trend(prices: List[float], window_short: int = 50, window_long: int = 200) -> str:
		"""Улучшенное определение тренда"""
		if len(prices) < window_long:
			return "insufficient_data"

		try:
			sma_short = IndicatorManager.SMA_series(prices, window_short)
			sma_long = IndicatorManager.SMA_series(prices, window_long)

			if len(sma_short) < 2 or len(sma_long) < 2:
				return "range"

			last_short = sma_short.iloc[-1] if hasattr(sma_short, 'iloc') else sma_short[-1]
			last_long = sma_long.iloc[-1] if hasattr(sma_long, 'iloc') else sma_long[-1]
			prev_short = sma_short.iloc[-2] if hasattr(sma_short, 'iloc') and len(sma_short) > 1 else last_short
			prev_long = sma_long.iloc[-2] if hasattr(sma_long, 'iloc') and len(sma_long) > 1 else last_long

			price_momentum = prices[-1] > prices[-min(5, len(prices) - 1)]

			if last_short > last_long and prev_short > prev_long and price_momentum:
				return "uptrend"
			elif last_short < last_long and prev_short < prev_long and not price_momentum:
				return "downtrend"

			return "range"

		except Exception as e:
			print(f"Ошибка в detect_trend: {str(e)}")
			return "range"

	@staticmethod
	def detect_correction(high_prices: List[float], low_prices: List[float],
						  close_prices: List[float], threshold: float = 0.38) -> bool:
		"""Улучшенное определение коррекции"""
		if len(high_prices) < 10 or len(low_prices) < 10:
			return False

		try:
			lookback = min(20, len(high_prices))
			recent_high = max(high_prices[-lookback:])
			recent_low = min(low_prices[-lookback:])
			current_close = close_prices[-1]

			price_range = recent_high - recent_low
			if price_range == 0:
				return False

			if current_close < recent_high - threshold * price_range:
				return True
			elif current_close > recent_low + threshold * price_range:
				return True

			return False

		except Exception as e:
			print(f"Ошибка в detect_correction: {e}")
			return False

	@staticmethod
	def calculate_trend_strength(prices: List[float], volume: List[float] = None) -> float:
		"""Расчет силы тренда (0-1)"""
		if len(prices) < 20:
			return 0.0

		try:
			price_change = (prices[-1] - prices[-20]) / prices[-20]
			volatility = np.std([(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))])

			if volatility > 0:
				strength = min(abs(price_change) / volatility, 1.0)
			else:
				strength = abs(price_change)

			return float(strength)

		except Exception as e:
			print(f"Ошибка в calculate_trend_strength: {e}")
			return 0.0

	@classmethod
	def analyze_market(cls, df: pd.DataFrame) -> Dict[str, Union[str, bool, float, Dict]]:
		"""Комплексный анализ рыночной ситуации"""
		try:
			prices = df['close'].tolist() if 'close' in df.columns else []
			high = df['high'].tolist() if 'high' in df.columns else []
			low = df['low'].tolist() if 'low' in df.columns else []
			volume = df['volume'].tolist() if 'volume' in df.columns else None

			if len(prices) < 50:
				return {
					'trend': 'insufficient_data',
					'correction': False,
					'rsi': 50.0,
					'volatility': 0.0,
					'trend_strength': 0.0,
					'analysis_quality': 'low'
				}

			trend = cls.detect_trend(prices)
			correction = cls.detect_correction(high, low, prices)
			rsi_value = cls.RSI(prices)
			volatility_value = cls.VOLATILITY_value(df) if len(df) > 0 else 0.0
			trend_strength = cls.calculate_trend_strength(prices, volume)

			sma_50 = cls.SMA(prices, 50)
			sma_200 = cls.SMA(prices, 200) if len(prices) >= 200 else sma_50
			price_vs_sma50 = (prices[-1] / sma_50 - 1) * 100 if sma_50 else 0

			analysis_quality = 'high' if len(prices) >= 200 else 'medium' if len(prices) >= 100 else 'low'

			return {
				'trend': trend,
				'correction': correction,
				'rsi': round(rsi_value, 2),
				'volatility': round(volatility_value, 2),
				'trend_strength': round(trend_strength, 3),
				'sma_50': round(sma_50, 4) if sma_50 else None,
				'sma_200': round(sma_200, 4) if sma_200 else None,
				'price_vs_sma50_pct': round(price_vs_sma50, 2),
				'analysis_quality': analysis_quality,
				'data_points': len(prices)
			}

		except Exception as e:
			print(f"Ошибка в analyze_market: {e}")
			return {
				'trend': 'error',
				'correction': False,
				'rsi': 50.0,
				'volatility': 0.0,
				'trend_strength': 0.0,
				'analysis_quality': 'error'
			}