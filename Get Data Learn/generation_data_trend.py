# trend_generator.py
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
import os


class TrendType(Enum):
	UPTREND = 1
	DOWNTREND = 2
	SIDEWAYS = 0


class TrendDataGenerator:
	def __init__(self, base_price_range=(50, 150), base_volume_range=(100, 1000)):
		self.base_price_range = base_price_range
		self.volume_range = base_volume_range

	def generate_single_trend(
			self,
			length=60,
			trend_type=TrendType.UPTREND,
			noise_level=1.0,
			trend_strength=0.5
	):
		"""Генерация одного тренда"""
		x = np.linspace(0, 1, length)

		# Базовые параметры
		base_price = np.random.uniform(*self.base_price_range)
		volatility = np.random.uniform(0.5, 2.0)

		# Генерация тренда
		if trend_type == TrendType.UPTREND:
			trend = np.linspace(0, trend_strength * 20, length)
		elif trend_type == TrendType.DOWNTREND:
			trend = np.linspace(0, -trend_strength * 20, length)
		else:  # Боковой тренд
			trend = np.zeros(length)
			volatility *= 2  # Увеличиваем волатильность

		# Генерация цен
		prices = base_price + trend + np.random.normal(0, volatility, length)
		noise = np.random.normal(0, noise_level, length)
		close_prices = prices + noise

		# Создание свечей
		data = {
			'open': close_prices - np.random.uniform(0.1, 0.5, length),
			'high': close_prices + np.abs(np.random.normal(0, 0.8, length)),
			'low': close_prices - np.abs(np.random.normal(0, 0.8, length)),
			'close': close_prices,
			'volume': np.random.randint(*self.volume_range, length),
			'trend': trend_type.value
		}

		return pd.DataFrame(data)

	def generate_dataset(
			self,
			n_samples=100,
			save_path='trend_data.csv',
			visualize=False
	):
		"""Генерация полного набора данных"""
		dfs = []
		trend_stats = {
			'uptrend': 0,
			'downtrend': 0,
			'sideways': 0
		}

		for _ in range(n_samples):
			# Выбор типа тренда
			trend_type = np.random.choice([
				TrendType.UPTREND,
				TrendType.DOWNTREND,
				TrendType.SIDEWAYS
			], p=[0.4, 0.4, 0.2])

			# Генерация данных
			df = self.generate_single_trend(
				trend_type=trend_type,
				noise_level=np.random.uniform(0.5, 2.0),
				trend_strength=np.random.uniform(0.3, 1.0)
			)
			dfs.append(df)

			# Обновление статистики
			if trend_type == TrendType.UPTREND:
				trend_stats['uptrend'] += 1
			elif trend_type == TrendType.DOWNTREND:
				trend_stats['downtrend'] += 1
			else:
				trend_stats['sideways'] += 1

		full_df = pd.concat(dfs, ignore_index=True)

		# Вывод статистики
		print("\nСтатистика сгенерированных данных:")
		print(f"Всего сегментов: {n_samples}")
		print(f"Восходящих трендов: {trend_stats['uptrend']}")
		print(f"Нисходящих трендов: {trend_stats['downtrend']}")
		print(f"Боковых движений: {trend_stats['sideways']}")
		print(f"Общее количество свечей: {len(full_df)}")

		# Сохранение данных
		full_df.to_csv(save_path, index=False)
		print(f"\nДанные сохранены в {os.path.abspath(save_path)}")

		# Визуализация
		if visualize:
			self._visualize_sample(full_df.iloc[:200])

		return full_df

	def _visualize_sample(self, df):
		"""Визуализация сгенерированных данных"""
		plt.figure(figsize=(15, 8))

		# Цена
		plt.plot(df['close'], label='Цена', alpha=0.7)

		# Разметка трендов
		colors = {
			0: 'gray',
			1: 'green',
			2: 'red'
		}

		for trend_type, color in colors.items():
			mask = df['trend'] == trend_type
			plt.scatter(
				df[mask].index,
				df[mask]['close'],
				color=color,
				alpha=0.3,
				label=[
					'Боковой',
					'Восходящий',
					'Нисходящий'
				][trend_type]
			)

		plt.title("Пример сгенерированных данных (первые 200 свечей)")
		plt.legend()
		plt.show()


# Пример использования
if __name__ == "__main__":
	OUTPUT_PATH = 'data/test/sintetic/trend_data.csv'
	generator = TrendDataGenerator()
	data = generator.generate_dataset(
		n_samples=500,
		save_path=OUTPUT_PATH,
		visualize=True
	)


# Статистика сгенерированных данных:
# Всего сегментов: 500
# Восходящих трендов: 191
# Нисходящих трендов: 218
# Боковых движений: 91
# Общее количество свечей: 30000