# pattern_generator.py
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
import os


class TrendType(Enum):
	BULLISH = 1
	BEARISH = 2


class TrianglePatternGenerator:
	def __init__(self, base_price_range=(90, 100), base_volume_range=(100, 1000)):
		"""
		Инициализация генератора тестовых данных

		:param base_price_range: диапазон базовых цен (мин, макс)
		:param base_volume_range: диапазон объемов (мин, макс)
		"""
		self.base_price_range = base_price_range
		self.volume_range = base_volume_range

	def validate_detector(self, df):
		detected = sum(df['pattern'] > 0)
		generated = sum(df['pattern'] > 0)  # Для синтетических данных
		print(f"Эффективность детектора: {detected / generated:.2%}")
		return df[df['pattern'] != df['detected_pattern']]  # Возврат расхождений

	def generate_single_pattern(
			self,
			length=60,
			trend_type=TrendType.BULLISH,
			noise_level=0.5,
			pattern_length=20,
			pattern_position='end'
	):
		"""
		Генерация одного паттерна

		:param length: общее количество свечей
		:param trend_type: тип паттерна (BULLISH/BEARISH)
		:param noise_level: уровень шума
		:param pattern_length: длина паттерна в свечах
		:param pattern_position: позиция паттерна ('start', 'middle', 'end')
		:return: DataFrame с сгенерированными данными
		"""
		x = np.linspace(0, 1, length)

		# Определяем позицию паттерна
		if pattern_position == 'start':
			pattern_start = 0
		elif pattern_position == 'middle':
			pattern_start = (length - pattern_length) // 2
		else:  # 'end'
			pattern_start = length - pattern_length

		pattern_end = pattern_start + pattern_length

		# Базовые линии цены
		base_price = np.random.uniform(*self.base_price_range)

		# Генерация линий треугольника
		# Увеличьте амплитуду паттернов
		if trend_type == TrendType.BULLISH:
			upper_line = np.linspace(base_price + 8, base_price, length)  # было +5
			lower_line = np.linspace(base_price - 8, base_price - 2, length)
		else:  # BEARISH
			upper_line = np.linspace(base_price + 8, base_price + 2, length)
			lower_line = np.linspace(base_price - 8, base_price, length)

		# Сужение паттерна только в заданной области
		for i in range(length):
			if not (pattern_start <= i < pattern_end):
				upper_line[i] = base_price + np.random.uniform(2, 5)
				lower_line[i] = base_price - np.random.uniform(2, 5)

		# Генерация цен
		close_prices = []
		for i in range(length):
			if pattern_start <= i < pattern_end:
				close = np.random.uniform(lower_line[i], upper_line[i])
			else:
				close = base_price + np.random.uniform(-3, 3)
			close_prices.append(close)

		# Добавление шума
		noise = np.random.normal(0, noise_level, length)
		close_prices = np.array(close_prices) + noise

		# Создание свечей и разметки
		pattern_labels = np.zeros(length)
		pattern_labels[pattern_start:pattern_end] = trend_type.value

		data = {
			'open': close_prices - np.random.uniform(0.1, 0.3, length),
			'high': close_prices + np.random.uniform(0.2, 0.5, length),
			'low': close_prices - np.random.uniform(0.2, 0.5, length),
			'close': close_prices,
			'volume': np.random.randint(*self.volume_range, length),
			'pattern': pattern_labels  # Используем заранее созданный массив меток
		}

		return pd.DataFrame(data)

	def generate_test_dataset(
			self,
			n_samples=100,
			pattern_length=20,
			save_path='triangle_patterns.csv',
			visualize=False
	):
		"""
		Генерация полного тестового набора данных

		:param n_samples: количество паттернов для генерации
		:param pattern_length: длина каждого паттерна
		:param save_path: путь для сохранения CSV файла
		:param visualize: визуализировать первые 200 свечей
		:return: сгенерированный DataFrame
		"""
		dfs = []
		pattern_stats = {
			'bullish': 0,
			'bearish': 0,
			'no_pattern': 0
		}

		for i in range(n_samples):
			# Случайный выбор типа паттерна
			trend_type = np.random.choice([TrendType.BULLISH, TrendType.BEARISH])

			# Генерация паттерна
			df = self.generate_single_pattern(
				trend_type=trend_type,
				pattern_length=pattern_length,
				noise_level=0.2,
				pattern_position=np.random.choice(['start', 'middle', 'end'])
			)
			dfs.append(df)

			# Обновляем статистику
			if trend_type == TrendType.BULLISH:
				pattern_stats['bullish'] += 1
			else:
				pattern_stats['bearish'] += 1

			# Добавление случайных данных без паттерна
			if i % 3 == 0:
				random_df = self.generate_single_pattern(
					pattern_length=0,  # Без паттерна
					noise_level=1.0  # Больше шума
				)
				dfs.append(random_df)
				pattern_stats['no_pattern'] += 1

		full_df = pd.concat(dfs, ignore_index=True)

		# Вывод статистики
		total_patterns = pattern_stats['bullish'] + pattern_stats['bearish']
		print("\nСтатистика сгенерированных данных:")
		print(f"Всего сегментов данных: {len(dfs)}")
		print(f"Из них содержат паттерны: {total_patterns}")
		print(f"  - Бычьих треугольников: {pattern_stats['bullish']}")
		print(f"  - Медвежьих треугольников: {pattern_stats['bearish']}")
		print(f"Случайных данных без паттернов: {pattern_stats['no_pattern']}")
		print(f"Общее количество свечей: {len(full_df)}")
		print(f"Из них помечено как паттерн: {sum(full_df['pattern'] > 0)}")

		# Сохранение в CSV
		full_df.to_csv(
			OUT_FILE_PATH,
			columns=[col for col in full_df.columns if col != 'pattern'],
			index=False
		)
		full_df.to_csv(save_path, index=False)
		print(f"\nДанные сохранены в {os.path.abspath(save_path)}")

		# Визуализация
		if visualize:
			self._visualize_sample(full_df[:200])

		return full_df

	def _visualize_sample(self, df):
		"""Визуализация сгенерированных данных"""
		plt.figure(figsize=(15, 6))
		plt.plot(df['close'], label='Цена')

		bullish = df[df['pattern'] == 1]
		bearish = df[df['pattern'] == 2]

		plt.scatter(
			bullish.index,
			bullish['close'],
			color='green',
			alpha=0.5,
			label='Бычий треугольник'
		)
		plt.scatter(
			bearish.index,
			bearish['close'],
			color='red',
			alpha=0.5,
			label='Медвежий треугольник'
		)

		plt.title("Пример сгенерированных данных (первые 200 свечей)")
		plt.legend()
		plt.show()


# Пример использования (можно раскомментировать для теста)
if __name__ == "__main__":
	OUT_FILE_PATH = 'data/test/sintetic/output.csv'
	RESULT_FILE_PATH = 'data/test/sintetic/result.csv'
	generator = TrianglePatternGenerator()
	data = generator.generate_test_dataset(
		n_samples=200,
		pattern_length=15,
		save_path=RESULT_FILE_PATH,
		visualize=True
	)


# Статистика сгенерированных данных:
# Всего сегментов данных: 267
# Из них содержат паттерны: 200
#   - Бычьих треугольников: 101
#   - Медвежьих треугольников: 99
# Случайных данных без паттернов: 67
# Общее количество свечей: 16020
# Из них помечено как паттерн: 3000