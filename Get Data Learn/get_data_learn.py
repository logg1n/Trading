import argparse
import time
import pandas as pd
import os

from pathlib import Path
from typing import List, Dict
from datetime import datetime

from bybit_client import BybitTradingClient
from cursor import Cursor


def parse_arguments():
	parser = argparse.ArgumentParser(
		description="Скрипт для получения исторических данных с Bybit"
	)
	parser.add_argument('--pair_tf', type=str, nargs='+', required=True,
						help='Таймфрейм (например "BTCUSDT 60" или "SOLUSDT 15 60 240 BTCUSDT 15")')
	parser.add_argument('--date', type=str, required=True,
						help='Дата начала в формате DD.MM.YYYY')

	return parser.parse_args()


def fetch_data(client: BybitTradingClient, cursor: Cursor, pair: str, tf: str):
	while True:
		try:
			data = client.get_historical_data(
				symbol=pair,
				timeframe=tf,
				start=cursor.curs,
			)

			if data is None or data.empty:
				print(f"Нет данных для {pair} {tf}")
				break

			yield data
			cursor.next_cursor()
			time.sleep(1)  # Задержка для API

		except Exception as e:
			print(f"Ошибка для {pair} {tf}: {str(e)}")
			break


def save_to_file(data: pd.DataFrame, pair: str, tf: str):
	"""
	Сохраняет данные по пути: data/{пара}/{пара}_{tf}.csv
	Пример: data/BTCUSDT/BTCUSDT_15m.csv
	"""
	try:
		# Создаем путь к директории
		dir_path = Path("data") / pair
		dir_path.mkdir(parents=True, exist_ok=True)

		# Формируем имя файла
		filename = f"{pair}_{tf}.csv"
		filepath = dir_path / filename

		# Проверка входных данных
		if not isinstance(data, pd.DataFrame) or data.empty:
			print("Ошибка: Неверный формат данных или пустой DataFrame")
			return

		# Сохранение данных
		if filepath.exists():
			# Загрузка существующих данных
			existing = pd.read_csv(filepath)

			# Объединение и удаление дубликатов, если времени нет, используем индекс
			combined = pd.concat([existing, data])

			# Сохраняем только если есть изменения
			if len(combined) > len(existing):
				combined.to_csv(filepath, index=False)
				print(f"Обновлен файл: {filepath}")
				print(f"Добавлено строк: {len(combined) - len(existing)}")
				print(f"Всего строк: {len(combined)}")
			else:
				print("Нет новых данных для сохранения")
		else:
			data.to_csv(filepath, index=False)
			print(f"Создан новый файл: {filepath}")
			print(f"Сохранено строк: {len(data)}")

	except Exception as e:
		print(f"Критическая ошибка при сохранении: {str(e)}")


def main():
	global args
	args = parse_arguments()

	client = BybitTradingClient()
	parts = " ".join(args.pair_tf).split()

	pairs = {}
	current_key = None

	for part in parts:
		if part.isalpha():  # Определяем, что это торговая пара
			current_key = part.upper()
			pairs[current_key] = []  # Всегда создаем список
		else:
			pairs[current_key].append(int(part))


	for pair, timeframes in pairs.items():
		for tf in timeframes:
			print(f"\nОбработка {pair} {tf}...")
			cursor = Cursor(tf, args.date)

			for chunk in fetch_data(client, cursor, pair, tf):
				save_to_file(chunk, pair, tf)


if __name__ == "__main__":
	main()