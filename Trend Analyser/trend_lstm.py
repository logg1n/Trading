import torch
import torch.nn as nn
import torch.nn.functional as F


class TrendLSTMModel(nn.Module):
	def __init__(self, input_size, hidden_size = 128, num_trend_classes=3):
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			batch_first=True,
			bidirectional=True
		)
		# Добавить в модель:
		self.dropout = nn.Dropout(0.5)  # После LSTM слоя

		# Мультитаргетные выходы
		self.trend_head = nn.Sequential(
			nn.Linear(hidden_size * 2, 32),
			nn.ReLU(),
			nn.Linear(32, num_trend_classes)
		)

		# Исправленный выход для коррекции (1 нейрон вместо 2)
		self.correction_head = nn.Sequential(
			nn.Linear(hidden_size * 2, 16),
			nn.ReLU(),
			nn.Linear(16, 1)  # Изменено с 2 на 1
		)

	def forward(self, x):
		lstm_out, _ = self.lstm(x)
		# В метод forward:
		lstm_out = self.dropout(lstm_out)
		last_output = lstm_out[:, -1, :]

		trend = self.trend_head(last_output)
		correction = self.correction_head(last_output)

		return trend, correction