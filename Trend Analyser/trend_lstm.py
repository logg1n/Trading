# trend_lstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class AttentionLayer(nn.Module):
	"""
	Механизм внимания для взвешивания важных временных шагов в последовательности.
	"""

	def __init__(self, hidden_dim: int):
		super().__init__()
		self.attention = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.Tanh(),
			nn.Linear(hidden_dim // 2, 1),
			nn.Softmax(dim=1)
		)

	def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
		# lstm_output shape: (batch_size, seq_len, hidden_dim)
		attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
		weighted_output = torch.sum(lstm_output * attention_weights, dim=1)
		return weighted_output


class ResidualBlock(nn.Module):
	"""
	Остаточный блок для улучшения градиентного потока в глубоких сетях.
	"""

	def __init__(self, hidden_dim: int, dropout: float = 0.3):
		super().__init__()
		self.block = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim)
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = x
		out = self.block(x)
		out += residual  # Residual connection
		out = F.relu(out)
		out = self.dropout(out)
		return out


class TrendLSTMModel(nn.Module):
	"""
	Улучшенная LSTM модель для прогнозирования трендов и коррекций.
	"""

	def __init__(self,
				 input_size: int,
				 hidden_size: int = 128,
				 num_layers: int = 2,
				 num_trend_classes: int = 3,
				 use_attention: bool = True,
				 use_residual: bool = True,
				 dropout: float = 0.3):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_attention = use_attention
		self.use_residual = use_residual

		# Многослойная двунаправленная LSTM
		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			bidirectional=True,
			dropout=dropout if num_layers > 1 else 0.0
		)

		# Dropout слои
		self.lstm_dropout = nn.Dropout(dropout)
		self.feature_dropout = nn.Dropout(dropout)

		# Механизм внимания
		if use_attention:
			self.attention = AttentionLayer(hidden_size * 2)

		# Общие features после LSTM
		lstm_output_size = hidden_size * 2  # bidirectional

		# Остаточные блоки
		if use_residual:
			self.residual_blocks = nn.Sequential(
				ResidualBlock(lstm_output_size, dropout),
				ResidualBlock(lstm_output_size, dropout)
			)

		# Головка для классификации тренда
		self.trend_head = nn.Sequential(
			nn.Linear(lstm_output_size, lstm_output_size // 2),
			nn.BatchNorm1d(lstm_output_size // 2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
			nn.BatchNorm1d(lstm_output_size // 4),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(lstm_output_size // 4, num_trend_classes)
		)

		# Головка для классификации коррекции
		self.correction_head = nn.Sequential(
			nn.Linear(lstm_output_size, lstm_output_size // 2),
			nn.BatchNorm1d(lstm_output_size // 2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(lstm_output_size // 2, 1),
			nn.Sigmoid()
		)

		# Вспомогательные головки для регуляризации
		self.volatility_head = nn.Sequential(
			nn.Linear(lstm_output_size, lstm_output_size // 4),
			nn.ReLU(),
			nn.Linear(lstm_output_size // 4, 1),
			nn.ReLU()
		)

		# Инициализация весов - ИСПРАВЛЕННАЯ ВЕРСИЯ
		self._initialize_weights()

	def _initialize_weights(self):
		"""
		ИСПРАВЛЕННАЯ инициализация весов.
		"""
		for name, module in self.named_modules():
			if isinstance(module, nn.Linear):
				# Инициализация линейных слоев
				nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
				if module.bias is not None:
					nn.init.constant_(module.bias, 0.1)
			elif isinstance(module, nn.LSTM):
				# Инициализация LSTM слоев
				for name_param, param in module.named_parameters():
					if 'weight_ih' in name_param:
						nn.init.xavier_uniform_(param.data)
					elif 'weight_hh' in name_param:
						nn.init.orthogonal_(param.data)
					elif 'bias' in name_param:
						nn.init.constant_(param.data, 0)
						# Установка bias forget gate в 1 для улучшения обучения
						n = param.size(0)
						param.data[(n // 4):(n // 2)].fill_(1.0)
			elif isinstance(module, nn.BatchNorm1d):
				# Инициализация BatchNorm
				nn.init.constant_(module.weight, 1)
				nn.init.constant_(module.bias, 0)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
		"""
		Прямой проход модели.
		"""
		batch_size, seq_len, _ = x.shape

		# LSTM слой
		lstm_out, (hidden, cell) = self.lstm(x)
		lstm_out = self.lstm_dropout(lstm_out)

		# Применение механизма внимания или использование последнего скрытого состояния
		if self.use_attention:
			context_vector = self.attention(lstm_out)
		else:
			context_vector = lstm_out[:, -1, :]

		# Применение остаточных блоков
		if self.use_residual:
			context_vector = self.residual_blocks(context_vector)

		context_vector = self.feature_dropout(context_vector)

		# Прогноз тренда
		trend_logits = self.trend_head(context_vector)

		# Прогноз коррекции
		correction_probs = self.correction_head(context_vector)

		# Вспомогательные прогнозы
		volatility_pred = self.volatility_head(context_vector)

		# Дополнительная информация для мониторинга
		auxiliary_outputs = {
			'context_vector': context_vector,
			'volatility_pred': volatility_pred,
			'lstm_norm': torch.norm(lstm_out, dim=2).mean() if lstm_out.numel() > 0 else torch.tensor(0.0)
		}

		if self.use_attention:
			with torch.no_grad():
				attention_weights = self.attention.attention(lstm_out)
				auxiliary_outputs['attention_weights'] = attention_weights

		return trend_logits, correction_probs.squeeze(-1), auxiliary_outputs

	def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Возвращает веса внимания для визуализации.
		"""
		if not self.use_attention:
			raise ValueError("Attention mechanism is not enabled")

		with torch.no_grad():
			lstm_out, _ = self.lstm(x)
			attention_weights = self.attention.attention(lstm_out)
			return attention_weights.squeeze(-1)

	def predict_proba(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
		"""
		Возвращает вероятности предсказаний.
		"""
		with torch.no_grad():
			trend_logits, correction_probs, aux_outputs = self.forward(x)

			trend_probs = F.softmax(trend_logits, dim=1)

			return {
				'trend_probs': trend_probs,
				'correction_probs': correction_probs,
				'trend_pred': torch.argmax(trend_probs, dim=1),
				'correction_pred': (correction_probs > 0.5).long(),
				'confidence': torch.max(trend_probs, dim=1)[0] * correction_probs
			}


class MultiTimeframeTrendModel(nn.Module):
	"""
	Модель для анализа нескольких таймфреймов одновременно.
	"""

	def __init__(self, input_size: int, hidden_size: int = 128, num_timeframes: int = 3):
		super().__init__()

		self.num_timeframes = num_timeframes

		# Отдельные модели для каждого таймфрейма
		self.timeframe_models = nn.ModuleList([
			TrendLSTMModel(input_size, hidden_size)
			for _ in range(num_timeframes)
		])

		# Слой для объединения признаков
		fusion_size = hidden_size * 2 * num_timeframes
		self.fusion_layer = nn.Sequential(
			nn.Linear(fusion_size, fusion_size // 2),
			nn.BatchNorm1d(fusion_size // 2),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(fusion_size // 2, hidden_size * 2),
			nn.BatchNorm1d(hidden_size * 2),
			nn.ReLU()
		)

		# Финальные головки
		self.trend_head = nn.Linear(hidden_size * 2, 3)
		self.correction_head = nn.Sequential(
			nn.Linear(hidden_size * 2, 1),
			nn.Sigmoid()
		)

		# Инициализация весов
		self._initialize_weights()

	def _initialize_weights(self):
		"""Инициализация весов для multi-timeframe модели"""
		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
				if module.bias is not None:
					nn.init.constant_(module.bias, 0.1)
			elif isinstance(module, nn.BatchNorm1d):
				nn.init.constant_(module.weight, 1)
				nn.init.constant_(module.bias, 0)

	def forward(self, multi_timeframe_inputs: Tuple[torch.Tensor, ...]):
		"""
		Args:
			multi_timeframe_inputs: кортеж тензоров для каждого таймфрейма
		"""
		if len(multi_timeframe_inputs) != self.num_timeframes:
			raise ValueError(f"Expected {self.num_timeframes} timeframes, got {len(multi_timeframe_inputs)}")

		# Получаем контекстные векторы от каждой модели таймфрейма
		context_vectors = []
		for i, x in enumerate(multi_timeframe_inputs):
			_, _, aux_outputs = self.timeframe_models[i](x)
			context_vectors.append(aux_outputs['context_vector'])

		# Объединяем все контекстные векторы
		fused_features = torch.cat(context_vectors, dim=1)
		fused_features = self.fusion_layer(fused_features)

		# Финальные предсказания
		trend_logits = self.trend_head(fused_features)
		correction_probs = self.correction_head(fused_features).squeeze(-1)

		return trend_logits, correction_probs


# Утилиты для работы с моделью
def count_parameters(model: nn.Module) -> int:
	"""Подсчет количества обучаемых параметров модели."""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
	"""Создание текстового описания архитектуры модели."""
	summary = []
	summary.append(f"Model: {model.__class__.__name__}")
	summary.append(f"Total parameters: {count_parameters(model):,}")
	summary.append("\nLayer breakdown:")

	for name, module in model.named_children():
		num_params = sum(p.numel() for p in module.parameters())
		summary.append(f"  {name}: {num_params:,} parameters")

	return "\n".join(summary)


# Упрощенная версия для тестирования
class SimpleTrendLSTMModel(nn.Module):
	"""
	Упрощенная версия модели для быстрого тестирования.
	"""

	def __init__(self, input_size, hidden_size=64, num_trend_classes=3):
		super().__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
		self.dropout = nn.Dropout(0.3)

		# Упрощенные головки
		self.trend_head = nn.Linear(hidden_size * 2, num_trend_classes)
		self.correction_head = nn.Sequential(
			nn.Linear(hidden_size * 2, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		lstm_out, _ = self.lstm(x)
		last_output = lstm_out[:, -1, :]
		last_output = self.dropout(last_output)

		trend = self.trend_head(last_output)
		correction = self.correction_head(last_output)

		return trend, correction.squeeze(-1), {}