import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 output_size: int = 3,
                 num_layers: int = 3,
                 lstm_dropout: float = 0.4,
                 fc_dropout: float = 0.4,
                 attention_hidden: int = None,
                 bidirectional: bool = False,
                 apply_softmax: bool = True):
        """

        Args:
                input_size: Размер входных признаков
                hidden_size: Размер скрытого состояния LSTM
                output_size: Размер выходного слоя
                num_layers: Количество слоев LSTM
                lstm_dropout: Dropout для LSTM
                fc_dropout: Dropout перед выходным слоем
                attention_hidden: Размер скрытого слоя механизма внимания (None = hidden_size)
                bidirectional: Использовать двунаправленную LSTM
                apply_softmax: Применять softmax к выходу
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.apply_softmax = apply_softmax
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM слой
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Механизм внимания
        attention_hidden = hidden_size if attention_hidden is None else attention_hidden
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
            nn.Softmax(dim=1)
        )

        # Выходной слой
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        self.dropout = nn.Dropout(fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass модели

        Args:
                x: Входной тензор [batch_size, seq_len, input_size]

        Returns:
                Выходной тензор [batch_size, output_size]
        """
        # Проверка типа данных
        x = x.float()

        # LSTM слой
        # [batch_size, seq_len, hidden_size * num_directions]
        lstm_out, _ = self.lstm(x)

        # Механизм внимания
        attention_weights = self.attention(
            lstm_out)  # [batch_size, seq_len, 1]
        # [batch_size, seq_len]
        attention_weights = attention_weights.squeeze(-1)

        # Взвешенная сумма
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(
            1)  # [batch_size, hidden_size * num_directions]

        # Классификация
        out = self.dropout(context)
        out = self.fc(out)

        if self.apply_softmax:
            return F.softmax(out, dim=1)
        return out

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает веса внимания для визуализации"""
        with torch.no_grad():
            x = x.float()
            lstm_out, _ = self.lstm(x)
            attention_weights = self.attention(lstm_out).squeeze(-1)
            return attention_weights
