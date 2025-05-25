import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from datetime import datetime

from lstm import LSTMModel
from Patterns.triangle import Triangle
from Patterns.triangle import TrendType
from Tools.indicator_manager import IndicatorManager


# Определение функции логирования
def log_step(message):
   timestamp = datetime.now().strftime('%H:%M:%S')
   print(f"[{timestamp}] {message}")

# Класс для управления обучением и дообучением
class ModelTrainer:
   def __init__(self, model_path="new_lstm_model.pth"):
      self.model_path = model_path
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.best_loss = float('inf')
      self.features = ["body", "upper_shadow", "SMA_20", "RSI_14"]
      self.sequence_length = 60

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

   def initialize_model(self, input_size=4, hidden_size=64, output_size=3):  # output_size=3
      model = LSTMModel(input_size, hidden_size, output_size).to(self.device)

      if os.path.exists(self.model_path):
         model.load_state_dict(torch.load(self.model_path, map_location=self.device))
         log_step(f"Загружена существующая модель из {self.model_path}")
      else:
         log_step("Инициализирована новая модель")

      return model

   def load_and_preprocess_data(self, file_path):
      """Загрузка и предварительная обработка данных из файла"""
      try:
         data = pd.read_csv(file_path)
         log_step(f"Данные успешно загружены. Размер: {data.shape[0]} строк, {data.shape[1]} столбцов")

         # Разметка паттернов
         data = self.detect_pattern(data)

         # Создание признаков
         data = self.create_features(data)

         return data
      except Exception as e:
         log_step(f"Ошибка загрузки данных: {str(e)}")
         raise

   def detect_pattern(self, raw_data: pd.DataFrame) -> pd.DataFrame:
      """Разметка паттернов в данных с использованием онлайн-детектора"""
      detector_bullish = Triangle(raw_data, trend_type=TrendType.BULLISH)
      detector_bearish = Triangle(raw_data, trend_type=TrendType.BEARISH)
      patterns = []
      log_step(f"Обработка {len(raw_data)} свечей для поиска паттернов...")

      for idx, row in raw_data.iterrows():
         # Обновляем индексы детекторов
         detector_bullish.current_index = idx
         detector_bearish.current_index = idx

         # Проверяем паттерны через единый вызов find_pattern
         is_bullish = detector_bullish.find_pattern(
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"]
         )

         is_bearish = detector_bearish.find_pattern(
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"]
         )

         # Сохраняем результат
         if is_bullish and not is_bearish:  # Приоритет бычьим паттернам
            patterns.append(1)
         elif is_bearish and not is_bullish:
            patterns.append(2)
         else:
            patterns.append(0)

      # Анализ и логирование результатов
      raw_data["pattern_label"] = patterns
      total_patterns = sum(1 for x in patterns if x > 0)
      log_step(f"Разметка завершена. Всего найдено {total_patterns} паттернов")
      log_step(f"  - Бычьих: {patterns.count(1)} ({patterns.count(1) / len(patterns):.2%})")
      log_step(f"  - Медвежьих: {patterns.count(2)} ({patterns.count(2) / len(patterns):.2%})")

      return raw_data

   def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
      """Создание признаков для модели"""
      log_step("Вычисление базовых признаков свечей...")
      data["body"] = (data["close"] - data["open"]).astype(np.float32)
      data["upper_shadow"] = (data["high"] - data[["open", "close"]].max(axis=1)).astype(np.float32)

      log_step("Вычисление индикаторов...")
      data["SMA_20"] = IndicatorManager.SMA_series(data['close'].tolist(), 20).astype(np.float32)
      data["EMA_20"] = IndicatorManager.EMA_series(data['close'].tolist(), 20).astype(np.float32)
      data["SMA_50"] = IndicatorManager.SMA_series(data['close'].tolist(), 50).astype(np.float32)
      data["EMA_50"] = IndicatorManager.EMA_series(data['close'].tolist(), 50).astype(np.float32)
      data["RSI_14"] = IndicatorManager.RSI_series(data['close'].tolist(), 14).astype(np.float32)

      cleaned_data = data.dropna()
      log_step(f"Признаки созданы. Удалено {len(data) - len(cleaned_data)} строк с NaN")
      return cleaned_data

   def prepare_sequences(self, data: pd.DataFrame):
      """Подготовка последовательностей с двойной проверкой типов"""
      X, y = [], []
      for i in range(len(data) - self.sequence_length):
         seq_features = data.iloc[i:i + self.sequence_length][self.features]
         # Двойное преобразование: сначала в numpy float32, затем в torch float32
         seq_features = seq_features.astype(np.float32).values
         target = int(data.iloc[i + self.sequence_length]["pattern_label"])
         X.append(seq_features)
         y.append(target)

      # Явное создание тензоров с проверкой типов
      X_tensor = torch.as_tensor(np.array(X), dtype=torch.float32)
      y_tensor = torch.as_tensor(np.array(y), dtype=torch.long)

      # Финалная проверка
      assert X_tensor.dtype == torch.float32, "X должен быть float32"
      assert y_tensor.dtype == torch.long, "y должен быть long"

      return X_tensor, y_tensor

   def create_dataloaders(self, X, y, val_split=0.2):
      """Создание DataLoader'ов с проверкой типов"""
      # Разделение данных
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



   def find_data_files(self):
      global PATH_DATA
      """Поиск файлов с данными"""
      data_files = glob.glob(PATH_DATA)
      if not data_files:
         raise FileNotFoundError(f"Не найдено файлов по шаблону: {PATH_DATA}")

      log_step(f"Найдено {len(data_files)} файлов для обработки:")
      for file in data_files:
         log_step(f" - {file}")

      return data_files

   def calculate_class_weights(self, data):
         """Автоматический расчет весов классов с обработкой нулевых значений"""
         class_counts = data['pattern_label'].value_counts().sort_index()

         # Добавляем небольшое значение чтобы избежать деления на 0
         class_counts = class_counts + 1

         # Увеличим вес миноритарных классов
         class_weights = torch.tensor([
            1.0,  # N
            class_counts[0] / class_counts[1] * 2,  # B
            class_counts[0] / class_counts[2] * 2  # M
         ], dtype=torch.float32).to(self.device)

         # weights = 1. / class_counts
         # weights = weights / weights.sum() * len(class_counts)
         # class_weights = torch.tensor(weights.values, dtype=torch.float32).to(self.device)

         return class_weights

   def train_model(self, model, train_loader, val_loader, class_weights, epochs=50):
         """Обучение модели со всеми исправлениями"""
         model = model.to(self.device)
         criterion = nn.CrossEntropyLoss(weight=class_weights)
         optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
         scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

         best_loss = float('inf')
         patience = 0

         for epoch in range(epochs):
            model.train()
            total_loss = 0
            all_preds = []
            all_labels = []

            for batch_x, batch_y in train_loader:
               # Гарантируем правильные типы и устройство
               batch_x = batch_x.float().to(self.device)
               batch_y = batch_y.long().to(self.device)

               optimizer.zero_grad()
               outputs = model(batch_x)
               loss = criterion(outputs, batch_y)
               loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
               optimizer.step()

               total_loss += loss.item()
               _, preds = torch.max(outputs, 1)
               all_preds.extend(preds.cpu().numpy())
               all_labels.extend(batch_y.cpu().numpy())

            # Валидация
            val_loss, val_acc, val_report = self.validate_model(model, val_loader, criterion)
            scheduler.step(val_loss)

            # Логирование
            log_step(f"Epoch {epoch + 1}/{epochs} | "
                     f"Train Loss: {total_loss / len(train_loader):.4f} | "
                     f"Val Loss: {val_loss:.4f} | "
                     f"LR: {optimizer.param_groups[0]['lr']:.2e} | ")
#                    f"Val Acc: [N: {val_class_acc[0]:.2%} B: {val_class_acc[1]:.2%} M: {val_class_acc[2]:.2%}]")

            # Early stopping
            if val_loss < best_loss:
               best_loss = val_loss
               patience = 0
               torch.save(model.state_dict(), self.model_path)
               log_step(f"Модель сохранена (Val Loss: {val_loss:.4f})")
            else:
               patience += 1
               if patience >= 5:
                  log_step("Early stopping: нет улучшений 5 эпох подряд")
                  break

         return model

   def validate_model(self, model, val_loader, criterion):
      """Расширенная валидация с подробными метриками"""
      model.eval()
      total_loss = 0
      all_preds = []
      all_labels = []

      with torch.no_grad():
         for batch_x, batch_y in val_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.long().to(self.device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

      # Подробный отчет по классам
      report = classification_report(
         all_labels, all_preds,
         target_names=['N', 'B', 'M'],
         output_dict=True,
         zero_division=0
      )

      # Логирование распределения предсказаний
      unique, counts = np.unique(all_preds, return_counts=True)
      pred_dist = dict(zip(['N', 'B', 'M'], counts))
      log_step(f"Распределение предсказаний: {pred_dist}")

      return (total_loss / len(val_loader),
              report['accuracy'],
              report)

   def train_on_all_data(self, initial_epochs=50, incremental_epochs=10):
      """Полный цикл обучения с улучшенной обработкой"""
      try:
         data_files = self.find_data_files()
         if not data_files:
            log_step("Не найдено файлов для обучения")
            return None

         # Первичное обучение
         if not os.path.exists(self.model_path):
            log_step("Инициализация нового обучения...")
            initial_data = self.load_and_preprocess_data(data_files[0])

            # Расчет весов классов
            class_weights = self.calculate_class_weights(initial_data)
            log_step(f"Веса классов: {class_weights.cpu().numpy()}")

            X, y = self.prepare_sequences(initial_data)
            train_loader, val_loader = self.create_dataloaders(X, y)

            model = self.initialize_model()
            model = self.train_model(model, train_loader, val_loader,
                                     class_weights, initial_epochs)
            remaining_files = data_files[1:]
         else:
            log_step("Загрузка существующей модели...")
            model = self.initialize_model()
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

               model = self.train_model(
                  model, train_loader, val_loader,
                  class_weights, incremental_epochs
               )

            except Exception as e:
               log_step(f"Ошибка при обработке {file}: {str(e)}")
               continue

         log_step("Обучение завершено успешно!")
         return model

      except Exception as e:
         log_step(f"Критическая ошибка: {str(e)}")
         import traceback
         log_step(traceback.format_exc())
         return None

   def test_model(self, test_file_path):
      """Тестирование модели на новых данных без обучения"""
      try:
         # 1. Загрузка модели
         model = self.initialize_model()
         if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель {self.model_path} не найдена!")

         # 2. Загрузка тестовых данных
         test_data = self.load_and_preprocess_data(test_file_path)
         X_test, y_test = self.prepare_sequences(test_data)

         # 3. Подготовка DataLoader
         test_tensor = torch.FloatTensor(X_test).to(self.device)
         test_dataset = TensorDataset(test_tensor)
         test_loader = DataLoader(test_dataset, batch_size=32)

         # 4. Прогнозирование
         model.eval()
         predictions = []
         with torch.no_grad():
            for batch in test_loader:
               outputs = model(batch[0])
               _, predicted = torch.max(outputs.data, 1)
               predictions.extend(predicted.cpu().numpy())

         # 5. Анализ результатов
         if 'pattern_label' in test_data.columns:
            y_true = test_data['pattern_label'].values[self.sequence_length:]
            accuracy = sum(p == t for p, t in zip(predictions, y_true)) / len(y_true)
            log_step(f"Точность на тестовых данных: {accuracy:.2%}")

            # Матрица ошибок
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, predictions)
            log_step("Матрица ошибок:\n" + str(cm))

         return predictions

      except Exception as e:
         log_step(f"Ошибка тестирования: {str(e)}")
         return None


# Основной блок выполнения
if __name__ == "__main__":
   PATH_DATA = "data/train/BTCUSDT/*.csv"
   TEST_DATA = "data/test/sintetic/output.csv"
   trainer = ModelTrainer()


   # Режим работы (train или test)
   mode = "test"  # Меняйте на "train" для обучения

   if mode == "train":
      # Обычное обучение
      trainer.train_on_all_data(
         initial_epochs=50,
         incremental_epochs=10
      )
   elif mode == "test":
      # Тестирование на конкретном файле
      test_predictions = trainer.test_model(TEST_DATA)

      # Визуализация результатов
      if test_predictions is not None:
         test_data = pd.read_csv(TEST_DATA)
         test_data = test_data.iloc[trainer.sequence_length:]  # Обрезаем начало

         import matplotlib.pyplot as plt

         plt.figure(figsize=(15, 6))
         plt.plot(test_data['close'], label='Цена')

         # Отметки бычьих паттернов
         bullish = [i for i, p in enumerate(test_predictions) if p == 1]
         plt.scatter(bullish, test_data.iloc[bullish]['close'],
                     color='green', label='Бычьи предсказания')

         # Отметки медвежьих паттернов
         bearish = [i for i, p in enumerate(test_predictions) if p == 2]
         plt.scatter(bearish, test_data.iloc[bearish]['close'],
                     color='red', label='Медвежьи предсказания')

         plt.legend()
         plt.title("Результаты тестирования модели")
         plt.show()