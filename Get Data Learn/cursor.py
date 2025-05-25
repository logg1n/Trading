from datetime import datetime
from typing import Dict, List


class Cursor:
   def __init__(self, timeframe: str, date: str):
      self.limit: int = 1000
      self.timeframe = timeframe
      self.curs = self.datetime_to_milliseconds(date, "%d.%m.%Y")
      self.time_intervals:  Dict[str, int] = {
         "1": 1,
         "3": 3,
         "5": 5,
         "15": 15,
         "30": 30,
         "60": 60,
         "120": 120,
         "240": 240,
         "360": 360,
         "720": 720,
         "D": 1440,  # 1 день = 1440 минут
         "M": 43200,  # 1 месяц (30 дней) = 43200 минут
         "W": 10080  # 1 неделя (7 дней) = 10080 минут
      }

   def next_cursor(self) -> None:
      self.curs = (self.curs + (self.time_intervals.get(self.timeframe) * self.limit) * 60 * 1000)
            
   def datetime_to_milliseconds(self, date_str: str, date_format: str = "%d.%m.%Y") -> int | None:
      """
      Преобразует дату и время в миллисекунды.

      Формат даты и времени:
      %d - День (01-31)
      %m - Месяц (01-12)
      %Y - Год (0001-9999)
      %H - Часы (00-23)
      %M - Минуты (00-59)

      Пример: "23.02.2025 18:30" с форматом "%d.%m.%Y %H:%M"

      :param date_str: Дата и время в виде строки.
      :param date_format: Формат строки даты и времени.
      :return: Количество миллисекунд с начала эпохи.
      """
      dt = datetime.strptime(date_str, date_format)
      milliseconds = int(dt.timestamp() * 1000)
      return milliseconds

   def milliseconds_to_datetime(self, milliseconds: int, date_format: str = "%d.%m.%Y %H:%M") -> str:
         """
         Преобразует миллисекунды в дату и время.

         Формат даты и времени:
         %d - День (01-31)
         %m - Месяц (01-12)
         %Y - Год (0001-9999)
         %H - Часы (00-23)
         %M - Минуты (00-59)

         Пример: 1745550600000 миллисекунд -> "23.02.2025 18:30" с форматом "%d.%m.%Y %H:%M"

         :param milliseconds: Количество миллисекунд с начала эпохи.
         :param date_format: Формат строки даты и времени.
         :return: Дата и время в виде строки.
         """
         dt = datetime.fromtimestamp(milliseconds / 1000.0)
         date_str = dt.strftime(date_format)
         return date_str
