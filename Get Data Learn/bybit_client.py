import logging
import os
import time
from typing import List

import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()


class BybitTradingClient:
   def __init__(self):
      self.limit = 1000
      self.__api_key = os.getenv("DEMO_API_KEY")
      self.__secret_key = os.getenv("DEMO_SECRET_KEY")
      self._endpoint = os.getenv("ENDPOINT")

      self.session = HTTP(api_key=self.__api_key, api_secret=self.__secret_key)
      self.session.endpoint = self._endpoint
      self.logger = logging.getLogger("Bybit.Trading")

   def get_historical_data(
           self,
           symbol: str,
           timeframe: str,
           start: int ,
   ) -> pd.DataFrame:
      """Получение и обработка исторических данных
      args: start -> The start timestamp (ms)
      args: stop ->The stop timestamp (ms)
      args: interval -> 1,3,5,15,30,60,120,240,360,720,D,M,W
      """
      try:
         response = self.session.get_kline(
            category="linear",
            symbol=symbol,
            interval=timeframe,
            start=start,
            limit=self.limit
         )

         if response["retCode"] != 0:
            self.logger.error(f"API error: {response['retMsg']}")
            return pd.DataFrame()

         # Получаем колонки из ответа API
         raw_data: List[List] = response["result"]["list"]
         datas = [l[1:-1] for l in raw_data]

         # Определяем актуальные имена колонок
         columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
         ]

         df = pd.DataFrame(datas, columns=columns)

         # Конвертация типов
         numeric_cols = ["open", "high", "low", "close", "volume"]
         df[numeric_cols] = df[numeric_cols].astype(float)
         df[numeric_cols[-1]] = df[numeric_cols[-1]].astype(int)

         return df

      except Exception as e:
         self.logger.error(f"Data processing error for {symbol}: {str(e)}")
         return pd.DataFrame()
