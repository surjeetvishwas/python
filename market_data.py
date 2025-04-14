# -*- coding: utf-8 -*-
"""market data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QUvLZiONQDpvC6pKMWAvzhKcbs86UWTi
"""

import requests

API_KEY = "r34QULM8Zfr9QtblDCQfsiZC2vfXW0H5"
API_URL = "https://api.exchangerate-api.com/v4/latest/USD"

class MarketData:
    """Obtiene la tasa libre de riesgo y tipo de cambio desde la API."""

    @staticmethod
    def get_risk_free_rate():
        try:
            response = requests.get(API_URL)
            data = response.json()
            return data.get("rates", {}).get("MXN", 0.05)  # Default 0.05 si falla
        except Exception as e:
            print(f"Error obteniendo tasa libre de riesgo: {e}")
            return 0.05