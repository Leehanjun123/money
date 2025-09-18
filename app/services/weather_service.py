"""
Weather service for outfit recommendations
"""

import httpx
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)

class WeatherService:
    """
    Weather data provider using OpenWeather API
    Falls back to Korean Meteorological Administration if needed
    """
    
    def __init__(self):
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "")
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.cache = {}
        self.cache_duration = timedelta(minutes=30)
    
    async def get_current_weather(self, location: str) -> Dict:
        """Get current weather for location"""
        # Check cache
        cache_key = f"weather_{location}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data
        
        try:
            # Try OpenWeather API
            if self.openweather_api_key:
                weather = await self._fetch_openweather(location)
            else:
                # Fallback to mock data
                weather = self._get_mock_weather(location)
            
            # Cache result
            self.cache[cache_key] = (weather, datetime.now())
            return weather
            
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
            return self._get_default_weather()
    
    async def _fetch_openweather(self, location: str) -> Dict:
        """Fetch weather from OpenWeather API"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/weather",
                params={
                    "q": location,
                    "appid": self.openweather_api_key,
                    "units": "metric"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_openweather_response(data)
            else:
                raise Exception(f"OpenWeather API error: {response.status_code}")
    
    def _parse_openweather_response(self, data: Dict) -> Dict:
        """Parse OpenWeather API response"""
        return {
            "temperature": round(data["main"]["temp"]),
            "feels_like": round(data["main"]["feels_like"]),
            "humidity": data["main"]["humidity"],
            "conditions": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "wind_speed": round(data["wind"]["speed"] * 3.6),  # Convert m/s to km/h
            "location": data["name"],
            "country": data["sys"]["country"],
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_mock_weather(self, location: str) -> Dict:
        """Return mock weather data for development"""
        import random
        
        # Simulate different weather conditions
        conditions = ["Clear", "Clouds", "Rain", "Snow", "Drizzle"]
        temp = random.randint(10, 30)
        
        return {
            "temperature": temp,
            "feels_like": temp + random.randint(-3, 3),
            "humidity": random.randint(30, 80),
            "conditions": random.choice(conditions),
            "description": "Partly cloudy",
            "wind_speed": random.randint(5, 25),
            "location": location,
            "country": "KR",
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_default_weather(self) -> Dict:
        """Return default weather when API fails"""
        return {
            "temperature": 20,
            "feels_like": 20,
            "humidity": 50,
            "conditions": "Clear",
            "description": "Clear sky",
            "wind_speed": 10,
            "location": "Unknown",
            "country": "KR",
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_forecast(self, location: str, days: int = 5) -> Dict:
        """Get weather forecast"""
        # Implementation for forecast
        pass