"""
Smart outfit recommendation service
Combines AI analysis, weather data, and user preferences
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RecommendationService:
    """
    Advanced outfit recommendation engine
    Uses rule-based + ML approach for personalized suggestions
    """
    
    def __init__(self, ai_service, weather_service, cache_service):
        self.ai_service = ai_service
        self.weather_service = weather_service
        self.cache_service = cache_service
        
        # Style rules for different occasions
        self.occasion_rules = {
            "casual": {
                "top": ["tshirt", "shirt", "sweater", "hoodie"],
                "bottom": ["jeans", "shorts", "casual_pants"],
                "shoes": ["sneakers", "loafers", "sandals"],
                "formality": 0.2
            },
            "business": {
                "top": ["shirt", "blouse", "blazer"],
                "bottom": ["dress_pants", "skirt", "formal_pants"],
                "shoes": ["dress_shoes", "heels", "loafers"],
                "formality": 0.8
            },
            "date": {
                "top": ["shirt", "blouse", "sweater", "dress"],
                "bottom": ["jeans", "skirt", "dress_pants"],
                "shoes": ["dress_shoes", "heels", "boots"],
                "formality": 0.6
            },
            "sport": {
                "top": ["tshirt", "tank_top", "sports_bra"],
                "bottom": ["shorts", "leggings", "track_pants"],
                "shoes": ["sneakers", "running_shoes"],
                "formality": 0.1
            },
            "party": {
                "top": ["dress", "shirt", "blouse"],
                "bottom": ["skirt", "dress_pants", "jeans"],
                "shoes": ["heels", "dress_shoes", "boots"],
                "formality": 0.7
            }
        }
        
        # Weather-based recommendations
        self.weather_rules = {
            "cold": {  # < 10°C
                "layers": ["coat", "jacket", "sweater"],
                "avoid": ["shorts", "sandals", "tank_top"],
                "accessories": ["scarf", "gloves", "beanie"]
            },
            "cool": {  # 10-18°C
                "layers": ["light_jacket", "cardigan", "blazer"],
                "suitable": ["jeans", "long_sleeve"],
                "optional": ["scarf"]
            },
            "warm": {  # 18-25°C
                "suitable": ["tshirt", "shirt", "light_pants"],
                "optional": ["light_cardigan"],
                "avoid": ["heavy_coat"]
            },
            "hot": {  # > 25°C
                "suitable": ["tshirt", "shorts", "dress", "sandals"],
                "avoid": ["jacket", "sweater", "boots"],
                "accessories": ["sunglasses", "hat"]
            },
            "rainy": {
                "required": ["raincoat", "waterproof_jacket"],
                "avoid": ["suede", "canvas_shoes"],
                "accessories": ["umbrella"]
            }
        }
        
        # Color harmony rules
        self.color_harmony = {
            "monochromatic": ["black-gray-white", "navy-blue-light_blue"],
            "complementary": ["blue-orange", "red-green", "yellow-purple"],
            "analogous": ["blue-green-teal", "red-orange-yellow"],
            "neutral_safe": ["black", "white", "gray", "navy", "beige"]
        }
    
    async def get_outfit_recommendations(
        self,
        user_id: str,
        occasion: str = "casual",
        weather_override: Optional[Dict] = None,
        num_outfits: int = 3
    ) -> List[Dict]:
        """
        Generate smart outfit recommendations
        """
        try:
            # Get user's wardrobe
            wardrobe = await self._get_user_wardrobe(user_id)
            if not wardrobe:
                return self._get_default_recommendations()
            
            # Get weather data
            weather = weather_override or await self._get_weather_for_user(user_id)
            
            # Get user preferences
            preferences = await self._get_user_preferences(user_id)
            
            # Generate outfit combinations
            outfits = await self._generate_outfits(
                wardrobe, occasion, weather, preferences, num_outfits
            )
            
            # Score and rank outfits
            scored_outfits = await self._score_outfits(outfits, occasion, weather)
            
            # Add styling tips
            final_recommendations = self._add_styling_tips(scored_outfits, weather)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return self._get_default_recommendations()
    
    async def _get_user_wardrobe(self, user_id: str) -> List[Dict]:
        """Get user's clothing items from database"""
        # This would connect to actual database
        # For now, return mock data
        return [
            {"id": 1, "type": "top", "subtype": "tshirt", "color": "white", "style": "casual"},
            {"id": 2, "type": "top", "subtype": "shirt", "color": "blue", "style": "business"},
            {"id": 3, "type": "bottom", "subtype": "jeans", "color": "blue", "style": "casual"},
            {"id": 4, "type": "bottom", "subtype": "dress_pants", "color": "black", "style": "business"},
            {"id": 5, "type": "shoes", "subtype": "sneakers", "color": "white", "style": "casual"},
            {"id": 6, "type": "outerwear", "subtype": "jacket", "color": "black", "style": "casual"}
        ]
    
    async def _get_weather_for_user(self, user_id: str) -> Dict:
        """Get weather for user's location"""
        # Default to Seoul for now
        location = "Seoul"
        weather = await self.weather_service.get_current_weather(location)
        return weather
    
    async def _get_user_preferences(self, user_id: str) -> Dict:
        """Get user style preferences"""
        # This would fetch from database
        return {
            "favorite_colors": ["blue", "black", "white"],
            "avoided_colors": ["yellow", "orange"],
            "style_preference": "casual_smart",
            "fit_preference": "regular"
        }
    
    async def _generate_outfits(
        self,
        wardrobe: List[Dict],
        occasion: str,
        weather: Dict,
        preferences: Dict,
        num_outfits: int
    ) -> List[Dict]:
        """Generate outfit combinations using smart rules"""
        outfits = []
        occasion_rule = self.occasion_rules.get(occasion, self.occasion_rules["casual"])
        
        # Group items by type
        grouped = {
            "top": [item for item in wardrobe if item["type"] == "top"],
            "bottom": [item for item in wardrobe if item["type"] == "bottom"],
            "shoes": [item for item in wardrobe if item["type"] == "shoes"],
            "outerwear": [item for item in wardrobe if item["type"] == "outerwear"]
        }
        
        # Generate combinations
        for _ in range(num_outfits * 2):  # Generate extra to filter later
            outfit = {}
            
            # Select top
            if grouped["top"]:
                suitable_tops = [
                    t for t in grouped["top"]
                    if self._is_suitable_for_weather(t, weather)
                ]
                if suitable_tops:
                    outfit["top"] = random.choice(suitable_tops)
            
            # Select bottom
            if grouped["bottom"]:
                suitable_bottoms = [
                    b for b in grouped["bottom"]
                    if self._is_suitable_for_weather(b, weather)
                ]
                if suitable_bottoms:
                    outfit["bottom"] = random.choice(suitable_bottoms)
            
            # Select shoes
            if grouped["shoes"]:
                outfit["shoes"] = random.choice(grouped["shoes"])
            
            # Add outerwear if needed
            if self._needs_outerwear(weather) and grouped["outerwear"]:
                outfit["outerwear"] = random.choice(grouped["outerwear"])
            
            if outfit:
                outfits.append(outfit)
        
        # Remove duplicates and return best combinations
        unique_outfits = self._remove_duplicate_outfits(outfits)
        return unique_outfits[:num_outfits]
    
    def _is_suitable_for_weather(self, item: Dict, weather: Dict) -> bool:
        """Check if item is suitable for weather conditions"""
        temp = weather.get("temperature", 20)
        conditions = weather.get("conditions", "clear")
        
        # Temperature-based filtering
        if temp < 10:  # Cold
            if item["subtype"] in ["shorts", "tank_top", "sandals"]:
                return False
        elif temp > 25:  # Hot
            if item["subtype"] in ["sweater", "jacket", "boots"]:
                return False
        
        # Rain filtering
        if "rain" in conditions.lower():
            if item["subtype"] in ["canvas_shoes", "suede"]:
                return False
        
        return True
    
    def _needs_outerwear(self, weather: Dict) -> bool:
        """Determine if outerwear is needed"""
        temp = weather.get("temperature", 20)
        wind_speed = weather.get("wind_speed", 0)
        
        return temp < 18 or wind_speed > 20 or "rain" in weather.get("conditions", "").lower()
    
    async def _score_outfits(
        self,
        outfits: List[Dict],
        occasion: str,
        weather: Dict
    ) -> List[Dict]:
        """Score outfits based on multiple factors"""
        scored = []
        
        for outfit in outfits:
            score = 0.0
            
            # Occasion appropriateness (0-40 points)
            occasion_score = self._calculate_occasion_score(outfit, occasion)
            score += occasion_score * 40
            
            # Weather suitability (0-30 points)
            weather_score = self._calculate_weather_score(outfit, weather)
            score += weather_score * 30
            
            # Color harmony (0-20 points)
            color_score = self._calculate_color_harmony(outfit)
            score += color_score * 20
            
            # Style consistency (0-10 points)
            style_score = self._calculate_style_consistency(outfit)
            score += style_score * 10
            
            outfit["score"] = score
            outfit["breakdown"] = {
                "occasion": occasion_score,
                "weather": weather_score,
                "color": color_score,
                "style": style_score
            }
            
            scored.append(outfit)
        
        # Sort by score
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        return scored
    
    def _calculate_occasion_score(self, outfit: Dict, occasion: str) -> float:
        """Calculate how appropriate outfit is for occasion"""
        rule = self.occasion_rules.get(occasion, self.occasion_rules["casual"])
        score = 0.0
        items_count = 0
        
        for item_type, item in outfit.items():
            if item_type in ["top", "bottom", "shoes"]:
                items_count += 1
                if item.get("style") in rule.get(item_type, []):
                    score += 1.0
                elif item.get("subtype") in rule.get(item_type, []):
                    score += 0.8
                else:
                    score += 0.3
        
        return score / items_count if items_count > 0 else 0
    
    def _calculate_weather_score(self, outfit: Dict, weather: Dict) -> float:
        """Calculate weather appropriateness"""
        temp = weather.get("temperature", 20)
        conditions = weather.get("conditions", "clear")
        
        score = 1.0
        
        # Temperature scoring
        if temp < 10:  # Cold
            if "outerwear" not in outfit:
                score -= 0.3
            if outfit.get("bottom", {}).get("subtype") == "shorts":
                score -= 0.5
        elif temp > 25:  # Hot
            if "outerwear" in outfit:
                score -= 0.3
            if outfit.get("top", {}).get("subtype") in ["sweater", "hoodie"]:
                score -= 0.4
        
        # Rain scoring
        if "rain" in conditions.lower():
            if outfit.get("shoes", {}).get("subtype") in ["canvas_shoes", "suede"]:
                score -= 0.3
        
        return max(0, score)
    
    def _calculate_color_harmony(self, outfit: Dict) -> float:
        """Calculate color harmony score"""
        colors = []
        for item_type, item in outfit.items():
            if isinstance(item, dict) and "color" in item:
                colors.append(item["color"])
        
        if len(colors) < 2:
            return 0.8  # Single color or no colors
        
        # Check for safe combinations
        if all(c in self.color_harmony["neutral_safe"] for c in colors):
            return 1.0  # All neutrals always work
        
        # Check for known good combinations
        color_pair = f"{colors[0]}-{colors[1]}"
        for harmony_type, combinations in self.color_harmony.items():
            if harmony_type != "neutral_safe":
                for combo in combinations:
                    if color_pair in combo or f"{colors[1]}-{colors[0]}" in combo:
                        return 0.9
        
        # Default moderate score for unknown combinations
        return 0.6
    
    def _calculate_style_consistency(self, outfit: Dict) -> float:
        """Calculate style consistency across items"""
        styles = []
        for item_type, item in outfit.items():
            if isinstance(item, dict) and "style" in item:
                styles.append(item["style"])
        
        if not styles:
            return 0.5
        
        # All same style = perfect consistency
        if len(set(styles)) == 1:
            return 1.0
        
        # Mix of casual and smart casual is ok
        if set(styles).issubset({"casual", "casual_smart", "smart_casual"}):
            return 0.8
        
        # Business and formal mix
        if set(styles).issubset({"business", "formal", "business_casual"}):
            return 0.8
        
        # Otherwise lower score
        return 0.5
    
    def _remove_duplicate_outfits(self, outfits: List[Dict]) -> List[Dict]:
        """Remove duplicate outfit combinations"""
        unique = []
        seen = set()
        
        for outfit in outfits:
            # Create a unique key for the outfit
            key = tuple(sorted([
                (k, v.get("id")) for k, v in outfit.items()
                if isinstance(v, dict) and "id" in v
            ]))
            
            if key not in seen:
                seen.add(key)
                unique.append(outfit)
        
        return unique
    
    def _add_styling_tips(self, outfits: List[Dict], weather: Dict) -> List[Dict]:
        """Add personalized styling tips to each outfit"""
        enhanced = []
        
        for idx, outfit in enumerate(outfits):
            tips = []
            
            # Weather-based tips
            temp = weather.get("temperature", 20)
            if temp < 15:
                tips.append("레이어링을 활용해 따뜻하고 스타일리시하게")
            elif temp > 25:
                tips.append("가벼운 소재로 시원하게 연출")
            
            # Occasion-based tips
            if outfit.get("score", 0) > 80:
                tips.append("완벽한 조합이에요!")
            
            # Color tips
            if outfit.get("breakdown", {}).get("color", 0) > 0.8:
                tips.append("컬러 매치가 훌륭해요")
            
            # Accessory suggestions
            if weather.get("conditions") == "sunny":
                tips.append("선글라스를 추가하면 멋스러워요")
            
            outfit["styling_tips"] = tips
            outfit["ranking"] = idx + 1
            outfit["confidence"] = min(95, outfit.get("score", 0) + random.randint(0, 10))
            
            enhanced.append(outfit)
        
        return enhanced
    
    def _get_default_recommendations(self) -> List[Dict]:
        """Return default recommendations when data is unavailable"""
        return [
            {
                "ranking": 1,
                "outfit": {
                    "top": {"type": "top", "description": "흰색 티셔츠"},
                    "bottom": {"type": "bottom", "description": "청바지"},
                    "shoes": {"type": "shoes", "description": "스니커즈"}
                },
                "styling_tips": ["무난하고 편안한 데일리룩"],
                "confidence": 85
            }
        ]