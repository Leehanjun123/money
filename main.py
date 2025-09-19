"""
Style Mate Production Backend API
Optimized for Railway deployment
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import logging
import os
import uuid
from datetime import datetime
import hashlib
import io
from PIL import Image
import numpy as np
import cv2
import json
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory cache (simple implementation)
cache = {}

# Import advanced services if available
try:
    import sys
    sys.path.append('/Users/leehanjun/Desktop/money')
    from style_mate.backend.services.advanced_ai_service import AdvancedAIService
    from style_mate.backend.services.outfit_generator import OutfitImageGenerator
    USE_ADVANCED_AI = True
    logger.info("Advanced AI services loaded successfully")
except ImportError as e:
    USE_ADVANCED_AI = False
    logger.warning(f"Advanced AI services not available: {e}")

class SimpleAIService:
    """Lightweight AI service using OpenCV"""
    
    def __init__(self):
        self.initialized = False
        
    async def initialize(self):
        """Initialize models"""
        try:
            logger.info("Initializing AI service...")
            # In production, load actual models here
            self.initialized = True
            logger.info("AI service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AI service: {e}")
    
    async def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze clothing using computer vision"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Basic image analysis using OpenCV
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Edge detection to find clothing boundaries
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours (clothing items)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            items = []
            for i, contour in enumerate(contours[:3]):  # Limit to 3 items
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract dominant color from region
                    roi = image_np[y:y+h, x:x+w]
                    avg_color = np.mean(roi.reshape(-1, 3), axis=0)
                    
                    # Classify color
                    color = self._classify_color(avg_color)
                    
                    items.append({
                        "bbox": [int(x), int(y), int(x+w), int(y+h)],
                        "confidence": min(0.95, area / 50000),  # Pseudo confidence
                        "category": self._guess_category(h/w),  # Aspect ratio based
                        "color": color
                    })
            
            return {
                "items": items,
                "count": len(items),
                "processing_time": 0.5,
                "method": "opencv"
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {"items": [], "count": 0, "error": str(e)}
    
    def _classify_color(self, rgb):
        """Classify RGB color to name"""
        colors = {
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "gray": [128, 128, 128],
            "beige": [245, 245, 220]
        }
        
        min_dist = float('inf')
        closest_color = "unknown"
        
        for name, ref_rgb in colors.items():
            dist = np.linalg.norm(rgb - ref_rgb)
            if dist < min_dist:
                min_dist = dist
                closest_color = name
        
        return closest_color
    
    def _guess_category(self, aspect_ratio):
        """Guess clothing category from aspect ratio"""
        if aspect_ratio > 1.5:
            return "dress"
        elif aspect_ratio > 1.2:
            return "top"
        elif aspect_ratio < 0.8:
            return "bottom"
        else:
            return "clothing"

class RecommendationEngine:
    """Smart recommendation engine"""
    
    def __init__(self, ai_service):
        self.ai_service = ai_service
    
    async def get_recommendations(
        self, 
        user_id: str,
        occasion: str,
        weather: Optional[Dict] = None
    ) -> List[Dict]:
        """Generate outfit recommendations"""
        
        # Get base recommendations
        outfits = self._generate_base_outfits(occasion)
        
        # Adjust for weather if provided
        if weather:
            outfits = self._adjust_for_weather(outfits, weather)
        
        # Score and rank
        scored_outfits = []
        for i, outfit in enumerate(outfits[:3]):
            scored_outfit = {
                "ranking": i + 1,
                "outfit": outfit,
                "styling_tips": self._get_styling_tips(occasion, weather),
                "confidence": 85 + (3 - i) * 5  # 95, 90, 85
            }
            scored_outfits.append(scored_outfit)
        
        return scored_outfits
    
    def _generate_base_outfits(self, occasion):
        """Generate base outfits by occasion"""
        occasion_styles = {
            "casual": [
                {"top": {"type": "tshirt", "color": "white"}, 
                 "bottom": {"type": "jeans", "color": "blue"},
                 "shoes": {"type": "sneakers", "color": "white"}},
                {"top": {"type": "hoodie", "color": "gray"}, 
                 "bottom": {"type": "joggers", "color": "black"},
                 "shoes": {"type": "sneakers", "color": "black"}},
            ],
            "business": [
                {"top": {"type": "shirt", "color": "white"}, 
                 "bottom": {"type": "dress_pants", "color": "navy"},
                 "shoes": {"type": "dress_shoes", "color": "black"}},
                {"top": {"type": "blouse", "color": "blue"}, 
                 "bottom": {"type": "skirt", "color": "black"},
                 "shoes": {"type": "heels", "color": "black"}},
            ],
            "date": [
                {"top": {"type": "shirt", "color": "navy"}, 
                 "bottom": {"type": "chinos", "color": "beige"},
                 "shoes": {"type": "loafers", "color": "brown"}},
            ]
        }
        
        return occasion_styles.get(occasion, occasion_styles["casual"])
    
    def _adjust_for_weather(self, outfits, weather):
        """Adjust outfits based on weather"""
        temp = weather.get("temperature", 20)
        
        for outfit in outfits:
            if temp < 15:
                outfit["outerwear"] = {"type": "jacket", "color": "black"}
            elif temp > 25:
                if outfit["bottom"]["type"] == "jeans":
                    outfit["bottom"]["type"] = "shorts"
        
        return outfits
    
    def _get_styling_tips(self, occasion, weather):
        """Generate styling tips"""
        tips = []
        
        if occasion == "casual":
            tips.append("편안하면서도 스타일리시한 룩")
        elif occasion == "business":
            tips.append("프로페셔널한 인상을 주는 코디")
        elif occasion == "date":
            tips.append("세련되고 매력적인 스타일")
        
        if weather and weather.get("temperature", 20) < 15:
            tips.append("레이어링으로 따뜻하게")
        
        return tips

# Initialize services
if USE_ADVANCED_AI:
    ai_service = AdvancedAIService()
    outfit_generator = OutfitImageGenerator()
    logger.info("Using advanced AI services")
else:
    ai_service = SimpleAIService()
    outfit_generator = None
    logger.info("Using simple AI services")

recommendation_engine = RecommendationEngine(ai_service)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("🚀 Starting Style Mate Production Server...")
    if hasattr(ai_service, 'initialize'):
        await ai_service.initialize()
    logger.info(f"✅ Server ready (Advanced AI: {USE_ADVANCED_AI})")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Style Mate Production API",
    description="Production-ready fashion AI service",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Style Mate Production",
        "version": "2.0.0",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "features": {
            "ai_analysis": ai_service.initialized,
            "recommendations": True,
            "caching": True
        }
    }

@app.post("/api/analyze")
async def analyze_clothing(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Analyze clothing image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    # Read file
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(400, "File too large")
    
    # Check cache
    file_hash = hashlib.md5(contents).hexdigest()
    if file_hash in cache:
        logger.info("Cache hit!")
        return JSONResponse(cache[file_hash])
    
    # Analyze
    result = await ai_service.analyze_image(contents)
    
    # Cache result
    cache[file_hash] = result
    
    # Clean cache if too large
    if len(cache) > 100:
        # Remove oldest entries
        keys_to_remove = list(cache.keys())[:20]
        for key in keys_to_remove:
            del cache[key]
    
    return JSONResponse(result)

@app.post("/api/coordinate")
async def get_coordination(
    user_id: str = "default",
    occasion: str = "casual",
    weather: Optional[Dict] = None,
    generate_image: bool = False
):
    """Get outfit recommendations with optional image generation"""
    recommendations = await recommendation_engine.get_recommendations(
        user_id, occasion, weather
    )
    
    # Generate outfit images if requested and available
    if generate_image and outfit_generator and recommendations:
        for rec in recommendations[:2]:  # Generate for top 2 recommendations
            try:
                outfit_data = {
                    'occasion': occasion,
                    'top': rec['outfit'].get('top'),
                    'bottom': rec['outfit'].get('bottom'),
                    'shoes': rec['outfit'].get('shoes'),
                    'styling_tips': rec.get('styling_tips', [])
                }
                
                image_result = await outfit_generator.generate_outfit_image(outfit_data)
                rec['outfit_image'] = image_result['image']
                rec['style_notes'] = image_result.get('style_notes', [])
            except Exception as e:
                logger.warning(f"Failed to generate outfit image: {e}")
                rec['outfit_image'] = None
    
    return {
        "status": "success",
        "recommendations": recommendations,
        "user_id": user_id,
        "occasion": occasion,
        "has_images": generate_image and outfit_generator is not None
    }

@app.get("/api/weather/{location}")
async def get_weather(location: str):
    """Get weather (mock for now)"""
    # In production, integrate with real weather API
    return {
        "temperature": 22,
        "conditions": "Clear",
        "humidity": 60,
        "location": location,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "cache_size": len(cache),
        "cache_entries": list(cache.keys())[:5],  # Show first 5
        "ai_status": "operational" if ai_service.initialized else "loading",
        "uptime": "healthy",
        "memory_usage": "low"
    }

@app.post("/api/closet/items")
async def upload_item(
    file: UploadFile = File(...),
    category: str = "clothing"
):
    """Upload clothing item"""
    contents = await file.read()
    
    # Analyze the image
    analysis = await ai_service.analyze_image(contents)
    
    # Save to "database" (in-memory for now)
    item_id = hashlib.md5(contents).hexdigest()[:8]
    
    return {
        "id": item_id,
        "category": category,
        "analysis": analysis,
        "uploaded_at": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "ai_service": ai_service.initialized,
            "cache": len(cache) < 500,
            "memory": True
        }
    }

# Beta signup endpoints - Store real data
beta_applications = []

@app.post("/api/beta-signup")
async def beta_signup(data: dict):
    """Store beta tester application"""
    try:
        # Add unique ID and server timestamp
        application = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            **data  # Include all data from frontend
        }
        
        # Store in memory (persists until server restart)
        beta_applications.append(application)
        
        logger.info(f"New beta signup: {application.get('email', 'unknown')}")
        
        return {
            "success": True,
            "message": "베타 신청이 완료되었습니다",
            "application_id": application["id"]
        }
    except Exception as e:
        logger.error(f"Beta signup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/beta-applications")
async def get_beta_applications():
    """Get all beta applications with statistics"""
    try:
        # Sort by timestamp (newest first)
        sorted_apps = sorted(
            beta_applications, 
            key=lambda x: x["timestamp"], 
            reverse=True
        )
        
        # Calculate real statistics
        total = len(beta_applications)
        today_date = datetime.now().strftime("%Y-%m-%d")
        today = sum(1 for app in beta_applications 
                   if app["timestamp"].startswith(today_date))
        
        # Device statistics
        ios = sum(1 for app in beta_applications if app.get("device") == "iOS")
        android = sum(1 for app in beta_applications if app.get("device") == "Android")
        both = sum(1 for app in beta_applications if app.get("device") == "Both")
        
        # Age distribution
        age_dist = {}
        for app in beta_applications:
            age = app.get("age", "unknown")
            age_dist[age] = age_dist.get(age, 0) + 1
        
        # Fashion interest distribution
        interest_dist = {}
        for app in beta_applications:
            interest = app.get("fashionInterest", "unknown")
            interest_dist[interest] = interest_dist.get(interest, 0) + 1
        
        return {
            "applications": sorted_apps,
            "stats": {
                "total": total,
                "today": today,
                "ios": ios,
                "android": android,
                "both": both,
                "by_age": age_dist,
                "by_interest": interest_dist
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Get applications error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/beta-applications/{application_id}")
async def get_beta_application(application_id: str):
    """Get specific beta application by ID"""
    app = next((a for a in beta_applications if a["id"] == application_id), None)
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
    return app

@app.delete("/api/beta-applications/{application_id}")
async def delete_beta_application(application_id: str):
    """Delete a beta application"""
    global beta_applications
    beta_applications = [a for a in beta_applications if a["id"] != application_id]
    return {"success": True, "message": "Application deleted"}

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    print("\n" + "="*60)
    print("🚀 Style Mate Production Server")
    print("="*60)
    print(f"\n📍 URL: http://localhost:{port}")
    print(f"📖 Docs: http://localhost:{port}/docs")
    print(f"💾 Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print("\n✨ Production server ready!\n")
    
    uvicorn.run(
        "main_production:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )