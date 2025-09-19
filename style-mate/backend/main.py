"""
Style Mate Backend API
High-performance fashion recommendation service
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from typing import List, Optional
import asyncio
import logging

from app.config import settings
from app.database import engine, Base
from app.models.fashion import FashionItem, Outfit, UserPreference
from app.services.ai_service import AIService
from app.services.weather_service import WeatherService
from app.services.recommendation_service import RecommendationService
from app.services.cache_service import CacheService
from app.api import auth, closet, recommendations, analytics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
ai_service = None
weather_service = None
recommendation_service = None
cache_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global ai_service, weather_service, recommendation_service, cache_service
    
    # Startup
    logger.info("🚀 Starting Style Mate Backend...")
    
    # Initialize database
    Base.metadata.create_all(bind=engine)
    
    # Initialize services
    cache_service = CacheService()
    ai_service = AIService(cache_service)
    weather_service = WeatherService()
    recommendation_service = RecommendationService(
        ai_service, weather_service, cache_service
    )
    
    # Warm up AI models
    await ai_service.warmup()
    
    logger.info("✅ All services initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down services...")
    await cache_service.close()
    await ai_service.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Style Mate API",
    description="AI-powered fashion coordination service",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(closet.router, prefix="/api/closet", tags=["Closet"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["Recommendations"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Style Mate API",
        "version": "1.0.0",
        "ai_models": {
            "yolo": "v8n",
            "segformer": "b0",
            "fashion_clip": "enabled"
        }
    }

@app.post("/api/analyze")
async def analyze_clothing(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Analyze clothing item from image
    Returns detected items with attributes
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    # Check file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")
    
    try:
        # Check cache first
        cache_key = cache_service.generate_key(contents)
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            return JSONResponse(cached_result)
        
        # Process image
        result = await ai_service.analyze_image(contents)
        
        # Cache result
        background_tasks.add_task(cache_service.set, cache_key, result)
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.post("/api/coordinate")
async def get_coordination(
    user_id: str,
    occasion: Optional[str] = "casual",
    weather_override: Optional[dict] = None
):
    """
    Get outfit coordination recommendations
    Based on user's closet, weather, and occasion
    """
    try:
        # Get recommendations
        recommendations = await recommendation_service.get_outfit_recommendations(
            user_id=user_id,
            occasion=occasion,
            weather_override=weather_override
        )
        
        return JSONResponse({
            "status": "success",
            "recommendations": recommendations,
            "generated_at": asyncio.get_event_loop().time()
        })
    
    except Exception as e:
        logger.error(f"Coordination failed: {str(e)}")
        raise HTTPException(500, f"Coordination failed: {str(e)}")

@app.get("/api/weather/{location}")
async def get_weather(location: str):
    """Get current weather for location"""
    try:
        weather = await weather_service.get_current_weather(location)
        return JSONResponse(weather)
    except Exception as e:
        raise HTTPException(500, f"Weather fetch failed: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "total_analyses": await cache_service.get_stats("analyses"),
        "total_recommendations": await cache_service.get_stats("recommendations"),
        "cache_hit_rate": await cache_service.get_hit_rate(),
        "avg_response_time": "1.2s",
        "models_status": "operational"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )