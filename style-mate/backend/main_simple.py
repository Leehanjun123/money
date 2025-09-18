"""
Style Mate Backend API - Simplified Version
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
import random
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="Style Mate API",
    description="AI-powered fashion coordination service",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Style Mate API",
        "version": "1.0.0",
        "message": "백엔드 서버가 정상 작동 중입니다! 🚀"
    }

@app.post("/api/analyze")
async def analyze_clothing(file: UploadFile = File(...)):
    """
    Mock analyze clothing item from image
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "파일은 이미지여야 합니다")
    
    # Mock AI analysis result
    result = {
        "items": [
            {
                "category": random.choice(["top", "bottom", "dress", "outerwear"]),
                "confidence": random.uniform(0.85, 0.99),
                "bbox": [100, 100, 300, 400]
            }
        ],
        "attributes": [
            {
                "color": random.choice(["black", "white", "blue", "red", "gray"]),
                "pattern": random.choice(["solid", "striped", "patterned"]),
                "style": random.choice(["casual", "formal", "sporty"])
            }
        ],
        "processing_time": random.uniform(0.5, 2.0)
    }
    
    return JSONResponse(result)

@app.post("/api/coordinate")
async def get_coordination(
    user_id: str = "default_user",
    occasion: Optional[str] = "casual"
):
    """
    Get outfit coordination recommendations
    """
    recommendations = []
    
    for i in range(3):
        outfit = {
            "ranking": i + 1,
            "outfit": {
                "top": {
                    "type": random.choice(["shirt", "tshirt", "sweater"]),
                    "color": random.choice(["white", "black", "blue"]),
                    "description": f"상의 {i+1}"
                },
                "bottom": {
                    "type": random.choice(["jeans", "chinos", "shorts"]),
                    "color": random.choice(["blue", "black", "beige"]),
                    "description": f"하의 {i+1}"
                },
                "shoes": {
                    "type": random.choice(["sneakers", "loafers", "boots"]),
                    "color": random.choice(["white", "black", "brown"]),
                    "description": f"신발 {i+1}"
                }
            },
            "styling_tips": [
                f"{occasion} 스타일에 완벽해요!",
                "오늘 날씨에 딱 맞는 코디에요"
            ],
            "confidence": random.randint(85, 98)
        }
        recommendations.append(outfit)
    
    return JSONResponse({
        "status": "success",
        "recommendations": recommendations,
        "generated_at": datetime.now().isoformat()
    })

@app.get("/api/weather/{location}")
async def get_weather(location: str):
    """Get mock weather for location"""
    weather = {
        "temperature": random.randint(15, 25),
        "feels_like": random.randint(14, 26),
        "humidity": random.randint(40, 70),
        "conditions": random.choice(["Clear", "Cloudy", "Partly Cloudy"]),
        "description": "맑음",
        "wind_speed": random.randint(5, 15),
        "location": location,
        "country": "KR",
        "timestamp": datetime.now().isoformat()
    }
    return JSONResponse(weather)

@app.get("/api/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "total_analyses": random.randint(100, 500),
        "total_recommendations": random.randint(200, 1000),
        "cache_hit_rate": f"{random.randint(60, 90)}%",
        "avg_response_time": f"{random.uniform(0.8, 1.5):.1f}s",
        "models_status": "operational",
        "active_users": random.randint(10, 50)
    }

@app.post("/api/closet/items")
async def add_closet_item(file: UploadFile = File(...)):
    """Add item to closet"""
    return {
        "id": f"item_{random.randint(1000, 9999)}",
        "message": "아이템이 옷장에 추가되었습니다!",
        "category": "top",
        "added_at": datetime.now().isoformat()
    }

@app.get("/api/closet/items")
async def get_closet_items():
    """Get user's closet items"""
    items = []
    categories = ["top", "bottom", "shoes", "outerwear"]
    colors = ["black", "white", "blue", "gray", "beige"]
    
    for i in range(10):
        items.append({
            "id": f"item_{i}",
            "category": random.choice(categories),
            "color": random.choice(colors),
            "wear_count": random.randint(0, 20),
            "last_worn": f"{random.randint(1, 30)}일 전",
            "favorite": random.choice([True, False])
        })
    
    return {
        "items": items,
        "total": len(items)
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Style Mate Backend Server Starting...")
    print("="*50)
    print("\n📍 Server URL: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🔄 Interactive API: http://localhost:8000/redoc")
    print("\n✨ 서버가 시작되었습니다! 브라우저에서 확인해보세요.\n")
    
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )